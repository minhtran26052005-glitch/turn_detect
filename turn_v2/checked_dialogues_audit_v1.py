"""Independent raw-chunk, cache, split and protected-input audit of added rows."""
from collections import defaultdict,Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
from .common import normalize
from .monolingual_data import encode_tail
from .checked_dialogues_prepare_v1 import OUT,PARENT,ROOT,SOURCE,sha,save


def run():
    from transformers import AutoTokenizer
    receipt=json.loads((OUT/'dataset_manifest.json').read_text());cache=OUT/'cache'
    for n,meta in receipt['files'].items():assert sha(cache/(n+'.npy'))==meta['sha256']
    assert sha(OUT/'train_vi_en.jsonl')==receipt['train_sha256']
    arrays={n:np.load(cache/(n+'.npy'),mmap_mode='r') for n in receipt['files']}
    for n in ['ids','lengths','labels','logits','language_ids','source_row_indices']:
        assert np.array_equal(arrays[n][:200000],np.load(PARENT/'cache'/(n+'.npy'),mmap_mode='r'))
    assert np.all(arrays['teacher_mask'][:200000]==1) and np.all(arrays['teacher_mask'][200000:]==0)
    raw=defaultdict(list);family_splits=defaultdict(set)
    def fam(name,identity):return name+':'+re.sub(r'\.s\d+\.(?:vi|en)$','',identity)
    for name,digest in receipt['raw_sources_sha256'].items():
        p=SOURCE/'raw'/name;assert sha(p)==digest
        for line in p.open(encoding='utf-8'):
            r=json.loads(line);split=r.get('split','train');family_splits[fam(name,r['dialogue_id'])].add(split)
            if split=='train':
                key=(name,r['dialogue_id'],r['turn_idx'],r['speaker'],r['src_lang'])
                raw[key].append({k:r[k] for k in ['source_raw','chunk_in_turn','keep','starts_mid_sentence','ends_mid_sentence','checks','drop_reasons']})
    assert all(len(v)==1 for v in family_splits.values())
    paragraphs={}
    db=sqlite3.connect((OUT/'additional_paragraphs.sqlite').as_uri()+'?mode=ro',uri=True)
    for (record,) in db.execute("SELECT record FROM paragraphs WHERE split='train'"):
        p=json.loads(record);key=(p['origin'],p['dialogue_id'],p['turn_idx'],p['speaker'],p['language'])
        rows=sorted(raw[key],key=lambda x:x['chunk_in_turn']);assert [r['chunk_in_turn'] for r in rows]==list(range(len(rows)))
        assert all(r['keep'] is True and not r['checks'].get('hard') and not r['drop_reasons'] for r in rows)
        assert not rows[0]['starts_mid_sentence'] and not rows[-1]['ends_mid_sentence']
        assert all(a['ends_mid_sentence']==b['starts_mid_sentence'] for a,b in zip(rows,rows[1:]))
        text=' '.join(r['source_raw'].strip() for r in rows);assert normalize(text).split()==p['words']
        p['raw_display']=text;paragraphs[key]=p
    tok=AutoTokenizer.from_pretrained(PARENT/'pretrained',local_files_only=True,token=False)
    def h(ids):return hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).digest()
    def encode(row):return encode_tail(tok,row['context'],row['text'],256)['input_ids']
    protected=set()
    for line in (PARENT/'dev_rows_vi_en.jsonl').open():protected.add(h(encode(json.loads(line))))
    for lang in ['vi','en']:
        with gzip.open(ROOT/'outputs/multilingual_stream_c75_test_20261102_v1'/('predictions_'+lang+'.jsonl.gz'),'rt') as f:
            for line in f:protected.add(h(encode(json.loads(line))))
    old={h(ids[:int(length)]) for ids,length in zip(arrays['ids'][:200000],arrays['lengths'][:200000])}
    seen=set();counts=Counter();origins=Counter();cases=[];case_counts=Counter();source_turns=set();n=0
    with (OUT/'train_vi_en.jsonl').open() as f:
        for i,line in enumerate(f):
            n=i+1
            if i<200000:continue
            r=json.loads(line);assert r['split']=='train' and r['language'] in ('vi','en')
            key=(r['origin'],r['dialogue_id'],r['turn_idx'],r['speaker'],r['language']);p=paragraphs[key];words=p['words']
            assert family_splits[fam(r['origin'],r['dialogue_id'])]=={'train'}
            a,b,c=r['context_start'],r['boundary'],r['future_end'];assert 0<=a<b<c<=len(words) and c-b in (1,2)
            assert r['context']==' '.join(words[a:b]) and r['text']==' '.join(words[b:c]) and b not in p['unknown']
            assert r['label']==('EOT' if b in p['eots'] else 'NEOT')
            if r['label']=='EOT':
                positions={len(normalize(p['raw_display'][:m.end()]).split()) for m in re.finditer(r'[.。]',p['raw_display'])}
                assert b in positions,'EOT has no actual source full stop'
            ids=encode(r);assert ids==arrays['ids'][i,:int(arrays['lengths'][i])].tolist()
            assert int(r['label']=='EOT')==int(arrays['labels'][i]) and int(r['language']=='en')==int(arrays['language_ids'][i])
            key_hash=h(ids);assert key_hash not in seen and key_hash not in protected and key_hash not in old
            seen.add(key_hash);counts[r['language']+':'+r['label']]+=1;origins[r['origin']]+=1;source_turns.add(key)
            ck=r['language']+':'+r['label']
            if case_counts[ck]<3:
                case_counts[ck]+=1;cases.append({**r,'raw_display':p['raw_display'],'word_count':len(words)})
    assert n==receipt['rows'] and sum(counts.values())==receipt['additional_rows'] and dict(counts)==receipt['additional_label_counts']
    migration=json.loads((OUT/'migration_receipt.json').read_text())
    for name,value in migration['dev_files_unchanged'].items():assert sha(OUT/name)==value==sha(PARENT/name)
    report={'verified':True,'total_train_rows':n,'parent_rows_preserved':200000,'additional_rows':sum(counts.values()),
        'additional_label_counts':dict(counts),'additional_source_rows_by_file':dict(origins),'unique_source_turns_in_subset':len(source_turns),
        'every_additional_row_checked':True,'raw_chunk_provenance_verified':True,'causal_one_two_word_future_verified':True,
        'chunk_edges_not_used_as_eot':True,'teacher_mask_zero_for_added_rows':True,'original_dev_test_in_train':False,
        'fixed_dev_c75_student_token_input_overlap':0,'parent_train_token_input_overlap':0,'duplicate_added_token_inputs':0,
        'dev_files_byte_identical':True,'raw_source_sha256':receipt['raw_sources_sha256'],'train_sha256':receipt['train_sha256'],
        'cache_receipt_sha256':sha(cache/'receipt.json'),'scope':'Import integrity audit of TRAIN punctuation-silver samples; not model quality or human/audio gold',
        'test_input_fields_only_used_for_dedup':True,'test_scored_or_used_for_selection':False,
        'limitations':receipt['overlap_limitations']}
    save('additional_data_audit.json',report);save('additional_data_review.json',{'audit':report,'cases':cases});db.close()
    print(json.dumps(report))


if __name__=='__main__':run()
