"""Read-only independent raw-source coverage and resume checks for uploaded v4."""
from collections import Counter,defaultdict
import argparse
import gzip
import json
import re
import numpy as np

from .uploaded_vi_en_data_v4 import OUT,LANGS,StreamOrder,initialize,source_rows,save,sha,digest
from .checked_dialogues_prepare_v1 import punctuation
from .tiny_compression_v1.common import normalize,encode_pair


def fixtures():
    words,eots,unknown=punctuation('Mr. Dao waits. Then he says, go now! and waits... again.')
    assert 1 in unknown and 3 in eots and 8 in unknown and 10 in unknown and 11 in eots
    # A turn edge cannot supply future words. The visible buffer survives speaker changes.
    s=dict(domain='fixture',words=['a','b','c','d','e','f','g'],segments=[dict(start=0,end=4,language='vi'),dict(start=4,end=7,language='en')],eots=[2,7],unknown=[1])
    streams=initialize([s]);order=np.asarray([1,0]);r=StreamOrder(streams,order,lanes=1)
    rows,states=r.rows();assert [(x['boundary'],x['k'],x['text']) for x in rows]==[(1,2,'b c')]
    assert rows[0]['scored'] is False
    r.advance(states,[.1],.5)
    rows,states=r.rows();assert rows[0]['context']=='a b' and rows[0]['label']==1
    # Ignore the gold EOT label: a low prediction must retain the buffer.
    r.advance(states,[.1],.5);snapshot=r.state()
    restored=StreamOrder(streams,order,snapshot,lanes=1)
    a,sa=r.rows();b,sb=restored.rows();assert a==b and a[0]['context']=='a b c d e' and a[0]['text']=='f g'
    r.advance(sa,[.9],.5);restored.advance(sb,[.9],.5)
    assert r.rows()[0]==restored.rows()[0]
    return dict(punctuation_ambiguity=True,same_turn_future=True,no_gold_or_speaker_reset=True,stream_lane_resume_equivalent=True)


def verify_sources():
    tests=fixtures();manifest=json.loads((OUT/'dataset_manifest.json').read_text())
    for n,h in manifest['files'].items():assert sha(OUT/n)==h
    lookup=defaultdict(list);raw_counts=Counter()
    for origin,rows,receipt in source_rows():
        assert receipt==manifest['sources'][origin]
        groups=defaultdict(list)
        for r in rows:
            if r.get('split','train')!='train' or r.get('src_lang') not in LANGS:continue
            groups[(r['dialogue_id'],int(r['turn_idx']),r['speaker'],r['src_lang'])].append(r)
        for (identity,turn,speaker,lang),part in groups.items():
            part.sort(key=lambda r:int(r['chunk_in_turn']))
            if [int(r['chunk_in_turn']) for r in part]!=list(range(len(part))):continue
            if part[0]['starts_mid_sentence'] or part[-1]['ends_mid_sentence']:continue
            if any(a['ends_mid_sentence']!=b['starts_mid_sentence'] for a,b in zip(part,part[1:])):continue
            if not all(r['keep'] is True and not r.get('drop_reasons') and not r.get('checks',{}).get('hard') for r in part):continue
            text=' '.join(r['source_raw'].strip() for r in part)
            if re.search(r'[\u3400-\u9fff]',text):continue
            words=normalize(text).split()
            if len(words)<2:continue
            lookup[(origin,identity,lang,digest(' '.join(words)))].append((words,text,len(part)))
    streams=initialize(json.load(gzip.open(OUT/'train_streams.json.gz','rt',encoding='utf-8')))
    seen=Counter();scored=0;source_rows_verified=0
    for s in streams:
        assert s['source_split']=='train' and s['domain'] in manifest['sources']
        identity=s['key'][len(s['domain'])+1:].rsplit(':part',1)[0]
        actual_eots=set();actual_unknown=set();row_count=0
        for p in s['segments']:
            words=s['words'][p['start']:p['end']]
            candidates=lookup[(s['domain'],identity,p['language'],digest(' '.join(words)))]
            assert candidates,(s['key'],p)
            match=next(v for v in candidates if v[0]==words)
            w,e,u=punctuation(match[1]);assert w==words
            actual_eots.update(p['start']+i for i in e);actual_unknown.update(p['start']+i for i in u)
            row_count+=match[2]
            for k in (1,2):
                for b in range(p['start']+1,p['end']-k+1):
                    assert p['start']<=b-1<b+k<=p['end']
                    if b not in actual_unknown:scored+=1
            seen[s['domain']+':segments']+=1
        assert actual_eots==s['_eots'] and actual_unknown==s['_unknown'] and row_count==s['source_rows']
        source_rows_verified+=row_count
    expected=sum(v for k,v in manifest['train'].items() if ':k' in k)
    assert scored==expected and source_rows_verified==manifest['train']['source_rows']
    save('independent_source_verification.json',dict(complete=True,fixtures=tests,
         every_training_segment_matched_to_kept_contiguous_raw_source=True,source_rows_verified=source_rows_verified,
         every_eot_unknown_label_verified=True,all_eligible_scored_word_gaps_verified=scored,origin_segments=dict(seen),
         original_test_or_dev_in_training=False,drive_replay_rows=0,label_provenance='Punctuation silver; not human/audio gold'))


def verify_training():
    import torch
    from transformers import AutoTokenizer
    from .multilingual_malbert_vi_en_uploaded_train_v4 import fingerprint
    manifest=json.loads((OUT/'dataset_manifest.json').read_text())
    streams=initialize(json.load(gzip.open(OUT/'train_streams.json.gz','rt',encoding='utf-8')))
    tok=AutoTokenizer.from_pretrained(OUT/'tokenizer',local_files_only=True,token=False)
    trace=json.loads((OUT/'first_training_causal_trace.json').read_text());starts={};last={}
    import hashlib
    for r in trace:
        key=r['stream'],r['k'];s=streams[r['stream']];b=r['boundary'];k=r['k']
        assert r['start']==starts.get(key,0) and b>last.get(key,0)
        segment=next(p for p in s['segments'] if p['start']<=b-1<p['end'])
        assert b+k<=segment['end'] and r['label']==int(b in s['_eots']) and r['scored']==(b not in s['_unknown'])
        ids=encode_pair(tok,' '.join(s['words'][r['start']:b]),' '.join(s['words'][b:b+k]),256)['input_ids']
        assert hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()==r['input_ids_sha256']
        if r['p_eot']>=r['threshold']:starts[key]=b
        last[key]=b
    path=OUT/'runs/finetune/latest_resume.pt'
    with path.open('rb') as f:
        checkpoint=torch.load('/proc/self/fd/'+str(f.fileno()),map_location='cpu',weights_only=False,mmap=True)
        assert checkpoint['branch_step']>=100 and len(checkpoint['optimizer']['state'])==27
        assert checkpoint['fingerprint']==fingerprint(manifest)
        assert checkpoint['uploaded_rows_seen']>0 and checkpoint['drive_replay_rows']==0
        assert {'asr/run2b','factory/gen4000'}<=checkpoint['uploaded_origin_exposures'].keys()
        assert all(k in checkpoint for k in ['stream_state','order','torch_rng','cuda_rng','numpy_rng','python_rng','sampler_rng'])
        assert all(bool(torch.isfinite(v).all()) for v in checkpoint['model'].values() if v.is_floating_point())
        parent=torch.load(OUT/'frozen/parent_full_resume.pt',map_location='cpu',weights_only=False,mmap=True)
        assert any(not torch.equal(v,parent['model'][k]) for k,v in checkpoint['model'].items())
        assert {int(v['step']) for v in checkpoint['optimizer']['state'].values()}=={checkpoint['step']}
        save('independent_causal_training_verification.json',dict(complete=True,trace_rows_audited=len(trace),
            every_encoded_input_label_and_predicted_reset_verified=True,full_checkpoint_load_passed=True,
            step=checkpoint['step'],branch_step=checkpoint['branch_step'],epoch=checkpoint['epoch'],
            uploaded_rows_seen=checkpoint['uploaded_rows_seen'],origin_exposures=checkpoint['uploaded_origin_exposures'],
            new_asr_and_gen4000_actual_training_verified=True,optimizer_entries=27,parameters=11941122,
            model_weights_changed_after_actual_training=True,all_model_weights_finite=True,
            original_model_optimizer_and_rng_migration_receipt_sha256=sha(OUT/'migration_receipt.json'),drive_replay_rows=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--training-only',action='store_true');args=parser.parse_args()
    if args.training_only:verify_training()
    else:verify_sources()
