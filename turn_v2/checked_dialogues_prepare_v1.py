"""TRAIN-only, causal conversion of user-provided checked translation chunks."""
from collections import Counter, defaultdict
import bisect
from datetime import datetime, timezone
import gzip
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import unicodedata

from .common import normalize

ROOT=Path('/home/edabk/hoangbpm/turn_detect')
SOURCE=ROOT/'data/checked_dialogues_vi_en_20261103_v1'
PARENT=ROOT/'outputs/multilingual_malbert_vi_en_20261103_v2'
OUT=ROOT/'outputs/multilingual_malbert_vi_en_checked_20261103_v3'
SAMPLE_PER_LANGUAGE=100000


def sha(path):
    with Path(path).open('rb') as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)


def progress(stage,**values):
    save('preparation_status.json',dict(stage=stage,pid=os.getpid(),at_utc=datetime.now(timezone.utc).isoformat(),**values))


def family(identity):
    return re.sub(r'\.s\d+\.(?:vi|en)$','',identity)


def stitched(rows):
    """Use actual contiguous chunks only; never the privileged rest_of_sentence."""
    rows=sorted(rows,key=lambda r:r['chunk_in_turn'])
    if [r['chunk_in_turn'] for r in rows]!=list(range(len(rows))):return None,'noncontiguous_chunks'
    if rows[0]['starts_mid_sentence'] or rows[-1]['ends_mid_sentence']:return None,'dangling_fragment'
    if any(a['ends_mid_sentence']!=b['starts_mid_sentence'] for a,b in zip(rows,rows[1:])):return None,'fragment_flag_mismatch'
    if not all(r['keep'] is True and not r.get('checks',{}).get('hard') and not r.get('drop_reasons') for r in rows):return None,'not_kept_or_hard_check'
    return ' '.join(r['source_raw'].strip() for r in rows),None


def punctuation(text):
    """Align punctuation to normalized word gaps, leaving ambiguous endings unscored."""
    text=unicodedata.normalize('NFKC',text).lower();words=[];buf=[];eots=set();unknown=set()
    abbreviations={m.start() for m in re.finditer(r'(?<=\w)\.(?=\w)|\.(?:\s*\.)+|\b(?:mr|mrs|ms|dr|prof|sr|jr|vs|etc)\.',text)}
    for match in re.finditer(r'\b(?:[a-z]\.){2,}',text):
        abbreviations.update(i for i in range(match.start(),match.end()) if text[i]=='.')
    def flush():
        if buf:words.append(''.join(buf));buf.clear()
    for i,ch in enumerate(text):
        if ch.isspace() or unicodedata.category(ch).startswith('P'):
            flush();b=len(words)
            if ch in '.。':
                prev=text[i-1] if i else '';nxt=text[i+1] if i+1<len(text) else ''
                token=re.search(r'\b(\w+)\s*$',text[:i]);abbr=token and (token[1] in {'mr','mrs','ms','dr','prof','sr','jr','vs','etc'} or len(token[1])==1)
                if prev=='.' or nxt=='.' or (prev.isalnum() and nxt.isalnum()) or abbr or i in abbreviations:unknown.add(b)
                else:eots.add(b)
            elif ch in '?!？！…':unknown.add(b)
        else:buf.append(ch)
    flush();eots-=unknown
    assert words==normalize(text).split()
    return words,sorted(eots),sorted(unknown)


def eligible_rows(paragraph):
    words=paragraph['words'];eots=paragraph['eots'];unknown=set(paragraph['unknown'])
    for k in (1,2):
        for b in range(1,len(words)-k+1):
            if b in unknown:continue
            key=f"{paragraph['origin']}:{paragraph['dialogue_id']}:{paragraph['turn_idx']}:{b}:{k}"
            number=int.from_bytes(hashlib.sha256(key.encode()).digest()[:16],'big');mode=number%4
            if mode==0:start=0
            elif mode<3:start=eots[bisect.bisect_left(eots,b)-1] if bisect.bisect_left(eots,b) else 0
            else:start=(number//4)%b
            yield number,{'language':paragraph['language'],'context':' '.join(words[start:b]),'text':' '.join(words[b:b+k]),
                'label':'EOT' if b in eots else 'NEOT','split':'train','lookahead_words':k,
                'source_kind':'USER_CHECKED_SOURCE_TEXT','label_source':'PUNCTUATION_SILVER_NOT_HUMAN_AUDIO_GOLD',
                'origin':paragraph['origin'],'dialogue_id':paragraph['dialogue_id'],'family_id':paragraph['family_id'],
                'turn_idx':paragraph['turn_idx'],'speaker':paragraph['speaker'],'boundary':b,'context_start':start,
                'paragraph_words':len(words),'future_end':b+k,'context_mode':'rolling' if mode==0 else 'reference_reset_train_only' if mode<3 else 'fragment_train_only'}


def prepare():
    import numpy as np
    from transformers import AutoTokenizer
    from .monolingual_data import encode_tail
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'cache/receipt.json').exists():return json.loads((OUT/'cache/receipt.json').read_text())
    assert shutil.disk_usage(OUT).free>15*2**30 and shutil.disk_usage('/mnt/hdd500').free>13*2**30
    manifest=json.loads((SOURCE/'source_manifest.json').read_text());assert manifest['original_dev_test_remain_excluded']
    progress('AUDITING_USER_SOURCES');stats=Counter();groups=defaultdict(list);family_splits=defaultdict(set)
    expected={'factory_checked.jsonl':'01a8df34923011d038a2a30cce60774a69f07794756f41f44e8924341e29c9f8',
              'public_checked.jsonl':'96f999ac6feeaad29fadc23d0bda87f44d8ade337f088418e9fb6c6332e7c6e4'}
    for name,digest in expected.items():
        path=SOURCE/'raw'/name;assert sha(path)==digest
        with path.open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);stats[name+':rows']+=1
                split=r.get('split','train') if name=='public_checked.jsonl' else 'train'
                assert split in ('train','dev','test') and r['src_lang'] in ('vi','en')
                identity=r['dialogue_id'];fam=name+':'+family(identity);family_splits[fam].add(split)
                # Select text/metadata fields only; instruction/target/rest_of_sentence are never executable instructions or model inputs.
                small={k:r[k] for k in ['source_raw','chunk_in_turn','starts_mid_sentence','ends_mid_sentence','keep','checks','drop_reasons']}
                groups[(name,identity,int(r['turn_idx']),r['speaker'],r['src_lang'],fam,split)].append(small)
                stats[name+':'+split+':kept' if r['keep'] else name+':'+split+':rejected']+=1
    mixed={f for f,s in family_splits.items() if len(s)>1};assert not mixed,'Source family crosses original splits'
    db=sqlite3.connect(OUT/'additional_paragraphs.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS paragraphs(id INTEGER PRIMARY KEY,split TEXT,language TEXT,family TEXT,record TEXT)')
    assert db.execute('SELECT COUNT(*) FROM paragraphs').fetchone()[0]==0,'Preserve partial build for audit; do not append duplicates'
    protected_paragraphs=set();base=sqlite3.connect((ROOT/'outputs/multilingual_stream_v2_20261102/dataset/paragraphs.sqlite').as_uri()+'?mode=ro',uri=True)
    for split,record in base.execute("SELECT split,record FROM paragraphs WHERE split!='train'"):
        words=json.loads(record)['words']
        if len(words)>=8:protected_paragraphs.add(hashlib.sha256(' '.join(words).encode()).digest())
    base.close();protected_families=set();parsed=[]
    for (origin,identity,turn,speaker,lang,fam,split),chunks in groups.items():
        text,reason=stitched(chunks)
        if reason:stats['excluded:'+reason]+=1;continue
        if re.search(r'[\d\u3400-\u9fff]',text):stats['excluded:digits_or_han_not_spoken_vien']+=1;continue
        words,eots,unknown=punctuation(text)
        if len(words)<2:stats['excluded:short_turn']+=1;continue
        p={'origin':origin,'dialogue_id':identity,'family_id':fam,'turn_idx':turn,'speaker':speaker,'language':lang,'words':words,'eots':eots,'unknown':unknown,'source_split':split}
        key=hashlib.sha256(' '.join(words).encode()).digest()
        if split!='train' and len(words)>=8:protected_paragraphs.add(key)
        parsed.append((p,key))
    del groups
    for p,key in parsed:
        if p['source_split']=='train' and key in protected_paragraphs:protected_families.add(p['family_id'])
    seen_paragraphs=set();heaps={l:[] for l in ('vi','en')};candidate_counts=Counter();seq=0
    progress('SAMPLING_ADDITIONAL_TRAIN',paragraphs=len(parsed))
    for p,key in parsed:
        split=p['source_split']
        if split=='train' and p['family_id'] in protected_families:stats['excluded:protected_family_paragraph']+=1;continue
        if split=='train' and key in seen_paragraphs:stats['excluded:duplicate_train_paragraph']+=1;continue
        if split=='train':seen_paragraphs.add(key)
        db.execute('INSERT INTO paragraphs(split,language,family,record) VALUES(?,?,?,?)',(split,p['language'],p['family_id'],json.dumps(p,ensure_ascii=False)))
        stats[split+':paragraphs:'+p['language']]+=1
        if split!='train':continue
        heap=heaps[p['language']]
        for rank,row in eligible_rows(p):
            seq+=1;candidate_counts[p['language']+':'+row['label']]+=1;item=(-rank,seq,row)
            if len(heap)<130000:heapq.heappush(heap,item)
            elif rank < -heap[0][0]:heapq.heapreplace(heap,item)
    db.commit();assert db.execute("SELECT COUNT(*) FROM (SELECT family FROM paragraphs GROUP BY family HAVING COUNT(DISTINCT split)>1)").fetchone()[0]==0
    tok=AutoTokenizer.from_pretrained(PARENT/'pretrained',local_files_only=True,token=False)
    def encoded(row):return encode_tail(tok,row['context'],row['text'],256)['input_ids']
    def input_hash(ids):return hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).digest()
    protected_inputs=set();progress('HASHING_PROTECTED_MODEL_INPUTS')
    with (PARENT/'dev_rows_vi_en.jsonl').open() as f:
        for line in f:protected_inputs.add(input_hash(encoded(json.loads(line))))
    for lang in ('vi','en'):
        path=ROOT/'outputs/multilingual_stream_c75_test_20261102_v1'/('predictions_'+lang+'.jsonl.gz')
        with gzip.open(path,'rt') as f:
            for line in f:
                r=json.loads(line);protected_inputs.add(input_hash(encoded(r)))
    for (record,) in db.execute("SELECT record FROM paragraphs WHERE split!='train'"):
        p=json.loads(record)
        for _,row in eligible_rows(p):protected_inputs.add(input_hash(encoded(row)))
    old={k:np.load(PARENT/'cache'/(k+'.npy'),mmap_mode='r') for k in ['ids','lengths','labels','logits','language_ids','source_row_indices']}
    assert len(old['labels'])==200000
    old_inputs={input_hash(ids[:int(n)]) for ids,n in zip(old['ids'],old['lengths'])}
    selected=[];seen_inputs={};conflicts=set();selected_counts=Counter()
    for lang in ('vi','en'):
        progress('FILTERING_TOKENIZED_TRAIN',language=lang)
        for _,_,row in sorted(heaps[lang],key=lambda x:-x[0]):
            ids=encoded(row);key=input_hash(ids)
            if key in protected_inputs:stats['excluded:protected_exact_model_input']+=1;continue
            if key in old_inputs:stats['excluded:already_in_parent_train']+=1;continue
            if key in seen_inputs:
                if seen_inputs[key]!=row['label']:conflicts.add(key)
                stats['excluded:duplicate_additional_input']+=1;continue
            seen_inputs[key]=row['label'];row['_ids']=ids;row['_hash']=key;selected.append(row);selected_counts[lang]+=1
            if selected_counts[lang]>=SAMPLE_PER_LANGUAGE:break
    selected=[r for r in selected if r['_hash'] not in conflicts];stats['excluded:conflicting_selected_inputs']=len(conflicts)
    assert len(selected)>100000 and set(r['language'] for r in selected)=={'vi','en'}
    assert not {r['family_id'] for r in selected}&{row[0] for row in db.execute("SELECT DISTINCT family FROM paragraphs WHERE split!='train'")}
    cache=OUT/'cache';cache.mkdir(exist_ok=True);n=200000+len(selected)
    specs={'ids':(np.uint16,(n,256)),'lengths':(np.uint16,(n,)),'labels':(np.uint8,(n,)),'logits':(np.float16,(n,2)),
           'language_ids':(np.uint8,(n,)),'source_row_indices':(np.uint32,(n,)),'teacher_mask':(np.uint8,(n,))}
    arrays={k:np.lib.format.open_memmap(cache/(k+'.npy'),mode='w+',dtype=t,shape=s) for k,(t,s) in specs.items()}
    for k in old:arrays[k][:200000]=old[k]
    arrays['teacher_mask'][:200000]=1;arrays['teacher_mask'][200000:]=0;arrays['logits'][200000:]=0
    train=OUT/'train_vi_en.jsonl';assert not train.exists()
    counts=Counter({'vi':100000,'en':100000});labels=Counter()
    with train.open('w',encoding='utf-8') as f:
        with (PARENT/'train_vi_en.jsonl').open() as src:shutil.copyfileobj(src,f)
        for j,row in enumerate(selected,200000):
            ids=row.pop('_ids');row.pop('_hash');assert row['future_end']<=row['paragraph_words'] and len(row['text'].split()) in (1,2)
            assert normalize(row['context'])==row['context'] and normalize(row['text'])==row['text']
            arrays['ids'][j]=0;arrays['ids'][j,:len(ids)]=ids;arrays['lengths'][j]=len(ids);arrays['labels'][j]=row['label']=='EOT'
            arrays['language_ids'][j]=int(row['language']=='en');arrays['source_row_indices'][j]=j
            counts[row['language']]+=1;labels[row['language']+':'+row['label']]+=1;f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if j%4096==0:progress('WRITING_VERIFIED_ADDITIONAL_CACHE',done=j,total=n)
        f.flush();os.fsync(f.fileno())
    for a in arrays.values():a.flush()
    for p in cache.glob('*.npy'):
        with p.open('rb') as f:os.fsync(f.fileno())
    db.close();result={'complete':True,'rows':n,'counts':dict(counts),'active_languages':['vi','en'],'parent_rows':200000,
        'additional_rows':len(selected),'additional_label_counts':dict(labels),'candidate_points':dict(candidate_counts),'audit_counts':dict(stats),
        'raw_sources_sha256':expected,'source_manifest_sha256':sha(SOURCE/'source_manifest.json'),'train_sha256':sha(train),
        'tokenizer_sha256':sha(PARENT/'tokenizer/tokenizer.json'),'parent_cache_sha256':sha(PARENT/'cache/receipt.json'),
        'files':{k:{'sha256':sha(cache/(k+'.npy')),'bytes':(cache/(k+'.npy')).stat().st_size} for k in specs},
        'source_module_sha256':sha(__file__),'protected_model_input_hashes':len(protected_inputs),'protected_family_exclusions':len(protected_families),
        'protected_exact_token_input_overlap_after_filtering':0,'original_dev_test_in_train':False,'source_target_double_counting':False,
        'parser':'Contiguous source_raw chunks from the same turn/speaker/language; rest_of_sentence and target never used',
        'sampling':'Lowest deterministic SHA priorities, at most100000 eligible 1/2-word points per language; not all candidate gaps',
        'labels':'Punctuation silver, not human/audio gold; ellipsis/abbreviation and ?/! endpoints unscored',
        'kd':'Parent rows retain frozen logits; additional rows use CE only, teacher_mask=0, no fabricated teacher targets',
        'overlap_limitations':'Exact normalized full-paragraph family exclusion and actual student token-window input hashing; not exhaustive paraphrase or all possible reset windows',
        'test_predictions_used_for_selection':False,'test_input_fields_used_only_for_dedup':True}
    save('dataset_manifest.json',result);save('cache/receipt.json',result);progress('COMPLETE',rows=n,additional_rows=len(selected));return result


if __name__=='__main__':
    try:print(json.dumps(prepare()))
    except Exception as e:
        import traceback
        save('preparation_error.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
