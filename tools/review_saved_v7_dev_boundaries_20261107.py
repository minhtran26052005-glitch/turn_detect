"""Extract a deterministic, source-stratified sample of saved DEV decisions."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT=Path('/home/edabk/hoangbpm/turn_detect/outputs')
JOB=ROOT/'multilingual_malbert_vi_en_sentence_mix_20261107_v7'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def word_spans(text):
    spans=[]; start=None
    for i,ch in enumerate(text):
        if ch.isspace() or unicodedata.category(ch).startswith('P'):
            if start is not None:spans.append((start,i));start=None
        elif start is None:start=i
    if start is not None:spans.append((start,len(text)))
    return spans


def main():
    report_path=JOB/'runs/finetune/dev_step_00480000.json'
    report=json.loads(report_path.read_text())
    mon_path=JOB/'dev_monitor.json.gz';trace_path=JOB/'runs_finetune_dev_00480000_dev_trace.jsonl.gz'
    streams=json.load(gzip.open(mon_path,'rt',encoding='utf-8'))
    originals={s['key']:s for s in json.load(gzip.open(ROOT/'malbert_eval_zip_20261104_v2/prepared_streams.json.gz','rt',encoding='utf-8'))}
    groups=defaultdict(list);counts=Counter();h=hashlib.sha256()
    with gzip.open(trace_path,'rt',encoding='utf-8') as f:
        for line in f:
            h.update(line.rstrip('\n').encode())
            r=json.loads(line)
            if not r['scored']:continue
            s=streams[r['stream']];b=r['boundary']
            assert bool(r['label'])==(b in set(s['eots']))
            assert r['predicted_eot']==(r['p_eot']>=report['dev']['threshold'])
            case=('tp' if r['label'] else 'fp') if r['predicted_eot'] else ('fn' if r['label'] else 'tn')
            counts[r['language']+'|'+case]+=1
            if case=='tn':continue
            rank=hashlib.sha256(('v7-semantic-review-20261107|'+s['key']+'|'+str(b)).encode()).hexdigest()
            groups[(r['language'],s['domain'],case)].append((rank,r))
    assert h.hexdigest()==report['dev']['causal_input_trace_digest']
    for lang in ('vi','en'):
        for case in ('tp','fp','fn','tn'):
            assert counts[lang+'|'+case]==report['dev']['by_language'][lang][case]
    sample=[]
    for key,rows in sorted(groups.items()):
        for _,r in sorted(rows,key=lambda v:v[0])[:1 if key[2]=='tp' else 2]:
            s=streams[r['stream']];original=originals[s['key']];b=r['boundary']
            seg=next(p for p in original['segments'] if p['start']<b and b+2<=p['end'])
            a,z=seg['start'],seg['end'];text=seg['display'];spans=word_spans(unicodedata.normalize('NFKC',text))
            text=unicodedata.normalize('NFKC',text)
            assert [text[x:y].lower() for x,y in spans]==s['words'][a:z]
            cut=spans[b-a-1][1]
            low=max(a,b-32);high=min(z,b+26)
            lo=spans[low-a][0];hi=spans[high-a-1][1]
            if high==z:hi=len(text)
            marked=text[lo:cut]+' ⟂ '+text[cut:hi]
            sample.append(dict(id='D%02d'%(len(sample)+1),language=key[0],domain=key[1],case=key[2],
                               stream_key=s['key'],boundary=b,context_start=r['context_start'],
                               p_eot=r['p_eot'],encoded_sha256=r['encoded_sha256'],
                               actual_model_context_tail=' '.join(s['words'][max(r['context_start'],b-70):b]),
                               actual_model_future=' '.join(s['words'][b:b+2]),
                               original_source_excerpt=marked,
                               full_original_turn=text if len(text)<2000 else None,
                               excerpt_contains_offline_reviewer_future_beyond_model=True))
    out=dict(complete=True,at_utc=datetime.now(timezone.utc).isoformat(),branch_step=480000,
             global_step=report['step'],threshold=report['dev']['threshold'],report_sha256=digest(report_path),
             monitor_sha256=digest(mon_path),trace_sha256=digest(trace_path),
             all_saved_trace_decisions_and_confusion_verified=True,
             selection='Two hash-ranked FP and FN plus one TP per language/domain, source-stratified, not population representative',
             model_inference_performed=False,training_or_labels_changed=False,rows=sample)
    (JOB/'semantic_boundary_review_sample_20261107.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':main()
