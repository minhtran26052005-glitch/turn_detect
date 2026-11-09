"""New DEV only: threshold selection uses actual predicted-reset calibration rollouts."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import numpy as np

from .uploaded_vi_en_curriculum_data_v6 import OUT, LANGS, StreamOrder, clean, initialize, save, status
from .malbert_eval_zip_v2 import kind, metric
from .tiny_compression_v1.common import encode_pair


def predictor(model, tok, device='cuda', teacher=False):
    import torch
    def predict(rows):
        # Canonical encoding, with no language/source metadata sent to the model.
        seqs=[encode_pair(tok,r['context'],r['text'],256)['input_ids'] for r in rows]
        values=[]
        for start in range(0,len(seqs),32):
            part=seqs[start:start+32];width=max(map(len,part))
            ids=torch.zeros((len(part),width),dtype=torch.long,device=device);mask=torch.zeros_like(ids)
            for j,seq in enumerate(part):ids[j,:len(seq)]=torch.as_tensor(seq,device=device);mask[j,:len(seq)]=1
            with torch.inference_mode():
                logits=model(input_ids=ids,attention_mask=mask).logits if teacher else model(ids,mask)
                probs=logits.float().softmax(-1)[:,1].cpu().tolist()
            assert all(math.isfinite(p) and 0<=p<=1 for p in probs)
            values.extend(probs)
        return values,seqs
    return predict


def rollout(predict, streams, threshold, trace_name=None):
    initialize(streams);order=np.asarray([2*i+1 for i in range(len(streams))])
    runner=StreamOrder(streams,order,lanes=64);counts={l:Counter() for l in LANGS};domains=defaultdict(Counter)
    inferred=0;digest=hashlib.sha256();cases=[];case_counts=Counter()
    trace=gzip.open(OUT/trace_name,'wt',encoding='utf-8',compresslevel=3) if trace_name else None
    try:
        while True:
            rows,states=runner.rows()
            if not rows:break
            probs,seqs=predict(rows)
            for r,p,seq in zip(rows,probs,seqs):
                s=streams[r['stream']];b=r['boundary'];assert len(r['text'].split())==2
                assert r['context']==' '.join(s['words'][r['start']:b]) and s['split']=='dev'
                item=dict(stream=r['stream'],boundary=b,context_start=r['start'],p_eot=p,predicted_eot=p>=threshold,
                          scored=r['scored'],label=r['label'],language=r['language'],
                          encoded_sha256=hashlib.sha256(np.asarray(seq,dtype='<u4').tobytes()).hexdigest())
                raw=json.dumps(item,separators=(',',':'));digest.update(raw.encode());inferred+=1
                if trace:trace.write(raw+'\n')
                if r['scored']:
                    case=kind(bool(r['label']),p>=threshold);counts[r['language']][case]+=1
                    domains[(s['domain'],r['language'])][case]+=1
                    key=(s['domain'],r['language'],case)
                    if case_counts[key]<2:
                        case_counts[key]+=1
                        cases.append(dict(domain=s['domain'],language=r['language'],case=case,
                            context_tail=' '.join(r['context'].split()[-50:]),future=r['text'],p_eot=p,stream_key=s['key']))
            runner.advance(states,probs,threshold)
    finally:
        if trace:trace.close()
    by={l:metric(c) for l,c in counts.items()}
    return dict(threshold=float(threshold),by_language=by,score=min((m['eot_f1'] or 0) for m in by.values()),
        by_domain=[dict(domain=d,language=l,**metric(c)) for (d,l),c in sorted(domains.items())],
        inferences=inferred,causal_input_trace_digest=digest.hexdigest(),cases=cases,
        gold_reset_in_inference=False,source_speaker_reset=False,language_id_supplied=False,max_future_words=2,
        scope='Uploaded original DEV, punctuation silver; not TEST or independent human/audio turn gold')


def evaluate(predict, calibration, monitor, tag):
    grids=[]
    for t in np.linspace(.1,.9,9):
        result=rollout(predict,calibration,float(t));grids.append(result)
        status('CALIBRATING_NEW_CAUSAL_DEV',tag=tag,threshold=float(t),f1={l:m['eot_f1'] for l,m in result['by_language'].items()})
    best=max(grids,key=lambda r:(r['score'],sum((m['eot_f1'] or 0) for m in r['by_language'].values())))
    t=best['threshold']
    result=rollout(predict,monitor,t,tag.replace('/','_')+'_dev_trace.jsonl.gz')
    result['calibration']=[dict(threshold=r['threshold'],by_language=r['by_language'],score=r['score'],
        inferences=r['inferences'],causal_input_trace_digest=r['causal_input_trace_digest']) for r in grids]
    result['threshold_selection']='Worst-language F1 on separate original DEV calibration families, actual predicted resets at every threshold'
    result['monitor_families_disjoint_from_calibration']=True
    result['test_used']=False
    save(tag+'.json',result)
    return result


def retained(report, teacher):
    checks={}
    for l in LANGS:
        a,b=report['by_language'][l],teacher['by_language'][l]
        assert a['samples']==b['samples'] and a['actual_eot']==b['actual_eot']
        checks[l]=(a['eot_f1'] or 0)>=(b['eot_f1'] or 0)-.02 and (a['eot_precision'] or 0)>=(b['eot_precision'] or 0)-.02 and (a['eot_recall'] or 0)>=(b['eot_recall'] or 0)-.03
    return all(checks.values()),checks
