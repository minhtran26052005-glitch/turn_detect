"""VI/EN-only DEV calibration, predicted-reset rollouts and retention."""
from collections import Counter
import numpy as np
from .factory_stream_test_v1 import metric, kind

LANGUAGES=('vi','en')


def evaluate(predict,rows,streams):
    assert rows and all(r['language'] in LANGUAGES for r in rows)
    assert streams and all(set(s['language_by_word'])<=set(LANGUAGES) for s in streams)
    probs=predict(rows)
    splits={l:([p for p,r in zip(probs,rows) if r['language']==l],
               [r['label']=='EOT' for r in rows if r['language']==l]) for l in LANGUAGES}
    def score(p,y,t):return metric(Counter(kind(bool(a),b>=t) for a,b in zip(y,p)))
    t=float(max(np.linspace(.1,.95,86),key=lambda v:min(score(*splits[l],v)['eot_f1'] for l in LANGUAGES)))
    counts={l:Counter() for l in LANGUAGES};starts=[0]*len(streams)
    for b in range(1,max(len(s['words']) for s in streams)-1):
        batch=[];indices=[]
        for i,s in enumerate(streams):
            if b+2>len(s['words']):continue
            batch.append({'context':' '.join(s['words'][starts[i]:b]),'text':' '.join(s['words'][b:b+2])});indices.append(i)
        if not batch:continue
        for i,p in zip(indices,predict(batch)):
            s=streams[i]
            if b not in s['unknown']:counts[s['language_by_word'][b-1]][kind(b in s['eots'],p>=t)]+=1
            if p>=t:starts[i]=b
    by={l:metric(counts[l]) for l in LANGUAGES}
    return dict(threshold=t,by_language=by,static_by_language={l:score(*splits[l],t) for l in LANGUAGES},
        score=min(x['eot_f1'] for x in by.values()),scope='VI/EN-only synthetic punctuation causal predicted-reset DEV; not human/audio gold',
        gold_reset_in_inference=False,language_id_supplied=False,max_future_words=2,active_languages=list(LANGUAGES))


def retained(report,teacher):
    assert set(report['by_language'])==set(LANGUAGES)
    checks={}
    for l in LANGUAGES:
        a,b=report['by_language'][l],teacher['by_language'][l]
        checks[l]=a['eot_f1']>=b['eot_f1']-.02 and (a['eot_precision'] or 0)>=(b['eot_precision'] or 0)-.02 and (a['eot_recall'] or 0)>=(b['eot_recall'] or 0)-.03
    return all(checks.values()),checks
