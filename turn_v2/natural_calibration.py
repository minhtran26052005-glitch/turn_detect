"""Exact decision-boundary calibration on development, never on held data."""
import numpy as np
from .common import LABEL2ID
from .metrics import classification_metrics


def summarize_exact(rows,probs):
    probs=np.asarray(probs,dtype=np.float64)
    labels=np.array([LABEL2ID[r['label']] for r in rows]);languages=np.array([r['language'] for r in rows])
    slices={};thresholds={}
    for lang in ['vi','en','zh']:
        language=languages==lang
        sources=sorted({r['evaluation_slice'] for r in rows if r['language']==lang})
        masks={source:np.array([r['language']==lang and r['evaluation_slice']==source for r in rows]) for source in sources}
        unique=np.unique(probs[language])
        # Midpoints cover every distinct >= decision partition, including very
        # narrow probability ranges that a .005 grid cannot separate.
        candidates=np.unique(np.concatenate([[0.,.5,1.],unique,(unique[:-1]+unique[1:])/2]))
        best=None;best_key=None
        for threshold in candidates:
            metrics={source:classification_metrics(labels[mask],probs[mask],float(threshold)) for source,mask in masks.items()}
            key=(min(m['joint_score'] for m in metrics.values()),-sum(m['false_eot'] for m in metrics.values()),-abs(threshold-.5))
            if best_key is None or key>best_key:best=(float(threshold),metrics);best_key=key
        thresholds[lang]=best[0];slices.update({lang+':'+source:value for source,value in best[1].items()})
    natural={lang:slices[lang+':natural'] for lang in ['vi','en','zh']}
    score=min(m['joint_score'] for m in slices.values())
    return {'thresholds':thresholds,'slices':slices,'natural_languages':natural,'minimum_dev_slice_joint_score':score,
        'development_target_met':all(m['accuracy']>.9 and m['eot_f1']>.9 for m in natural.values()),
        'development_all_slices_over_90':score>.9,'natural_hold_false_eot_below_10_percent':all(m['false_eot_rate']<=.1 for m in natural.values()),
        'independent_real_target_achieved':False,'calibration':'Exact dev score boundaries; weights fixed; no held predictions used'}
