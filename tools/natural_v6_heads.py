"""Train small semantic heads on frozen encoders; choose using dev only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.common import file_hash,read_jsonl,write_json
from turn_v2.metrics import calibrate_on_dev,classification_metrics


def main(args):
    root=Path('outputs/natural_v6_20261025');output=root/(args.encoder+'_heads');output.mkdir(exist_ok=False)
    features=np.load(root/(args.encoder+'_features/features.npz'))
    x=features['features'].astype(np.float64);labels=features['labels'];languages=features['language'];splits=features['split']
    rows=[r for split in ['train','dev'] for r in read_jsonl(Path('data_v6/turn_context_20261025')/(split+'.jsonl'))]
    assert np.array_equal(features['case_id'],np.array([r['case_id'] for r in rows]))
    if args.encoder=='fusion':raise ValueError('Use an explicit separate fusion protocol if needed')
    candidates=[];winners={};winner_probs=np.zeros(len(rows));winners_train={}
    # Inverse class/call weights reduce domination by long calls. Inputs contain
    # only text-derived semantic vectors and text lengths, no call identifiers.
    call_counts=Counter((r['language'],r['label'],r['session_group']) for r in rows if r['split']=='train')
    for lang in ['vi','en','zh']:
        train=(languages==lang)&(splits=='train');dev=(languages==lang)&(splits=='dev')
        positions=np.flatnonzero(train);target=labels[train]
        raw=np.array([1./call_counts[(rows[i]['language'],rows[i]['label'],rows[i]['session_group'])] for i in positions])
        weight=raw.copy()
        for label in [0,1]:weight[target==label]*=.5*len(target)/raw[target==label].sum()
        best=-1
        for c in [.1,1.,10.,100.,1000.]:
            model=LogisticRegression(C=c,solver='liblinear',max_iter=2000,random_state=20261025)
            model.fit(x[train],target,sample_weight=weight)
            probs=model.predict_proba(x[dev])[:,1];metric=calibrate_on_dev(labels[dev],probs)
            train_metric=classification_metrics(target,model.predict_proba(x[train])[:,1],metric['threshold'])
            record={'language':lang,'C':c,'dev':metric,'train_fit':train_metric}
            candidates.append(record)
            if metric['joint_score']>best:
                best=metric['joint_score'];path=output/(lang+'_head.joblib')
                joblib.dump({'classifier':model,'threshold':metric['threshold'],'language':lang},path,compress=3)
                winners[lang]={**record,'head_bytes':path.stat().st_size,'head_sha256':file_hash(path),'path':str(path)}
                winner_probs[dev]=probs;winner_probs[train]=model.predict_proba(x[train])[:,1]
    np.savez_compressed(output/'predictions.npz',prob_EOT=winner_probs,labels=labels,language=languages,split=splits)
    result={'status':'DEV_SELECTED_SEMANTIC_HEADS','encoder':args.encoder,'winners':winners,'candidates':candidates,
        'minimum_dev_joint_score':min(m['dev']['joint_score'] for m in winners.values()),
        'dev_target_met':all(m['dev']['accuracy']>.9 and m['dev']['eot_f1']>.9 for m in winners.values()),
        'all_train_fit_over_90':all(m['train_fit']['accuracy']>.9 and m['train_fit']['eot_f1']>.9 for m in winners.values()),
        'source_sha256':file_hash(Path(__file__)),'features_sha256':file_hash(root/(args.encoder+'_features/features.npz')),
        'sealed_loaded':False,'classification_inputs':'Semantic text features; no call/label/timestamp features'}
    write_json(output/'report.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='candidates'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--encoder',choices=['qwen','minilm'],required=True);main(parser.parse_args())
