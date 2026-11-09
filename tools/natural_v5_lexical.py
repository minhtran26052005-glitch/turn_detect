"""Small regularized text baseline; dev only, held cohort stays unopened."""
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.common import LABEL2ID,digest,file_hash,read_jsonl,write_json
from turn_v2.metrics import calibrate_on_dev,classification_metrics
from turn_v2.stream_v4_train import encode_stream


def main():
    output=Path('outputs/natural_v5_20261025/lexical_dev');output.mkdir(parents=True,exist_ok=False)
    train=list(read_jsonl(Path('data_v5/canonical_20261025/train.jsonl')))
    dev=list(read_jsonl(Path('data_v5/canonical_20261025/dev.jsonl')))
    replay=list(read_jsonl(Path('data_v4/causal_20261025/train.jsonl')))
    tokenizer=AutoTokenizer.from_pretrained('outputs/stream_v4_20261025/runs/stream_v4_campaign_20261025/round_02/best',local_files_only=True)
    def features(row):
        values=encode_stream(tokenizer,row);cut=values.index(tokenizer.sep_token_id)
        current=tokenizer.decode(values[cut+1:-1]);context=tokenizer.decode(values[1:cut])
        return 'CTX '+context+' ENDCTX CURRENT '+current+' ENDCURRENT TAIL '+current[-80:]+' ENDTAIL'
    descriptions=[];selected={};selected_predictions=np.zeros(len(dev));rng=np.random.default_rng(20261028)
    for lang in ['vi','en','zh']:
        native=[r for r in train if r['language']==lang]
        index=np.array([i for i,r in enumerate(dev) if r['language']==lang]);validation=[dev[i] for i in index]
        target=np.array([LABEL2ID[r['label']] for r in validation])
        best=-1.;winner=None
        for mode in ['natural_only','natural_with_replay']:
            rows=[{**r,'fit_source':'natural'} for r in native]
            if mode.endswith('with_replay'):
                for label in ['NEOT','EOT']:
                    pool=[r for r in replay if r['language']==lang and r['label']==label]
                    sample=rng.choice(len(pool),min(len(pool),1000 if lang=='vi' else 400),replace=False)
                    rows.extend({**pool[i],'fit_source':'replay'} for i in sample)
            labels=np.array([LABEL2ID[r['label']] for r in rows])
            counts=Counter((r['fit_source'],r['label']) for r in rows)
            weights=np.array([len(rows)*(.5 if mode=='natural_only' else .4 if r['fit_source']=='natural' else .1)/counts[(r['fit_source'],r['label'])] for r in rows])
            vectorizer=FeatureUnion([('chars',TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=1,max_features=60000,sublinear_tf=True)),
                ('words',TfidfVectorizer(ngram_range=(1,2),token_pattern=r'(?u)\b\w+\b',min_df=1,max_features=30000,sublinear_tf=True))])
            matrix=vectorizer.fit_transform([features(r) for r in rows]);dx=vectorizer.transform([features(r) for r in validation])
            for c in [.25,1.,4.]:
                model=LogisticRegression(C=c,solver='liblinear',max_iter=1000,random_state=20261028)
                model.fit(matrix,labels,sample_weight=weights)
                probs=model.predict_proba(dx)[:,1];metrics=calibrate_on_dev(target,probs)
                record={'language':lang,'mode':mode,'C':c,'metrics':metrics,'fit_samples':len(rows),'dimensions':matrix.shape[1]}
                descriptions.append(record)
                if metrics['joint_score']>best:
                    best=metrics['joint_score'];winner=record
                    path=output/(lang+'_'+mode+'_C'+str(c)+'.joblib')
                    joblib.dump({'vectorizer':vectorizer,'classifier':model,'threshold':metrics['threshold'],
                        'feature_contract':'CTX decoded budgeted history ENDCXT CURRENT decoded current tail ENDCURRENT TAIL last80chars ENDTAIL'},path,compress=3)
                    winner={**record,'path':str(path),'bytes':path.stat().st_size,'sha256':file_hash(path)}
                    selected_predictions[index]=probs
        selected[lang]=winner
    labels=np.array([LABEL2ID[r['label']] for r in dev]);langs=np.array([r['language'] for r in dev])
    np.savez_compressed(output/'predictions.npz',labels=labels,language=langs,prob_EOT=selected_predictions)
    # Apply the selected VI threshold to the existing legacy dev, no retuning.
    vi_model=joblib.load(selected['vi']['path'])
    legacy=[r for r in read_jsonl(Path('data_v4/causal_20261025/dev.jsonl')) if r['language']=='vi']
    probs=vi_model['classifier'].predict_proba(vi_model['vectorizer'].transform([features(r) for r in legacy]))[:,1]
    legacy_metrics=classification_metrics(np.array([LABEL2ID[r['label']] for r in legacy]),probs,selected['vi']['metrics']['threshold'])
    report={'status':'DEV_ONLY_LEXICAL_BASELINE','selected':selected,'all_candidates':descriptions,
        'minimum_natural_dev_joint_score':min(r['metrics']['joint_score'] for r in selected.values()),
        'dev_three_languages_over_90':all(r['metrics']['accuracy']>.9 and r['metrics']['eot_f1']>.9 for r in selected.values()),
        'vi_legacy_dev_regression':legacy_metrics,'best_bundle_bytes':sum(r['bytes'] for r in selected.values()),
        'serialized_research_bundle_not_verified_deployment':True,'sklearn':importlib.metadata.version('scikit-learn'),
        'sealed_loaded':False,'canonical_manifest_sha256':file_hash(Path('data_v5/canonical_20261025/manifest.json')),
        'source_sha256':file_hash(Path(__file__))}
    write_json(output/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
