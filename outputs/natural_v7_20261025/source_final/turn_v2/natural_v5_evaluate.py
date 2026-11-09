"""One frozen held evaluation; previously exposed V4 material is regression."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import LABEL2ID,file_hash,read_jsonl,write_json,write_jsonl
from .metrics import classification_metrics
from .stream_v4_train import encode_rows,infer


def metrics(rows,probs,thresholds):
    labels=np.array([LABEL2ID[r['label']] for r in rows]);langs=np.array([r['language'] for r in rows])
    return {lang:classification_metrics(labels[langs==lang],probs[langs==lang],thresholds[lang]) for lang in sorted(set(langs))}


def run(args):
    report=json.loads((args.run/'report.json').read_text());checkpoint=Path(report['best_path'])
    if not args.allow_below_target_diagnostic:
        assert report['best_dev']['development_target_met'], 'Development below target; keep limited unseen cohort sealed'
    args.output.mkdir(parents=True,exist_ok=False);started=time.time()
    thresholds=report['best_dev']['thresholds']
    assert file_hash(checkpoint/'model.safetensors')==report['checkpoint_sha256']
    manifest=json.loads((args.data/'manifest.json').read_text())
    for split in ['dev','sealed']:assert file_hash(args.data/(split+'.jsonl'))==manifest['files'][split]['sha256']
    initial=json.loads((args.run/'initial_dev.json').read_text());baseline_thresholds=initial['thresholds']
    baseline_path=Path('runs/stream_v4_campaign_20261025/round_02/best')
    freeze={'selected_path':str(checkpoint),'weights_sha256':report['checkpoint_sha256'],'thresholds':thresholds,
        'selection_report_sha256':file_hash(args.run/'report.json'),'canonical_manifest_sha256':file_hash(args.data/'manifest.json'),
        'sealed_sha256':manifest['files']['sealed']['sha256'],'baseline_path':str(baseline_path),
        'baseline_sha256':file_hash(baseline_path/'model.safetensors'),'baseline_thresholds_from_initial_dev':baseline_thresholds,
        'test_status':'STARTED_AFTER_DEV_SELECTION; NO_RETUNING','started_unix':started}
    write_json(args.output/'frozen_before_test.json',freeze)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(checkpoint);model=AutoModelForSequenceClassification.from_pretrained(checkpoint,dtype=torch.float32).cuda().eval()
    sealed=list(read_jsonl(args.data/'sealed.jsonl'));dev=list(read_jsonl(args.data/'dev.jsonl'))
    old_pilot=list(read_jsonl(Path('data_v4/vi_natural_pause_20261025/cases.jsonl')))+list(read_jsonl(Path('data_v4/en_zh_natural_pause_20261025/cases.jsonl')))
    results={}
    for name,rows in [('natural_dev',dev),('new_held_natural',sealed),('v4_70_seen_regression',old_pilot),
        ('vi_legacy_full_dev',[{**r,'language':'vi'} for r in read_jsonl(Path('data_v2/frozen/dev.jsonl'))])]:
        ids,lengths=encode_rows(rows,tokenizer);probs=infer(model,ids,lengths)
        results[name]=metrics(rows,probs,thresholds)
        np.savez_compressed(args.output/(name+'_predictions.npz'),labels=np.array([LABEL2ID[r['label']] for r in rows]),
            prob_EOT=probs,language=np.array([r['language'] for r in rows]))
        if name=='new_held_natural':
            write_jsonl(args.output/'held_cases_predictions.jsonl',[{**r,'prob_EOT':float(p),
                'prediction':'EOT' if p>=thresholds[r['language']] else 'NEOT','threshold':thresholds[r['language']]} for r,p in zip(rows,probs,strict=True)])
        print(json.dumps({'scope':name,'metrics':results[name]}),flush=True)
    crawl=list(read_jsonl(Path('data_v4/fresh_crawl_20261025/review_cases.jsonl')))
    ids,lengths=encode_rows(crawl,tokenizer);probs=infer(model,ids,lengths)
    write_jsonl(args.output/'crawl_seen_unlabeled_predictions.jsonl',[{**r,'prob_EOT':float(p),'prediction':'EOT' if p>=thresholds[r['language']] else 'NEOT',
        'threshold':thresholds[r['language']],'label':None} for r,p in zip(crawl,probs,strict=True)])
    del model;gc.collect();torch.cuda.empty_cache()
    baseline=AutoModelForSequenceClassification.from_pretrained(baseline_path,dtype=torch.float32).cuda().eval()
    ids,lengths=encode_rows(sealed,tokenizer);baseline_probs=infer(baseline,ids,lengths)
    results['v4_control_matched_new_held']=metrics(sealed,baseline_probs,baseline_thresholds)
    np.savez_compressed(args.output/'v4_control_new_held_predictions.npz',labels=np.array([LABEL2ID[r['label']] for r in sealed]),prob_EOT=baseline_probs,language=np.array([r['language'] for r in sealed]))
    measured=all(m['accuracy']>.9 and m['eot_f1']>.9 for m in results['new_held_natural'].values())
    outcome={'status':'HELD_EVALUATION_COMPLETE','metrics':results,'selected_path':str(checkpoint),'weights_sha256':report['checkpoint_sha256'],
        'thresholds':thresholds,'automatic_public_held_three_language_target_met':measured,'human_verified_real_world_quality_certified':False,
        'held_cases':len(sealed),'held_conversations':manifest['files']['sealed']['calls'],'held_reference':'Automatic observed floor/VAD, no human adjudication',
        'crawl_cases':len(crawl),'crawl_gold_labels':0,'crawl_accuracy':None,'crawl_eot_f1':None,'old_holdout_evaluated':False,
        'accepted_serving_default_changed':False,'elapsed_seconds':time.time()-started}
    write_json(args.output/'report.json',outcome)
    freeze['test_status']='COMPLETED; NOW_SEEN_REGRESSION_ONLY';write_json(args.output/'frozen_before_test.json',freeze)
    write_json(args.data/'sealed_consumed.json',{'status':'SEEN_REGRESSION_ONLY','report_sha256':file_hash(args.output/'report.json'),
        'selected_weights_sha256':report['checkpoint_sha256'],'no_retuning_from_this_test':True})
    print(json.dumps(outcome),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path('data_v5/canonical_20261025'))
    parser.add_argument('--run',type=Path,default=Path('runs/natural_v5_train_20261025'))
    parser.add_argument('--allow-below-target-diagnostic',action='store_true',help='Explicitly consume the held cohort for diagnosis even below target')
    parser.add_argument('--output',type=Path,default=Path('outputs/natural_v5_20261025/final_eval'));run(parser.parse_args())
