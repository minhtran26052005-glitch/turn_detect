"""Freeze selected round and score held material once, without tuning on it."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import LABEL2ID, file_hash, read_jsonl, write_json, write_jsonl
from .metrics import classification_metrics
from .stream_v4_train import encode_rows, infer


def language_metrics(rows, probs, thresholds):
    labels=np.asarray([LABEL2ID[r['label']] for r in rows]);langs=np.asarray([r['language'] for r in rows])
    return {lang:classification_metrics(labels[langs==lang],probs[langs==lang],thresholds[lang]) for lang in sorted(set(langs))}


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    campaign=json.loads((args.campaign/'campaign.json').read_text(encoding='utf-8'))
    assert campaign['status']=='BOUNDED_ROUNDS_FINISHED_REAL_TARGET_NOT_CERTIFIED' and campaign['best_path']
    checkpoint=Path(campaign['best_path'])
    selected=json.loads((checkpoint.parent/'best_dev.json').read_text(encoding='utf-8'))
    if args.followup:
        followup=json.loads((args.followup/'report.json').read_text(encoding='utf-8'))
        assert followup['status']=='FOLLOWUP_FINISHED_REAL_TARGET_NOT_CERTIFIED'
        checkpoint=Path(followup['best_path'])
        selected=followup['best_record']
        assert file_hash(checkpoint/'model.safetensors')==followup['checkpoint_sha256']
    thresholds=selected['thresholds']
    identity={'selected_path':str(checkpoint),'weights_sha256':file_hash(checkpoint/'model.safetensors'),'thresholds':thresholds,
        'selected_dev_sha256':file_hash(checkpoint.parent/'best_dev.json'),'test_status':'STARTED_AFTER_SELECTION; NO_RETUNING',
        'sealed_sha256':file_hash(args.data/'sealed.jsonl'),'fresh_crawl_sha256':file_hash(args.crawl),'started_unix':time.time(),
        'followup_used':bool(args.followup),'selection_basis':'Development only; first local held results not opened before follow-up protocol was fixed'}
    if args.vi_pilot and args.en_zh_pilot:
        base=Path('runs/tri_teacher_20261024/best')
        baseline_sha=file_hash(base/'model.safetensors')
        assert baseline_sha=='e962d06668aeceaeb7c05116a1a55e664f5f3f21e9c7eaf1d250b922028a4b80'
        baseline_thresholds=json.loads((args.campaign/'initial_dev.json').read_text())['thresholds']
        identity['baseline_weights_sha256']=baseline_sha
        identity['baseline_dev_thresholds']=baseline_thresholds
    write_json(args.output/'frozen_before_test.json',identity)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(checkpoint)
    model=AutoModelForSequenceClassification.from_pretrained(checkpoint,dtype=torch.float32).cuda().eval()
    all_results={}
    scopes=[('dev',list(read_jsonl(args.data/'dev.jsonl'))),('sealed_read_speech',list(read_jsonl(args.data/'sealed.jsonl'))),
        ('vi_full_legacy_dev',[{**r,'language':'vi'} for r in read_jsonl(Path('data_v2/frozen/dev.jsonl'))])]
    if args.vi_pilot:
        pilot=list(read_jsonl(args.vi_pilot))
        assert len(pilot)==30 and all(not row['is_human_gold'] for row in pilot)
        identity['fresh_vi_sha256']=file_hash(args.vi_pilot)
        write_json(args.output/'frozen_before_test.json',identity)
        scopes.append(('fresh_vi_natural_auto_labels',pilot))
    if args.en_zh_pilot:
        en_zh=list(read_jsonl(args.en_zh_pilot))
        assert len(en_zh)==40 and all(not row['is_human_gold'] for row in en_zh)
        identity['fresh_en_zh_sha256']=file_hash(args.en_zh_pilot)
        write_json(args.output/'frozen_before_test.json',identity)
        scopes.append(('fresh_en_zh_natural_auto_labels',en_zh))
    for name,rows in scopes:
        ids,lengths=encode_rows(rows,tok);probs=infer(model,ids,lengths)
        metrics=language_metrics(rows,probs,thresholds)
        all_results[name]=metrics
        np.savez_compressed(args.output/(name+'_predictions.npz'),labels=np.array([LABEL2ID[r['label']] for r in rows]),prob_EOT=probs,language=np.array([r['language'] for r in rows]))
        if name.startswith('fresh_'):
            write_jsonl(args.output/(name+'_predictions.jsonl'),[{**row,'prob_EOT':float(p),'prediction':'EOT' if p>=thresholds[row['language']] else 'NEOT',
                'threshold':thresholds[row['language']]} for row,p in zip(rows,probs,strict=True)])
        print(json.dumps({'scope':name,'metrics':metrics}),flush=True)
    rows=list(read_jsonl(args.crawl));ids,lengths=encode_rows(rows,tok);probs=infer(model,ids,lengths)
    traces=[{**row,'prob_EOT':float(p),'prediction':'EOT' if p>=thresholds[row['language']] else 'NEOT','threshold':thresholds[row['language']],
        'is_gold':False,'label_status':'PENDING_INDEPENDENT_REVIEW'} for row,p in zip(rows,probs,strict=True)]
    write_jsonl(args.output/'fresh_crawl_predictions.jsonl',traces)
    if args.vi_pilot and args.en_zh_pilot:
        # Matched-input comparison: same causal codec and fresh cases, distinct
        # per-candidate thresholds frozen on dev before seeing these cases.
        import gc
        del model
        gc.collect();torch.cuda.empty_cache()
        baseline=AutoModelForSequenceClassification.from_pretrained(base,dtype=torch.float32).cuda().eval()
        baseline_rows=pilot+en_zh
        ids,lengths=encode_rows(baseline_rows,tok);base_probs=infer(baseline,ids,lengths)
        all_results['original_teacher_matched_fresh_auto_labels']=language_metrics(baseline_rows,base_probs,baseline_thresholds)
        np.savez_compressed(args.output/'original_teacher_fresh_predictions.npz',labels=np.array([LABEL2ID[r['label']] for r in baseline_rows]),
            prob_EOT=base_probs,language=np.array([r['language'] for r in baseline_rows]))
    result={'status':'EVALUATED_TARGET_NOT_CERTIFIED','selected_round':selected.get('round'),'dev_slices':selected['slices'],
        'metrics':all_results,'fresh_crawl_cases':len(rows),'fresh_crawl_gold_labels':0,'fresh_crawl_accuracy':None,'fresh_crawl_eot_f1':None,
        'frozen':identity,'old_holdout_evaluated':False,'all_real_languages_over_90':False,
        'reason':'VI real-pause gold absent; fresh crawl unlabeled; dev scores are development and read-speech held set is not spontaneous dialogue',
        'elapsed_seconds':time.time()-identity['started_unix']}
    write_json(args.output/'report.json',result)
    identity['test_status']='COMPLETED; SEEN_TEST_NOW_REGRESSION_ONLY';write_json(args.output/'frozen_before_test.json',identity)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--campaign',type=Path,default=Path('runs/stream_v4_campaign_20261025'))
    p.add_argument('--data',type=Path,default=Path('data_v4/causal_20261025'))
    p.add_argument('--crawl',type=Path,default=Path('data_v4/fresh_crawl_20261025/review_cases.jsonl'))
    p.add_argument('--followup',type=Path)
    p.add_argument('--vi-pilot',type=Path)
    p.add_argument('--en-zh-pilot',type=Path)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
