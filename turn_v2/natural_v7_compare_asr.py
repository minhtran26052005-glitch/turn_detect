"""Matched dev-only ASR route comparison with unchanged teacher/thresholds."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from .common import LABEL2ID,file_hash,read_jsonl,write_json
from .metrics import classification_metrics,calibrate_on_dev
from .natural_v6_finetune import encode_rows
from .stream_v4_train import infer


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    old=[r for r in read_jsonl(Path('data_v6/turn_context_20261025/dev.jsonl')) if r['language']=='vi']
    new=[r for r in read_jsonl(Path('data_v6/pho_text_20261025/dev.jsonl')) if r['language']=='vi']
    assert [r['case_id'] for r in old]==[r['case_id'] for r in new]
    assert [r['label'] for r in old]==[r['label'] for r in new]
    labels=np.array([LABEL2ID[r['label']] for r in old])
    path=Path('runs/natural_v5_train_20261025/round_02/best')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True);model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True,dtype=torch.float32).cuda().eval()
    ids,lengths=encode_rows(old,tokenizer);oldp=infer(model,ids,lengths,batch_size=32)
    selected=calibrate_on_dev(labels,oldp);threshold=selected['threshold']
    identity={'teacher_path':str(path),'weights_sha256':file_hash(path/'model.safetensors'),'threshold_fixed_from_old_dev':threshold,
        'old_sha256':file_hash(Path('data_v6/turn_context_20261025/dev.jsonl')),'new_sha256':file_hash(Path('data_v6/pho_text_20261025/dev.jsonl')),
        'scope':'Matched Vietnamese development; ASR route including its decoding changed, model/codec/cases/labels/threshold fixed',
        'held_loaded':False}
    write_json(args.output/'frozen_before_new_ASR_scores.json',identity)
    ids,lengths=encode_rows(new,tokenizer);newp=infer(model,ids,lengths,batch_size=32)
    np.savez_compressed(args.output/'predictions.npz',labels=labels,old_prob_EOT=oldp,new_prob_EOT=newp,case_id=np.array([r['case_id'] for r in old]))
    result={'status':'MATCHED_ASR_DEV_COMPARISON','old_whisper_small':selected,'new_phowhisper_same_threshold':classification_metrics(labels,newp,threshold),
        'mean_absolute_probability_change':float(np.mean(np.abs(oldp-newp))),'changed_decisions':int(np.sum((oldp>=threshold)!=(newp>=threshold))),
        'identity':identity,'elapsed_seconds':time.monotonic()-started,'independent_test_claim':False,'human_ASR_reference_available':False}
    write_json(args.output/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('outputs/natural_v7_20261025/matched_ASR_dev'))
    main(parser.parse_args())
