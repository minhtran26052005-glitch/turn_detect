"""Colab-only synthetic shape timing of TRAIN forward/backward; no optimizer writes."""
import gc
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback

ROOT=Path('outputs/translation_segmenter_v15d_20261026')
JOB=ROOT/'full_epoch_v1'
CONTROL=JOB/'throughput_probe_v2'


def state(stage,**fields):
    from .translation_segmenter_v15c import atomic_json
    atomic_json(CONTROL/'status.json',{'stage':stage,'at_utc':time.time(),**fields})


def main():
    import fcntl
    import numpy as np
    import torch
    import transformers.utils as u
    import transformers.utils.import_utils as ui
    u.is_torchvision_available=ui.is_torchvision_available=lambda:False
    u.is_librosa_available=ui.is_librosa_available=lambda:False
    from transformers import AutoConfig,AutoModelForSequenceClassification
    from .translation_segmenter_v15c import LABELS,atomic_json
    torch.set_num_threads(2)
    checkpoint=ROOT/'teachers/vi/latest_resume.pt';before=checkpoint.stat()
    state('WAITING_FOR_SHARED_GPU_LOCK')
    with (JOB/'GPU_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state('LOADING_EXISTING_CHECKPOINT_FOR_TIMING_ONLY')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        cfg=AutoConfig.from_pretrained(ROOT/'teachers/vi/best',local_files_only=True)
        model=AutoModelForSequenceClassification.from_config(cfg,dtype=torch.float32).cuda()
        model.load_state_dict(saved['model'],strict=True);del saved;gc.collect()
        model.gradient_checkpointing_enable();model.train()
        reports={}
        for width in (64,128,256):
            state('PREPARING_SYNTHETIC_SHAPE_TIMING_ONLY',sequence_width=width)
            batches=[]
            for _ in range(2):
                array=np.full((32,width),10,dtype=np.int64);array[:,0]=1;array[:,-3]=2;array[:,-1]=2
                mask=np.ones_like(array);target=torch.arange(32,device='cuda')%2
                batches.append((torch.from_numpy(array).cuda(),torch.from_numpy(mask).cuda(),target))
            elapsed=[]
            for repeat in range(4):
                model.zero_grad(set_to_none=True);torch.cuda.synchronize();started=time.perf_counter()
                for ids,mask,target in batches:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model(input_ids=ids,attention_mask=mask).logits.float()
                        loss=torch.nn.functional.cross_entropy(logits,target,label_smoothing=.01)/2
                    assert torch.isfinite(loss);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                torch.cuda.synchronize();duration=time.perf_counter()-started
                if repeat:elapsed.append(duration)
                state('TIMING_TRAIN_FORWARD_BACKWARD_NO_OPTIMIZER_STEP',sequence_width=width,repeat=repeat,seconds=duration)
            reports[str(width)]={'seconds_per_64_row_backward':float(np.mean(elapsed)),'timed_repeats':3,'max_sequence_widths':[x[0].shape[1] for x in batches],'microbatch':32,'accumulation':2,'adamw_step_included':False}
            del batches;gc.collect();torch.cuda.empty_cache()
        del model;gc.collect();torch.cuda.empty_cache()
    after=checkpoint.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    result={'complete':True,'sequence_widths':reports,'shared_model_architecture':'mdeberta-v3-base','representative_weights_language':'vi','synthetic_tokens_only':True,'dataset_reads_or_writes':False,'optimizer_steps':0,'checkpoint_unchanged':True,'test_or_dev_scored':False,'optimizer_and_storage_overhead_not_included':True,'not_a_quality_evaluation':True}
    atomic_json(CONTROL/'report.json',result);state('COMPLETE',**result)


if __name__=='__main__':
    CONTROL.mkdir(exist_ok=True)
    try:main()
    except Exception as exc:
        from .translation_segmenter_v15c import atomic_json
        atomic_json(CONTROL/'error.json',{'error':repr(exc),'traceback':traceback.format_exc()});state('FAILED_PRESERVED',error=repr(exc));raise
