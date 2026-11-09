"""Colab-only timing of real TRAIN-shaped forward/backward; no optimizer writes."""
import gc
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback

ROOT=Path('outputs/translation_segmenter_v15d_20261026')
JOB=ROOT/'full_epoch_v1'
CONTROL=JOB/'throughput_probe_v1'


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
        path=JOB/'datasets/cycle000084_full_epoch_contiguous_v1/dataset.sqlite'
        db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True,timeout=2)
        reports={}
        for language in ('vi','en','zh'):
            raw=db.execute("SELECT tokens,label FROM records WHERE language=? AND split='train' AND disposition='accepted' ORDER BY example_id LIMIT 64",(language,)).fetchall()
            assert len(raw)==64
            encoded=[np.frombuffer(x[0],dtype='<u4') for x in raw]
            batches=[]
            for start in (0,32):
                selected=encoded[start:start+32];width=max(map(len,selected))
                array=np.zeros((32,width),dtype=np.int64);mask=np.zeros_like(array)
                for i,ids in enumerate(selected):array[i,:len(ids)]=ids;mask[i,:len(ids)]=1
                batches.append((torch.from_numpy(array).cuda(),torch.from_numpy(mask).cuda(),torch.tensor([LABELS[label] for _,label in raw[start:start+32]],device='cuda')))
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
                state('TIMING_TRAIN_FORWARD_BACKWARD_NO_OPTIMIZER_STEP',language=language,repeat=repeat,seconds=duration)
            reports[language]={'seconds_per_64_row_backward':float(np.mean(elapsed)),'timed_repeats':3,'max_sequence_widths':[x[0].shape[1] for x in batches],'microbatch':32,'accumulation':2,'adamw_step_included':False}
            del batches;gc.collect();torch.cuda.empty_cache()
        db.close();del model;gc.collect();torch.cuda.empty_cache()
    after=checkpoint.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    result={'complete':True,'languages':reports,'shared_model_architecture':'mdeberta-v3-base','representative_weights_language':'vi','actual_train_tokens_only':True,'optimizer_steps':0,'checkpoint_unchanged':True,'test_or_dev_scored':False,'not_a_quality_evaluation':True}
    atomic_json(CONTROL/'report.json',result);state('COMPLETE',**result)


if __name__=='__main__':
    CONTROL.mkdir(exist_ok=True)
    try:main()
    except Exception as exc:
        from .translation_segmenter_v15c import atomic_json
        atomic_json(CONTROL/'error.json',{'error':repr(exc),'traceback':traceback.format_exc()});state('FAILED_PRESERVED',error=repr(exc));raise
