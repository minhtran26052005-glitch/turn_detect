"""One-shot GPU runtime check with zero optimizer steps, no test scoring.

Exercises the real full-epoch loader/checkpoint path on private hardlinked
copies. Two synthetic token rows check BF16 forward/backward; AdamW never steps.
All real teacher files/receipts must retain inode/size/mtime/bytes unchanged.
"""
import gc
import json
import os
from pathlib import Path
import traceback

from . import translation_segmenter_full_train_v1 as train
from . import translation_segmenter_full_data_v2 as data
from .translation_segmenter_v15c import atomic_json

REAL=data.ROOT
JOB=data.JOB/'runtime_preflight_v1'


def main():
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification
    assert not (data.JOB/'current_dataset.json').exists(),'Run preflight only before any new training generation is ready'
    from .translation_segmenter_full_campaign_v1 import live
    assert not live(json.loads((REAL/'child.json').read_text())),'Real train child still active'
    scratch=JOB/'private_runtime';scratch.mkdir(parents=True,exist_ok=False)
    original_factory=AutoModelForSequenceClassification.from_config
    original_optimizer=torch.optim.AdamW
    results={};holder={}
    def factory(*args,**kwargs):
        model=original_factory(*args,**kwargs);holder['model']=model;return model
    class CheckOptimizer(original_optimizer):
        def load_state_dict(self,state):
            result=super().load_state_dict(state)
            model=holder['model'];model.train()
            x=torch.tensor([[1,10,2,20,2],[1,11,2,21,2]],device='cuda',dtype=torch.long)
            y=torch.tensor([0,1],device='cuda',dtype=torch.long)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(input_ids=x,attention_mask=torch.ones_like(x)).logits.float()
                loss=torch.nn.functional.cross_entropy(logits,y,label_smoothing=.01)
            assert logits.shape==(2,2) and torch.isfinite(loss)
            loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            self.zero_grad(set_to_none=True)
            holder['probe']={'bf16_forward_backward_finite':True,'restored_optimizer_states':len(self.state),'optimizer_steps_taken':0}
            return result
    AutoModelForSequenceClassification.from_config=staticmethod(factory)
    torch.optim.AdamW=CheckOptimizer
    try:
        train.ROOT=scratch;train.JOB=scratch/'full_epoch_v1'
        ids=np.array([[1,10,2,20,2],[1,11,2,21,2]],dtype=np.int32)
        encoded=(ids,np.array([5,5],dtype=np.int64),np.array([0,1],dtype=np.int64),np.array([1,2],dtype=np.int64))
        data.cache=lambda *_:encoded
        manifest={'generation':'runtime_preflight_no_training','source_cycle':84,'database_sha256':'synthetic-no-data-scored',
                  'fingerprints':{lang:{'train':{'rows':2,'sha256':'synthetic-loader-preflight'}} for lang in ('vi','en','zh')}}
        for lang in ('vi','en','zh'):
            assert not (data.JOB/'current_dataset.json').exists()
            source=REAL/'teachers'/lang;target=scratch/'teachers'/lang;target.mkdir(parents=True)
            before=source.joinpath('latest_resume.pt').stat();receipt=(source/'resume_receipt.json').read_bytes()
            os.link(source/'latest_resume.pt',target/'latest_resume.pt');(target/'resume_receipt.json').write_bytes(receipt)
            (target/'best').mkdir()
            for p in (source/'best').iterdir():
                if p.is_file():os.link(p,target/'best'/p.name)
            report=train.train_chunk(lang,manifest,1,budget=0)
            after=(source/'latest_resume.pt').stat()
            assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
            assert (source/'resume_receipt.json').read_bytes()==receipt
            assert report['steps_in_chunk']==0 and report['position']==0
            results[lang]={**holder['probe'],'real_teacher_checkpoint_and_receipt_unchanged':True,'loader_cursor_save_succeeded':True}
            atomic_json(JOB/'partial_report.json',{'languages':results,'model_quality_evaluation':False})
            holder.clear();gc.collect();torch.cuda.empty_cache()
            # Delete only self-created private binary copies/links, never real files.
            for p in target.rglob('*'):
                if p.is_file() and p.suffix in ('.pt','.safetensors'):
                    assert p.resolve().is_relative_to(scratch.resolve());p.unlink()
    finally:
        AutoModelForSequenceClassification.from_config=original_factory
        torch.optim.AdamW=original_optimizer
    atomic_json(JOB/'report.json',{'at_utc':data.now(),'complete':True,'languages':results,
      'all3_runtime_loader_and_bf16_gradient_paths_passed':True,'optimizer_steps_taken':0,
      'test_or_dev_rows_scored':0,'model_quality_evaluation':False,'real_checkpoints_unchanged':True})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic_json(JOB/'error.json',{'at_utc':data.now(),'error':repr(exc),'traceback':traceback.format_exc()});raise
