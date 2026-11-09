"""Separate low-memory local campaign while the Colab job is unreachable.

Train only final3 encoder layers and pooler/classifier, preserve the base file.
Checkpoints store exact trainable deltas with a pinned base checkpoint identity.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import transformers.utils as hf_utils
import transformers.utils.import_utils as hf_imports
# This text-only process does not require the incompatible local vision/audio
# optional dependencies. No installed package or global environment is edited.
hf_utils.is_torchvision_available = hf_imports.is_torchvision_available = lambda: False
hf_utils.is_librosa_available = hf_imports.is_librosa_available = lambda: False
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from safetensors.torch import save_file, load_file

from .common import LABEL2ID, file_hash, read_jsonl, write_json
from .stream_v4_train import encode_rows, infer, summarize


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(20261025);np.random.seed(20261025)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(args.model,local_files_only=True,dtype=torch.float32).cuda()
    for name,param in model.named_parameters():
        param.requires_grad_(name.startswith(('deberta.encoder.layer.9.','deberta.encoder.layer.10.','deberta.encoder.layer.11.','pooler.','classifier.')))
    trainable={name:param for name,param in model.named_parameters() if param.requires_grad}
    identity={'mode':'SEPARATE_LOCAL_RECOVERY_BRANCH; no Colab process altered','base_path':str(args.model.resolve()),
        'base_sha256':file_hash(args.model/'model.safetensors'),'trainable_parameters':sum(p.numel() for p in trainable.values()),
        'gpu':torch.cuda.get_device_name(0),'rounds':args.rounds,'steps_per_round':args.steps,'microbatch':4,'accumulation':2,
        'data_manifest_sha256':file_hash(args.data/'manifest.json'),'source_sha256':file_hash(Path(__file__)),
        'no_independent_real_vi_gold':True,'colab_state':'UNKNOWN_AFTER_HTTP404'}
    write_json(args.output/'identity.json',identity)
    train=list(read_jsonl(args.data/'train.jsonl'));dev=list(read_jsonl(args.data/'dev.jsonl'))
    x,l=encode_rows(train,tok);dx,dl=encode_rows(dev,tok);labels=np.array([LABEL2ID[r['label']] for r in train],dtype=np.int64)
    pools={}
    for lang in ['vi','en','zh']:
        for source in {r['source'] for r in train if r['language']==lang}:
            pools[(lang,source)]=np.array([i for i,r in enumerate(train) if r['language']==lang and r['source']==source])
    initial=summarize(dev,infer(model,dx,dl,batch_size=8));write_json(args.output/'initial_dev.json',initial)
    best=-1.;best_path=None;history=[];rng=np.random.default_rng(20261025)
    for round_id in range(1,args.rounds+1):
        lr=[2e-5,1e-5,5e-6][round_id-1]
        optimizer=torch.optim.AdamW(trainable.values(),lr=lr,weight_decay=.01)
        folder=args.output/f'round_{round_id:02d}';folder.mkdir()
        for step in range(1,args.steps+1):
            model.train();optimizer.zero_grad(set_to_none=True);losses=[]
            for micro in range(2):
                lang=['vi','en','zh'][(2*(step-1)+micro)%3]
                source='vi_legacy_rehearsal' if lang=='vi' else ('livekit_PREVIOUSLY_EXPOSED_development' if rng.random()<.8 else 'scicom_read_speech')
                pool=pools[(lang,source)];index=rng.choice(pool,4,replace=False);width=int(l[index].max())
                ids=torch.from_numpy(x[index,:width].astype(np.int64)).cuda()
                mask=torch.arange(width,device='cuda')[None]<torch.from_numpy(l[index].astype(np.int64)).cuda()[:,None]
                logits=model(input_ids=ids,attention_mask=mask.long()).logits.float()
                loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(labels[index]).cuda(),label_smoothing=.01)/2
                assert torch.isfinite(loss);loss.backward();losses.append(float(loss.detach())*2)
            torch.nn.utils.clip_grad_norm_(trainable.values(),1.)
            for group in optimizer.param_groups:group['lr']=lr*min(1.,step/20)*(.2+.8*.5*(1+math.cos(math.pi*step/args.steps)))
            optimizer.step()
            if step%20==0:
                progress={'stage':'LOCAL_TRAINING','round':round_id,'step':step,'steps_per_round':args.steps,'loss':float(np.mean(losses)),'elapsed_seconds':time.monotonic()-started,
                    'cuda_allocated_bytes':torch.cuda.memory_allocated()}
                write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
            if step%100==0 or step==args.steps:
                probs=infer(model,dx,dl,batch_size=8);metrics=summarize(dev,probs)
                record={'round':round_id,'step':step,**metrics};history.append(record);write_json(args.output/'history.json',history)
                if metrics['minimum_slice_joint_score']>best:
                    best=metrics['minimum_slice_joint_score'];best_path=folder/'trainable_delta.safetensors'
                    save_file({name:param.detach().cpu().contiguous() for name,param in trainable.items()},str(best_path))
                    write_json(folder/'best_dev.json',record);np.savez_compressed(folder/'dev_predictions.npz',labels=np.array([LABEL2ID[r['label']] for r in dev]),prob_EOT=probs)
                print(json.dumps({'stage':'LOCAL_DEV_EVALUATED',**record}),flush=True)
        del optimizer;gc.collect();torch.cuda.empty_cache()
        if best>.9:break
    write_json(args.output/'report.json',{'status':'LOCAL_ROUNDS_COMPLETE; REAL_TARGET_NOT_CERTIFIED','history':history,'best_path':str(best_path),
        'best_minimum_dev_score':best,'checkpoint_sha256':file_hash(best_path),'base_sha256':identity['base_sha256'],
        'independent_real_three_language_success':False,'elapsed_seconds':time.monotonic()-started,'colab_state_not_inferred':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('outputs/trilingual_20261024/runs/tri_teacher_20261024/best'))
    p.add_argument('--data',type=Path,default=Path('data_v4/causal_20261025'))
    p.add_argument('--rounds',type=int,choices=[1,2,3],default=3);p.add_argument('--steps',type=int,default=200)
    run(p.parse_args())
