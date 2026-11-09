"""Text-only adaptation to new call-isolated natural conversations."""
import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import LABEL2ID, file_hash, read_jsonl, write_json
from .metrics import classification_metrics
from .stream_v4_train import encode_rows, infer


def summarize(rows,probs):
    labels=np.array([LABEL2ID[r['label']] for r in rows]);thresholds={};slices={}
    for lang in ['vi','en','zh']:
        sources=sorted({r['evaluation_slice'] for r in rows if r['language']==lang})
        masks={source:np.array([r['language']==lang and r['evaluation_slice']==source for r in rows]) for source in sources}
        candidates=[]
        for threshold in np.linspace(.05,.95,181):
            metrics={source:classification_metrics(labels[mask],probs[mask],threshold) for source,mask in masks.items()}
            candidates.append((min(m['joint_score'] for m in metrics.values()),-sum(m['false_eot'] for m in metrics.values()),-abs(threshold-.5),threshold,metrics))
        selected=max(candidates,key=lambda value:value[:3]);thresholds[lang]=float(selected[3])
        slices.update({lang+':'+source:value for source,value in selected[4].items()})
    score=min(m['joint_score'] for m in slices.values())
    natural={lang:slices[lang+':natural'] for lang in ['vi','en','zh']}
    target=all(m['accuracy']>.9 and m['eot_f1']>.9 for m in natural.values())
    return {'thresholds':thresholds,'slices':slices,'natural_languages':natural,'minimum_dev_slice_joint_score':score,
        'development_target_met':target,'development_all_slices_over_90':score>.9,
        'natural_hold_false_eot_below_10_percent':all(m['false_eot_rate']<=.1 for m in natural.values()),
        'independent_real_target_achieved':False,'scope':'Development automatic floor labels; not independent human gold'}


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(20261027)
    rng=np.random.default_rng(20261027)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((args.data/'manifest.json').read_text())
    for split in ['train','dev']:assert file_hash(args.data/(split+'.jsonl'))==manifest['files'][split]['sha256']
    train=[{**r,'training_source':'natural'} for r in read_jsonl(args.data/'train.jsonl')]
    replay=list(read_jsonl(Path('data_v4/causal_20261025/train.jsonl')))
    train.extend({**r,'training_source':'replay'} for r in replay)
    dev=[{**r,'evaluation_slice':'natural'} for r in read_jsonl(args.data/'dev.jsonl')]
    dev.extend({**r,'evaluation_slice':'legacy_vi'} for r in read_jsonl(Path('data_v4/causal_20261025/dev.jsonl')) if r['language']=='vi')
    model=AutoModelForSequenceClassification.from_pretrained(args.model,dtype=torch.float32).cuda()
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    x,lengths=encode_rows(train,tokenizer);dx,dl=encode_rows(dev,tokenizer)
    labels=np.array([LABEL2ID[r['label']] for r in train],dtype=np.int64)
    groups=defaultdict(list)
    for i,row in enumerate(train):groups[(row['language'],row['training_source'],row['label'])].append(i)
    groups={key:np.array(value) for key,value in groups.items()}
    write_json(args.output/'identity.json',{'initial_path':str(args.model),'initial_sha256':file_hash(args.model/'model.safetensors'),
        'canonical_data_sha256':file_hash(args.data/'manifest.json'),'replay_train_sha256':file_hash(Path('data_v4/causal_20261025/train.jsonl')),
        'source_sha256':file_hash(Path(__file__)),'codec_source_sha256':file_hash(Path('turn_v2/stream_v4_train.py')),
        'natural_train_samples':manifest['files']['train']['cases'],'replay_samples':len(replay),'dev_samples':len(dev),
        'rounds_max':args.rounds,'steps_per_round':args.steps,'microbatch':16,'accumulation':2,
        'stopping':'strict development target plus hold FPR<=10%, or six stale dev evaluations after at least two rounds, or maxrounds',
        'sealed_data_loaded':False,'old_v4_pilot_loaded':False,'input':'Text only; latest current tokens plus strictly past text context'})
    probs=infer(model,dx,dl);initial=summarize(dev,probs)
    write_json(args.output/'initial_dev.json',initial)
    np.savez_compressed(args.output/'initial_dev.npz',labels=np.array([LABEL2ID[r['label']] for r in dev]),prob_EOT=probs)
    best=initial['minimum_dev_slice_joint_score'];best_path=str(args.model);best_record=initial;history=[];stale=0;reason='MAX_ROUNDS'
    model.gradient_checkpointing_enable()
    for round_id in range(1,args.rounds+1):
        folder=args.output/f'round_{round_id:02d}';folder.mkdir()
        lr=[1e-5,5e-6,2e-6,1e-6][round_id-1]
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
        for step in range(1,args.steps+1):
            model.train();optimizer.zero_grad(set_to_none=True);losses=[]
            for micro in range(2):
                lang=['vi','en','zh'][(2*(step-1)+micro)%3]
                native_probability=.6 if lang=='vi' else .8
                source='natural' if rng.random()<native_probability else 'replay'
                index=np.concatenate([rng.choice(groups[(lang,source,label)],8,replace=len(groups[(lang,source,label)])<8) for label in ['NEOT','EOT']])
                rng.shuffle(index);width=int(lengths[index].max())
                ids=torch.from_numpy(x[index,:width].astype(np.int64)).cuda()
                mask=torch.arange(width,device='cuda')[None]<torch.from_numpy(lengths[index].astype(np.int64)).cuda()[:,None]
                logits=model(input_ids=ids,attention_mask=mask.long()).logits.float()
                loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(labels[index]).cuda(),label_smoothing=.02)/2
                assert torch.isfinite(loss);loss.backward();losses.append(float(loss.detach())*2)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            factor=min(1.,step/40)*(.2+.8*.5*(1+math.cos(math.pi*step/args.steps)))
            for group in optimizer.param_groups:group['lr']=lr*factor
            optimizer.step()
            if step%50==0:
                progress={'stage':'NATURAL_TEXT_TRAINING','round':round_id,'step':step,'steps_per_round':args.steps,
                    'loss':float(np.mean(losses)),'elapsed_seconds':time.monotonic()-started}
                write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
            if step%150==0 or step==args.steps:
                probs=infer(model,dx,dl);record={'round':round_id,'step':step,**summarize(dev,probs)}
                history.append(record);write_json(args.output/'history.json',history)
                if record['minimum_dev_slice_joint_score']>best:
                    best=record['minimum_dev_slice_joint_score'];best_path=str(folder/'best');best_record=record;stale=0
                    model.save_pretrained(best_path,safe_serialization=True);tokenizer.save_pretrained(best_path)
                    write_json(folder/'best_dev.json',record)
                    np.savez_compressed(folder/'dev_predictions.npz',labels=np.array([LABEL2ID[r['label']] for r in dev]),prob_EOT=probs)
                else:stale+=1
                write_json(args.output/'selection.json',{'status':'IN_PROGRESS','best_path':best_path,'best_dev':best_record,'stale_evaluations':stale,'sealed_data_loaded':False})
                print(json.dumps({'stage':'NATURAL_DEV',**record}),flush=True)
                if record['development_all_slices_over_90'] and record['natural_hold_false_eot_below_10_percent']:
                    reason='DEV_TARGET';break
                if round_id>=2 and stale>=6:reason='DEV_PLATEAU';break
        del optimizer,model;gc.collect();torch.cuda.empty_cache()
        if reason!='MAX_ROUNDS':break
        model=AutoModelForSequenceClassification.from_pretrained(best_path,dtype=torch.float32).cuda();model.gradient_checkpointing_enable()
    write_json(args.output/'report.json',{'status':'NATURAL_TEXT_CAMPAIGN_FINISHED_PENDING_HELD_EVALUATION','stop_reason':reason,
        'best_path':best_path,'best_dev':best_record,'history':history,'checkpoint_sha256':file_hash(Path(best_path)/'model.safetensors'),
        'elapsed_seconds':time.monotonic()-started,'sealed_data_loaded':False,'independent_real_target_achieved':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path('data_v5/canonical_20261025'))
    parser.add_argument('--model',type=Path,default=Path('runs/stream_v4_campaign_20261025/round_02/best'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--rounds',type=int,choices=[1,2,3,4],default=4)
    parser.add_argument('--steps',type=int,default=600);run(parser.parse_args())
