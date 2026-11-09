"""A private teacher/checkpoint/cache for ONE language, never mixed updates."""
import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import transformers.utils as hf_utils
import transformers.utils.import_utils as hf_imports
hf_utils.is_torchvision_available=hf_imports.is_torchvision_available=lambda:False
hf_utils.is_librosa_available=hf_imports.is_librosa_available=lambda:False
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from .common import LABEL2ID,file_hash,read_jsonl,write_json
from .metrics import calibrate_on_dev,classification_metrics
from .monolingual_data import encode_tail
from .trilingual_teacher import predict


def encoded(rows,tokenizer):
    ids=np.zeros((len(rows),256),np.int32);lengths=np.zeros(len(rows),np.int16)
    for i,r in enumerate(rows):
        value=encode_tail(tokenizer,r.get('context',''),r['text'])['input_ids'];ids[i,:len(value)]=value;lengths[i]=len(value)
    return ids,lengths,np.array([LABEL2ID[r['label']] for r in rows],dtype=np.int64)


def describe(rows,probs):
    labels=np.array([LABEL2ID[r['label']] for r in rows]);overall=calibrate_on_dev(labels,probs)
    slices={}
    for name in sorted({r['evaluation_slice'] for r in rows}):
        mask=np.array([r['evaluation_slice']==name for r in rows]);slices[name]=classification_metrics(labels[mask],probs[mask],overall['threshold'])
    return {'overall':overall,'slices':slices,'natural_target_met':slices['natural']['joint_score']>=.9}


def run(args):
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    manifest=json.loads((args.data/'manifest.json').read_text());assert manifest['language']==args.language
    rows={s:list(read_jsonl(args.data/(s+'.jsonl'))) for s in ['train','dev']}
    for s in rows:
        assert file_hash(args.data/(s+'.jsonl'))==manifest['split_sha256'][s]
        assert {r['language'] for r in rows[s]}=={args.language},'Cross-language contamination'
    torch.set_num_threads(2);torch.manual_seed(20261030);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available(),'Use the connected Colab GPU'
    local=Path(args.initial).is_dir();load_options={'local_files_only':True} if local else {'revision':args.revision}
    tokenizer=AutoTokenizer.from_pretrained(args.initial,**load_options)
    train_ids,lengths,labels=encoded(rows['train'],tokenizer);dev_ids,dev_len,dev_labels=encoded(rows['dev'],tokenizer)
    model=AutoModelForSequenceClassification.from_pretrained(args.initial,**load_options,num_labels=2,id2label={0:'NEOT',1:'EOT'},label2id=LABEL2ID,dtype=torch.float32).cuda()
    initial_probs=predict(model,dev_ids,dev_len,'cuda',32)
    initial=describe(rows['dev'],initial_probs);write_json(args.output/'initial_dev.json',initial)
    best=initial['overall']['joint_score'];best_step=0;history=[];stale=0
    # Save a private starting checkpoint too: an unsuccessful adaptation cannot
    # silently become a different language's best checkpoint.
    model.save_pretrained(args.output/'best',safe_serialization=True);tokenizer.save_pretrained(args.output/'best')
    np.savez_compressed(args.output/'dev_predictions.npz',labels=dev_labels,prob_EOT=initial_probs,language=np.array([args.language]*len(dev_labels)))
    write_json(args.output/'identity.json',{'language':args.language,'train_languages':[args.language],'dev_languages':[args.language],
        'shared_trainable_weights':False,'initial_checkpoint':str(args.initial),'initial_revision':None if local else args.revision,
        'initial_weights_sha256':file_hash(args.output/'best/model.safetensors'),
        'source_sha256':file_hash(Path(__file__)),'codec_sha256':file_hash(Path('turn_v2/monolingual_data.py')),'data_manifest_sha256':file_hash(args.data/'manifest.json'),
        'steps_max':args.steps,'microbatch':16,'accumulation':2,'selection':'Same-language dev joint accuracy/F1; report natural slice separately',
        'input':'Observed text only; newest current tokens, past context;256tokens','test_used':False})
    groups=defaultdict(list)
    for i,r in enumerate(rows['train']):groups[(r['training_source'],r['label'])].append(i)
    params=[{'params':[p for n,p in model.named_parameters() if not n.startswith(('classifier.','pooler.'))],'lr':1e-5},
        {'params':[p for n,p in model.named_parameters() if n.startswith(('classifier.','pooler.'))],'lr':5e-5}]
    optimizer=torch.optim.AdamW(params,weight_decay=.01);model.gradient_checkpointing_enable();rng=np.random.default_rng(20261030)
    for step in range(1,args.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        for micro in range(2):
            source='natural' if rng.random()<.5 else 'replay'
            index=np.array([rng.choice(groups[(source,label)]) for label in ['EOT','NEOT'] for _ in range(8)])
            width=int(lengths[index].max());x=torch.from_numpy(train_ids[index,:width].astype(np.int64)).cuda()
            mask=torch.arange(width,device='cuda')[None]<torch.from_numpy(lengths[index].astype(np.int64)).cuda()[:,None]
            logits=model(input_ids=x,attention_mask=mask.long()).logits.float()
            loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(labels[index]).cuda(),label_smoothing=.02)/2
            assert torch.isfinite(loss);loss.backward();losses.append(float(loss.detach())*2)
        factor=min(1.,step/30)*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        for p,base in zip(optimizer.param_groups,[1e-5,5e-5]):p['lr']=base*factor
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad);optimizer.step()
        if step%25==0:
            value={'language':args.language,'stage':'MONOLINGUAL_TEACHER_TRAINING','step':step,'steps_max':args.steps,'loss':float(np.mean(losses)),'elapsed_seconds':time.monotonic()-started}
            write_json(args.output/'progress.json',value);print(json.dumps(value),flush=True)
        if step%100==0 or step==args.steps:
            probs=predict(model,dev_ids,dev_len,'cuda',32);record={'step':step,**describe(rows['dev'],probs)};history.append(record)
            if record['overall']['joint_score']>best:
                best=record['overall']['joint_score'];best_step=step;stale=0
                model.save_pretrained(args.output/'best',safe_serialization=True);tokenizer.save_pretrained(args.output/'best')
                np.savez_compressed(args.output/'dev_predictions.npz',labels=dev_labels,prob_EOT=probs,language=np.array([args.language]*len(dev_labels)))
            else:stale+=1
            write_json(args.output/'history.json',history);print(json.dumps({'language':args.language,'development':record}),flush=True)
            if stale>=3:break
    del optimizer,model;gc.collect();torch.cuda.empty_cache()
    model=AutoModelForSequenceClassification.from_pretrained(args.output/'best',local_files_only=True,dtype=torch.float32).cuda().eval()
    selected=np.load(args.output/'dev_predictions.npz');dev=describe(rows['dev'],selected['prob_EOT']);write_json(args.output/'best_dev.json',dev)
    cache=args.output/'cache';cache.mkdir()
    for name,value in [('train_ids',train_ids),('train_lengths',lengths),('train_labels',labels),('train_language',np.array([args.language]*len(labels))),
        ('dev_ids',dev_ids),('dev_lengths',dev_len),('dev_labels',dev_labels),('dev_language',np.array([args.language]*len(dev_labels)))]:np.save(cache/(name+'.npy'),value)
    captured=[];hook=model.pooler.register_forward_hook(lambda module,inputs,output:captured.append(output.detach()))
    logits_out=np.lib.format.open_memmap(cache/'teacher_logits.npy',mode='w+',dtype=np.float32,shape=(len(labels),2))
    hidden_out=np.lib.format.open_memmap(cache/'teacher_hidden.npy',mode='w+',dtype=np.float16,shape=(len(labels),768));order=np.argsort(lengths)
    with torch.inference_mode():
        for start in range(0,len(labels),64):
            index=order[start:start+64];width=int(lengths[index].max());x=torch.from_numpy(train_ids[index,:width].astype(np.int64)).cuda()
            mask=torch.arange(width,device='cuda')[None]<torch.from_numpy(lengths[index].astype(np.int64)).cuda()[:,None]
            captured.clear();logits=model(input_ids=x,attention_mask=mask.long()).logits.float()
            logits_out[index]=logits.cpu().numpy();hidden_out[index]=captured[-1].cpu().numpy().astype(np.float16)
            if start%2048==0:write_json(args.output/'progress.json',{'language':args.language,'stage':'MONOLINGUAL_KD_CACHE','completed':start,'total':len(labels)})
    hook.remove();logits_out.flush();hidden_out.flush();torch.save(model.deberta.embeddings.word_embeddings.weight.detach().cpu(),cache/'teacher_embeddings.pt')
    write_json(args.output/'training_report.json',{'status':'TEACHER_AND_TRAIN_CACHE_COMPLETE','language':args.language,'steps':step,'best_step':best_step,
        'dev':{args.language:dev['overall']},'dev_slices':dev['slices'],'natural_target_met':dev['natural_target_met'],'train_samples':len(labels),'dev_samples':len(dev_labels),
        'data_manifest_sha256':file_hash(args.data/'manifest.json'),'checkpoint_sha256':file_hash(args.output/'best/model.safetensors'),
        'scope':manifest['scope'],'old_holdout_evaluated':False,'test_evaluated':False,'elapsed_seconds':time.monotonic()-started})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--language',choices=['en','zh'],required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--initial',default='microsoft/mdeberta-v3-base')
    p.add_argument('--revision',default='a0484667b22365f84929a935b5e50a51f71f159d');p.add_argument('--steps',type=int,default=400);run(p.parse_args())
