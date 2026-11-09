"""Supervised text-only Qwen teacher adaptation; dev selection, no held scores."""
import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from .natural_v8_lora_r2 import disable_unused_torchao
disable_unused_torchao()
from peft import LoraConfig,get_peft_model
from transformers import AutoModelForCausalLM,AutoTokenizer
from .common import LABEL2ID,digest,file_hash,input_id,read_jsonl,write_json
from .natural_v5_train import summarize
from .natural_v6_features import SYSTEM,visible_pair


def batch(ids,index,tokenizer):
    width=max(len(ids[i]) for i in index);assert width<=1536
    values=np.full((len(index),width),tokenizer.pad_token_id,dtype=np.int64);mask=np.zeros_like(values)
    for k,i in enumerate(index):values[k,-len(ids[i]):]=ids[i];mask[k,-len(ids[i]):]=1
    values=torch.from_numpy(values).cuda();mask=torch.from_numpy(mask).cuda();positions=(mask.cumsum(-1)-1).clamp_min(0)
    return {'input_ids':values,'attention_mask':mask,'position_ids':positions}


def pair_logits(model,inputs,weights):
    base=model.get_base_model()
    hidden=base.model(**inputs,use_cache=False,return_dict=True).last_hidden_state[:,-1].float()
    return F.linear(hidden,weights)


def predict(model,ids,tokenizer,weights):
    model.eval();probs=np.zeros(len(ids));order=np.argsort([len(x) for x in ids])
    with torch.inference_mode():
        for start in range(0,len(ids),8):
            index=order[start:start+8];logits=pair_logits(model,batch(ids,index,tokenizer),weights)
            assert torch.isfinite(logits).all();probs[index]=torch.softmax(logits,dim=-1)[:,1].cpu().numpy()
    return probs


def run(args):
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    torch.set_num_threads(2);torch.manual_seed(20261029);rng=np.random.default_rng(20261029)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Path('data_v6/pho_text_20261025');manifest=json.loads((data/'manifest.json').read_text())
    for split in ['train','dev']:assert file_hash(data/(split+'.jsonl'))==manifest['files'][split]['sha256']
    train=[{**r,'training_source':'natural'} for r in read_jsonl(data/'train.jsonl')]
    dev=[{**r,'evaluation_slice':'natural'} for r in read_jsonl(data/'dev.jsonl')]
    dev.extend({**r,'evaluation_slice':'legacy_vi'} for r in read_jsonl(Path('data_v4/causal_20261025/dev.jsonl')) if r['language']=='vi')
    protected={r['language']+input_id(r) for r in read_jsonl(data/'sealed.jsonl')}
    protected.update(r['language']+input_id(r) for r in dev)
    train=[r for r in train if r['language']+input_id(r) not in protected]
    replay=[r for r in read_jsonl(Path('data_v4/causal_20261025/train.jsonl')) if r['language']+input_id(r) not in protected]
    for lang in ['vi','en','zh']:
        for label in ['NEOT','EOT']:
            pool=sorted([r for r in replay if r['language']==lang and r['label']==label],key=input_id)
            train.extend({**r,'training_source':'replay'} for r in pool[:600])
    receipt=json.loads(Path('outputs/natural_v5_20261025/qwen_download.json').read_text());path=Path(receipt['local_path'])
    small=json.loads(Path('outputs/natural_v6_20261025/encoder_download.json').read_text())
    small_tokenizer=AutoTokenizer.from_pretrained(small['local_path'],local_files_only=True)
    base_tokenizer=AutoTokenizer.from_pretrained('runs/stream_v4_campaign_20261025/round_02/best',local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True);tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    def encode(row):
        packet=visible_pair(row,base_tokenizer,small_tokenizer)
        values=tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(packet,ensure_ascii=False)}],tokenize=True,add_generation_prompt=True,return_dict=False)
        assert isinstance(values,list) and values and all(isinstance(value,int) for value in values), 'Expected a single list of integer token IDs'
        return values
    train_ids=[encode(r) for r in train];dev_ids=[encode(r) for r in dev]
    assert max(map(len,train_ids+dev_ids))<=1536
    base=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
    config=LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    model=get_peft_model(base,config);model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    a=tokenizer.encode('A',add_special_tokens=False);b=tokenizer.encode('B',add_special_tokens=False);assert len(a)==len(b)==1
    weights=model.get_base_model().lm_head.weight[[b[0],a[0]]].detach().float()
    trainable=[p for p in model.parameters() if p.requires_grad];assert trainable
    labels=np.array([LABEL2ID[r['label']] for r in train],dtype=np.int64);groups=defaultdict(list)
    for i,r in enumerate(train):groups[(r['language'],r['training_source'],r['label'])].append(i)
    groups={k:np.array(v) for k,v in groups.items()}
    config_json=json.loads(json.dumps(config.to_dict(),default=lambda value:sorted(value) if isinstance(value,set) else str(value)))
    identity={'repo':receipt['repo'],'revision':receipt['revision'],'base_files':receipt['files'],'adapter_config':config_json,
        'trainable_parameters':sum(p.numel() for p in trainable),'base_dtype':'bfloat16','quantization':None,
        'class_logits':'Frozen LM-head rows B=NEOT,A=EOT; normalized relative pair score',
        'training':'Supervised binary crossentropy through LoRA; not frozen features/zero-shot or RL',
        'natural_data_manifest_sha256':file_hash(data/'manifest.json'),'source_sha256':file_hash(Path(__file__)),
        'input_codec_sha256':file_hash(Path('turn_v2/natural_v6_features.py')),'train_rows_with_replay':len(train),'dev_rows':len(dev),
        'max_prompt_tokens':max(map(len,train_ids+dev_ids)),'steps_max':180,'microbatch':2,'accumulation':6,
        'held_labels_used_for_training_selection':False,'held_text_hashes_used_for_replay_dedup':True,
        'not_a_small_deployment_model':True,'predeclared_stop':'three stale dev evaluations or180steps; evalevery45steps',
        'prompt':SYSTEM}
    write_json(args.output/'identity.json',identity)
    # A true backward pass at the longest training prompts verifies memory and
    # adapter gradients before spending time on full dev. No optimizer update.
    probe_index=np.argsort([len(x) for x in train_ids])[-2:]
    model.train();model.zero_grad(set_to_none=True)
    probe_logits=pair_logits(model,batch(train_ids,probe_index,tokenizer),weights)
    probe_loss=F.cross_entropy(probe_logits,torch.from_numpy(labels[probe_index]).cuda())
    assert torch.isfinite(probe_loss);probe_loss.backward()
    nonzero=sum(p.grad is not None and bool(torch.any(p.grad!=0)) for p in trainable)
    assert nonzero>0,'LoRA did not receive gradients'
    write_json(args.output/'backward_probe.json',{'loss':float(probe_loss.detach()),'adapter_tensors_with_nonzero_gradients':nonzero,
        'max_prompt_tokens':max(len(train_ids[i]) for i in probe_index),'peak_cuda_allocated':torch.cuda.max_memory_allocated(),
        'optimizer_step_performed':False})
    model.zero_grad(set_to_none=True);del probe_logits,probe_loss;torch.manual_seed(20261029)
    initial=summarize(dev,predict(model,dev_ids,tokenizer,weights));write_json(args.output/'initial_dev.json',initial)
    best=initial['minimum_dev_slice_joint_score'];best_record=initial;best_path=None;history=[];stale=0;reason='MAX_STEPS'
    optimizer=torch.optim.AdamW(trainable,lr=1e-4,weight_decay=.01)
    for step in range(1,181):
        model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        for lang in ['vi','en','zh']:
            for _ in range(2):
                source='natural' if rng.random()<.8 else 'replay'
                index=np.array([rng.choice(groups[(lang,source,label)]) for label in ['NEOT','EOT']]);rng.shuffle(index)
                logits=pair_logits(model,batch(train_ids,index,tokenizer),weights)
                loss=F.cross_entropy(logits,torch.from_numpy(labels[index]).cuda(),label_smoothing=.02)/6
                assert torch.isfinite(loss);loss.backward();losses.append(float(loss.detach())*6)
        gradnorm=torch.nn.utils.clip_grad_norm_(trainable,1.)
        assert torch.isfinite(gradnorm)
        factor=min(1.,step/12)*(.2+.8*.5*(1+math.cos(math.pi*step/180)))
        optimizer.param_groups[0]['lr']=1e-4*factor;optimizer.step()
        if step%5==0:
            progress={'stage':'SUPERVISED_QWEN_LORA','step':step,'steps_max':180,'loss':float(np.mean(losses)),
                'gradient_norm':float(gradnorm),'elapsed_seconds':time.monotonic()-started,'cuda_peak_allocated':torch.cuda.max_memory_allocated()}
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        if step%45==0:
            probs=predict(model,dev_ids,tokenizer,weights);record={'step':step,**summarize(dev,probs)}
            history.append(record);write_json(args.output/'history.json',history)
            if record['minimum_dev_slice_joint_score']>best:
                best=record['minimum_dev_slice_joint_score'];best_record=record;best_path=str(args.output/'best_adapter');stale=0
                model.save_pretrained(best_path,safe_serialization=True);tokenizer.save_pretrained(best_path)
                write_json(Path(best_path)/'best_dev.json',record)
                np.savez_compressed(args.output/'best_dev_predictions.npz',labels=np.array([LABEL2ID[r['label']] for r in dev]),prob_EOT=probs)
            else:stale+=1
            write_json(args.output/'selection.json',{'best_path':best_path,'best_dev':best_record,'stale_evaluations':stale,'held_classifier_scored':False})
            print(json.dumps({'stage':'SUPERVISED_QWEN_LORA_DEV',**record}),flush=True)
            if record['development_all_slices_over_90'] and record['natural_hold_false_eot_below_10_percent']:reason='DEV_TARGET';break
            if stale>=3:reason='DEV_PLATEAU';break
    write_json(args.output/'report.json',{'status':'SUPERVISED_QWEN_LORA_FINISHED','stop_reason':reason,'steps_completed':step,
        'best_path':best_path,'best_dev':best_record,'history':history,'held_classifier_scored':False,
        'adapter_sha256':file_hash(Path(best_path)/'adapter_model.safetensors') if best_path else None,
        'elapsed_seconds':time.monotonic()-started,'human_gold_quality_certified':False,'real_target_achieved':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('runs/natural_v8_lora_20261025_r3'));run(parser.parse_args())
