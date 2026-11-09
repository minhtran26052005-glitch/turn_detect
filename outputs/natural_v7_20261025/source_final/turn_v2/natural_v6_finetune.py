"""Fine-tune a smaller multilingual cross-encoder with past partner context."""
import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from .common import LABEL2ID,file_hash,normalize,read_jsonl,write_json
from .natural_v5_train import summarize
from .stream_v4_train import infer


def encode_packet(tokenizer,row,maximum=512):
    current=tokenizer.encode(normalize(row['text']),add_special_tokens=False)[-254:]
    context=tokenizer.encode(normalize(row.get('context','')),add_special_tokens=False)
    context=context[-(maximum-3-len(current)):]
    return [tokenizer.cls_token_id]+context+[tokenizer.sep_token_id]+current+[tokenizer.sep_token_id]


def encode_rows(rows,tokenizer):
    ids=np.zeros((len(rows),512),np.int32);lengths=np.zeros(len(rows),np.int16)
    for i,row in enumerate(rows):
        values=encode_packet(tokenizer,row);ids[i,:len(values)]=values;lengths[i]=len(values)
    return ids,lengths


def run(args):
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    torch.set_num_threads(2);torch.manual_seed(20261026);rng=np.random.default_rng(20261026)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    receipt=json.loads(Path('outputs/natural_v6_20261025/encoder_download.json').read_text());encoder=Path(receipt['local_path'])
    manifest=json.loads((args.data/'manifest.json').read_text())
    for split in ['train','dev']:assert file_hash(args.data/(split+'.jsonl'))==manifest['files'][split]['sha256']
    train=[{**r,'training_source':'natural'} for r in read_jsonl(args.data/'train.jsonl')]
    # Replay stays training-only, and the former Vietnamese deployment control
    # remains untouched even if this new experiment forgets its task.
    replay=[{**r,'training_source':'replay'} for r in read_jsonl(Path('data_v4/causal_20261025/train.jsonl'))]
    train+=replay
    dev=[{**r,'evaluation_slice':'natural'} for r in read_jsonl(args.data/'dev.jsonl')]
    dev.extend({**r,'evaluation_slice':'legacy_vi'} for r in read_jsonl(Path('data_v4/causal_20261025/dev.jsonl')) if r['language']=='vi')
    tokenizer=AutoTokenizer.from_pretrained(encoder,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(encoder,local_files_only=True,dtype=torch.float32,num_labels=2).cuda()
    # Frozen embeddings reduce memory and protect multilingual lexical coverage.
    for name,param in model.named_parameters():
        if 'embeddings.' in name:param.requires_grad_(False)
    body=[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith('classifier.')]
    head=[p for name,p in model.named_parameters() if p.requires_grad and name.startswith('classifier.')]
    x,lengths=encode_rows(train,tokenizer);dx,dl=encode_rows(dev,tokenizer)
    y=np.array([LABEL2ID[r['label']] for r in train],dtype=np.int64);groups=defaultdict(list)
    for i,row in enumerate(train):groups[(row['language'],row['training_source'],row['label'])].append(i)
    groups={key:np.array(value) for key,value in groups.items()}
    write_json(args.output/'identity.json',{'encoder':receipt['repo'],'revision':receipt['revision'],'model_files':receipt['files'],
        'data_manifest_sha256':file_hash(args.data/'manifest.json'),'source_sha256':file_hash(Path(__file__)),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'all_parameters':sum(p.numel() for p in model.parameters()),
        'data_train':manifest['files']['train']['cases'],'dev_samples':len(dev),'rounds_max':3,'steps_per_round':200,
        'microbatch':16,'accumulation':2,'frozen_embeddings':True,'sealed_loaded':False,
        'input':'Text-only current tail254tokens + previous partner tail up to512tokens total, normalize punctuation',
        'selection':'minimum natural-language/legacy-VI development accuracy/F1, thresholds dev-only',
        'initialization':'Pretrained multilingual encoder, new random EOT head; not a distilled accepted model'})
    best=-1.;best_path=None;best_record=None;stale=0;history=[];reason='MAX_ROUNDS'
    for round_id in range(1,4):
        lr=[2e-5,1e-5,5e-6][round_id-1]
        optimizer=torch.optim.AdamW([{'params':body,'lr':lr},{'params':head,'lr':lr*10}],weight_decay=.01)
        folder=args.output/f'round_{round_id:02d}';folder.mkdir()
        for step in range(1,201):
            model.train();optimizer.zero_grad(set_to_none=True);losses=[]
            for micro in range(2):
                lang=['vi','en','zh'][(2*(step-1)+micro)%3];source='natural' if rng.random()<.8 else 'replay'
                index=np.concatenate([rng.choice(groups[(lang,source,label)],8,replace=len(groups[(lang,source,label)])<8) for label in ['NEOT','EOT']])
                rng.shuffle(index);width=int(lengths[index].max());ids=torch.from_numpy(x[index,:width].astype(np.int64)).cuda()
                mask=torch.arange(width,device='cuda')[None]<torch.from_numpy(lengths[index].astype(np.int64)).cuda()[:,None]
                logits=model(input_ids=ids,attention_mask=mask.long()).logits.float()
                loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(y[index]).cuda(),label_smoothing=.02)/2
                assert torch.isfinite(loss);loss.backward();losses.append(float(loss.detach())*2)
            torch.nn.utils.clip_grad_norm_([*body,*head],1.)
            factor=min(1.,step/20)*(.2+.8*.5*(1+math.cos(math.pi*step/200)))
            optimizer.param_groups[0]['lr']=lr*factor;optimizer.param_groups[1]['lr']=lr*10*factor
            optimizer.step()
            if step%25==0:
                progress={'stage':'SMALL_CROSSENCODER_TRAINING','round':round_id,'step':step,'steps_per_round':200,
                    'loss':float(np.mean(losses)),'elapsed_seconds':time.monotonic()-started}
                write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
            if step%50==0:
                probs=infer(model,dx,dl,batch_size=64);record={'round':round_id,'step':step,**summarize(dev,probs)}
                history.append(record);write_json(args.output/'history.json',history)
                if record['minimum_dev_slice_joint_score']>best:
                    best=record['minimum_dev_slice_joint_score'];best_path=str(folder/'best');best_record=record;stale=0
                    model.save_pretrained(best_path,safe_serialization=True);tokenizer.save_pretrained(best_path)
                    write_json(folder/'best_dev.json',record);np.savez_compressed(folder/'dev_predictions.npz',prob_EOT=probs,labels=np.array([LABEL2ID[r['label']] for r in dev]))
                else:stale+=1
                write_json(args.output/'selection.json',{'best_path':best_path,'best_dev':best_record,'stale_evaluations':stale,'sealed_loaded':False})
                print(json.dumps({'stage':'SMALL_CROSSENCODER_DEV',**record}),flush=True)
                if record['development_all_slices_over_90'] and record['natural_hold_false_eot_below_10_percent']:reason='DEV_TARGET';break
                if round_id>=2 and stale>=4:reason='DEV_PLATEAU';break
        del optimizer;gc.collect();torch.cuda.empty_cache()
        if reason!='MAX_ROUNDS':break
    write_json(args.output/'report.json',{'status':'SMALL_CROSSENCODER_DEV_CAMPAIGN_FINISHED','stop_reason':reason,'best_path':best_path,
        'best_dev':best_record,'history':history,'checkpoint_sha256':file_hash(Path(best_path)/'model.safetensors'),
        'elapsed_seconds':time.monotonic()-started,'sealed_loaded':False,'real_target_achieved':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path('data_v6/turn_context_20261025'))
    parser.add_argument('--output',type=Path,default=Path('runs/natural_v6_minilm_finetune_20261025'));run(parser.parse_args())
