"""Train tiny students using gold labels, frozen teacher logits and pooled features."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import time
import unicodedata

import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from .common import LABEL2ID, encode_pair, file_hash, normalize, read_jsonl, write_json
from .metrics import calibrate_on_dev, classification_metrics, probabilities
from .tiny_student import TinyTurnGRU


def build_vocabulary(teacher,data,output):
    output.mkdir(parents=True,exist_ok=False)
    source=json.loads((teacher/'tokenizer.json').read_text())
    original=Tokenizer.from_file(str(teacher/'tokenizer.json'))
    observed=set();pending=[]
    for row in read_jsonl(data/'train.jsonl'):
        pending.extend([normalize(row.get('context','')),normalize(row['text'])])
        if len(pending)>=2048:
            for v in original.encode_batch(pending,add_special_tokens=False):observed.update(v.ids)
            pending=[]
    for v in original.encode_batch(pending,add_special_tokens=False):observed.update(v.ids)
    vocab=source['model']['vocab']
    for entry in sorted(source.get('added_tokens',[]),key=lambda v:v['id']):
        if entry['id']==len(vocab):vocab.append([entry['content'],0.])
    def fallback(token):
        text=token.replace('\u2581','')
        return token.startswith('<0x') or len(text)==0 or (len(text)==1 and (ord(text)<128 or 'LATIN' in unicodedata.name(text,'')))
    kept=sorted(observed|{i for i,(token,_) in enumerate(vocab) if fallback(token)}|{t['id'] for t in source['added_tokens']}|{0,1,2,3})
    remap={old:new for new,old in enumerate(kept)}
    compact=copy.deepcopy(source);compact['model']['vocab']=[vocab[i] for i in kept];compact['model']['unk_id']=remap[source['model']['unk_id']]
    for token in compact['added_tokens']:token['id']=remap[token['id']]
    for token in compact.get('post_processor',{}).get('special_tokens',{}).values():token['ids']=[remap[i] for i in token['ids']]
    Tokenizer.from_str(json.dumps(compact)).save(str(output/'tokenizer.json'))
    settings=json.loads((teacher/'tokenizer_config.json').read_text())
    if settings.get('added_tokens_decoder'):settings['added_tokens_decoder']={str(remap[int(i)]):v for i,v in settings['added_tokens_decoder'].items()}
    write_json(output/'tokenizer_config.json',settings)
    write_json(output/'config.json',{'model_type':'deberta-v2','vocab_size':len(kept),'pad_token_id':0})
    write_json(output/'kept_teacher_ids.json',kept)
    # Independent segmentation audit over every raw training input.
    tiny=Tokenizer.from_file(str(output/'tokenizer.json'));changed=0;pending=[]
    def audit(texts):
        return sum([kept[i] for i in b.ids]!=a.ids for a,b in zip(original.encode_batch(texts,add_special_tokens=False),tiny.encode_batch(texts,add_special_tokens=False),strict=True))
    for row in read_jsonl(data/'train.jsonl'):
        pending.extend([normalize(row.get('context','')),normalize(row['text'])])
        if len(pending)>=2048:changed+=audit(pending);pending=[]
    changed+=audit(pending)
    report={'source':'frozen train only; observed pieces plus Latin/ASCII/byte/special fallbacks','vocab_size':len(kept),'teacher_vocab':len(vocab),'changed_train_segmentation':changed,'train_sha256':file_hash(data/'train.jsonl'),'tokenizer_bytes':(output/'tokenizer.json').stat().st_size}
    write_json(output/'vocabulary_report.json',report)
    if changed:raise ValueError('Student vocabulary changed training segmentation')
    print(json.dumps(report),flush=True)


def evaluate(model,ids,lengths,labels,batch_size,device):
    model.eval();all_logits=[]
    with torch.inference_mode():
        for start in range(0,len(labels),batch_size):
            end=min(start+batch_size,len(labels));length=int(lengths[start:end].max())
            x=torch.from_numpy(np.asarray(ids[start:end,:length],dtype=np.int64)).to(device)
            mask=torch.arange(length,device=device)[None,:]<torch.from_numpy(lengths[start:end].astype(np.int64)).to(device)[:,None]
            all_logits.append(model(x,mask.long()).float().cpu().numpy())
    logits=np.concatenate(all_logits)
    if not np.isfinite(logits).all():raise ValueError('Non-finite student eval logits')
    return probabilities(logits)


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    device='cuda' if torch.cuda.is_available() else 'cpu'
    cache_report=json.loads((args.cache/'cache_report.json').read_text())
    assert cache_report['status']=='TRAIN_TARGETS_COMPLETE'
    assert file_hash(args.data/'train.jsonl')==cache_report['train_sha256']
    assert file_hash(args.data/'dev.jsonl')==cache_report['dev_sha256']
    marker=file_hash(args.data/'holdout_consumed.json')
    if not (args.vocabulary/'vocabulary_report.json').exists():build_vocabulary(args.teacher,args.data,args.vocabulary)
    kept=json.loads((args.vocabulary/'kept_teacher_ids.json').read_text())
    original_ids=np.load(args.cache/'train_ids.npy',mmap_mode='r')
    lengths=np.load(args.cache/'train_lengths.npy');labels=np.load(args.cache/'train_labels.npy')
    mapper=np.full(23078,-1,np.int32);mapper[kept]=np.arange(len(kept),dtype=np.int32)
    train_ids=mapper[original_ids]
    assert train_ids.min()>=0
    teacher_logits=np.load(args.cache/'teacher_logits.npy',mmap_mode='r');teacher_hidden=np.load(args.cache/'teacher_hidden.npy',mmap_mode='r')
    tok=AutoTokenizer.from_pretrained(args.vocabulary)
    dev_rows=list(read_jsonl(args.data/'dev.jsonl'));dev_ids=np.zeros((len(dev_rows),256),np.int32);dev_lengths=np.zeros(len(dev_rows),np.int16)
    for i,row in enumerate(dev_rows):
        ids=encode_pair(tok,row.get('context',''),row['text'],256)['input_ids'];dev_ids[i,:len(ids)]=ids;dev_lengths[i]=len(ids)
    dev_labels=np.array([LABEL2ID[r['label']] for r in dev_rows])
    config={'model_type':'deberta-v2','architecture':'tiny_turn_gru','vocab_size':len(kept),'embedding_size':args.embedding_size,'hidden_size':args.hidden_size,'layers':args.layers,'teacher_size':768,'pad_token_id':0}
    model=TinyTurnGRU(**config).to(device)
    embedding=torch.load(args.cache/'teacher_embeddings.pt',map_location='cpu',weights_only=True)[kept].float().to(device)
    mean=embedding.mean(0);_,_,basis=torch.pca_lowrank(embedding-mean,q=args.embedding_size,center=False,niter=5)
    with torch.no_grad():model.embedding.weight.copy_((embedding-mean)@basis[:,:args.embedding_size])
    del embedding,mean,basis
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    steps_per_epoch=math.ceil(len(labels)/args.batch_size);total_steps=steps_per_epoch*args.epochs;step=0
    best=-1.;bad_epochs=0;history=[];started=time.monotonic()
    identity={'teacher_sha256':cache_report['teacher_sha256'],'train_sha256':cache_report['train_sha256'],'dev_sha256':cache_report['dev_sha256'],'holdout_marker_sha256':marker,'seed':args.seed,'config':config,'training':vars(args)|{'data':str(args.data),'teacher':str(args.teacher),'cache':str(args.cache),'vocabulary':str(args.vocabulary),'output':str(args.output)},'source_sha256':{n:file_hash(Path(__file__).parent/n) for n in ['distill.py','tiny_student.py','common.py','metrics.py']}}
    write_json(args.output/'run_identity.json',identity)
    rng=np.random.default_rng(args.seed)
    for epoch in range(1,args.epochs+1):
        model.train();order=rng.permutation(len(labels));losses=[]
        for start in range(0,len(labels),args.batch_size):
            index=order[start:start+args.batch_size];length=int(lengths[index].max())
            x=torch.from_numpy(train_ids[index,:length].astype(np.int64)).to(device)
            mask=torch.arange(length,device=device)[None,:]<torch.from_numpy(lengths[index].astype(np.int64)).to(device)[:,None]
            y=torch.from_numpy(labels[index]).to(device)
            target=torch.from_numpy(np.array(teacher_logits[index],copy=True)).to(device)
            hidden_target=torch.from_numpy(np.array(teacher_hidden[index],dtype=np.float32,copy=True)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits,hidden=model(x,mask.long(),return_features=True)
            ce=F.cross_entropy(logits,y,label_smoothing=.02)
            kd=F.kl_div(F.log_softmax(logits/2.,dim=-1),F.softmax(target/2.,dim=-1),reduction='batchmean')*4
            representation=F.mse_loss(F.layer_norm(hidden,(768,)),F.layer_norm(hidden_target,(768,)))
            loss=.5*ce+.5*kd+.05*representation
            if not torch.isfinite(loss):raise ValueError('Non-finite distillation loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            factor=min(1.,(step+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*step/total_steps)))
            for group in optimizer.param_groups:group['lr']=args.lr*factor
            optimizer.step();step+=1;losses.append(float(loss.detach()))
            if step%100==0:
                progress={'stage':'training','epoch':epoch,'step':step,'total_steps':total_steps,'loss':float(np.mean(losses[-100:])),'elapsed_seconds':time.monotonic()-started}
                write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        probs=evaluate(model,dev_ids,dev_lengths,dev_labels,args.batch_size,device)
        metrics=classification_metrics(dev_labels,probs,.5)
        record={'epoch':epoch,'step':step,'loss':float(np.mean(losses)),'dev':metrics,'elapsed_seconds':time.monotonic()-started};history.append(record)
        write_json(args.output/'history.json',history);print(json.dumps(record),flush=True)
        if metrics['joint_score']>best:
            best=metrics['joint_score'];bad_epochs=0
            torch.save(model.state_dict(),args.output/'student.pt')
            np.savez_compressed(args.output/'dev_predictions.npz',labels=dev_labels,prob_EOT=probs,indices=np.arange(len(dev_labels)))
            write_json(args.output/'best_epoch.json',record)
        else:bad_epochs+=1
        if epoch>=4 and bad_epochs>=3:break
    selected=np.load(args.output/'dev_predictions.npz');calibrated=calibrate_on_dev(selected['labels'],selected['prob_EOT'])
    write_json(args.output/'config.json',config)
    for name in ['tokenizer.json','tokenizer_config.json']:shutil.copy2(args.vocabulary/name,args.output/name)
    contract={'schema':1,'architecture':'tiny_turn_gru','max_length':256,'threshold':calibrated['threshold'],'padding_to_multiple_of':8,'intra_op_threads':2,'normalization':'NFKC lowercase punctuation-to-space collapse-whitespace','input_format':'[CLS] context [SEP] text [SEP]; context left-truncated; pad right','label2id':LABEL2ID,'smoke_only':False,'teacher_sha256':cache_report['teacher_sha256'],'quality_scope':'Dev selection only; consumed holdout not evaluated','quality_status':'FULL_DEV_TARGET_MET' if calibrated['joint_score']>=.9 else 'DEV_TARGET_NOT_MET'}
    write_json(args.output/'turn_detector_config.json',contract)
    report={'status':contract['quality_status'],'architecture':config,'parameters_training':sum(p.numel() for p in model.parameters()),'parameters_deployment':sum(p.numel() for n,p in model.named_parameters() if not n.startswith('teacher_projection.')),'train_samples':len(labels),'dev_samples':len(dev_labels),'optimizer_steps':step,'best_epoch':json.loads((args.output/'best_epoch.json').read_text())['epoch'],'dev_selected_threshold':calibrated,'holdout_evaluated':False,'holdout_marker_unchanged':file_hash(args.data/'holdout_consumed.json')==marker,'elapsed_seconds':time.monotonic()-started}
    write_json(args.output/'training_report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name,default in [('data','data_v2/frozen'),('teacher','models/turn_v2_compact_20261024/onnx'),('cache','runs/distill_cache_20261024'),('vocabulary','runs/distill_vocabulary_20261024')]:p.add_argument('--'+name,type=Path,default=Path(default))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--hidden-size',type=int,default=64);p.add_argument('--embedding-size',type=int,default=64);p.add_argument('--layers',type=int,default=1);p.add_argument('--epochs',type=int,default=8);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--lr',type=float,default=.001);p.add_argument('--seed',type=int,default=42)
    run(p.parse_args())
