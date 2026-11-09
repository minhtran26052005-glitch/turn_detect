"""Version3: Python3.10 compatibility and matrix-only INT8 export."""
from collections import Counter
from datetime import datetime,timezone
import argparse
import copy
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np

from .factory_stream_data_v1 import sha
from .multilingual_three_teacher_distill_v1 import DATA,rows_after,validate_row
from .tiny_compression_v1.common import encode_pair
from .factory_stream_test_v1 import metric,kind

ROOT=Path('/home/edabk/hoangbpm/turn_detect')
SOURCE=ROOT/'outputs/multilingual_stream_v2_20261102'
OUT=ROOT/'outputs/multilingual_tiny_compression_20261102_v1'
STOP=False


def stop(*_):
    global STOP
    STOP=True


def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)


def status(stage,**values):
    record={'stage':stage,'pid':os.getpid(),'at_utc':datetime.now(timezone.utc).isoformat(),'main_training_interrupted':False,**values}
    save('status.json',record);print(json.dumps(record),flush=True)


def atomic_torch(path,value):
    import torch
    assert shutil.disk_usage(OUT).free>12*2**30
    tmp=path.with_suffix('.tmp');torch.save(value,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    tmp.replace(path)


def matrix_metrics(p,y,t):return metric(Counter(kind(bool(a),b>=t) for a,b in zip(y,p)))


def evaluate(predict,dev_rows,streams):
    probs=predict(dev_rows);splits={l:([p for p,r in zip(probs,dev_rows) if r['language']==l],[r['label']=='EOT' for r in dev_rows if r['language']==l]) for l in ['vi','en','zh']}
    t=float(max(np.linspace(.1,.95,86),key=lambda v:min(matrix_metrics(*splits[l],v)['eot_f1'] for l in ['vi','en'])))
    counts={l:Counter() for l in ['vi','en']};starts=[0]*len(streams)
    for b in range(1,max(len(s['words']) for s in streams)-1):
        rows=[];indices=[]
        for i,s in enumerate(streams):
            if b+2>len(s['words']):continue
            rows.append({'context':' '.join(s['words'][starts[i]:b]),'text':' '.join(s['words'][b:b+2])});indices.append(i)
        if not rows:continue
        for i,p in zip(indices,predict(rows)):
            s=streams[i]
            if b not in s['unknown']:counts[s['language_by_word'][b-1]][kind(b in s['eots'],p>=t)]+=1
            if p>=t:starts[i]=b
    by={l:metric(c) for l,c in counts.items()};by['zh']=matrix_metrics(*splits['zh'],t)
    return {'threshold':t,'by_language':by,'static_by_language':{l:matrix_metrics(*splits[l],t) for l in splits},
            'score':min(m['eot_f1'] for m in by.values()),'scope':'VI/EN causal predicted-reset synthetic DEV; ZH legacy punctuation DEV anchors; no test or human/audio gold',
            'gold_reset_in_inference':False,'language_id_supplied':False,'max_future_words':2}


def retained(report,teacher):
    checks={}
    for lang in ['vi','en','zh']:
        a,b=report['by_language'][lang],teacher['by_language'][lang]
        checks[lang]=a['eot_f1']>=b['eot_f1']-.02 and (a['eot_precision'] or 0)>=(b['eot_precision'] or 0)-.02 and (a['eot_recall'] or 0)>=(b['eot_recall'] or 0)-.03
    return all(checks.values()),checks


def tensor_predictor(model,tok,device,teacher=False,hidden=False):
    import torch
    def predict(rows):
        values=[];features=[]
        for start in range(0,len(rows),64):
            part=rows[start:start+64];seqs=[encode_pair(tok,r['context'],r['text'],256)['input_ids'] for r in part];n=max(map(len,seqs))
            ids=torch.full((len(part),n),tok.pad_token_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
            for i,seq in enumerate(seqs):ids[i,:len(seq)]=torch.as_tensor(seq,device=device);mask[i,:len(seq)]=1
            with torch.inference_mode():
                if teacher:
                    with torch.autocast('cuda',dtype=torch.bfloat16):output=model(input_ids=ids,attention_mask=mask,output_hidden_states=hidden)
                    logits=output.logits.float()
                    if hidden:features.extend(output.hidden_states[-1][:,0].float().cpu().numpy())
                else:logits=model(ids,mask).float()
            if hidden:values.extend(logits.cpu().numpy())
            else:values.extend(logits.softmax(-1)[:,1].cpu().tolist())
        return (np.asarray(values),np.asarray(features)) if hidden else values
    return predict


def make_dev():
    path=OUT/'dev_rows.jsonl'
    streams=json.loads((SOURCE/'dataset/dev_streams.json').read_text())
    if path.exists():return [json.loads(x) for x in path.read_text().splitlines()],streams
    pools={l:[] for l in ['vi','en','zh']}
    for s in streams:
        for b in range(1,len(s['words'])-1):
            if b in s['unknown']:continue
            lang=s['language_by_word'][b-1]
            pools[lang].append({'language':lang,'context':' '.join(s['words'][:b]),'text':' '.join(s['words'][b:b+2]),'label':'EOT' if b in s['eots'] else 'NEOT'})
    db=sqlite3.connect(DATA.as_uri()+'?mode=ro',uri=True);pools['zh']=[validate_row(r) for r in rows_after(db,'zh','dev','',1000)];db.close()
    rows=[pools[l][i] for i in range(max(map(len,pools.values()))) for l in ['vi','en','zh'] if i<len(pools[l])]
    with path.open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        f.flush();os.fsync(f.fileno())
    save('dev_data_receipt.json',{'counts':{l:len(v) for l,v in pools.items()},'sha256':sha(path),'source_dev_streams_sha256':sha(SOURCE/'dataset/dev_streams.json'),'test_used':False})
    return rows,streams


def build_cache(rows,teacher,tok,student_tok):
    import torch
    cache=OUT/'cache';cache.mkdir(exist_ok=True);n=len(rows)
    specs={'ids':((n,256),np.uint16),'lengths':((n,),np.int16),'labels':((n,),np.uint8),'logits':((n,2),np.float16),'hidden':((n,768),np.float16)}
    arrays={k:np.load(cache/(k+'.npy'),mmap_mode='r+') if (cache/(k+'.npy')).exists() else np.lib.format.open_memmap(cache/(k+'.npy'),mode='w+',dtype=dtype,shape=shape) for k,(shape,dtype) in specs.items()}
    for k,(shape,dtype) in specs.items():assert arrays[k].shape==shape and arrays[k].dtype==dtype
    progress=json.loads((cache/'progress.json').read_text()) if (cache/'progress.json').exists() else {'done':0}
    predictor=tensor_predictor(teacher,tok,'cuda',teacher=True,hidden=True);stats=Counter(progress.get('stats',{}))
    for start in range(progress['done'],n,64):
        if STOP:raise InterruptedError('Stop requested during cache; checkpoint cursor retained')
        part=rows[start:start+64];logits,features=predictor(part)
        for j,row in enumerate(part):
            index=start+j;ids=encode_pair(student_tok,row['context'],row['text'],256)['input_ids']
            assert len(row['text'].split())<=2,'Noncausal source lookahead'
            arrays['ids'][index]=0;arrays['ids'][index,:len(ids)]=ids;arrays['lengths'][index]=len(ids);arrays['labels'][index]=row['label']=='EOT'
            stats[row['language']+':unknown_pieces']+=ids.count(student_tok.unk_token_id)
        arrays['logits'][start:start+len(part)]=logits;arrays['hidden'][start:start+len(part)]=features
        end=start+len(part)
        if end%4096==0 or end==n:
            for a in arrays.values():a.flush()
            save('cache/progress.json',{'done':end,'total':n,'stats':dict(stats)});status('CACHING_FROZEN_TEACHER_TARGETS',rows_done=end,rows_total=n)
    report={'complete':True,'rows':n,'train_sha256':sha(OUT/'train.jsonl'),'teacher_weights_sha256':sha(OUT/'teacher/model.safetensors'),
            'student_tokenizer_sha256':sha(OUT/'vocabulary/tokenizer.json'),'unknown_pieces':dict(stats),'hidden_feature_source':'Frozen teacher last-layer CLS, TRAIN-only distillation target; not deployment input',
            'files':{k:{'sha256':sha(cache/(k+'.npy')),'bytes':(cache/(k+'.npy')).stat().st_size} for k in specs}}
    assert not any(stats.values()),'Byte-fallback tokenizer produced unknown pieces'
    save('cache/receipt.json',report);return arrays


def fit(name,hidden,init,epochs,group,cache,tok,dev_rows,streams,teacher_report):
    import torch
    from torch.nn import functional as F
    from .tiny_compression_v1.tiny_student import TinyTurnGRU
    from .tiny_compression_v1.qat_student import prepare_qat,latent_state,WEIGHTS
    folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training_report.json').exists():return folder,json.loads((folder/'training_report.json').read_text())
    config={'model_type':'bert','architecture':'tiny_turn_gru','vocab_size':len(tok),'embedding_size':64,'hidden_size':hidden,'layers':1,'teacher_size':768,'pad_token_id':0}
    torch.manual_seed(43);np.random.seed(43)
    full=folder/'latest_resume.pt';resume=torch.load(full,map_location='cpu',weights_only=False) if full.exists() else None
    model=TinyTurnGRU(**config).cuda()
    if resume:model.load_state_dict(resume['model'],strict=True)
    elif init:model.load_state_dict(torch.load(init/'student.pt',map_location='cpu',weights_only=True),strict=True)
    quantizers=None
    if group:
        scales=resume['quant_scales'] if resume else None;quantizers=prepare_qat(model,group,scales=scales)
    opt=torch.optim.AdamW(model.parameters(),lr=.001 if init is None else .0001,weight_decay=.01)
    rng=np.random.default_rng(43);n=len(cache['labels']);batch=256
    step=resume['step'] if resume else 0;epoch=resume['epoch'] if resume else 0;cursor=resume['cursor'] if resume else 0
    order=resume['order'] if resume else rng.permutation(n);best=resume['best_score'] if resume else -1.;best_passed=resume['best_passed'] if resume else False
    if resume:
        assert resume['cache_train_sha256']==sha(OUT/'train.jsonl')
        opt.load_state_dict(resume['optimizer']);rng.bit_generator.state=resume['numpy_rng'];torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state(resume['cuda_rng'])
    save(str(folder.relative_to(OUT)/'config.json'),config)
    for f in ['tokenizer.json','tokenizer_config.json']:shutil.copyfile(OUT/'vocabulary'/f,folder/f)
    def checkpoint():
        state={'model':latent_state(model) if group else {k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'step':step,
            'epoch':epoch,'cursor':cursor,'order':order,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
            'best_score':best,'best_passed':best_passed,'cache_train_sha256':sha(OUT/'train.jsonl'),
            'quant_scales':{k:q.scale.detach().cpu() for k,q in quantizers.items()} if group else None}
        atomic_torch(full,state)
    history=json.loads((folder/'history.json').read_text()) if (folder/'history.json').exists() else []
    gradients={}
    while epoch<epochs:
        model.train()
        for start in range(cursor,n,batch):
            if STOP:checkpoint();raise InterruptedError('Stop requested; student optimizer/RNG/order/cursor preserved')
            while shutil.disk_usage(OUT).free<13*2**30:status('WAITING_FOR_DISK_RESERVE',run=name);time.sleep(30)
            index=order[start:start+batch];length=int(cache['lengths'][index].max())
            x=torch.as_tensor(np.asarray(cache['ids'][index,:length],dtype=np.int64),device='cuda');lengths=torch.as_tensor(np.asarray(cache['lengths'][index],dtype=np.int64),device='cuda')
            mask=(torch.arange(length,device='cuda')[None,:]<lengths[:,None]).long();y=torch.as_tensor(np.asarray(cache['labels'][index],dtype=np.int64),device='cuda')
            logits_target=torch.as_tensor(np.asarray(cache['logits'][index],dtype=np.float32),device='cuda');hidden_target=torch.as_tensor(np.asarray(cache['hidden'][index],dtype=np.float32),device='cuda')
            opt.zero_grad(set_to_none=True);logits,features=model(x,mask,return_features=True)
            ce=F.cross_entropy(logits,y,label_smoothing=.02);kd=F.kl_div(F.log_softmax(logits/2.,dim=-1),F.softmax(logits_target/2.,dim=-1),reduction='batchmean')*4
            representation=F.mse_loss(F.layer_norm(features,(768,)),F.layer_norm(hidden_target,(768,)))
            loss=.5*ce+.5*kd+.05*representation;assert torch.isfinite(loss)
            loss.backward()
            if group and not gradients:
                for key in WEIGHTS:
                    parent,leaf=key.rsplit('.',1);g=getattr(model.get_submodule(parent).parametrizations,leaf).original.grad
                    gradients[key]=g is not None and bool(torch.isfinite(g).all()) and bool(g.abs().sum()>0)
                assert all(gradients.values());save(str(folder.relative_to(OUT)/'gradient_checks.json'),gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();step+=1;cursor=min(start+batch,n)
            if step%100==0:status('TRAINING_QAT_INT4' if group else 'DISTILLING_TINY_GRU',run=name,epoch=epoch+1,epochs=epochs,step=step,loss=float(loss.detach()),hidden_size=hidden)
            if step%250==0:checkpoint()
        model.eval();report=evaluate(tensor_predictor(model,tok,'cuda'),dev_rows,streams);passed,checks=retained(report,teacher_report)
        record={'epoch':epoch+1,'step':step,'dev':report,'retention_passed':passed,'retention_by_language':checks}
        history.append(record);save(str(folder.relative_to(OUT)/'history.json'),history)
        if (passed and not best_passed) or (passed==best_passed and report['score']>best):
            best=report['score'];best_passed=passed
            atomic_torch(folder/'student.pt',latent_state(model) if group else {k:v.detach().cpu() for k,v in model.state_dict().items()})
            if group:atomic_torch(folder/'quant_params.pt',{'group_size':group,'scales':{k:q.scale.detach().cpu() for k,q in quantizers.items()}})
            save(str(folder.relative_to(OUT)/'best_report.json'),record)
        epoch+=1;cursor=0;order=rng.permutation(n);checkpoint()
    selected=json.loads((folder/'best_report.json').read_text())
    contract={'schema':1,'architecture':'tiny_turn_gru','max_length':256,'threshold':selected['dev']['threshold'],'padding_to_multiple_of':8,'intra_op_threads':2,
        'normalization':'NFKC lowercase punctuation-to-space collapse-whitespace','input_format':'CLS visible buffer SEP next1/2words SEP; left-truncate history only',
        'label2id':{'NEOT':0,'EOT':1},'smoke_only':False,'quality_scope':'Synthetic DEV selection, not independent test/human audio gold',
        'training_method':'INT4_QAT_WITH_DISTILLATION' if group else 'KNOWLEDGE_DISTILLATION','lookahead_words':[1,2]}
    save(str(folder.relative_to(OUT)/'turn_detector_config.json'),contract)
    result={'complete':True,'run':name,'epochs':epochs,'steps':step,'selected':selected,'config':config,'group_size':group,'student_sha256':sha(folder/'student.pt'),
            'parameters_deployment':sum(p.numel() for name,p in model.named_parameters() if not name.startswith('teacher_projection.')),'cache_rows':n,'test_used':False}
    save(str(folder.relative_to(OUT)/'training_report.json'),result);del model,opt;gc.collect();torch.cuda.empty_cache();return folder,result


def ort_predictor(path,tok):
    import onnxruntime as ort
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;opts.add_session_config_entry('session.disable_quant_qdq','1')
    session=ort.InferenceSession(str(path/'model.onnx'),opts,providers=['CPUExecutionProvider'])
    def predict(rows):
        values=[]
        for start in range(0,len(rows),64):
            seqs=[encode_pair(tok,r['context'],r['text'],256)['input_ids'] for r in rows[start:start+64]];n=max(map(len,seqs));ids=np.zeros((len(seqs),n),np.int64);mask=np.zeros_like(ids)
            for i,seq in enumerate(seqs):ids[i,:len(seq)]=seq;mask[i,:len(seq)]=1
            logits=session.run(['logits'],{'input_ids':ids,'attention_mask':mask})[0].astype(np.float64);p=np.exp(logits-logits.max(-1,keepdims=True));values.extend((p/p.sum(-1,keepdims=True))[:,1].tolist())
        return values
    return predict


def export_float(student,dest,dev):
    import torch
    import onnx
    from .tiny_compression_v1.tiny_student import load_student
    from transformers import AutoTokenizer
    if (dest/'model.onnx').exists():return
    dest.mkdir(parents=True,exist_ok=True);model=load_student(student).float().eval();tok=AutoTokenizer.from_pretrained(student,local_files_only=True,token=False)
    seqs=[encode_pair(tok,r['context'],r['text'],256)['input_ids'] for r in dev[:3]];n=max(map(len,seqs));ids=torch.zeros((3,n),dtype=torch.long);mask=torch.zeros_like(ids)
    for i,s in enumerate(seqs):ids[i,:len(s)]=torch.as_tensor(s);mask[i,:len(s)]=1
    with torch.no_grad():torch.onnx.export(model,(ids,mask),str(dest/'model.onnx'),input_names=['input_ids','attention_mask'],output_names=['logits'],
        dynamic_axes={'input_ids':{0:'batch',1:'sequence'},'attention_mask':{0:'batch',1:'sequence'},'logits':{0:'batch'}},opset_version=18,dynamo=False,external_data=False)
    onnx.checker.check_model(str(dest/'model.onnx'))
    for f in ['config.json','tokenizer.json','tokenizer_config.json','turn_detector_config.json']:shutil.copyfile(student/f,dest/f)


def export_int8(source,dest):
    import onnx
    from onnx import numpy_helper,helper,TensorProto
    from .tiny_compression_v1.compress import sort_graph_with_captures
    if (dest/'model.onnx').exists():return
    dest.mkdir(parents=True,exist_ok=True);g=onnx.version_converter.convert_version(onnx.load(str(source/'model.onnx')),21);tensors=[];nodes=[];count=0
    for tensor in g.graph.initializer:
        if tensor.data_type!=TensorProto.FLOAT or len(tensor.dims)<2 or (len(tensor.dims)==2 and tensor.dims[0]==1):tensors.append(tensor);continue
        w=numpy_helper.to_array(tensor);width=w.shape[-1];blocks=(width+63)//64;z=np.pad(w,[(0,0)]*(w.ndim-1)+[(0,blocks*64-width)]).reshape(*w.shape[:-1],blocks,64)
        scale=np.maximum(np.max(z,axis=-1)/127,-np.min(z,axis=-1)/128);scale=np.maximum(scale,np.finfo(np.float16).tiny).astype(np.float16)
        codes=np.clip(np.rint(z/scale[...,None].astype(np.float32)),-128,127).astype(np.int8).reshape(*w.shape[:-1],blocks*64)[...,:width]
        name=tensor.name+'_i8';tensors.extend([numpy_helper.from_array(codes,name),numpy_helper.from_array(scale,name+'_scale')]);count+=1
        nodes.extend([helper.make_node('DequantizeLinear',[name,name+'_scale'],[name+'_half'],axis=w.ndim-1,block_size=64),helper.make_node('Cast',[name+'_half'],[tensor.name],to=TensorProto.FLOAT)])
    assert count==5
    original=list(g.graph.node);del g.graph.node[:];g.graph.node.extend(nodes+original);del g.graph.initializer[:];g.graph.initializer.extend(tensors);sort_graph_with_captures(g.graph)
    onnx.checker.check_model(g,full_check=True);onnx.save(g,str(dest/'model.onnx'))
    for f in ['config.json','tokenizer.json','tokenizer_config.json','turn_detector_config.json']:shutil.copyfile(source/f,dest/f)
    contract=json.loads((dest/'turn_detector_config.json').read_text());contract.update(storage_precision='INT8',compute_precision='FP32',onnx_session_options={'session.disable_quant_qdq':'1'});save(str(dest.relative_to(OUT)/'turn_detector_config.json'),contract)


def finalize_candidate(path,tok,dev,streams,teacher,bits):
    predictor=ort_predictor(path,tok);report=evaluate(predictor,dev,streams);passed,checks=retained(report,teacher)
    contract=json.loads((path/'turn_detector_config.json').read_text());contract.update(threshold=report['threshold'],languages=['vi','en','zh'],quality_status='DEV_RETENTION_PASS' if passed else 'EXPERIMENTAL_DEV_RETENTION_FAIL',
        onnx_session_options={'session.disable_quant_qdq':'1'},storage_precision=bits,compute_precision='FP32')
    save(str(path.relative_to(OUT)/'turn_detector_config.json'),contract)
    shutil.copyfile(ROOT/'turn_v2/tiny_compression_v1/portable_runner.py',path/'run_model.py')
    (path/'requirements.txt').write_text('numpy>=2.1\nonnxruntime==1.23.2\ntokenizers==0.23.3\n')
    record={'path':str(path),'storage_precision':bits,'compute_precision':'FP32','model_bytes':(path/'model.onnx').stat().st_size,'deployment_bytes':sum(p.stat().st_size for p in path.iterdir() if p.is_file() and p.name!='dev_report.json'),
        'model_sha256':sha(path/'model.onnx'),'dev':report,'retention_passed':passed,'retention_by_language':checks}
    save(str(path.relative_to(OUT)/'dev_report.json'),record);return record


def run():
    import torch
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    utils.is_torchvision_available=imports.is_torchvision_available=lambda:False
    utils.is_librosa_available=imports.is_librosa_available=lambda:False
    from transformers import AutoConfig,AutoModelForSequenceClassification,AutoTokenizer
    from safetensors.torch import load_file
    from types import SimpleNamespace
    os.nice(4);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.30)
    while True:
        if (OUT/'preparation_error.json').exists():raise RuntimeError('Preparation failed; inspect receipt, do not initialize another teacher')
        p=OUT/'preparation_status.json'
        if p.exists() and json.loads(p.read_text())['stage']=='COMPLETE':break
        status('WAITING_FOR_PREPARATION');time.sleep(15)
        if STOP:raise InterruptedError('Explicit stop while waiting')
    assert not (OUT/'final_report.json').exists(),'Never rerun a completed compression campaign'
    check=subprocess.run([sys.executable,'-m','turn_v2.tiny_compression_v1.self_checks'],cwd=ROOT,capture_output=True,text=True)
    (OUT/'self_checks.log').write_text(check.stdout+check.stderr);assert check.returncode==0,'INT4 numerical/gradient self-check failed'
    status('LOADING_FROZEN_TEACHER')
    tok=AutoTokenizer.from_pretrained(OUT/'teacher',local_files_only=True,token=False);cfg=AutoConfig.from_pretrained(OUT/'teacher',local_files_only=True,token=False)
    student_tok=AutoTokenizer.from_pretrained(OUT/'vocabulary',local_files_only=True,token=False);assert student_tok.pad_token_id==0 and len(student_tok)<65536
    teacher=AutoModelForSequenceClassification.from_config(cfg,dtype=torch.float32);teacher.load_state_dict(load_file(str(OUT/'teacher/model.safetensors')),strict=True);teacher.cuda().eval()
    dev,streams=make_dev()
    if not (OUT/'teacher_dev.json').exists():save('teacher_dev.json',evaluate(tensor_predictor(teacher,tok,'cuda',teacher=True),dev,streams))
    teacher_report=json.loads((OUT/'teacher_dev.json').read_text())
    rows=[json.loads(x) for x in (OUT/'train.jsonl').read_text().splitlines()]
    if not (OUT/'cache/receipt.json').exists():cache=build_cache(rows,teacher,tok,student_tok)
    else:cache={k:np.load(OUT/'cache'/(k+'.npy'),mmap_mode='r') for k in ['ids','lengths','labels','logits','hidden']}
    del rows,teacher;gc.collect();torch.cuda.empty_cache()
    candidates=[]
    for hidden in [128,192]:
        kd,_=fit(f'gru{hidden}_distill',hidden,None,8,None,cache,student_tok,dev,streams,teacher_report)
        control,_=fit(f'gru{hidden}_fp32_control',hidden,kd,4,None,cache,student_tok,dev,streams,teacher_report)
        fp=OUT/'candidates'/f'gru{hidden}_fp32';status('EXPORTING_FP32_CONTROL',hidden_size=hidden);export_float(control,fp,dev)
        candidates.append(finalize_candidate(fp,student_tok,dev,streams,teacher_report,'FP32'))
        i8=OUT/'candidates'/f'gru{hidden}_int8';export_int8(fp,i8);candidates.append(finalize_candidate(i8,student_tok,dev,streams,teacher_report,'INT8'))
        for group in [64,32]:
            qat,_=fit(f'gru{hidden}_qat_g{group}',hidden,kd,4,group,cache,student_tok,dev,streams,teacher_report)
            qpath=OUT/'candidates'/f'gru{hidden}_int4_g{group}';status('EXPORTING_PACKED_INT4',hidden_size=hidden,group_size=group)
            if not (qpath/'model.onnx').exists():
                from .tiny_compression_v1.export_qat_student import export
                export(SimpleNamespace(student=qat,output=qpath,dev=OUT/'dev_rows.jsonl'))
            candidates.append(finalize_candidate(qpath,student_tok,dev,streams,teacher_report,'INT4'))
        save('candidate_results.json',candidates)
        if any(c['retention_passed'] and c['storage_precision']=='INT4' for c in candidates):break
    passing=[c for c in candidates if c['retention_passed'] and c['storage_precision'] in ['INT4','INT8']]
    if not passing:
        save('final_report.json',{'complete':True,'accepted':False,'reason':'No compressed candidate retained all three languages within declared DEV margins','candidates':candidates,'teacher':teacher_report,'test_used':False});status('COMPLETE_NO_ACCEPTED_COMPRESSED_MODEL');return
    winner=min(passing,key=lambda c:c['deployment_bytes']);package=ROOT/'models/turn_multilingual_tiny_compressed_20261102_v1';model=package/'onnx'
    model.mkdir(parents=True,exist_ok=False)
    for p in Path(winner['path']).iterdir():
        if p.is_file() and p.name!='dev_report.json':shutil.copyfile(p,model/p.name)
    files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in model.iterdir() if p.is_file()}
    manifest={'status':'DEV_RETENTION_VERIFIED_EXPERIMENTAL_NOT_HUMAN_GOLD','files':files,'model_bytes':files['model.onnx']['bytes'],'deployment_bytes':sum(f['bytes'] for f in files.values()),
        'storage_precision':winner['storage_precision'],'execution_precision':'FP32, matrices dequantized by CPU runtime; not native INT4 compute','candidate':winner,
        'teacher':json.loads((OUT/'teacher_receipt.json').read_text()),'train_rows':300000,'train_per_language':100000,'tokenizer_train_only':True,'source_spoken_v9_used':False,'test_or_holdout_evaluated':False,
        'retention_margins':'At own DEV threshold, each language F1/precision drop<=2pp and recall drop<=3pp relative to frozen teacher','no_quality_target90_certified':True,'libraries_not_included_in_deployment_size':True}
    (package/'model_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (package/'README.md').write_text('Shared VI/EN/ZH tiny GRU distilled from the frozen trained multilingual student, then quantized. See model_manifest.json for DEV retention, real storage/compute precision, file hashes and input contract. Source is pre-spoken source722 + legacyZH, not new spokenv9. Synthetic punctuation DEV, not independent conversational test. No active teacher/model replaced.\n')
    status('BENCHMARKING_CPU',package=str(package))
    benchmarks={}
    for label,path,kind_name in [('teacher',OUT/'teacher','teacher')]+[(Path(c['path']).name,Path(c['path']),'onnx') for c in candidates]:
        proc=subprocess.run([sys.executable,'-m','turn_v2.multilingual_tiny_benchmark_v1','--kind',kind_name,'--model',str(path),'--cases',str(OUT/'dev_rows.jsonl')],cwd=ROOT,capture_output=True,text=True,timeout=300)
        (OUT/('benchmark_'+label+'.log')).write_text(proc.stdout+proc.stderr);assert proc.returncode==0,'CPU benchmark failed: '+label
        benchmarks[label]=json.loads(proc.stdout.strip().splitlines()[-1])
    save('cpu_benchmark.json',benchmarks)
    (package/'cpu_benchmark.json').write_text(json.dumps(benchmarks,indent=2)+'\n')
    archive=package.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in package.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(package.parent))
    with zipfile.ZipFile(archive) as z:
        for n,meta in files.items():assert hashlib.sha256(z.read(package.name+'/onnx/'+n)).hexdigest()==meta['sha256']
    final={'complete':True,'accepted':True,'winner':winner,'teacher':teacher_report,'package':str(package),'archive':str(archive),'archive_sha256':sha(archive),
        'manifest_sha256':sha(package/'model_manifest.json'),'candidates':candidates,'test_used':False,'main_training_interrupted':False,
        'cpu_benchmark':benchmarks,'completed_utc':datetime.now(timezone.utc).isoformat()}
    save('final_report.json',final);status('COMPLETE',package=str(package),model_bytes=winner['model_bytes'],deployment_bytes=manifest['deployment_bytes'])


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (OUT/'CAMPAIGN_LOCK').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except InterruptedError as e:status('STOPPED_BY_REQUEST',reason=str(e))
        except Exception as e:save('error.json',{'error':repr(e),'traceback':traceback.format_exc(),'at_utc':datetime.now(timezone.utc).isoformat()});status('ERROR',error=repr(e));raise
