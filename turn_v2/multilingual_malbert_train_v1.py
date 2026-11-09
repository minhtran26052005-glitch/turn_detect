"""Verified pretrained mALBERT -> causal CE/KD -> quality-gated QAT INT4."""
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile
import numpy as np

ROOT=Path('/home/edabk/hoangbpm/turn_detect')
OUT=ROOT/'outputs/multilingual_malbert_pretrained_20261103_v1'
OLD=ROOT/'outputs/multilingual_tiny_compression_20261102_v1'
PACKAGE=ROOT/'models/turn_multilingual_malbert_20261103_v1'
TRAIN_SHA='51cbacc8758a3f3efa5fc2cc2223747bba09cc045ccacca5668deb440b2c3c41'
PRETRAINED_SHA='3040f56fef13a7c0d6b0e6f608daf9d8cd8cc4f4a163844d652d0b3e80cbdc84'
TEACHER_SHA='15d4ebfc7fcba7c3c842b7b7815ac309cf349f0436dca69522d9d529295954cc'
STOP=False


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)


def status(stage,**values):
    v=dict(stage=stage,pid=os.getpid(),at_utc=datetime.now(timezone.utc).isoformat(),
           architecture='Pretrained mALBERT32k with EOT/NEOT head',parameters=11941122,
           large_teacher_must_remain_stopped=True,**values)
    save('status.json',v);print(json.dumps(v),flush=True)


def stopped():return STOP or (OUT/'STOP_REQUESTED').exists()


def reserve():
    while True:
        mount=subprocess.run(['findmnt','-n','-o','SOURCE,OPTIONS','/mnt/hdd500'],text=True,capture_output=True,check=True).stdout.strip()
        assert mount.startswith('/dev/sda1 ') and 'rw' in mount.split()[1].split(','),mount
        if shutil.disk_usage(OUT).free>=13*2**30 and shutil.disk_usage('/mnt/hdd500').free>=13*2**30:return
        status('WAITING_FOR_DISK_RESERVE',checkpoint_preserved=True)
        if stopped():raise InterruptedError('Explicit stop while waiting for disk reserve')
        time.sleep(30)


def atomic_torch(path,value):
    import torch
    reserve();tmp=path.with_name(path.name+'.tmp');torch.save(value,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    if path.exists() and path.name=='latest_resume.pt':
        link=path.with_name('previous_resume.link');assert not link.exists()
        os.link(path,link);link.replace(path.with_name('previous_resume.pt'))
    tmp.replace(path)


def new_model():
    from .malbert_turn_student_v1 import MalbertTurnStudent
    model,loading=MalbertTurnStudent.from_verified_pretrained(OUT/'pretrained',3)
    assert model.parameter_count()==11941122<20_000_000
    return model,loading


def prepare():
    import torch
    from transformers import AutoTokenizer
    from .tiny_compression_v1.common import encode_pair
    assert json.loads((OUT/'train_probe/final.json').read_text())['passed'],'TRAIN learnability probe must pass before long training'
    assert sha(OUT/'pretrained/model.safetensors')==PRETRAINED_SHA
    old=json.loads((OLD/'cache/receipt.json').read_text())
    assert old['complete'] and old['rows']==300000 and old['train_sha256']==TRAIN_SHA==sha(OLD/'train.jsonl')
    assert old['teacher_weights_sha256']==TEACHER_SHA
    for key in ('labels','logits'):assert sha(OLD/'cache'/(key+'.npy'))==old['files'][key]['sha256']
    tok=AutoTokenizer.from_pretrained(OUT/'pretrained',local_files_only=True,token=False)
    assert len(tok)==32000 and tok.pad_token_id==0 and tok.sep_token_id==3
    tok.save_pretrained(OUT/'tokenizer');assert (OUT/'tokenizer/tokenizer.json').exists()
    if not (OUT/'initial_student.pt').exists():
        model,loading=new_model();atomic_torch(OUT/'initial_student.pt',model.state_dict())
        save('initialization_receipt.json',dict(**loading,pretrained_model_sha256=PRETRAINED_SHA,
             train_probe_not_used_as_initial_weights=True,source_revision='d3f3c1addfec3420d3f306f674ed2087b34e0411'))
        del model;gc.collect()
    cache=OUT/'cache';cache.mkdir(exist_ok=True);labels=np.load(OLD/'cache/labels.npy',mmap_mode='r');n=len(labels)
    if not (cache/'receipt.json').exists():
        reserve();spec={'ids':(np.uint16,(n,256)),'lengths':(np.uint16,(n,))}
        arrays={k:np.load(cache/(k+'.npy'),mmap_mode='r+') if (cache/(k+'.npy')).exists()
                else np.lib.format.open_memmap(cache/(k+'.npy'),mode='w+',dtype=t,shape=s) for k,(t,s) in spec.items()}
        for k,(t,s) in spec.items():assert arrays[k].dtype==t and arrays[k].shape==s
        progress=json.loads((cache/'progress.json').read_text()) if (cache/'progress.json').exists() else {'done':0,'counts':{},'unknown':{},'pieces':{}}
        counts=Counter(progress['counts']);unknown=Counter(progress['unknown']);pieces=Counter(progress['pieces'])
        with (OLD/'train.jsonl').open(encoding='utf-8') as f:
            total=0
            for i,line in enumerate(f):
                total=i+1
                if i<progress['done']:continue
                if stopped():raise InterruptedError('Stop during resumable TRAIN tokenization')
                row=json.loads(line);assert row['split']=='train' and 1<=len(row['text'].split())<=2
                assert int(row['label']=='EOT')==int(labels[i]);lang=row['language']
                ids=encode_pair(tok,row['context'],row['text'],256)['input_ids']
                arrays['ids'][i]=0;arrays['ids'][i,:len(ids)]=ids;arrays['lengths'][i]=len(ids)
                counts[lang]+=1;unknown[lang]+=ids.count(tok.unk_token_id);pieces[lang]+=len(ids)
                if (i+1)%4096==0 or i+1==n:
                    for a in arrays.values():a.flush()
                    save('cache/progress.json',dict(done=i+1,total=n,counts=dict(counts),unknown=dict(unknown),pieces=dict(pieces)))
                    status('TOKENIZING_TRAIN_MALBERT',rows=i+1,total=n)
            assert total==n and counts=={'vi':100000,'en':100000,'zh':100000}
        fractions={l:unknown[l]/pieces[l] for l in counts}
        assert max(fractions.values())<.05,('Investigate pretrained tokenizer coverage before training',fractions)
        save('cache/receipt.json',dict(complete=True,rows=n,counts=dict(counts),unknown_fraction=fractions,
             train_sha256=TRAIN_SHA,teacher_sha256=TEACHER_SHA,pretrained_sha256=PRETRAINED_SHA,
             teacher_cache_receipt_sha256=sha(OLD/'cache/receipt.json'),all_labels_and_row_indices_verified=True,
             tokenizer_sha256=sha(OUT/'tokenizer/tokenizer.json'),ids_sha256=sha(cache/'ids.npy'),lengths_sha256=sha(cache/'lengths.npy')))
    receipt=json.loads((cache/'receipt.json').read_text());assert receipt['complete'] and receipt['pretrained_sha256']==PRETRAINED_SHA
    assert receipt['ids_sha256']==sha(cache/'ids.npy') and receipt['tokenizer_sha256']==sha(OUT/'tokenizer/tokenizer.json')
    data={k:np.load(cache/(k+'.npy'),mmap_mode='r') for k in ('ids','lengths')}
    data.update({k:np.load(OLD/'cache'/(k+'.npy'),mmap_mode='r') for k in ('labels','logits')})
    return tok,data


def attach_qat(model,group,scales=None):
    import torch
    from torch.nn.utils import parametrize
    from .tiny_compression_v1.qat_student import GroupInt4FakeQuant
    from .tiny_compression_v1.quantize_student4 import quantize_groups
    result={}
    for key,weight in [(k,p) for k,p in model.named_parameters() if p.ndim==2]:
        parent,leaf=key.rsplit('.',1)
        if scales is None:
            _,value,_=quantize_groups(weight.detach().cpu().numpy(),group);scale=torch.from_numpy(value).to(weight.device)
        else:scale=scales[key].to(weight.device)
        q=GroupInt4FakeQuant(scale,group);parametrize.register_parametrization(model.get_submodule(parent),leaf,q);result[key]=q
    return result


def latent(model,q):
    state={k:v.detach().cpu() for k,v in model.state_dict().items() if '.parametrizations.' not in k}
    for key in q:
        parent,leaf=key.rsplit('.',1);state[key]=getattr(model.get_submodule(parent).parametrizations,leaf).original.detach().cpu()
    return state


def fit(name,initial,epochs,group,tok,cache,dev,streams,teacher,continue_until_retained=False):
    import torch
    from torch.nn import functional as F
    from .multilingual_tiny_compress_v3 import evaluate,retained,tensor_predictor
    folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training_report.json').exists():return folder
    torch.manual_seed(43);np.random.seed(43);random.seed(43)
    path=folder/'latest_resume.pt';resume=torch.load(path,map_location='cpu',weights_only=False) if path.exists() else None
    model,_=new_model();model.load_state_dict(resume['model'] if resume else torch.load(initial,map_location='cpu',weights_only=True),strict=True);model.cuda()
    q=attach_qat(model,group,resume['quant_scales'] if resume else None) if group else {}
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5 if name=='finetune' else 1e-5,weight_decay=.01)
    n=len(cache['labels']);batch=16;rng=np.random.default_rng(43)
    epoch=resume['epoch'] if resume else 0;cursor=resume['cursor'] if resume else 0;step=resume['step'] if resume else 0
    order=resume['order'] if resume else rng.permutation(n);best=tuple(resume['best_key']) if resume else (-1.,-1.,-1.)
    passed_best=bool(best[0]) if resume else False
    fingerprint={'train_sha256':TRAIN_SHA,'cache_sha256':sha(OUT/'cache/receipt.json'),'pretrained_sha256':PRETRAINED_SHA,
        'code_sha256':{p:sha(ROOT/'turn_v2'/p) for p in ['malbert_turn_student_v1.py','multilingual_malbert_train_v1.py','malbert_turn_export_v1.py']}}
    if resume:
        assert resume['fingerprint']==fingerprint and resume['parameters']==11941122
        opt.load_state_dict(resume['optimizer']);rng.bit_generator.state=resume['sampler_rng']
        torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state(resume['cuda_rng']);np.random.set_state(resume['numpy_rng']);random.setstate(resume['python_rng'])
    save(str(folder.relative_to(OUT)/'identity.json'),dict(parameters=11941122,epochs_minimum=epochs,group_size=group,
         continue_until_retained=continue_until_retained,initial_weights_sha256=sha(initial),batch_size=batch,**fingerprint))
    def checkpoint():
        full=dict(model=latent(model,q),optimizer=opt.state_dict(),step=step,epoch=epoch,cursor=cursor,order=order,
             sampler_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
             numpy_rng=np.random.get_state(),python_rng=random.getstate(),best_key=best,parameters=11941122,fingerprint=fingerprint,
             quant_scales={k:v.scale.detach().cpu() for k,v in q.items()})
        atomic_torch(path,full)
        save(str(folder.relative_to(OUT)/'resume_receipt.json'),dict(step=step,epoch=epoch,cursor=cursor,
             optimizer_entries=len(opt.state),parameters=11941122,optimizer_rng_sampler_scales_included=True))
        if not (OUT/'first_checkpoint_verification.json').exists():
            x=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            assert x['step']>0 and x['optimizer']['state'] and len(x['order'])==n and x['fingerprint']==fingerprint
            save('first_checkpoint_verification.json',dict(run=name,step=step,optimizer_entries=len(x['optimizer']['state']),
                 checkpoint_sha256=sha(path),torch_load_passed=True,optimizer_rng_order_source_verified=True,parameters=11941122))
    def evaluate_now():
        nonlocal best,passed_best
        name_rel=str(folder.relative_to(OUT)/('dev_step_%08d.json'%step))
        if (OUT/name_rel).exists():record=json.loads((OUT/name_rel).read_text())
        else:
            status('EVALUATING_MALBERT_DEV',run=name,step=step,epoch=epoch+1)
            model.eval();result=evaluate(tensor_predictor(model,tok,'cuda'),dev,streams);passed,checks=retained(result,teacher)
            record=dict(run=name,step=step,epoch=epoch+1,dev=result,retention_passed=passed,retention_by_language=checks,
                        scope='Synthetic punctuation DEV only; no test/human/audio gold',test_used=False)
            save(name_rel,record)
        values=[x['eot_f1'] for x in record['dev']['by_language'].values()]
        key=(float(record['retention_passed']),min(values),float(np.mean(values)))
        if key>best:
            best=key;passed_best=record['retention_passed'];atomic_torch(folder/'student.pt',latent(model,q))
            if q:atomic_torch(folder/'quant_params.pt',dict(group_size=group,scales={k:v.scale.detach().cpu() for k,v in q.items()}))
            save(str(folder.relative_to(OUT)/'best_report.json'),record)
        save('latest_dev.json',record);model.train();return record
    while epoch<epochs or (continue_until_retained and not passed_best):
        model.train()
        for start in range(cursor,n,batch):
            if stopped():checkpoint();raise InterruptedError('Userstop at a preserved studentcheckpoint')
            if step%100==0:reserve()
            index=order[start:start+batch];width=int(cache['lengths'][index].max())
            ids=torch.tensor(np.asarray(cache['ids'][index,:width],dtype=np.int64),device='cuda')
            lengths=torch.tensor(np.asarray(cache['lengths'][index],dtype=np.int64),device='cuda')
            mask=(torch.arange(width,device='cuda')[None,:]<lengths[:,None]).long()
            y=torch.tensor(np.asarray(cache['labels'][index],dtype=np.int64),device='cuda')
            target=torch.tensor(np.asarray(cache['logits'][index],dtype=np.float32),device='cuda')
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=not bool(group)):logits=model(ids,mask)
            logits=logits.float();ce=F.cross_entropy(logits,y)
            kd=F.kl_div(F.log_softmax(logits/2,-1),F.softmax(target/2,-1),reduction='batchmean')*4
            kd_weight=0. if name=='finetune' and epoch<2 else .3
            loss=(1-kd_weight)*ce+kd_weight*kd;assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if group and step==0:
                checks={}
                for key in q:
                    parent,leaf=key.rsplit('.',1);g=getattr(model.get_submodule(parent).parametrizations,leaf).original.grad
                    checks[key]=g is not None and bool(torch.isfinite(g).all()) and bool(g.abs().sum()>0)
                assert all(checks.values());save(str(folder.relative_to(OUT)/'qat_gradient_checks.json'),checks)
            total=epochs*math.ceil(n/batch);base_lr=2e-5 if name=='finetune' else 1e-5
            lr=base_lr*min(1.,(step+1)/300)*(.2+.8*.5*(1+math.cos(math.pi*min(1.,step/total))))
            for pg in opt.param_groups:pg['lr']=lr
            opt.step();step+=1;cursor=min(start+batch,n)
            if step%50==0:status('QAT_MALBERT_INT4' if group else 'FINETUNING_PRETRAINED_MALBERT',run=name,step=step,
                 epoch=epoch+1,rows_seen=cursor,loss=float(loss.detach()),kd_weight=kd_weight,group_size=group)
            if step==100 or step%500==0:checkpoint()
            if step%1000==0:evaluate_now();checkpoint()
        evaluate_now();epoch+=1;cursor=0;order=rng.permutation(n);checkpoint()
    save(str(folder.relative_to(OUT)/'training_report.json'),dict(complete=True,run=name,step=step,epochs=epoch,parameters=11941122,
         selected=json.loads((folder/'best_report.json').read_text()),fingerprint=fingerprint,test_used=False))
    del model,opt;gc.collect();torch.cuda.empty_cache();return folder


def export_and_score(folder,name,bits,tok,dev,streams,teacher):
    import torch
    from .malbert_turn_export_v1 import export
    from .multilingual_tiny_compress_v3 import evaluate,retained,ort_predictor
    path=OUT/'candidates'/name
    if (path/'dev_report.json').exists():return json.loads((path/'dev_report.json').read_text())
    model,_=new_model();model.load_state_dict(torch.load(folder/'student.pt',map_location='cpu',weights_only=True),strict=True)
    quant=torch.load(folder/'quant_params.pt',map_location='cpu',weights_only=True) if bits==4 else None
    status('EXPORTING_EVALUATING_MALBERT_CPU',candidate=name,bits=bits)
    export(model,path,OUT/'tokenizer',dev,tok,bits=bits,quant=quant)
    del model;gc.collect()
    result=evaluate(ort_predictor(path,tok),dev,streams);passed,checks=retained(result,teacher)
    contract=dict(max_length=256,padding_to_multiple_of=8,intra_op_threads=2,lookahead_words=[1,2],threshold=result['threshold'],
         normalization='NFKC lowercase punctuation-to-space collapse-whitespace',
         input_format='CLS visible buffer SEP next1/2words SEP; left-truncate history only; segments derived from first SEP',
         languages=['vi','en','zh'],label2id={'NEOT':0,'EOT':1},parameters=11941122,hard_parameter_limit_exclusive=20_000_000,
         storage_precision='INT'+str(bits) if bits!=32 else 'FP32',compute_precision='FP32',gold_reset_in_inference=False,
         language_id_supplied=False,onnx_session_options={'session.disable_quant_qdq':'1'},smoke_only=False,
         quality_status='SILVER_DEV_RETENTION_PASS' if passed else 'RETENTION_FAIL')
    save(str(path.relative_to(OUT)/'turn_detector_config.json'),contract)
    shutil.copyfile(ROOT/'turn_v2/tiny_compression_v1/portable_runner.py',path/'run_model.py')
    (path/'requirements.txt').write_text('numpy>=2.1\nonnxruntime==1.23.2\ntokenizers==0.23.3\n')
    record=dict(candidate=name,path=str(path),parameters=11941122,storage_bits=bits,model_bytes=(path/'model.onnx').stat().st_size,
         model_sha256=sha(path/'model.onnx'),dev=result,retention_passed=passed,retention_by_language=checks,test_used=False)
    save(str(path.relative_to(OUT)/'dev_report.json'),record);return record


def finalize(candidates):
    passing=[c for c in candidates if c['retention_passed'] and c['parameters']<20_000_000]
    if not passing:
        save('final_report.json',dict(complete=True,accepted=False,candidates=candidates,test_used=False))
        status('COMPLETE_NO_ACCEPTED_MALBERT_MODEL');return
    winner=min(passing,key=lambda c:c['model_bytes']);target=PACKAGE/'onnx';target.mkdir(parents=True,exist_ok=False)
    names=['model.onnx','config.json','tokenizer.json','tokenizer_config.json','turn_detector_config.json','run_model.py','requirements.txt','export_receipt.json']
    for n in names:shutil.copyfile(Path(winner['path'])/n,target/n)
    bench={}
    for name,kind,path in [('teacher','teacher',OLD/'teacher'),('student','onnx',target)]:
        p=subprocess.run([sys.executable,'-m','turn_v2.multilingual_tiny_benchmark_v2','--kind',kind,'--model',str(path),'--cases',str(OLD/'dev_rows.jsonl')],cwd=ROOT,text=True,capture_output=True,timeout=360)
        save('benchmark_'+name+'.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0
        bench[name]=json.loads(p.stdout.strip().splitlines()[-1])
    files={n:dict(bytes=(target/n).stat().st_size,sha256=sha(target/n)) for n in names}
    manifest=dict(winner=winner,files=files,benchmark=bench,source_revision='d3f3c1addfec3420d3f306f674ed2087b34e0411',
         pretrained_backbone_preserved=True,parameters=11941122,packed_int4=winner['storage_bits']==4,
         compute_precision='FP32; packed matrices dequantized',quality_scope='Synthetic punctuation DEV retention, not independent human/audio gold',
         test_used=False,new_spoken_v9_used=False,large_teacher_restarted=False,backed_up_to_drive=False)
    (PACKAGE/'model_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=PACKAGE.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in PACKAGE.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(PACKAGE.parent))
    with zipfile.ZipFile(archive) as z:
        for n,x in files.items():assert hashlib.sha256(z.read(PACKAGE.name+'/onnx/'+n)).hexdigest()==x['sha256']
    save('final_report.json',dict(complete=True,accepted=True,winner=winner,candidates=candidates,benchmark=bench,
         package=str(PACKAGE),archive_sha256=sha(archive),manifest_sha256=sha(PACKAGE/'model_manifest.json'),test_used=False))
    status('COMPLETE',storage_bits=winner['storage_bits'],package=str(PACKAGE))


def run():
    import torch
    import transformers.utils as u
    import transformers.utils.import_utils as im
    u.is_torchvision_available=im.is_torchvision_available=lambda:False
    u.is_librosa_available=im.is_librosa_available=lambda:False
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.5);os.nice(5)
    torch.manual_seed(43);np.random.seed(43);random.seed(43)
    assert not (OUT/'final_report.json').exists(),'Do not restart a completed pretrained campaign'
    assert not Path('/proc/2827094').exists(),'Large teacher remains stopped by user request'
    tok,cache=prepare()
    from .multilingual_tiny_compress_v3 import make_dev
    dev,streams=make_dev();teacher=json.loads((OLD/'teacher_dev.json').read_text())
    fine=fit('finetune',OUT/'initial_student.pt',6,None,tok,cache,dev,streams,teacher,continue_until_retained=True)
    candidates=[export_and_score(fine,'finetuned_fp32',32,tok,dev,streams,teacher)]
    save('candidate_results.json',candidates)
    assert candidates[0]['retention_passed'],'Investigate actualCPUretention before QAT; preserve all fine-tuning state'
    control=fit('fp32_control',fine/'student.pt',2,None,tok,cache,dev,streams,teacher)
    for name,bits in [('fp32_control',32),('int8_control',8)]:
        candidates.append(export_and_score(control,name,bits,tok,dev,streams,teacher));save('candidate_results.json',candidates)
    for group in [64,32]:
        folder=fit('qat_g'+str(group),fine/'student.pt',2,group,tok,cache,dev,streams,teacher)
        candidates.append(export_and_score(folder,'int4_g'+str(group),4,tok,dev,streams,teacher));save('candidate_results.json',candidates)
    finalize(candidates)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    def stop(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (OUT/'CAMPAIGN_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except InterruptedError as e:status('STOPPED_BY_REQUEST',reason=str(e))
        except Exception as e:
            save('error.json',dict(error=repr(e),traceback=traceback.format_exc(),at_utc=datetime.now(timezone.utc).isoformat()))
            status('ERROR',error=repr(e));raise
