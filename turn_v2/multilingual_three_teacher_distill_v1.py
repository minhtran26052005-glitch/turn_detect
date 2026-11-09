"""Distill the three trained language teachers into one multilingual student.

Uses the immutable cycle-84 accepted train/dev SQLite data. No test data,
Qwen-generated pilot data, deployment, or language routing at student inference.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sqlite3
import time
import traceback

PROJECT=Path('/home/edabk/hoangbpm/turn_detect')
SOURCE=PROJECT/'outputs/translation_segmenter_v15d_20261026'
GEN='cycle000084_full_epoch_contiguous_v1'
DATA=SOURCE/'full_epoch_v1/datasets'/GEN/'dataset.sqlite'
OUT=Path('/mnt/hdd500/turn_detect_training/multilingual_distill_v1')
HF_CACHE=Path('/mnt/hdd500/turn_detect_training/huggingface_multilingual_distill_v1')
MODEL='distilbert/distilbert-base-multilingual-cased'
REV='45c032ab32cc946ad88a166f7cb282f58c753c2e'
LANGUAGES=('vi','en','zh')
LABELS={'NEOT':0,'EOT':1}
STOP=False


def request_stop(*_):
    global STOP
    STOP=True


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def status(stage,**extra):
    value={'stage':stage,'pid':os.getpid(),'at_unix':time.time(),'generation':GEN,
        'languages':list(LANGUAGES),'one_shared_student':True,'test_used':False,
        'qwen_pilot_used':False,'source':'punctuation_derived_corpus',**extra}
    atomic_json(OUT/'status.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)


def assert_storage():
    if shutil.disk_usage(PROJECT).free < 10*2**30 or shutil.disk_usage(OUT).free < 13*2**30:
        raise RuntimeError('LOW_DISK_PRESERVE_CHECKPOINT: require 10 GiB SSD and 13 GiB HDD for atomic checkpoint')


def rows_after(db,language,split,cursor,limit):
    # Frozen example_id is a SHA digest, yielding deterministic mixed source
    # order without a full-cache preparation or large OFFSET scans.
    sql=("SELECT example_id,tokens,input_hash,label,k,record_json FROM records "
      "INDEXED BY accepted_fingerprint_cover WHERE language=? AND split=? "
      "AND disposition='accepted' AND example_id>? ORDER BY example_id LIMIT ?")
    return db.execute(sql,(language,split,cursor,limit)).fetchall()


def validate_row(row):
    eid,raw,digest,label,k,record=row
    if not raw or len(raw)%4 or len(raw)>1024 or hashlib.sha256(raw).hexdigest()!=digest:
        raise ValueError('Frozen teacher tokens do not match source SHA')
    if label not in LABELS or k not in (1,2):raise ValueError('Unexpected label or lookahead')
    data=json.loads(record)
    if data['example_id']!=eid or data['label']!=label or data['lookahead_words']!=k:
        raise ValueError('Dataset JSON and saved teacher metadata differ')
    return data


def freeze_teachers(manifest):
    """Hardlink source files on the same filesystem so future replacement is safe."""
    result={}
    specs={'vi':('best/model.safetensors','best_report.json'),
           'en':('latest_resume.pt','resume_receipt.json'),
           'zh':('latest_resume.pt','resume_receipt.json')}
    for lang,(name,receipt_name) in specs.items():
        folder=SOURCE/'teachers'/lang;source=folder/name
        proof=json.loads((folder/receipt_name).read_text())
        if lang=='vi':
            assert proof['generation']==GEN and proof['epoch']==2
        else:
            assert proof['full_epoch_dataset_sha256']==manifest['database_sha256']
            assert proof['training_fingerprint']==manifest['fingerprints'][lang]['train']['sha256']
            assert proof['bytes']==source.stat().st_size
        frozen_dir=folder/'frozen_multilingual_distill_v1';frozen_dir.mkdir(exist_ok=True)
        frozen=frozen_dir/Path(name).name
        if not frozen.exists():os.link(source,frozen)
        elif not os.path.samefile(source,frozen):raise RuntimeError('Existing teacher freeze differs; do not overwrite')
        digest=sha(frozen)
        if lang=='vi':assert digest==proof['weights_sha256']
        result[lang]={'source':str(source),'frozen':str(frozen),'sha256':digest,'bytes':frozen.stat().st_size,
            'selection':'DEV-selected epoch 2' if lang=='vi' else 'frozen latest task checkpoint',
            'source_receipt':proof}
        status('FREEZING_TEACHERS',completed=list(result))
    return result


def prepare_batch(rows,student_tokenizer,teacher_tokenizer,device):
    import numpy as np
    import torch
    from .monolingual_data import encode_tail
    decoded=[validate_row(x) for x in rows]
    student=[encode_tail(student_tokenizer,r['context'],r['text'],256)['input_ids'] for r in decoded]
    teacher=[np.frombuffer(x[1],dtype='<u4').astype('int64') for x in rows]
    # Verify the teacher tokenizer and source field agree for every example.
    for i,r in enumerate(decoded):
        if list(teacher[i])!=encode_tail(teacher_tokenizer,r['context'],r['text'],256)['input_ids']:
            raise ValueError('Teacher tokenizer disagrees with immutable dataset')
    def pad(items,pad_id):
        width=max(len(x) for x in items)
        ids=torch.full((len(items),width),pad_id,dtype=torch.long,device=device)
        mask=torch.zeros_like(ids)
        for i,x in enumerate(items):
            ids[i,:len(x)]=torch.as_tensor(x,dtype=torch.long,device=device);mask[i,:len(x)]=1
        return ids,mask
    sid,smask=pad(student,student_tokenizer.pad_token_id)
    tid,tmask=pad(teacher,teacher_tokenizer.pad_token_id)
    return sid,smask,tid,tmask,torch.tensor([LABELS[x[3]] for x in rows],device=device)


def train(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    utils.is_torchvision_available=imports.is_torchvision_available=lambda:False
    utils.is_librosa_available=imports.is_librosa_available=lambda:False
    from transformers import AutoConfig,AutoModelForSequenceClassification,AutoTokenizer
    from safetensors.torch import load_file
    torch.set_num_threads(2);torch.manual_seed(20261030);np.random.seed(20261030);random.seed(20261030)
    assert torch.cuda.is_available(), 'CUDA GPU required'
    OUT.mkdir(parents=True,exist_ok=True);HF_CACHE.mkdir(parents=True,exist_ok=True)
    assert_storage()
    assert DATA.is_file() and Path('/mnt/hdd500').is_mount(), 'Dataset HDD must be mounted'
    manifest=json.loads((SOURCE/'full_epoch_v1/current_dataset.json').read_text())
    assert manifest['generation']==GEN and manifest['complete']
    proof=json.loads((DATA.parent/'verification_receipt.json').read_text())
    assert proof['complete'] and proof['database_sha256']==manifest['database_sha256']
    status('FREEZING_TEACHERS',completed=[])
    teachers_proof=freeze_teachers(manifest)
    config={'schema':1,'student_base':MODEL,'student_revision':REV,'teachers':teachers_proof,
       'generation':GEN,'dataset_sha256':manifest['database_sha256'],'fingerprints':manifest['fingerprints'],
       'model_inputs':'normalized context plus 1/2 future words; no language selector or punctuation',
       'label_order':['NEOT','EOT'],'sampling':'round-robin vi/en/zh; hashed-example-id order',
       'temperature':2.0,'kl_weight':0.5,'loss':'half cross entropy + half T^2 KL',
       'dataset_scope':'punctuation-derived; train/dev only; no Qwen pilot','dev_selection':'single shared threshold using min-language DEV F1',
       'qwen_used':False,'test_used':False}
    if (OUT/'recipe.json').exists():
        old=json.loads((OUT/'recipe.json').read_text())
        assert old['student_revision']==REV and old['dataset_sha256']==manifest['database_sha256']
        assert {k:v['sha256'] for k,v in old['teachers'].items()}=={k:v['sha256'] for k,v in teachers_proof.items()}
    else:atomic_json(OUT/'recipe.json',config)
    status('LOADING_STUDENT',model=MODEL,revision=REV)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REV,cache_dir=str(HF_CACHE),token=False)
    student=AutoModelForSequenceClassification.from_pretrained(MODEL,revision=REV,cache_dir=str(HF_CACHE),token=False,num_labels=2,
        id2label={0:'NEOT',1:'EOT'},label2id=LABELS).cuda()
    student.gradient_checkpointing_enable()
    tok_dir=OUT/'student_tokenizer';tok_dir.mkdir(exist_ok=True);tok.save_pretrained(tok_dir)
    teacher_tok=AutoTokenizer.from_pretrained(SOURCE/'teachers/vi/best',local_files_only=True)
    teachers={}
    for lang in LANGUAGES:
        status('LOADING_TEACHER',language=lang,loaded=list(teachers))
        cfg=AutoConfig.from_pretrained(SOURCE/'teachers/vi/best',local_files_only=True)
        model=AutoModelForSequenceClassification.from_config(cfg,dtype=torch.float32)
        path=teachers_proof[lang]['frozen']
        if lang=='vi':weights=load_file(path)
        else:
            loaded=torch.load(path,map_location='cpu',weights_only=False)
            assert loaded['training_fingerprint']==manifest['fingerprints'][lang]['train']['sha256']
            assert loaded['optimizer']['state'],'Source checkpoint is not a full trained checkpoint'
            weights=loaded['model']
        model.load_state_dict(weights,strict=True);model=model.to(device='cuda',dtype=torch.bfloat16).eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        teachers[lang]=model
        del weights,model
        if lang!='vi':del loaded
        gc.collect()
    db=sqlite3.connect(DATA.resolve().as_uri()+'?mode=ro',uri=True,timeout=30)
    db.execute('PRAGMA cache_size=-131072')
    opt=torch.optim.AdamW(student.parameters(),lr=2e-5,weight_decay=.01)
    ckpt=OUT/'latest_student.pt';step=0;cursors={x:'' for x in LANGUAGES};counts={x:0 for x in LANGUAGES};rounds={x:0 for x in LANGUAGES}
    if ckpt.exists():
        state=torch.load(ckpt,map_location='cpu',weights_only=False)
        assert state['generation']==GEN and state['dataset_sha256']==manifest['database_sha256']
        assert state['teacher_hashes']=={k:v['sha256'] for k,v in teachers_proof.items()}
        student.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
        step=state['step'];cursors=state['cursors'];counts=state['counts'];rounds=state['rounds']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        del state
    device=torch.device('cuda');loss_values=[]
    def save():
        assert_storage()
        state={'model':student.state_dict(),'optimizer':opt.state_dict(),'step':step,'cursors':cursors,'counts':counts,'rounds':rounds,
          'generation':GEN,'dataset_sha256':manifest['database_sha256'],'teacher_hashes':{k:v['sha256'] for k,v in teachers_proof.items()},
          'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
        tmp=OUT/'latest_student.tmp';torch.save(state,tmp);os.replace(tmp,ckpt)
        atomic_json(OUT/'resume_receipt.json',{'step':step,'bytes':ckpt.stat().st_size,'cursors':cursors,'counts':counts,'rounds':rounds,
          'dataset_sha256':manifest['database_sha256'],'teacher_hashes':state['teacher_hashes'],'optimizer_and_rng_included':True})
    def evaluate():
        status('EVALUATING_DEV',step=step)
        student.eval();all_results={}
        for lang in LANGUAGES:
            rows=rows_after(db,lang,'dev','',args.dev_samples)
            probs=[];labels=[]
            with torch.inference_mode():
                for begin in range(0,len(rows),args.batch_size):
                    subset=rows[begin:begin+args.batch_size]
                    sid,smask,_tid,_tmask,y=prepare_batch(subset,tok,teacher_tok,device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        p=student(input_ids=sid,attention_mask=smask).logits.float().softmax(-1)[:,1]
                    probs.extend(p.cpu().tolist());labels.extend(y.cpu().tolist())
            all_results[lang]=(np.array(probs),np.array(labels))
        def metrics(probs,labels,t):
            pred=probs>=t;positive=labels==1
            tp=int((pred&positive).sum());fp=int((pred&~positive).sum());fn=int((~pred&positive).sum())
            return {'samples':len(labels),'accuracy':float((pred==positive).mean()),'eot_f1':float(2*tp/max(1,2*tp+fp+fn)), 'tp':tp,'fp':fp,'fn':fn}
        candidates=np.linspace(.10,.90,81)
        threshold=max(candidates,key=lambda t:min(metrics(*all_results[l],t)['eot_f1'] for l in LANGUAGES))
        report={'step':step,'split':'dev','scope':'punctuation-derived, not human/audio conversational gold','shared_threshold':float(threshold),
          'by_language':{l:metrics(*all_results[l],threshold) for l in LANGUAGES},'sample_policy':'first hashed example_ids of each frozen DEV split',
          'test_used_for_selection':False,'same_model_for_all_languages':True}
        atomic_json(OUT/f'dev_step_{step:08d}.json',report);atomic_json(OUT/'latest_dev.json',report)
        student.train();return report
    status('TRAINING',step=step,counts=counts,loss=None)
    while not STOP:
        assert_storage()
        lang=LANGUAGES[step%len(LANGUAGES)]
        rows=rows_after(db,lang,'train',cursors[lang],args.batch_size)
        if not rows:
            rounds[lang]+=1;cursors[lang]='';rows=rows_after(db,lang,'train','',args.batch_size)
        if not rows:raise RuntimeError('Empty frozen train language')
        sid,smask,tid,tmask,y=prepare_batch(rows,tok,teacher_tok,device)
        opt.zero_grad(set_to_none=True);student.train()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            reference=teachers[lang](input_ids=tid,attention_mask=tmask).logits.float()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            actual=student(input_ids=sid,attention_mask=smask).logits.float()
            supervised=F.cross_entropy(actual,y)
            t=2.;distilled=F.kl_div(F.log_softmax(actual/t,dim=-1),F.softmax(reference/t,dim=-1),reduction='batchmean')*(t*t)
            loss=.5*supervised+.5*distilled
        if not torch.isfinite(loss):raise RuntimeError('Non-finite distillation loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1.0);opt.step()
        step+=1;cursors[lang]=rows[-1][0];counts[lang]+=len(rows);loss_values.append(float(loss.detach()))
        if step%10==0:
            status('TRAINING',step=step,current_language=lang,counts=counts,rounds=rounds,loss=sum(loss_values[-30:])/min(30,len(loss_values)),
              cuda_peak_reserved_gib=round(torch.cuda.max_memory_reserved()/2**30,2))
        if step%args.save_every==0:save()
        if step%args.eval_every==0:
            save();evaluate()
    save();status('STOPPED_BY_REQUEST',step=step,counts=counts)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--save-every',type=int,default=100);parser.add_argument('--eval-every',type=int,default=1000)
    parser.add_argument('--dev-samples',type=int,default=1000);args=parser.parse_args()
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (SOURCE/'full_epoch_v1/GPU_LOCK').open('a') as gpu:
            fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:train(args)
            except Exception as e:
                atomic_json(OUT/'error.json',{'at_unix':time.time(),'error':repr(e),'traceback':traceback.format_exc()})
                status('ERROR',error=repr(e));raise


if __name__=='__main__':main()
