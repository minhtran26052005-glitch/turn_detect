"""New authorized shared-student branch: dense streaming negatives + ZH replay."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sqlite3
import time
import traceback

from .factory_stream_data_v1 import Sampler, sha
from .monolingual_data import encode_tail
from .multilingual_three_teacher_distill_v1 import DATA as REPLAY, rows_after, validate_row, SOURCE, MODEL, REV, HF_CACHE

OUT=Path('/home/edabk/hoangbpm/turn_detect/outputs/multilingual_stream_v2_20261102')
STOP=False


def stop(*_):
    global STOP
    STOP=True


def save_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def status(stage,**more):
    record={'stage':stage,'pid':os.getpid(),'at_utc':datetime.now(timezone.utc).isoformat(),'source':'Qwen synthetic punctuation streaming + legacy ZH replay',**more}
    save_json(OUT/'status.json',record);print(json.dumps(record,ensure_ascii=False),flush=True)


def metrics(probs,labels,t):
    import numpy as np
    probs=np.asarray(probs);labels=np.asarray(labels,dtype=bool);pred=probs>=t
    tp=int((pred&labels).sum());tn=int((~pred&~labels).sum());fp=int((pred&~labels).sum());fn=int((~pred&labels).sum());n=tp+tn+fp+fn
    return {'samples':n,'accuracy':(tp+tn)/n if n else None,'eot_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'tp':tp,'tn':tn,'fp':fp,'fn':fn}


def train(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    utils.is_torchvision_available=imports.is_torchvision_available=lambda:False
    utils.is_librosa_available=imports.is_librosa_available=lambda:False
    from transformers import AutoConfig,AutoModelForSequenceClassification,AutoTokenizer
    from safetensors.torch import save_file
    torch.set_num_threads(2);os.nice(3)
    dataset=OUT/'dataset';manifest=json.loads((dataset/'manifest.json').read_text());assert manifest['complete']
    dataset_sha=sha(dataset/'manifest.json')
    status('VERIFYING_NEW_DATASET',branch_step=0)
    for name,meta in manifest['files'].items():assert sha(dataset/name)==meta['sha256']
    initial=json.loads((OUT/'warm_start_receipt.json').read_text());assert sha(OUT/'warm_start.pt')==initial['sha256']
    assert initial['optimizer_and_rng_preserved']
    recipe={'student_base':MODEL,'revision':REV,'dataset_manifest_sha256':dataset_sha,'source_commit':manifest['source_frozen_commit_index'],
            'trainer_source_sha256':sha(__file__),'sampler_source_sha256':sha(Sampler.__init__.__code__.co_filename),
            'warm_start':initial,'loss':'unweighted CE on dense synthetic streaming references + legacy ZH accepted references; no old teacher KL on new negatives',
            'schedule':['vi','en','vi','en','zh'],'batch_size':args.batch_size,'learning_rate':'inherit AdamW 2e-5 and optimizer state',
            'labels':'All eligible word gaps; full stop EOT, other gaps NEOT, question/exclamation endpoints ignored',
            'source_groups_split_before_expansion':True,'exposed_200_review_dialogues_excluded':True,'sealed_test_scored':False,
            'training_authorized_by_user_20261102':True,'qwen_labels_are_silver_not_human_gold':True,
            'dev_threshold':'one threshold selected on causal rolling all-gap DEV VI/EN; evaluate predicted-reset rollout at that locked threshold',
            'best_selection':'min(VI reset EOT F1, EN reset EOT F1, legacy ZH EOT F1), DEV only',
            'baseline_probes_used_for_selection':False}
    if (OUT/'recipe.json').exists():assert json.loads((OUT/'recipe.json').read_text())==recipe
    else:save_json(OUT/'recipe.json',recipe)
    checkpoint=OUT/'latest_student.pt';state=torch.load(checkpoint if checkpoint.exists() else OUT/'warm_start.pt',map_location='cpu',weights_only=False)
    resuming=checkpoint.exists()
    if resuming:assert state['dataset_manifest_sha256']==dataset_sha
    else:assert state['dataset_sha256']==initial['source_dataset_sha256']
    assert state['optimizer']['state'] and all(k in state for k in ['torch_rng','cuda_rng','numpy_rng','python_rng'])
    cfg=AutoConfig.from_pretrained(MODEL,revision=REV,cache_dir=str(HF_CACHE),local_files_only=True,token=False);cfg.num_labels=2;cfg.id2label={0:'NEOT',1:'EOT'};cfg.label2id={'NEOT':0,'EOT':1}
    tok=AutoTokenizer.from_pretrained('/mnt/hdd500/turn_detect_training/multilingual_distill_v1/student_tokenizer',local_files_only=True,token=False)
    model=AutoModelForSequenceClassification.from_config(cfg,dtype=torch.float32);model.load_state_dict(state['model'],strict=True);model.cuda();model.gradient_checkpointing_enable()
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01);opt.load_state_dict(state['optimizer'])
    branch_step=state['branch_step'] if resuming else 0;base_step=state['base_step'] if resuming else state['step']
    sampler=Sampler(dataset,state=state['sampler'] if resuming else None)
    zh_cursor=state['zh_cursor'] if resuming else state['cursors']['zh'];zh_round=state['zh_round'] if resuming else state['rounds']['zh']
    counts=state['counts'] if resuming else {'vi':0,'en':0,'zh':0};best=state.get('best_score',-1.) if resuming else -1.
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    if resuming:assert sampler.preview_sha('vi')==state['next_vi_batch_preview_sha256']
    del state;gc.collect()
    tok_dir=OUT/'student_tokenizer';tok.save_pretrained(tok_dir);cfg.save_pretrained(tok_dir)
    replay=sqlite3.connect(REPLAY.resolve().as_uri()+'?mode=ro',uri=True,timeout=30)
    dev_streams=json.loads((dataset/'dev_streams.json').read_text())
    dev_zh=[validate_row(r) for r in rows_after(replay,'zh','dev','',1000)]
    def batch(rows):
        seqs=[encode_tail(tok,r['context'],r['text'],256)['input_ids'] for r in rows];width=max(map(len,seqs))
        ids=torch.full((len(rows),width),tok.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
        for i,seq in enumerate(seqs):ids[i,:len(seq)]=torch.as_tensor(seq,device='cuda');mask[i,:len(seq)]=1
        return ids,mask
    def predict(rows):
        result=[]
        for start in range(0,len(rows),args.batch_size):
            ids,mask=batch(rows[start:start+args.batch_size])
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):result.extend(model(input_ids=ids,attention_mask=mask).logits.float().softmax(-1)[:,1].cpu().tolist())
        return result
    def evaluate(baseline=False):
        nonlocal best
        status('EVALUATING_STREAMING_DEV',branch_step=branch_step,step=base_step+branch_step);model.eval()
        static={'vi':([],[]),'en':([],[])};rows=[]
        for s in dev_streams:
            for b in range(1,len(s['words'])-1):
                if b in s['unknown']:continue
                rows.append({'context':' '.join(s['words'][:b]),'text':' '.join(s['words'][b:b+2]),'language':s['language_by_word'][b-1],'label':b in s['eots']})
        probabilities=predict(rows)
        for r,p in zip(rows,probabilities):static[r['language']][0].append(p);static[r['language']][1].append(r['label'])
        threshold=float(max(np.linspace(.10,.95,86),key=lambda t:min(metrics(*static[l],t)['eot_f1'] for l in ('vi','en'))))
        rollout={'vi':([],[]),'en':([],[])};starts=[0]*len(dev_streams)
        for b in range(1,max(len(s['words']) for s in dev_streams)-1):
            subset=[];indexes=[]
            for i,s in enumerate(dev_streams):
                if b+2>len(s['words']):continue
                subset.append({'context':' '.join(s['words'][starts[i]:b]),'text':' '.join(s['words'][b:b+2])});indexes.append(i)
            if not subset:continue
            for i,p in zip(indexes,predict(subset)):
                s=dev_streams[i]
                if b not in s['unknown']:
                    lang=s['language_by_word'][b-1];rollout[lang][0].append(p);rollout[lang][1].append(b in s['eots'])
                if p>=threshold:starts[i]=b
        zh_probs=predict(dev_zh);zh_labels=[r['label']=='EOT' for r in dev_zh]
        result={'step':base_step+branch_step,'branch_step':branch_step,'split':'dev','phase':'BASELINE_BEFORE_NEW_TRAINING' if baseline else 'NEW_TRAINING_DEV',
                'shared_threshold':threshold,'by_language':{l:metrics(*rollout[l],threshold) for l in ('vi','en')},
                'static_rolling_by_language':{l:metrics(*static[l],threshold) for l in ('vi','en')},
                'scope':'VI/EN synthetic punctuation every-gap causal predicted-reset streaming DEV; ZH legacy punctuation anchors, not comparable human/audio gold',
                'stream_count':len(dev_streams),'threshold_selected_on_rolling_dev_only':True,'gold_resets_in_eval':False,'sealed_test_used':False,'old_probes_used':False}
        result['by_language']['zh']=metrics(zh_probs,zh_labels,threshold)
        score=min(m['eot_f1'] for m in result['by_language'].values());result['selection_score']=score
        if score>best:
            best=score;folder=OUT/'best';folder.mkdir(exist_ok=True);tmp=folder/'model.safetensors.tmp'
            save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(tmp));tmp.replace(folder/'model.safetensors');cfg.save_pretrained(folder);tok.save_pretrained(folder);save_json(OUT/'best_report.json',result)
        save_json(OUT/f'dev_branch_step_{branch_step:08d}.json',result);save_json(OUT/'latest_dev.json',result)
        if baseline:save_json(OUT/'baseline_dev.json',result)
        model.train();return result
    def save_checkpoint():
        assert shutil.disk_usage(OUT).free>=12*2**30,'Keep atomic checkpoint reserve'
        state={'model':model.state_dict(),'optimizer':opt.state_dict(),'step':base_step+branch_step,'base_step':base_step,'branch_step':branch_step,
               'sampler':sampler.state_dict(),'zh_cursor':zh_cursor,'zh_round':zh_round,'counts':counts,'best_score':best,
               'dataset_manifest_sha256':dataset_sha,'next_vi_batch_preview_sha256':sampler.preview_sha('vi'),
               'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
        tmp=OUT/'latest_student.tmp';torch.save(state,tmp)
        with tmp.open('rb') as f:os.fsync(f.fileno())
        if checkpoint.exists():
            link=OUT/'previous_resume.link';assert not link.exists();os.link(checkpoint,link);link.replace(OUT/'previous_resume.pt')
        tmp.replace(checkpoint)
        receipt={'step':base_step+branch_step,'branch_step':branch_step,'bytes':checkpoint.stat().st_size,'optimizer_entries':len(opt.state),
                 'optimizer_and_rng_included':True,'sampler_state_included':True,'dataset_manifest_sha256':dataset_sha,
                 'next_vi_batch_preview_sha256':state['next_vi_batch_preview_sha256'],'counts':counts,'epochs':sampler.epochs,'zh_cursor':zh_cursor}
        if branch_step==100:
            check=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert check['branch_step']==100 and check['optimizer']['state'] and check['sampler']==state['sampler'] and check['next_vi_batch_preview_sha256']==state['next_vi_batch_preview_sha256']
            assert torch.equal(check['torch_rng'],state['torch_rng']) and check['zh_cursor']==zh_cursor
            assert check['dataset_manifest_sha256']==dataset_sha
            receipt['first_checkpoint_torch_load_verified']=True
            receipt['optimizer_step_max']=max(float(s['step']) for s in check['optimizer']['state'].values())
            assert receipt['optimizer_step_max']>=initial['optimizer_step_max']+100
            save_json(OUT/'first_checkpoint_verification.json',receipt);del check;gc.collect()
        save_json(OUT/'resume_receipt.json',receipt)
    save_checkpoint()
    if branch_step==0 and not (OUT/'baseline_dev.json').exists():evaluate(baseline=True);save_checkpoint()
    schedule=('vi','en','vi','en','zh');losses=[]
    while not STOP:
        if shutil.disk_usage(OUT).free<13*2**30:
            status('WAITING_FOR_DISK_RESERVE',branch_step=branch_step,step=base_step+branch_step);time.sleep(30);continue
        lang=schedule[branch_step%5]
        if lang=='zh':
            source_rows=rows_after(replay,'zh','train',zh_cursor,args.batch_size)
            if not source_rows:zh_cursor='';zh_round+=1;source_rows=rows_after(replay,'zh','train','',args.batch_size)
            rows=[validate_row(r) for r in source_rows]
        else:rows=[sampler.row(lang) for _ in range(args.batch_size)]
        ids,mask=batch(rows);labels=torch.as_tensor([r['label']=='EOT' for r in rows],dtype=torch.long,device='cuda');opt.zero_grad(set_to_none=True);model.train()
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(model(input_ids=ids,attention_mask=mask).logits.float(),labels)
        assert torch.isfinite(loss),'Non-finite streaming loss'
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();branch_step+=1
        if lang=='zh':zh_cursor=source_rows[-1][0]
        for r in rows:counts[r.get('language',lang)]+=1
        losses.append(float(loss.detach()));losses=losses[-30:]
        if branch_step%10==0:status('TRAINING_NEW_STREAMING',step=base_step+branch_step,base_step=base_step,branch_step=branch_step,current_language=lang,counts=counts,epochs=sampler.epochs,loss=sum(losses)/len(losses),cuda_peak_reserved_gib=round(torch.cuda.max_memory_reserved()/2**30,2))
        if branch_step%100==0:save_checkpoint()
        if branch_step%args.eval_every==0:evaluate();save_checkpoint()
    save_checkpoint();status('STOPPED_BY_REQUEST',step=base_step+branch_step,branch_step=branch_step,counts=counts)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,default=64);p.add_argument('--eval-every',type=int,default=1000);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (OUT/'campaign.lock').open('a') as own,(SOURCE/'full_epoch_v1/GPU_LOCK').open('a') as gpu:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:train(args)
        except Exception as error:
            save_json(OUT/'error.json',{'error':repr(error),'traceback':traceback.format_exc()});status('ERROR',error=repr(error));raise
