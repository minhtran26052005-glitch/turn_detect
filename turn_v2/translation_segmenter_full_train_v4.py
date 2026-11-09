"""Colab-only full-coverage training chunks; resume exact optimizer + cursor.

Languages rotate every 500 optimizer steps, retaining one immutable permutation
until every eligible row has been consumed exactly once. DEV selects checkpoints
and learning-rate reductions. No test, reserve or legacy-dev scoring occurs.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

from . import translation_segmenter_full_data_v2 as data
from .translation_segmenter_epoch_sampling_v1 import epoch_seed,make_order,batches,planned_steps,verify_complete
from . import translation_segmenter_v15c as base

ROOT=data.ROOT
JOB=data.JOB
STOP=False
CHUNK_STEPS=500
KEY='selection_score_min_accuracy_f1_period_recall_and_internal_punctuation_specificity'


def stop(*_):
    global STOP
    STOP=True


def state(stage,**fields):
    value={'campaign':'translation_segmenter_full_epoch_v1','status':'RUNNING','stage':stage,
           'heartbeat_utc':data.now(),'pid':os.getpid(),'quality_target_certified':False,
           'qwen_judge_enabled':False,**fields}
    base.atomic_json(ROOT/'status.json',value);base.atomic_json(JOB/'training_status.json',value)
    language=fields.get('language')
    if language in data.LANGUAGES:
        base.atomic_json(ROOT/'teachers'/language/'progress.json',value)
    print(json.dumps(value),flush=True)


def convergence(history):
    """Plateau on DEV controls LR; avoid training indefinitely to fit label noise."""
    if not history:return {'lr_scale':1.,'converged':False,'epochs':0}
    best=float('inf');best_score=float('-inf');stale=0;scale=1.;train_stale=0;best_train=float('inf')
    for r in history:
        loss=r['dev_cross_entropy'];tl=r['train_weighted_loss']
        score=r.get(KEY,0.)
        if loss<best-.001 or score>best_score+.001:stale=0
        else:stale+=1
        best=min(best,loss);best_score=max(best_score,score)
        if tl<best_train*(1-.005):best_train=tl;train_stale=0
        else:train_stale+=1
        if stale and stale%2==0:scale=max(.01,scale*.5)
    done=len(history)>=3 and stale>=6 and (train_stale>=3 or scale<=.0625)
    return {'lr_scale':scale,'converged':done,'epochs':len(history),'dev_stale_epochs':stale,
            'train_stale_epochs':train_stale,'reason':'DEV_SATURATION_KEEP_BEST' if done else 'CONTINUE_FULL_EPOCHS',
            'loss_zero_required':False,'test_used':False}


def history_for(manifest,language):
    folder=JOB/'runs'/manifest['generation']/language
    return [json.loads(p.read_text()) for p in sorted(folder.glob('epoch_*.json'))]


def preserve_previous_recipe(folder):
    archived=folder/'previous_recipe_v15d'
    complete=archived/'archive_receipt.json'
    if complete.exists():return
    archived.mkdir(exist_ok=True)
    paths=[folder/name for name in ('latest_resume.pt','best_report.json','last_training_report.json','identity.json','resume_receipt.json')]
    paths+=list((folder/'best').iterdir())
    receipt={}
    for source in paths:
        if not source.is_file():continue
        target=archived/source.relative_to(folder);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():os.link(source,target)
        expected=base.file_sha(source)
        assert base.file_sha(target)==expected,'Preserved starting checkpoint differs'
        receipt[str(source.relative_to(folder))]={'sha256':expected,'bytes':target.stat().st_size}
    assert 'latest_resume.pt' in receipt
    base.atomic_json(complete,{'at_utc':data.now(),'weights_optimizer_rng_before_full_epoch_training':True,'files':receipt})


def train_chunk(language,manifest,epoch,budget=CHUNK_STEPS):
    import numpy as np
    import torch
    import transformers.utils as u
    import transformers.utils.import_utils as ui
    u.is_torchvision_available=ui.is_torchvision_available=lambda:False
    u.is_librosa_available=ui.is_librosa_available=lambda:False
    from transformers import AutoConfig,AutoModelForSequenceClassification,AutoTokenizer
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(2)
    folder=ROOT/'teachers'/language;checkpoint=folder/'latest_resume.pt'
    assert checkpoint.exists(),'Never initialize from base while resuming full epochs'
    prior_receipt=json.loads((folder/'resume_receipt.json').read_text())
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert checkpoint.stat().st_size==prior_receipt['bytes']
    assert saved['total_steps']==prior_receipt['total_steps']
    assert saved['optimizer']['state'],'Optimizer resume is mandatory'
    preserve_previous_recipe(folder)
    state('LOADING_FULL_EPOCH_DATA_CACHE',language=language,cycle=manifest['source_cycle'],epoch=epoch,
          generation=manifest['generation'])
    encoded=data.cache(manifest,language,'train');n=len(encoded[0]);fp=manifest['fingerprints'][language]['train']['sha256']
    prior=saved.get('full_epoch',{})
    history=history_for(manifest,language);control=convergence(history)
    if prior.get('generation')==manifest['generation'] and prior.get('epoch')==epoch:
        assert prior['fingerprint']==fp and prior['eligible_rows']==n
        order=prior['order'];visits=prior['visits'];position=int(prior['position'])
        loss_sum=float(prior['loss_sum']);loss_den=float(prior['loss_den']);epoch_steps=int(prior['epoch_steps'])
    else:
        if prior:assert prior['position']==prior['eligible_rows'],'Finish previous frozen epoch before new epoch/data'
        order=make_order(n,epoch_seed(language,manifest['generation'],epoch));visits=np.zeros(n,dtype=np.uint8)
        position=0;loss_sum=0.;loss_den=0.;epoch_steps=0
    assert np.array_equal(np.sort(order),np.arange(n,dtype=np.int64))
    expected_visits=np.zeros(n,dtype=np.uint8);expected_visits[order[:position]]=1
    assert np.array_equal(visits,expected_visits),'Saved sampler proof disagrees with cursor'
    cfg=AutoConfig.from_pretrained(folder/'best',local_files_only=True)
    model=AutoModelForSequenceClassification.from_config(cfg,dtype=torch.float32).cuda()
    model.load_state_dict(saved['model'],strict=True);model.gradient_checkpointing_enable()
    optimizer=torch.optim.AdamW([
       {'params':[p for k,p in model.named_parameters() if not k.startswith(('classifier.','pooler.'))],'lr':1e-5},
       {'params':[p for k,p in model.named_parameters() if k.startswith(('classifier.','pooler.'))],'lr':5e-5}],weight_decay=.01)
    optimizer.load_state_dict(saved['optimizer']);total_steps=int(saved['total_steps'])
    torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    labels=encoded[2];counts=np.bincount(labels,minlength=2);assert min(counts)>0
    weights=torch.tensor(n/(2*counts),dtype=torch.float32,device='cuda')
    previous_numpy_rng=saved['numpy_rng'];del saved;gc.collect()
    start_time=time.monotonic();executed=0;last_save=start_time
    def save():
        extended={'generation':manifest['generation'],'fingerprint':fp,'epoch':epoch,
          'eligible_rows':n,'position':position,'order':order,'visits':visits,'epoch_steps':epoch_steps,
          'loss_sum':loss_sum,'loss_den':loss_den,'lr_scale':control['lr_scale'],
          'sampler':'PERMUTATION_WITHOUT_REPLACEMENT_V1','recipe':data.RECIPE}
        temp=folder/'latest_full_epoch.tmp'
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
           'cycle':manifest['source_cycle'],'cycle_step':epoch_steps,'total_steps':total_steps,
           'training_fingerprint':fp,'numpy_rng':previous_numpy_rng,'torch_rng':torch.get_rng_state(),
           'cuda_rng':torch.cuda.get_rng_state(),'full_epoch':extended},temp)
        temp.replace(checkpoint)
        base.atomic_json(folder/'resume_receipt.json',{'cycle':manifest['source_cycle'],'cycle_step':epoch_steps,
          'total_steps':total_steps,'bytes':checkpoint.stat().st_size,'training_fingerprint':fp,
          'full_epoch_generation':manifest['generation'],'full_epoch_dataset_sha256':manifest['database_sha256'],
          'full_epoch_number':epoch,'full_epoch_cursor':position,'full_epoch_rows':n,
          'full_epoch_complete':position==n,'epoch_weighted_loss':loss_sum/loss_den if loss_den else None,'weights_optimizer_rng_and_sampler':True,
          'required_restore_module':'turn_v2.restore_full_epoch_campaign_v4',
          'checkpoint':str(checkpoint),'weights_and_optimizer_remote_only':True})
    # Pin adopted AdamW/RNG plus new permutation before taking any new gradient.
    save()
    for begin,end,micros in batches(order,position):
        if executed>=budget or STOP:break
        model.train();optimizer.zero_grad(set_to_none=True)
        indices=order[begin:end];denominator=float((n/(2*counts))[labels[indices]].sum())
        step_loss=0.
        for idx in micros:
            width=int(encoded[1][idx].max())
            ids=torch.from_numpy(encoded[0][idx,:width].astype(np.int64)).cuda()
            lens=torch.from_numpy(encoded[1][idx].copy()).cuda()
            mask=torch.arange(width,device='cuda')[None]<lens[:,None]
            target=torch.from_numpy(labels[idx].copy()).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(input_ids=ids,attention_mask=mask.long()).logits.float()
                numerator=torch.nn.functional.cross_entropy(logits,target,weight=weights,reduction='sum',label_smoothing=.01)
                loss=numerator/denominator
            assert torch.isfinite(loss);loss.backward();step_loss+=float(numerator.detach())
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);assert torch.isfinite(grad)
        for group,lr in zip(optimizer.param_groups,(1e-5,5e-5)):group['lr']=lr*control['lr_scale']
        optimizer.step();visits[indices]+=1;position=end;epoch_steps+=1;total_steps+=1;executed+=1
        loss_sum+=step_loss;loss_den+=denominator
        if executed%10==0 or position==n:
            elapsed=time.monotonic()-start_time
            state('TRAINING_FULL_EPOCH',language=language,cycle=manifest['source_cycle'],epoch=epoch,
              generation=manifest['generation'],step=epoch_steps,steps=planned_steps(n),total_steps=total_steps,
              unique_rows_seen=position,eligible_rows=n,coverage=position/n,
              loss=step_loss/denominator,epoch_weighted_loss=loss_sum/loss_den,
              lr_scale=control['lr_scale'],seconds_per_optimizer_step=elapsed/executed,
              remaining_epoch_seconds=(planned_steps(n)-epoch_steps)*elapsed/executed)
        if executed%100==0 or position==n or time.monotonic()-last_save>300:
            save();last_save=time.monotonic()
    save()
    if position==n:
        proof=verify_complete(order,position,visits)
        run=JOB/'runs'/manifest['generation']/language;run.mkdir(parents=True,exist_ok=True)
        base.atomic_json(run/f'coverage_epoch_{epoch:04d}.json',{'at_utc':data.now(),'generation':manifest['generation'],
                         'language':language,'epoch':epoch,'training_fingerprint':fp,**proof})
        state('EVALUATING_FULL_DEV_AFTER_COMPLETE_EPOCH',language=language,cycle=manifest['source_cycle'],epoch=epoch)
        dev=data.cache(manifest,language,'dev')
        db=sqlite3.connect('file:'+str(JOB/'datasets'/manifest['generation']/'dataset.sqlite')+'?mode=ro',uri=True)
        rows=[json.loads(x[0]) for x in db.execute("SELECT record_json FROM records WHERE language=? AND split='dev' AND disposition='accepted' ORDER BY example_id",(language,))];db.close()
        report,probs=base.eval_report(model,dev,rows,language,manifest['source_cycle'],'full_epoch_dev')
        # Natural-frequency DEV cross entropy is separate from balanced TRAIN objective.
        clipped=np.clip(probs,1e-7,1-1e-7);y=dev[2]
        report.update(at_utc=data.now(),epoch=epoch,generation=manifest['generation'],
           training_recipe=data.RECIPE,train_weighted_loss=loss_sum/loss_den,
           dev_cross_entropy=float(-(y*np.log(clipped)+(1-y)*np.log(1-clipped)).mean()),
           coverage=proof,loss_smoothing=.01,loss_weighting='inverse_label_frequency',test_used_for_selection=False)
        tokenizer=AutoTokenizer.from_pretrained(folder/'best',local_files_only=True)
        previous=json.loads((folder/'best_report.json').read_text()) if (folder/'best_report.json').exists() else {}
        if previous.get('generation')!=manifest['generation'] or report[KEY]>previous[KEY]:
            # Replace files atomically; archived hardlinks retain their old inodes.
            stage=folder/f'best_staging_{manifest["generation"]}_{epoch:04d}'
            stage.mkdir(exist_ok=True)
            model.save_pretrained(stage,safe_serialization=True);tokenizer.save_pretrained(stage)
            for p in stage.iterdir():
                if p.is_file():p.replace(folder/'best'/p.name)
            stage.rmdir()
            report['weights_sha256']=base.file_sha(folder/'best/model.safetensors')
            base.atomic_json(folder/'best_report.json',report)
        base.atomic_json(run/f'epoch_{epoch:04d}.json',report)
        base.atomic_json(folder/'last_training_report.json',report)
        base.atomic_json(run/'convergence.json',convergence(history_for(manifest,language)))
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
    return {'epoch_complete':position==n,'position':position,'rows':n,'steps_in_chunk':executed}


if __name__=='__main__':
    import argparse,signal,traceback
    p=argparse.ArgumentParser();p.add_argument('--language',choices=data.LANGUAGES,required=True)
    p.add_argument('--generation',required=True);p.add_argument('--epoch',type=int,required=True)
    p.add_argument('--budget',type=int,default=CHUNK_STEPS);a=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    m=json.loads((JOB/'datasets'/a.generation/'manifest.json').read_text())
    try:
        import fcntl
        with (JOB/'GPU_LOCK').open('a') as gpu_lock:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX)
            train_chunk(a.language,m,a.epoch,a.budget)
    except Exception as exc:
        base.atomic_json(JOB/f'train_error_{a.language}.json',{'at_utc':data.now(),'error':repr(exc),'traceback':traceback.format_exc()})
        state('TRAIN_ERROR_PRESERVED',language=a.language,error=repr(exc));raise
