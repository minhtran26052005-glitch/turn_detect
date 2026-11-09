"""Prepare additional TRAIN while v2 runs; then preserve state and launch v3."""
import copy
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .checked_dialogues_prepare_v1 import OUT,PARENT,ROOT,prepare,sha,save


def state(stage,**values):
    save('cutover_v2_status.json',dict(stage=stage,pid=os.getpid(),at_utc=datetime.now(timezone.utc).isoformat(),**values))


def alive(launch,module):
    p=Path('/proc')/str(launch['pid'])
    if not p.exists():return False
    s=(p/'stat').read_text();ticks=s[s.rfind(')')+2:].split()[19]
    return ticks==str(launch['start_ticks']) and module in (p/'cmdline').read_bytes().decode()


def link_same(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert sha(source)==sha(target)
    else:os.link(source,target)


def fingerprint(receipt):
    names=['malbert_turn_student_v1.py','multilingual_malbert_vi_en_checked_train_v3.py','malbert_turn_export_v2.py','bilingual_stream_evaluation_v1.py','checked_dialogues_prepare_v1.py']
    return {'train_sha256':receipt['train_sha256'],'cache_sha256':sha(OUT/'cache/receipt.json'),
        'pretrained_sha256':'3040f56fef13a7c0d6b0e6f608daf9d8cd8cc4f4a163844d652d0b3e80cbdc84','active_languages':['vi','en'],
        'code_sha256':{n:sha(ROOT/'turn_v2'/n) for n in names}}


def preserve_and_migrate(receipt):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    if (OUT/'migration_receipt.json').exists():
        result=json.loads((OUT/'migration_receipt.json').read_text());assert result['cache_sha256']==sha(OUT/'cache/receipt.json');return result
    launch=json.loads((PARENT/'launcher.json').read_text());assert launch['pid']==914852 and str(launch['start_ticks'])=='60017417'
    state('READY_FOR_SAFE_CHECKPOINT_HANDOVER',parent_pid=launch['pid'],additional_rows=receipt['additional_rows'])
    if alive(launch,'turn_v2.multilingual_malbert_vi_en_train_v2'):
        marker=PARENT/'STOP_REQUESTED'
        with marker.open('w') as f:f.write('User authorized additional checked VI/EN TRAIN data; preserve full checkpoint before versioned handover.\n');f.flush();os.fsync(f.fileno())
        assert marker.read_text().startswith('User authorized additional checked')
        while alive(launch,'turn_v2.multilingual_malbert_vi_en_train_v2'):time.sleep(2)
    parent_status=json.loads((PARENT/'status.json').read_text());assert parent_status['stage']=='STOPPED_BY_REQUEST',parent_status
    assert not (PARENT/'error.json').exists(),'Do not hide a failed predecessor'
    pin=OUT/'parent_full_resume.pt';link_same(PARENT/'runs/finetune/latest_resume.pt',pin)
    original=torch.load(pin,map_location='cpu',weights_only=False,mmap=True);assert original['parameters']==11941122 and len(original['optimizer']['state'])==27
    assert original['fingerprint']['active_languages']==['vi','en']
    assert original['fingerprint']['cache_sha256']==sha(PARENT/'cache/receipt.json')
    for name,value in original['fingerprint']['code_sha256'].items():assert sha(ROOT/'turn_v2'/name)==value
    for folder in ['pretrained','tokenizer']:
        for source in (PARENT/folder).rglob('*'):
            if source.is_file():link_same(source,OUT/folder/source.relative_to(PARENT/folder))
    for name in ['initial_student.pt','dev_rows_vi_en.jsonl','dev_streams_vi_en.json','teacher_vi_en_dev.json']:
        link_same(PARENT/name,OUT/name)
    if not (OUT/'initialization_receipt.json').exists():
        save('initialization_receipt.json',{'mode':'INHERITED_FULL_RESUME_CHECKPOINT',
            'parent_checkpoint_sha256':sha(pin),'model_optimizer_rng_inherited':True,
            'new_base_initialization_used_for_training':False,'parameters':11941122,
            'pretrained_source_revision':'d3f3c1addfec3420d3f306f674ed2087b34e0411'})
    n=receipt['rows'];old_n=200000
    assert len(original['order'])==old_n
    extra=np.random.default_rng(20261103).permutation(np.arange(old_n,n,dtype=np.int64))
    order=np.concatenate([original['order'],extra])
    assert len(np.unique(order))==n and np.array_equal(order[:old_n],original['order'])
    child=dict(original);child['order']=order;child['epoch']=0;child['best_key']=(-1.,-1.,-1.)
    child['fingerprint']=fingerprint(receipt);child['additional_rows_seen']=0
    child['warm_ce_completed']=original['epoch']>=2
    old_total=6*math.ceil(old_n/16);new_total=6*math.ceil(n/16)
    child['lr_step_offset']=(original['step']+original.get('lr_step_offset',0.))/old_total*new_total-original['step']
    target=OUT/'runs/finetune/latest_resume.pt';target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists()
    tmp=target.with_name('migrating_resume.pt');torch.save(child,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    tmp.replace(target);loaded=torch.load(target,map_location='cpu',weights_only=False,mmap=True)
    assert all(torch.equal(value,loaded['model'][key]) for key,value in original['model'].items())
    def equal(a,b):
        if isinstance(a,torch.Tensor):return torch.equal(a,b)
        if isinstance(a,np.ndarray):return np.array_equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    assert equal(original['optimizer'],loaded['optimizer'])
    assert all(equal(original[k],loaded[k]) for k in ['torch_rng','cuda_rng','numpy_rng','python_rng','sampler_rng'])
    assert loaded['step']==original['step'] and loaded['cursor']==original['cursor']
    result={'authorized_active_languages':['vi','en'],'user_authorized_additional_train':True,'model_optimizer_rng_preserved':True,
        'all_model_tensors_equal':True,'all_optimizer_tensors_equal':True,'rng_and_sampler_state_equal':True,
        'parent_pid':launch['pid'],'parent_start_ticks':launch['start_ticks'],'parent_step':original['step'],
        'parent_epoch':original['epoch'],'parent_cursor':original['cursor'],'parent_rows':old_n,'active_rows':n,
        'additional_rows':receipt['additional_rows'],'remaining_parent_order_preserved_before_any_new_rows':True,
        'new_branch_epoch':0,'lr_step_offset':child['lr_step_offset'],'optimizer_entries':27,'parameters':11941122,
        'parent_checkpoint_sha256':sha(pin),'migrated_checkpoint_sha256':sha(target),'cache_sha256':sha(OUT/'cache/receipt.json'),
        'dev_files_unchanged':{name:sha(OUT/name) for name in ['dev_rows_vi_en.jsonl','dev_streams_vi_en.json','teacher_vi_en_dev.json']},
        'dataset_fingerprint_changed_only_in_versioned_child':True,'test_used':False}
    save('migration_receipt.json',result)
    save('recipe.json',{'parameters':11941122,'active_languages':['vi','en'],'additional_data':receipt,'migration':result,
        'loss':'Parent rows retain CE+frozen-KL; additional rows are CE only via teacher_mask=0',
        'dev':'Byte-identical original VI/EN rolling calibration and predicted-reset DEV; imported DEV/TEST excluded',
        'state':'Model/AdamW27/TorchCUDA_numpyPythonRNG/sampler retained. Finish remaining parent row order before additional rows.',
        'deployment':'Same strict VI/EN retention gates, then FP32/INT8/real packed INT4 QAT and CPU verification; no automatic replacement',
        'new_spoken_colab_imported':False,'human_audio_gold':False,'drive_backup_configured':False})
    return result


def run():
    state('PREPARING_ADDITIONAL_TRAIN_PARENT_CONTINUES')
    receipt=prepare();migration=preserve_and_migrate(receipt)
    launch_path=OUT/'launcher.json'
    if launch_path.exists():
        launch=json.loads(launch_path.read_text());assert alive(launch,'turn_v2.multilingual_malbert_vi_en_checked_train_v3'),'Existing child stopped; inspect its checkpoint before explicit resume'
    else:
        assert not (Path('/proc')/str(migration['parent_pid'])).exists()
        with (OUT/'worker.log').open('ab') as log:
            p=subprocess.Popen([sys.executable,'-u','-m','turn_v2.multilingual_malbert_vi_en_checked_train_v3'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        time.sleep(.5);proc=Path('/proc')/str(p.pid);raw=(proc/'stat').read_text()
        launch={'pid':p.pid,'start_ticks':raw[raw.rfind(')')+2:].split()[19],
            'module':'turn_v2.multilingual_malbert_vi_en_checked_train_v3','boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
        save('launcher.json',launch)
    state('WAITING_FOR_ACTUAL_ADDITIONAL_DATA_TRAINING',trainer=launch,parent_step=migration['parent_step'])
    import torch
    while alive(launch,'turn_v2.multilingual_malbert_vi_en_checked_train_v3'):
        path=OUT/'runs/finetune/latest_resume.pt'
        with path.open('rb') as f:
            x=torch.load('/proc/self/fd/'+str(f.fileno()),map_location='cpu',weights_only=False,mmap=True)
            if x.get('additional_rows_seen',0)>0:
                assert x['step']>migration['parent_step'] and len(x['optimizer']['state'])==27
                assert {int(s['step']) for s in x['optimizer']['state'].values()}=={x['step']}
                assert len(x['order'])==receipt['rows'] and x['fingerprint']==fingerprint(receipt)
                f.seek(0);h=hashlib.sha256()
                for b in iter(lambda:f.read(2**20),b''):h.update(b)
                proof={'verified':True,'trainer':launch,'step':x['step'],'epoch':x['epoch'],'cursor':x['cursor'],
                    'additional_rows_seen':x['additional_rows_seen'],'optimizer_entries':27,'rng_order_and_source_verified':True,
                    'parameters':11941122,'active_languages':['vi','en'],'checkpoint_sha256':h.hexdigest(),
                    'new_cache_sha256':sha(OUT/'cache/receipt.json'),'original_dev_test_excluded':True,'test_used':False}
                save('independent_training_verification.json',proof);state('TRAINING_ADDITIONAL_DATA_VERIFIED',**proof);return
        time.sleep(10)
    raise RuntimeError('Child exited before a checkpoint proved additional data was trained; preserve all files')


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);os.nice(10)
    with (OUT/'IMPORT_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception as e:
            save('cutover_v2_error.json',{'error':repr(e),'traceback':traceback.format_exc(),'at_utc':datetime.now(timezone.utc).isoformat()})
            state('ERROR',error=repr(e));raise
