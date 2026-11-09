"""SSH-only graceful cutover from the old distill worker to the new data branch."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

PROJECT=Path('/home/edabk/hoangbpm/turn_detect')
OUT=PROJECT/'outputs/multilingual_stream_v2_20261102'
OLD=Path('/mnt/hdd500/turn_detect_training/multilingual_distill_v1')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(name,r):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');data=json.dumps(r,ensure_ascii=False,indent=2)+'\n'
    with t.open('w') as f:f.write(data);f.flush();os.fsync(f.fileno())
    t.replace(p)


def status(stage,**kw):
    write('cutover_status.json',{'stage':stage,'pid':os.getpid(),'at_utc':datetime.now(timezone.utc).isoformat(),**kw})


def identity(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    s=(p/'stat').read_text();t=s[s.rfind(')')+2:].split()
    return {'pid':pid,'start_ticks':t[19],'state':t[0],'cmdline':(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()}


def run():
    import torch
    sys.path.insert(0,str(PROJECT))
    from turn_v2.factory_stream_data_v1 import Sampler
    assert sys.platform!='win32' and OUT.is_dir()
    preflight=json.loads((OUT/'preflight_receipt.json').read_text());assert preflight['passed']
    assert sha(PROJECT/'turn_v2/factory_stream_train_v1.py')==preflight['new_trainer_source_sha256']
    assert sha(PROJECT/'turn_v2/factory_stream_data_v1.py')==preflight['sampler_source_sha256']
    assert sha(OUT/'dataset/manifest.json')==preflight['dataset_manifest_sha256']
    assert shutil.disk_usage(OUT).free>=18*2**30
    sampler=Sampler(OUT/'dataset');state=sampler.state_dict();a=[sampler.row('vi') for _ in range(32)];sampler.load_state_dict(state);assert a==[sampler.row('vi') for _ in range(32)]
    write('training_runtime_preflight.json',{'passed':True,'python':sys.executable,'sampler_resume_verified':True,'rows':32})
    assert not (OUT/'launcher.json').exists(),'New worker already launched; inspect instead of duplicating'
    old=identity(2611438)
    if old and old['state']!='Z':
        assert old['start_ticks']=='27127617' and '-m turn_v2.multilingual_three_teacher_distill_v1' in old['cmdline']
        status('STOPPING_OLD_AT_SAFE_CHECKPOINT',old_process=old)
        os.kill(old['pid'],signal.SIGTERM)
        for _ in range(120):
            live=identity(old['pid'])
            if not live or live['start_ticks']!=old['start_ticks'] or live['state']=='Z':break
            time.sleep(5)
        else:raise RuntimeError('Old worker did not exit; checkpoint preserved, no forced termination')
    stopped=json.loads((OLD/'status.json').read_text());receipt=json.loads((OLD/'resume_receipt.json').read_text())
    assert stopped['stage']=='STOPPED_BY_REQUEST' and stopped['step']==receipt['step']
    assert receipt['optimizer_and_rng_included']
    status('COPYING_FINAL_WARM_START',base_step=stopped['step'])
    source=OLD/'latest_student.pt';digest=sha(source);target=OUT/'warm_start.pt'
    if target.exists():assert sha(target)==digest
    else:
        tmp=OUT/'warm_start.pt.tmp';shutil.copyfile(source,tmp)
        with tmp.open('rb') as f:os.fsync(f.fileno())
        assert sha(tmp)==digest;tmp.replace(target)
    state=torch.load(target,map_location='cpu',weights_only=False)
    assert state['step']==stopped['step'] and state['optimizer']['state'] and all(k in state for k in ['torch_rng','cuda_rng','numpy_rng','python_rng','cursors'])
    warm={'sha256':digest,'bytes':target.stat().st_size,'step':state['step'],'source_dataset_sha256':state['dataset_sha256'],
          'optimizer_and_rng_preserved':True,'optimizer_entries':len(state['optimizer']['state']),
          'optimizer_step_max':max(float(s['step']) for s in state['optimizer']['state'].values()),
          'source_cursors':state['cursors'],'source_rounds':state['rounds'],'old_process':old,'old_stop_status':stopped,
          'source':str(source),'new_data_cursors_start_fresh':True,'reason':'Explicit user request for a new streaming version from generated data, not base initialization.'}
    write('warm_start_receipt.json',warm);del state
    env=os.environ.copy();env['LD_LIBRARY_PATH']='/home/edabk/miniconda3/envs/.VLLM/lib/python3.10/site-packages/nvidia/cu13/lib'
    with (OUT/'worker.log').open('a') as log:
        proc=subprocess.Popen([str(PROJECT/'.venv/bin/python'),'-u','-m','turn_v2.factory_stream_train_v1','--batch-size','64','--eval-every','1000'],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(1);assert proc.poll() is None,'New worker exited during launch'
    launch={**identity(proc.pid),'module':'turn_v2.factory_stream_train_v1','base_step':warm['step'],'source_commit':722,'dataset_manifest_sha256':preflight['dataset_manifest_sha256']}
    write('launcher.json',launch);status('NEW_WORKER_LAUNCHED',new_process=launch)
    for _ in range(180):
        if (OUT/'error.json').exists():raise RuntimeError('New trainer error; inspect error.json, do not auto resume old campaign')
        live=identity(proc.pid)
        if not live or live['state']=='Z':raise RuntimeError('New trainer exited; inspect retained logs/checkpoints')
        if (OUT/'first_checkpoint_verification.json').exists():
            proof=json.loads((OUT/'first_checkpoint_verification.json').read_text());assert proof['branch_step']==100 and proof['first_checkpoint_torch_load_verified']
            status('TRAINING_VERIFIED',new_process=launch,first_checkpoint=proof);return
        time.sleep(5)
    status('NEW_WORKER_CONTINUES_MONITOR',new_process=launch)


if __name__=='__main__':
    with (OUT/'CUTOVER_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception as error:
            write('cutover_error.json',{'error':repr(error),'traceback':traceback.format_exc()});status('ERROR',error=repr(error));raise
