"""Resume the existing uploaded-data campaign after its host reboot, without resetting state."""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .uploaded_vi_en_data_v4 import ROOT,OUT,sha,save

MODULE='turn_v2.multilingual_malbert_vi_en_uploaded_train_v4'
BOOT=Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def state(stage,**values):
    raw=Path('/proc/self/stat').read_text()
    save('recovery_status.json',dict(stage=stage,pid=os.getpid(),start_ticks=raw[raw.rfind(')')+2:].split()[19],
        boot_id=BOOT,at_utc=datetime.now(timezone.utc).isoformat(),**values))


def workers():
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if MODULE.encode() in args:found.append(int(p.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return found


def alive(launch):
    p=Path('/proc')/str(launch['pid'])
    try:
        raw=(p/'stat').read_text();cmd=(p/'cmdline').read_bytes().split(b'\0')
        return launch['boot_id']==BOOT and raw[raw.rfind(')')+2:].split()[19]==launch['start_ticks'] and MODULE.encode() in cmd
    except (FileNotFoundError,PermissionError,ProcessLookupError):return False


def mount_ready():
    rows=[r.split() for r in Path('/proc/mounts').read_text().splitlines() if r.split()[1]=='/mnt/hdd500']
    if not rows or rows[0][0]!='/dev/sda1' or 'rw' not in rows[0][3].split(','):return False
    uuid=subprocess.run(['findmnt','-n','-o','UUID','/mnt/hdd500'],text=True,capture_output=True,check=True).stdout.strip()
    assert uuid.upper()=='37622A5A7A5A4B57',('Wrong disk identity',uuid)
    return min(shutil.disk_usage(ROOT).free,shutil.disk_usage('/mnt/hdd500').free)>=13*2**30


def audit(path):
    import torch
    import numpy as np
    from .multilingual_malbert_vi_en_uploaded_train_v4 import fingerprint
    manifest=json.loads((OUT/'dataset_manifest.json').read_text())
    for n,h in manifest['files'].items():assert sha(OUT/n)==h,(n,'dataset hash mismatch')
    x=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    assert x['parameters']==11941122 and len(x['optimizer']['state'])==27
    assert x['fingerprint']==fingerprint(manifest)
    assert x['step']==x['parent_step']+x['branch_step']
    assert {int(s['step']) for s in x['optimizer']['state'].values()}=={x['step']}
    assert all(k in x for k in ['model','torch_rng','cuda_rng','numpy_rng','python_rng','sampler_rng','stream_state','order'])
    n=manifest['train']['streams']*2
    assert len(x['order'])==n and np.array_equal(np.sort(x['order']),np.arange(n))
    assert x['stream_state']['next']==x['cursor'] and 0<=x['cursor']<=n
    assert len(x['stream_state']['slots'])<=16
    assert all(0<=s['stream']<n//2 and s['k'] in (1,2) and 0<=s['start']<s['b'] for s in x['stream_state']['slots'])
    assert all(bool(torch.isfinite(v).all()) for v in x['model'].values() if v.is_floating_point())
    assert x['drive_replay_rows']==0 and {'daily','factory/gen500','factory/gen4000','asr/run2b'}<=x['uploaded_origin_exposures'].keys()
    return dict(step=x['step'],branch_step=x['branch_step'],epoch_zero_based=x['epoch'],cursor=x['cursor'],
        active_lanes=len(x['stream_state']['slots']),uploaded_rows_seen=x['uploaded_rows_seen'],
        uploaded_origin_exposures=x['uploaded_origin_exposures'],parameters=11941122,optimizer_entries=27,
        fingerprint=x['fingerprint'],checkpoint_sha256=sha(path),all_rng_sampler_conversation_lanes_verified=True,
        all_model_weights_finite=True)


def run():
    import torch
    torch.set_num_threads(2)
    launch=json.loads((OUT/'launcher.json').read_text())
    assert launch['boot_id']!=BOOT,'This recovery is scoped to an actual host reboot'
    assert not workers(),'An active worker already exists; never duplicate'
    assert not (OUT/'STOP_REQUESTED').exists(),'Honor an explicit user stop'
    folder=OUT/('recovery_'+BOOT.split('-')[0]);folder.mkdir(exist_ok=True)
    pin=folder/'full_resume_pin.pt';source=OUT/'runs/finetune/latest_resume.pt'
    if not pin.exists():os.link(source,pin)
    for n in ['launcher.json','status.json','runs/finetune/resume_receipt.json']:
        target=folder/('before_'+Path(n).name)
        if not target.exists():shutil.copyfile(OUT/n,target)
    proof=audit(pin);save(str(folder.relative_to(OUT)/'checkpoint_verification.json'),proof)
    state('WAITING_FOR_HDD_REMOUNT_AFTER_REBOOT',checkpoint=proof,previous_launcher=launch,
          required_mount='/dev/sda1 /mnt/hdd500 rw',required_uuid='37622A5A7A5A4B57',sudo_password_required=True)
    while not mount_ready():
        assert not workers(),'Another worker appeared; retain guard'
        if (OUT/'STOP_REQUESTED').exists():raise InterruptedError('User stop while waiting for storage')
        time.sleep(15)
    assert not workers() and sha(source)==proof['checkpoint_sha256']
    state('STARTING_FULL_STATE_RESUME_AFTER_REMOUNT',checkpoint_step=proof['step'])
    with (OUT/'worker.log').open('ab') as f:
        p=subprocess.Popen([sys.executable,'-u','-m',MODULE],cwd=ROOT,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(.5);proc=Path('/proc')/str(p.pid);raw=(proc/'stat').read_text()
    current=dict(pid=p.pid,start_ticks=raw[raw.rfind(')')+2:].split()[19],boot_id=BOOT,module=MODULE,
        resumed_from_checkpoint_step=proof['step'],recovery_folder=str(folder))
    assert alive(current);save('launcher.json',current)
    state('VERIFYING_RESUMED_TRAINING_AFTER_REBOOT',worker=current,checkpoint_step=proof['step'])
    while alive(current):
        # The worker atomically replaces this file; audit a pinned, stable snapshot.
        if source.stat().st_ino!=pin.stat().st_ino:
            verify_pin=folder/'first_resumed_full_checkpoint.pt'
            if not verify_pin.exists():os.link(source,verify_pin)
            after=audit(verify_pin)
            if after['step']>proof['step']:
                assert after['fingerprint']==proof['fingerprint'] and after['uploaded_rows_seen']>proof['uploaded_rows_seen']
                result=dict(complete=True,verified=True,boot_id=BOOT,worker=current,before=proof,after=after,
                    full_optimizer_rng_sampler_resume=True,base_model_reset=False,all_dataset_hashes_unchanged=True,
                    disk_mount_verified=True,model_or_threshold_selected_from_test=False)
                save('recovery_training_verification.json',result)
                state('RECOVERY_COMPLETE_TRAINING_VERIFIED',worker=current,step=after['step'],branch_step=after['branch_step'])
                return
        time.sleep(5)
    raise RuntimeError('Resumed worker exited before a new full checkpoint; do not launch again blindly')


if __name__=='__main__':
    with (OUT/'REBOOT_RECOVERY_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception as e:
            save('recovery_error.json',dict(error=repr(e),traceback=traceback.format_exc()))
            state('RECOVERY_ERROR',error=repr(e));raise
