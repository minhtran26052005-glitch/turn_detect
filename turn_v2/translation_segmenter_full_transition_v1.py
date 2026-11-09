"""Adopt user-authorized full epochs after current child finishes normally.

Only the old launcher is briefly frozen. Never reset/interrupt Colab or signal
its train child. If this controller fails before handoff, resume old launcher.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from . import translation_segmenter_full_data_v2 as data
from .translation_segmenter_full_campaign_v1 import live
from .translation_segmenter_v15c import atomic_json,file_sha

ROOT=data.ROOT
JOB=data.JOB
CONTROL=JOB/'transition_v1'


def read(p):return json.loads(Path(p).read_text())


def state(stage,**fields):
    atomic_json(CONTROL/'status.json',{'at_utc':data.now(),'stage':stage,**fields})


def replace_backup():
    folder=ROOT/'drive_backup_v1';a=read(folder/'launcher.json')
    if a['command'][-1]=='turn_v2.translation_segmenter_drive_backup_v2':
        assert live(a);return
    assert a['command'][-1]=='turn_v2.translation_segmenter_drive_backup_v1'
    while live(a):
        if read(folder/'status.json').get('stage')=='SYNCED':
            # Freeze and recheck before termination; never interrupt a copy pass.
            os.kill(a['pid'],signal.SIGSTOP)
            if read(folder/'status.json').get('stage')!='SYNCED':
                os.kill(a['pid'],signal.SIGCONT);time.sleep(2);continue
            atomic_json(CONTROL/'previous_backup_launcher.json',a)
            os.kill(a['pid'],signal.SIGTERM);os.kill(a['pid'],signal.SIGCONT)
            break
        state('WAITING_FOR_CURRENT_DRIVE_BACKUP_PASS_TO_COMMIT');time.sleep(5)
    while live(a):time.sleep(1)
    cmd=[sys.executable,'-u','-m','turn_v2.translation_segmenter_drive_backup_v2']
    with (folder/'worker_v2.log').open('a') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1'})
    record={'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],
      'command':cmd,'source_sha256':file_sha(Path(__file__).with_name('translation_segmenter_drive_backup_v2.py')),
      'drive_path':a['drive_path'],'log':str(folder/'worker_v2.log')}
    atomic_json(folder/'launcher.json',record);atomic_json(CONTROL/'new_backup_launcher.json',record)
    time.sleep(2)
    assert live(record),'New backup worker failed; inspect worker_v2.log'


def supervise():
    parent=read(ROOT/'launcher.json');assert live(parent),'Old campaign identity mismatch'
    assert 'turn_v2.translation_segmenter_v15d3' in parent['command']
    paused=False;handed_off=False
    atomic_json(CONTROL/'original_launcher.json',parent)
    try:
        assert (Path('/proc')/str(parent['pid'])/'stat').read_text().split()[2] not in ('T','t'),'Launcher already paused by another action'
        atomic_json(CONTROL/'pause_receipt.json',{'at_utc':data.now(),'parent':parent,
           'user_authorization':'Audit all vi/en/zh data and change training to complete epochs and DEV/loss convergence',
           'child_not_signalled':True})
        os.kill(parent['pid'],signal.SIGSTOP);paused=True
        # Inspect OS children AFTER the freeze; child.json can race with Popen.
        children=[]
        pids=(Path('/proc')/str(parent['pid'])/'task'/str(parent['pid'])/'children').read_text().split()
        for pid in pids:
            proc=Path('/proc')/pid
            argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]
            if not argv:continue  # exited zombie, to be reaped by old parent
            assert 'turn_v2.translation_segmenter_v15d2' in argv and '--language' in argv,argv
            children.append({'pid':int(pid),'start_ticks':(proc/'stat').read_text().split()[21],'command':argv})
        atomic_json(CONTROL/'children_after_freeze.json',children)
        while any(live(child) for child in children):
            state('WAITING_FOR_CURRENT_TRAIN_CHILD_NORMAL_EXIT',parent=parent,children=children)
            time.sleep(5)
        # Parent handler runs after CONT; completed child cannot be interrupted.
        assert live(parent);os.kill(parent['pid'],signal.SIGTERM);os.kill(parent['pid'],signal.SIGCONT);paused=False
        while live(parent):time.sleep(2)
        atomic_json(CONTROL/'old_campaign_finished_receipt.json',{'at_utc':data.now(),'parent':parent,'child_finished_normally':True})
        replace_backup()
        cmd=[sys.executable,'-u','-m','turn_v2.translation_segmenter_full_campaign_v1']
        record={'pid':os.getpid(),'start_ticks':(Path('/proc')/str(os.getpid())/'stat').read_text().split()[21],
           'command':cmd,'module':'turn_v2.translation_segmenter_full_campaign_v1',
           'source_sha256':file_sha(Path(__file__).with_name('translation_segmenter_full_campaign_v1.py')),
           'log':str(CONTROL/'worker.log'),'max_cycles':0,'transition_receipt':str(CONTROL/'old_campaign_finished_receipt.json')}
        atomic_json(ROOT/'launcher.json',record)
        state('HANDOFF_TO_FULL_EPOCH_CAMPAIGN',new_launcher=record);handed_off=True
        os.execv(sys.executable,cmd)
    finally:
        if paused and live(parent):os.kill(parent['pid'],signal.SIGCONT)
        if not handed_off and live(parent):
            atomic_json(CONTROL/'safety_resume_receipt.json',{'at_utc':data.now(),'parent':parent,'resumed':True})


def watchdog():
    while not (CONTROL/'old_campaign_finished_receipt.json').exists():
        if (CONTROL/'pause_receipt.json').exists() and not live(read(CONTROL/'launcher.json')):
            parent=read(CONTROL/'original_launcher.json')
            if live(parent):os.kill(parent['pid'],signal.SIGCONT)
            atomic_json(CONTROL/'watchdog_receipt.json',{'at_utc':data.now(),'old_parent_resumed_if_live':True});return
        time.sleep(5)


if __name__=='__main__':
    if '--watchdog' in sys.argv:watchdog()
    else:
        try:supervise()
        except Exception as exc:
            atomic_json(CONTROL/'error.json',{'at_utc':data.now(),'error':repr(exc),'traceback':traceback.format_exc()})
            state('FAILED_PRESERVED',error=repr(exc));raise
