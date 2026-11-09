"""User-authorized GPU priority: freeze only the campaign launcher, not child.

The current train child finishes normally. Evaluation/control/finalizer continue
on their existing frozen state. Finally resume the exact launcher identity.
No kernel reset/interrupt or campaign STOP_REQUESTED marker is used.
"""
import argparse
import os
from pathlib import Path
import signal
import time
import traceback

from .translation_segmenter_v15d_test_v2 import atomic_json,now,process_matches,read_json

ROOT=Path('outputs/translation_segmenter_v15d_20261026')
JOB=ROOT/'diagnostic_test_c75_v1'
CONTROL=JOB/'priority_window_v1'


def resumed(record,reason):
    if process_matches(record):os.kill(record['pid'],signal.SIGCONT)
    atomic_json(CONTROL/'resume_receipt.json',{'at_utc':now(),'reason':reason,'parent':record,'identity_still_matches':process_matches(record)})


def status(stage,**fields):
    atomic_json(CONTROL/'status.json',{'at_utc':now(),'stage':stage,**fields})


def supervise():
    parent=read_json(ROOT/'launcher.json');child=read_json(ROOT/'child.json')
    if not process_matches(parent):raise RuntimeError('Campaign parent identity unavailable')
    if not process_matches(child) or '--phase' not in child['command'] or child['command'][child['command'].index('--phase')+1]!='train':
        raise RuntimeError('Current train child already finished; do not apply this mid-train pause protocol blindly')
    if (Path('/proc')/str(parent['pid'])/'stat').read_text().split()[2] in ('T','t'):
        raise RuntimeError('Campaign launcher was already stopped by another action')
    atomic_json(CONTROL/'pause_receipt.json',{'at_utc':now(),'user_authorization':'Prioritize test after the current train, then automatically continue campaign',
        'parent':parent,'current_train_child_untouched':child,'pause_scope':'single launcher PID only; current child completes normally'})
    reason='exception_safety_resume'
    def terminate(*_):raise SystemExit('Priority supervisor terminated; resume launcher in finally')
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    try:
        os.kill(parent['pid'],signal.SIGSTOP)
        while process_matches(child):
            status('WAITING_FOR_CURRENT_TRAIN_CHILD_TO_FINISH',parent=parent,child=child)
            time.sleep(5)
        status('EXISTING_TEST_AND_CONTROLS_HAVE_GPU_PRIORITY',parent=parent)
        deadline=time.monotonic()+7200
        while True:
            failures=[JOB/'continuation_v2_error.json',JOB/'analysis_v2/error.json',JOB/'finalizer_v1/error.json',JOB/'reserve_audit_v1/error.json']
            failure=next((p for p in failures if p.exists()),None)
            if failure:
                reason='upstream_failure_resume_campaign';raise RuntimeError('Preserved evaluation error: '+str(failure))
            final=JOB/'finalizer_v1/status.json';reserve=JOB/'reserve_audit_v1/status.json'
            done=final.exists() and reserve.exists() and read_json(final).get('stage')=='COMPLETE' and read_json(reserve).get('stage')=='COMPLETE'
            if done:
                reason='test_controls_reports_and_reserve_audit_complete';break
            if not process_matches(parent):raise RuntimeError('Campaign launcher identity disappeared')
            if time.monotonic()>deadline:
                reason='priority_lease_elapsed_resume_campaign_without_killing_evaluation';break
            status('EXISTING_TEST_AND_CONTROLS_HAVE_GPU_PRIORITY',parent=parent)
            time.sleep(5)
    finally:
        resumed(parent,reason)
    status('CAMPAIGN_RESUMED',reason=reason,parent=parent)


def watchdog():
    record=read_json(CONTROL/'launcher.json');started=time.monotonic()
    while not (CONTROL/'resume_receipt.json').exists():
        receipt=CONTROL/'pause_receipt.json'
        if receipt.exists() and (not process_matches(record) or time.monotonic()-started>9000):
            parent=read_json(receipt)['parent'];resumed(parent,'independent_watchdog_resume_after_supervisor_exit_or_lease')
            atomic_json(CONTROL/'watchdog_receipt.json',{'at_utc':now(),'resumed_parent':parent});return
        time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watchdog',action='store_true');args=parser.parse_args()
    try:
        watchdog() if args.watchdog else supervise()
    except Exception as exc:
        atomic_json(CONTROL/'error.json',{'at_utc':now(),'error':repr(exc),'traceback':traceback.format_exc()})
        status('FAILED_PRESERVED_CAMPAIGN_RESUME_ATTEMPTED',error=repr(exc));raise
