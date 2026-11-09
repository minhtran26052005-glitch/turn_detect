"""Serialized continuation of already authorized text-only experiments.

Wait for the preserved ASR process. Never interrupt/restart a prior process or
silently repeat a submitted step. Human quality success is never inferred.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/content/turn_detect_v2')
EVIDENCE=ROOT/'outputs/natural_v7_20261025'


def record(stage,**extra):
    value={'stage':stage,'unix':time.time(),**extra}
    (EVIDENCE/'pipeline_progress.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(value),flush=True)


def run_job(name,module,timeout):
    assert not (ROOT/'jobs'/name).exists(), 'Inspect previous step; never resubmit a partially known job'
    record('RUNNING_STEP',job=name,module=module)
    result=subprocess.run([sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout',str(timeout),
        '--',sys.executable,'-u','-m',module],cwd=ROOT)
    assert result.returncode==0,(name,result.returncode)
    state=json.loads((ROOT/'jobs'/name/'status.json').read_text())
    assert state['status']=='COMPLETED' and state['exit_code']==0


def main():
    assert ROOT.is_dir();EVIDENCE.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    while True:
        state=json.loads((ROOT/'jobs/natural_v6_pho_repair_20261025/status.json').read_text())
        if state['status']=='COMPLETED':assert state['exit_code']==0;break
        assert state['status']=='RUNNING',state
        assert time.monotonic()-started<3600,'ASR wait timed out; original process not interrupted'
        record('WAIT_FOR_PRESERVED_ASR',observed_status=state['status'])
        time.sleep(15)
    run_job('natural_v7_compare_asr_20261025','turn_v2.natural_v7_compare_asr',600)
    run_job('natural_v7_text_20261025','turn_v2.natural_v7_train',3600)
    teacher=json.loads((ROOT/'runs/natural_v7_text_20261025/report.json').read_text())
    candidates=[{'kind':'mdeberta','path':teacher['best_path'],'weights_sha256':teacher['checkpoint_sha256'],
        'dev':teacher['best_dev'],'selection_report':'runs/natural_v7_text_20261025/report.json'}]
    if not teacher['best_dev']['development_all_slices_over_90']:
        run_job('natural_v8_lora_20261025','turn_v2.natural_v8_lora',5400)
        lora=json.loads((ROOT/'runs/natural_v8_lora_20261025/report.json').read_text())
        if lora['best_path']:
            candidates.append({'kind':'qwen_lora','path':lora['best_path'],'adapter_sha256':lora['adapter_sha256'],
                'dev':lora['best_dev'],'selection_report':'runs/natural_v8_lora_20261025/report.json',
                'pinned_base_receipt':'outputs/natural_v5_20261025/qwen_download.json'})
    best=max(candidates,key=lambda r:r['dev']['minimum_dev_slice_joint_score'])
    # This freeze is created before any held prediction is calculated. Any
    # later test is a diagnostic if dev remains below the requested target.
    selected={'status':'FROZEN_AFTER_DEV_BEFORE_HELD','selected':best,'candidates':candidates,'held_classifier_scored':False,
        'selection_basis':'Same natural dev293 + legacyVi3600, minimum slice accuracy/F1; no held/crawl scores',
        'user_quality_objective_achieved':False,'accepted_serving_default_changed':False,
        'data_manifest_sha256':hashlib.file_digest((ROOT/'data_v6/pho_text_20261025/manifest.json').open('rb'),'sha256').hexdigest(),
        'elapsed_seconds':time.monotonic()-started}
    (EVIDENCE/'selected_before_held.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    record('DEV_CAMPAIGNS_FINISHED_AWAITING_FINAL_DIAGNOSTIC',selected_kind=best['kind'],selected_path=best['path'],
        best_dev_minimum=best['dev']['minimum_dev_slice_joint_score'],quality_target_claimed=False)


if __name__=='__main__':main()
