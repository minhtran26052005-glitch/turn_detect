"""Finalize selection after isolated LoRA compatibility recovery, no reruns."""
import hashlib
import json
from pathlib import Path
import time


def main():
    root=Path('/content/turn_detect_v2');evidence=root/'outputs/natural_v7_20261025'
    assert json.loads((root/'jobs/natural_campaign_pipeline_20261025/status.json').read_text())['status']=='FAILED'
    assert json.loads((root/'jobs/natural_v8_lora_20261025/status.json').read_text())['status']=='FAILED'
    assert json.loads((root/'jobs/natural_v7_text_20261025/status.json').read_text())['status']=='COMPLETED'
    started=time.monotonic()
    while True:
        state=json.loads((root/'jobs/natural_v8_lora_20261025_r2/status.json').read_text())
        if state['status']=='COMPLETED':assert state['exit_code']==0;break
        assert state['status']=='RUNNING',state
        assert time.monotonic()-started<5400
        (evidence/'recovery_progress.json').write_text(json.dumps({'stage':'WAIT_FOR_ISOLATED_LORA_R2','unix':time.time()}))
        time.sleep(15)
    teacher=json.loads((root/'runs/natural_v7_text_20261025/report.json').read_text())
    lora=json.loads((root/'runs/natural_v8_lora_20261025_r2/report.json').read_text())
    candidates=[{'kind':'mdeberta','path':teacher['best_path'],'weights_sha256':teacher['checkpoint_sha256'],
        'dev':teacher['best_dev'],'selection_report':'runs/natural_v7_text_20261025/report.json'}]
    if lora['best_path']:
        candidates.append({'kind':'qwen_lora','path':lora['best_path'],'adapter_sha256':lora['adapter_sha256'],'dev':lora['best_dev'],
            'selection_report':'runs/natural_v8_lora_20261025_r2/report.json','pinned_base_receipt':'outputs/natural_v5_20261025/qwen_download.json',
            'compatibility_receipt':'outputs/natural_v8_20261025/r2_compatibility.json'})
    best=max(candidates,key=lambda c:c['dev']['minimum_dev_slice_joint_score'])
    output=evidence/'selected_before_held.json';assert not output.exists()
    selected={'status':'FROZEN_AFTER_DEV_BEFORE_HELD','selected':best,'candidates':candidates,'held_classifier_scored':False,
        'selection_basis':'Same naturaldev293 + legacyVi3600, minimum slice accuracy/F1; no held/crawl scores',
        'user_quality_objective_achieved':False,'accepted_serving_default_changed':False,
        'original_pipeline_failed_preserved':True,'completed_control_flow_receipt':'jobs/natural_campaign_recovery_20261025/status.json',
        'data_manifest_sha256':hashlib.file_digest((root/'data_v6/pho_text_20261025/manifest.json').open('rb'),'sha256').hexdigest()}
    output.write_text(json.dumps(selected,indent=2),encoding='utf-8')
    result={'stage':'RECOVERED_DEV_CAMPAIGNS_FINISHED_AWAITING_FINAL_DIAGNOSTIC','selected_kind':best['kind'],'selected_path':best['path'],
        'best_dev_minimum':best['dev']['minimum_dev_slice_joint_score'],'held_classifier_scored':False,'unix':time.time()}
    (evidence/'recovery_progress.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
