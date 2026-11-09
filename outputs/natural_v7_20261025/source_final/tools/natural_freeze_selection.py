"""Freeze the final candidate after the preserved compatibility recoveries."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.common import LABEL2ID,read_jsonl,file_hash
from turn_v2.natural_calibration import summarize_exact


def main():
    root=Path('/content/turn_detect_v2');evidence=root/'outputs/natural_v7_20261025'
    for name in ['natural_v7_text_20261025','natural_v8_lora_20261025_r3']:
        status=json.loads((root/'jobs'/name/'status.json').read_text());assert status['status']=='COMPLETED' and status['exit_code']==0
    calibration=json.loads((evidence/'exact_dev_calibration.json').read_text());candidates=calibration['candidates']
    assert calibration['status']=='EXACT_DEV_CALIBRATION_FROZEN_BEFORE_HELD' and not calibration['held_predictions_used']
    assert file_hash(root/'turn_v2/natural_calibration.py')==calibration['calibrator_sha256']
    rows=[{**r,'evaluation_slice':'natural'} for r in read_jsonl(root/'data_v6/pho_text_20261025/dev.jsonl')]
    rows.extend({**r,'evaluation_slice':'legacy_vi'} for r in read_jsonl(root/'data_v4/causal_20261025/dev.jsonl') if r['language']=='vi')
    for candidate in candidates:
        predictions=root/candidate['dev_predictions_path'];assert file_hash(predictions)==candidate['dev_predictions_sha256']
        assert file_hash(root/candidate['selection_report'])==candidate['selection_report_sha256']
        scores=np.load(predictions);assert np.array_equal(scores['labels'],np.array([LABEL2ID[r['label']] for r in rows]))
        assert summarize_exact(rows,scores['prob_EOT'])==candidate['dev']
        assert {lang:metric['samples'] for lang,metric in candidate['dev']['natural_languages'].items()}=={'vi':145,'en':79,'zh':69}
        assert candidate['dev']['slices']['vi:legacy_vi']['samples']==3600
        filename='adapter_model.safetensors' if candidate['kind']=='qwen_lora' else 'model.safetensors'
        expected=candidate.get('adapter_sha256',candidate.get('weights_sha256'))
        with (root/candidate['path']/filename).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected
    selected=max(candidates,key=lambda c:c['dev']['minimum_dev_slice_joint_score'])
    output=evidence/'selected_before_held.json';assert not output.exists()
    receipt={'status':'FROZEN_AFTER_DEV_BEFORE_HELD','selected':selected,'candidates':candidates,'held_classifier_scored':False,
        'selection_basis':'Same naturaldev293 + legacyVi3600, minimum slice accuracy/F1; no held/crawl classifier scores',
        'original_pipeline_failed_preserved':True,'completed_control_flow_receipt':'jobs/natural_freeze_selection_20261025/status.json',
        'recoveries':'r1 rejected unused torchao version; r2 rejected tokenizer return mapping; r3 uses process-only optional probe and explicit integer token IDs. First two failed before optimizer.',
        'data_manifest_sha256':hashlib.file_digest((root/'data_v6/pho_text_20261025/manifest.json').open('rb'),'sha256').hexdigest(),
        'exact_calibration_sha256':file_hash(evidence/'exact_dev_calibration.json'),
        'user_quality_objective_achieved':False,'accepted_serving_default_changed':False}
    output.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({'status':receipt['status'],'selected_kind':selected['kind'],'selected_path':selected['path'],
        'minimum_dev_score':selected['dev']['minimum_dev_slice_joint_score']}),flush=True)


if __name__=='__main__':main()
