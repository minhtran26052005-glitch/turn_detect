"""Post-training dev calibration with full provenance, before held inference."""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.common import LABEL2ID,file_hash,read_jsonl,write_json
from turn_v2.natural_calibration import summarize_exact

ROOT=Path('outputs/natural_v7_20261025')


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    output=ROOT/'exact_dev_calibration.json';assert not output.exists(),'Keep the already frozen calibration'
    run=ROOT/'runs/natural_v7_text_20261025';teacher=load(run/'report.json');initial=load(run/'initial_dev.json');identity=load(run/'identity.json')
    base_identities={item['path']:item['sha256'] for item in identity['candidate_baselines']}
    v8root=Path('outputs/natural_v8_20261025');lora_run=v8root/'runs/natural_v8_lora_20261025_r3';lora=load(lora_run/'report.json')
    rows=[{**r,'evaluation_slice':'natural'} for r in read_jsonl(Path('data_v6/pho_text_20261025/dev.jsonl'))]
    rows.extend({**r,'evaluation_slice':'legacy_vi'} for r in read_jsonl(Path('data_v4/causal_20261025/dev.jsonl')) if r['language']=='vi')
    candidates=[];seen=set()
    def add(kind,path,sha,prediction_path,local_prediction,source_report,local_report,raw_dev):
        if (kind,path) in seen:return
        seen.add((kind,path));prediction=np.load(local_prediction)
        assert np.array_equal(prediction['labels'],np.array([LABEL2ID[r['label']] for r in rows]))
        calibrated=summarize_exact(rows,prediction['prob_EOT'])
        candidate={'kind':kind,'path':path,'adapter_sha256' if kind=='qwen_lora' else 'weights_sha256':sha,
            'dev':calibrated,'dev_predictions_path':Path(prediction_path).as_posix(),'dev_predictions_sha256':file_hash(local_prediction),
            'selection_report':source_report,'selection_report_sha256':file_hash(local_report),
            'original_grid_minimum_dev_score':raw_dev['minimum_dev_slice_joint_score']}
        if kind=='qwen_lora':candidate['pinned_base_receipt']='outputs/natural_v5_20261025/qwen_download.json'
        candidates.append(candidate)
    for i,item in enumerate(initial['candidates']):
        local=run/f'initial_baseline_{i:02d}.npz'
        add('mdeberta',item['baseline_path'],base_identities[item['baseline_path']],
            Path('runs/natural_v7_text_20261025')/local.name,local,'runs/natural_v7_text_20261025/initial_dev.json',run/'initial_dev.json',item)
    add('mdeberta',teacher['best_path'],teacher['checkpoint_sha256'],teacher['best_dev_predictions_path'],ROOT/teacher['best_dev_predictions_path'],
        'runs/natural_v7_text_20261025/report.json',run/'report.json',teacher['best_dev'])
    if lora['best_path']:
        add('qwen_lora',lora['best_path'],lora['adapter_sha256'],'runs/natural_v8_lora_20261025_r3/best_dev_predictions.npz',
            lora_run/'best_dev_predictions.npz','runs/natural_v8_lora_20261025_r3/report.json',lora_run/'report.json',lora['best_dev'])
    selected=max(candidates,key=lambda c:c['dev']['minimum_dev_slice_joint_score'])
    result={'status':'EXACT_DEV_CALIBRATION_FROZEN_BEFORE_HELD','candidates':candidates,'selected_kind':selected['kind'],'selected_path':selected['path'],
        'calibrator_sha256':file_hash(Path('turn_v2/natural_calibration.py')),'source_sha256':file_hash(Path(__file__)),
        'data_manifest_sha256':file_hash(Path('data_v6/pho_text_20261025/manifest.json')),
        'natural_dev_sha256':file_hash(Path('data_v6/pho_text_20261025/dev.jsonl')),'legacy_dev_sha256':file_hash(Path('data_v4/causal_20261025/dev.jsonl')),
        'development_cases':len(rows),'held_predictions_used':False,'weights_changed_by_calibration':False,
        'note':'Weights/checkpoints retained by their training dev policies; only thresholds and final candidate comparison now use exact development boundaries'}
    write_json(output,result)
    print(json.dumps({'selected_kind':selected['kind'],'selected_path':selected['path'],
        'candidates':[{'kind':c['kind'],'path':c['path'],'grid_score':c['original_grid_minimum_dev_score'],'exact_score':c['dev']['minimum_dev_slice_joint_score'],
        'natural':c['dev']['natural_languages']} for c in candidates]}),flush=True)


if __name__=='__main__':main()
