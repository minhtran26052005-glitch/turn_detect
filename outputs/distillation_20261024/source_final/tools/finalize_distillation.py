"""Promote the smallest tested full-dev passing student and its portable runner."""
from __future__ import annotations
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.common import file_hash,write_json


def main():
    root=Path('outputs/distillation_20261024')
    protocol=json.loads((root/'protocol.json').read_text())
    data=json.loads(Path('data_v2/frozen/manifest.json').read_text())
    control=Path(protocol['teacher'])
    control_manifest=json.loads((control.parent/'model_manifest.json').read_text())
    for name,item in control_manifest['files'].items():
        assert file_hash(control/name)==item['sha256']
    assert file_hash(Path('data_v2/frozen/holdout_consumed.json'))==protocol['holdout_marker_sha256']
    for split in ['train','dev']:assert file_hash(Path('data_v2/frozen')/(split+'.jsonl'))==data['split_sha256'][split]
    candidates=[]
    runs=list((root/'runs').glob('*/training_report.json'))
    assert len(runs)==6,'All six authorized training/refinement trials must be collected'
    for training_path in runs:
        training=json.loads(training_path.read_text())
        for precision in ['int8','fp16','fp32']:
            report_path=training_path.parent/('quality_'+precision)/'report.json'
            report=json.loads(report_path.read_text())
            assert report['samples']==data['splits']['dev']['rows']==17833
            source=Path(report['model'])
            assert file_hash(source/'model.onnx')==report['model_sha256']
            assert report['holdout_marker_unchanged'] and not report['holdout_evaluated']
            assert report['dev_sha256']==data['split_sha256']['dev']
            metrics=report['dev_selected_threshold']
            candidates.append({'run':training_path.parent.name,'precision':precision,'path':str(source),
                'report':str(report_path.as_posix()),'model_bytes':report['model_bytes'],
                'model_core_bytes':sum(p.stat().st_size for p in source.iterdir() if p.is_file()),
                'dev':metrics,'parameters':training['parameters_deployment'],
                'passes':metrics['accuracy']>=.9 and metrics['eot_f1']>=.9})
    passing=sorted([r for r in candidates if r['passes']],key=lambda r:(r['model_core_bytes'],-r['dev']['joint_score']))
    if not passing:raise ValueError('No student reached both targets')
    selected=passing[0];source=Path(selected['path']);destination=Path('models/turn_v2_distilled_min_20261024/onnx')
    destination.mkdir(parents=True,exist_ok=False)
    for name in ['model.onnx','config.json','tokenizer.json','tokenizer_config.json','turn_detector_config.json']:shutil.copy2(source/name,destination/name)
    contract=json.loads((destination/'turn_detector_config.json').read_text())
    assert contract['threshold']==selected['dev']['threshold']
    contract.pop('compression_dev_report',None)
    contract.update({'quality_status':'SMALLEST_TESTED_STUDENT_FULL_DEV_TARGET_MET','quality_scope':'Full frozen dev selection; no fresh holdout or real-dialogue guarantee','distillation_run':selected['run'],'selected_dev_report':selected['report']})
    write_json(destination/'turn_detector_config.json',contract)
    shutil.copy2('turn_v2/tiny_runtime.py',destination/'run_model.py')
    (destination/'requirements.txt').write_text('numpy\nonnxruntime==1.23.2\ntokenizers=='+importlib.metadata.version('tokenizers')+'\n')
    files={p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(destination.iterdir()) if p.is_file()}
    total=sum(v['bytes'] for v in files.values())
    manifest={'status':'SMALLEST_TESTED_STUDENT_FULL_DEV_TARGET_MET','model_dir':str(destination.resolve()),
        'files':files,'onnx_bytes':files['model.onnx']['bytes'],'deployment_bytes':total,
        'deployment_includes':'ONNX, tokenizer/configuration, portable Python runner, requirements; installed runtime libraries excluded',
        'teacher_onnx_bytes':(control/'model.onnx').stat().st_size,'teacher_deployment_bytes':control_manifest['deployment_bytes'],
        'deployment_reduction_factor':control_manifest['deployment_bytes']/total,'source_run':selected['run'],
        'parameters_deployment':selected['parameters'],'storage_precision':selected['precision'],
        'dev_metrics':selected['dev'],'quality_scope':contract['quality_scope'],'holdout_evaluated':False,
        'selected_report':selected['report'],'candidates_tested':len(candidates),'training_trials':len(runs)}
    write_json(destination.parent/'model_manifest.json',manifest)
    write_json(root/'selection.json',{'criterion':'Smallest actual five-file model core among all full-dev passing variants; identical portable runner overhead added after selection','candidates':candidates,'selected':selected,'manifest':manifest})
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
