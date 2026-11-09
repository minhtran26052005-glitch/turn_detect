"""Verify trained receipts, 18 dev evaluations, portable model, Colab and package."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import onnx

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.common import file_hash,write_json
from turn_v2.metrics import classification_metrics


def main():
    root=Path('outputs/distillation_20261024');model=Path('models/turn_v2_distilled_min_20261024');folder=model/'onnx'
    manifest=json.loads((model/'model_manifest.json').read_text(encoding='utf-8'));selection=json.loads((root/'selection.json').read_text(encoding='utf-8'));protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'));data=json.loads(Path('data_v2/frozen/manifest.json').read_text(encoding='utf-8'))
    checks={}
    def check(name,value):checks[name]=bool(value)
    for name,item in manifest['files'].items():
        check(name+'_bytes',(folder/name).stat().st_size==item['bytes']);check(name+'_hash',file_hash(folder/name)==item['sha256'])
    check('entire_deployment_size',sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())==manifest['deployment_bytes'])
    check('under_one_percent_of_teacher_package',manifest['deployment_bytes']<protocol['teacher_deployment_bytes']/100)
    for split in ['train','dev','holdout']:check(split+'_preserved',file_hash(Path('data_v2/frozen')/(split+'.jsonl'))==data['split_sha256'][split])
    check('consumed_marker_preserved',file_hash(Path('data_v2/frozen/holdout_consumed.json'))==protocol['holdout_marker_sha256'])
    control=Path(protocol['teacher']);control_manifest=json.loads((control.parent/'model_manifest.json').read_text(encoding='utf-8'))
    check('teacher_files_preserved',all(file_hash(control/n)==i['sha256'] for n,i in control_manifest['files'].items()))
    check('teacher_reconstruction_parity',json.loads((root/'cache/teacher_parity.json').read_text(encoding='utf-8'))['passed'])
    cache=json.loads((root/'cache/cache_report.json').read_text(encoding='utf-8'))
    check('all_train_teacher_targets',cache['status']=='TRAIN_TARGETS_COMPLETE' and cache['train_samples']==315174 and cache['train_sha256']==data['split_sha256']['train'])
    check('train_vocabulary_segmentation_unchanged',json.loads((root/'vocabulary/vocabulary_report.json').read_text(encoding='utf-8'))['changed_train_segmentation']==0)
    check('18_measured_candidates',len(selection['candidates'])==18)
    check('six_completed_training_trials',len(list((root/'jobs').glob('*/status.json')))==6 and all(json.loads(p.read_text(encoding='utf-8'))['status']=='COMPLETED' and json.loads(p.read_text(encoding='utf-8'))['exit_code']==0 for p in (root/'jobs').glob('*/status.json')))
    preserved_hashes={file_hash(p) for directory in root.glob('source*') for p in directory.rglob('*.py')}
    check('executed_training_sources_preserved',all(h in preserved_hashes for p in (root/'runs').glob('*/run_identity.json') for h in json.loads(p.read_text(encoding='utf-8'))['source_sha256'].values()))
    check('all_variants_full_dev_only',all(json.loads(Path(c['report']).read_text(encoding='utf-8'))['samples']==17833 and not json.loads(Path(c['report']).read_text(encoding='utf-8'))['holdout_evaluated'] for c in selection['candidates']))
    passing=[c for c in selection['candidates'] if c['passes']]
    check('smallest_tested_passing_core',selection['selected']['model_core_bytes']==min(c['model_core_bytes'] for c in passing))
    report=json.loads(Path(manifest['selected_report']).read_text(encoding='utf-8'));predictions=np.load(Path(manifest['selected_report']).parent/'dev_predictions.npz')
    control_predictions=np.load('outputs/compression_20261024/weight_only_full_dev_cached/dev_predictions.npz')
    check('full_dev_prediction_order',np.array_equal(predictions['indices'],np.arange(17833)))
    check('same_reference_labels',np.array_equal(predictions['labels'],control_predictions['labels']))
    contract=json.loads((folder/'turn_detector_config.json').read_text(encoding='utf-8'));metrics=classification_metrics(predictions['labels'],predictions['prob_EOT'],contract['threshold'])
    for key in ['accuracy','eot_f1','eot_precision','eot_recall','false_eot','missed_eot']:check('recomputed_'+key,metrics[key]==manifest['dev_metrics'][key])
    check('both_90_targets',metrics['accuracy']>=.9 and metrics['eot_f1']>=.9)
    check('same_evaluated_model',file_hash(folder/'model.onnx')==report['model_sha256'])
    onnx.checker.check_model(str(folder/'model.onnx'));check('onnx_checker',True)
    portable=json.loads((root/'portable_runtime_parity.json').read_text(encoding='utf-8'));check('portable_full_dev_inputs_and_132_predictions',portable['passed'] and portable['full_dev_input_encoding_rows']==17833 and portable['changed_encoded_rows']==0)
    smoke=json.loads((root/'standalone_smoke.json').read_text(encoding='utf-8'));check('standalone_no_torch_or_transformers',smoke['passed'] and not smoke['torch_imported'] and not smoke['transformers_imported'])
    check('real_unicode_smoke','hà nội' in smoke['demos'][0]['text'])
    nb_receipt=json.loads((root/'notebook_verification.json').read_text(encoding='utf-8'));check('actual_colab_notebook_cell',nb_receipt['passed'] and nb_receipt['selected_tags']==['distillation_result'])
    notebook=json.loads(Path('a.ipynb').read_text(encoding='utf-8'));cell=next(c for c in notebook['cells'] if c['cell_type']=='code' and 'distillation_result' in c.get('metadata',{}).get('tags',[]))
    remote=json.loads(''.join(o['text'] for o in cell['outputs'] if o['output_type']=='stream'))
    deltas=[abs(a['prob_EOT']-b['prob_EOT']) for a,b in zip(smoke['demos'],remote['demos'],strict=True)]
    check('windows_colab_demo_parity',max(deltas)<1e-6 and all(a['text']==b['text'] and a['prediction']==b['prediction'] for a,b in zip(smoke['demos'],remote['demos'],strict=True)))
    write_json(root/'windows_colab_parity.json',{'passed':checks['windows_colab_demo_parity'],'max_probability_delta':max(deltas),'samples':len(deltas)})
    check('colab_model_hashes',json.loads((root/'colab_model_sync.json').read_text(encoding='utf-8'))['status']=='ok' and remote['hashes_verified'])
    check('33_tests_passed',json.loads((root/'unit_tests.json').read_text(encoding='utf-8'))['passed']==33)
    benchmark=json.loads((root/'benchmark.json').read_text(encoding='utf-8'));check('benchmark_matched_inputs_threads',benchmark['threads']==2 and benchmark['batch_size']==1 and benchmark['results']['compact']['model_sha256']==manifest['files']['model.onnx']['sha256'])
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            if tag=='a':self.links.extend(v for k,v in attrs if k=='href')
    links=Links();links.feed((root/'review.html').read_text(encoding='utf-8'));check('review_links_exist',all((root/p).resolve().is_file() for p in links.links))
    check('model_card_exists',(model/'README.md').is_file())
    archive=model.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(model.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(model.parent).as_posix())
    with zipfile.ZipFile(archive) as z:
        check('zip_model_hashes',all(hashlib.sha256(z.read(model.name+'/onnx/'+n)).hexdigest()==i['sha256'] for n,i in manifest['files'].items()))
    result={'status':'PASS' if all(checks.values()) else 'FAIL','verified_at_utc':datetime.now(timezone.utc).isoformat(),'passed':sum(checks.values()),'total':len(checks),'checks':checks,'model_sha256':manifest['files']['model.onnx']['sha256'],'deployment_bytes':manifest['deployment_bytes'],'zip':{'path':str(archive),'bytes':archive.stat().st_size,'sha256':file_hash(archive)},'dev':metrics,'scope':manifest['quality_scope']}
    write_json(root/'completion_validation.json',result);print(json.dumps(result,indent=2))
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':main()
