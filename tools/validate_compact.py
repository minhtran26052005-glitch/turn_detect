"""Verify the promoted files, scored dev metrics, and actual execution receipts."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import onnx

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.common import file_hash, write_json
from turn_v2.metrics import classification_metrics


def main():
    root = Path('outputs/compression_20261024')
    model = Path('models/turn_v2_compact_20261024/onnx')
    manifest = json.loads((model.parent/'model_manifest.json').read_text())
    report = json.loads((root/'weight_only_full_dev_cached/report.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    data_manifest = json.loads(Path('data_v2/frozen/manifest.json').read_text())
    contract = json.loads((model/'turn_detector_config.json').read_text())
    checks = {}
    def check(name, condition):
        checks[name] = bool(condition)
    for filename, identity in manifest['files'].items():
        check(filename+'_bytes', (model/filename).stat().st_size==identity['bytes'])
        check(filename+'_sha256', file_hash(model/filename)==identity['sha256'])
    total = sum(p.stat().st_size for p in model.rglob('*') if p.is_file())
    check('whole_deployment_within_budget', total<=protocol['size_budget_bytes'])
    check('deployment_size_matches_manifest',total==manifest['deployment_bytes'])
    check('evaluated_weights_match',file_hash(model/'model.onnx')==report['model_sha256'])
    check('teacher_preserved',file_hash(Path(protocol['reference']))==protocol['reference_sha256'])
    for split in ('train','dev','holdout'):
        check('frozen_'+split+'_preserved',file_hash(Path('data_v2/frozen')/(split+'.jsonl'))==data_manifest['split_sha256'][split])
    check('consumed_holdout_marker_preserved',file_hash(Path('data_v2/frozen/holdout_consumed.json'))==protocol['holdout_marker_sha256'])
    check('holdout_not_evaluated',not report['holdout_evaluated'] and not manifest['holdout_evaluated'])
    predictions = np.load(root/'weight_only_full_dev_cached/dev_predictions.npz')
    reference = np.load('outputs/training_trial_20261024/dev_predictions.npz')
    check('full_dev_scored',len(predictions['labels'])==data_manifest['splits']['dev']['rows']==report['samples'])
    check('full_dev_order',np.array_equal(predictions['indices'],np.arange(report['samples'])))
    check('dev_reference_labels_match',np.array_equal(predictions['labels'],reference['labels']))
    metrics = classification_metrics(predictions['labels'],predictions['prob_EOT'],contract['threshold'])
    for key in ('accuracy','eot_f1','eot_precision','eot_recall','false_eot','missed_eot'):
        check('recomputed_'+key,metrics[key]==manifest['dev_metrics'][key])
    check('accuracy_target',metrics['accuracy']>=.9)
    check('eot_f1_target',metrics['eot_f1']>=.9)
    check('threshold_selected_on_dev',contract['threshold']==report['dev_selected_threshold']['threshold'])
    check('runtime_options_match_evaluation',contract['onnx_session_options']==report['cpu']['session_options'])
    parity = json.loads((root/'runtime_folding_parity.json').read_text())
    check('runtime_folding_parity',parity['passed'] and parity['model_sha256']==report['model_sha256'])
    check('windows_colab_probability_parity',json.loads((root/'windows_colab_parity.json').read_text())['passed'])
    onnx.checker.check_model(str(model/'model.onnx'))
    check('onnx_checker',True)
    for name in ('colab_reconnected','colab_final_verification'):
        result = json.loads((root/(name+'.json')).read_text())
        check(name,result['status']=='ok')
    nb = json.loads((root/'notebook_verification.json').read_text())
    check('actual_colab_notebook_cell_passed',nb['passed'] and nb['selected_tags']==['compression_result'])
    check('windows_unit_tests',json.loads((root/'unit_tests_windows.json').read_text())['passed']==30)
    colab_tests = json.loads((root/'colab_unit_tests_final_source.json').read_text())
    check('colab_unit_tests_final_source',colab_tests['status']=='ok')
    frozen_source = json.loads((root/'source_weight_only_cached/source_manifest.json').read_text())
    for filename,identity in frozen_source['source_sha256'].items():
        if filename.startswith('turn_v2'):
            check('executed_source_'+Path(filename).name,file_hash(Path(filename))==identity)
    crawl = json.loads((root/'crawl_smoke/report.json').read_text())
    check('live_crawl_predictions',crawl['status']=='INFERENCE_COMPLETED' and crawl['predictions']>0 and not crawl['holdout_evaluated'])
    bench = json.loads((root/'benchmark.json').read_text())
    check('benchmark_same_weights',bench['results']['compact']['model_sha256']==report['model_sha256'])
    check('benchmark_matched_threads',bench['threads']==4 and bench['batch_size']==1)
    check('readable_review_exists',(root/'review.html').is_file())
    result = {'verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'PASS' if all(checks.values()) else 'FAIL','passed':sum(checks.values()),
        'total':len(checks),'checks':checks,'model_sha256':report['model_sha256'],
        'model_bytes':report['model_bytes'],'deployment_bytes':total,'dev_metrics':metrics,
        'quality_scope':'Dev selection only; no fresh independent holdout claim'}
    write_json(root/'completion_validation.json',result)
    print(json.dumps(result,indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
