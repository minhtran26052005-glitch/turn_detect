"""Recompute all development claims and verify temporal/source controls."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.common import LABEL2ID,file_hash,input_id,read_jsonl,write_json
from turn_v2.metrics import classification_metrics

ROOT=Path('outputs/natural_v7_20261025');CHECKS=[]


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def check(name,value):CHECKS.append({'check':name,'passed':bool(value)})


def compare(expected,actual,name):
    for key in ['samples','accuracy','eot_f1','false_eot','missed_eot','false_eot_rate','confusion_matrix_NEOT_EOT']:
        check(name+':'+key,math.isclose(expected[key],actual[key],abs_tol=1e-12,rel_tol=0) if isinstance(expected[key],float) else expected[key]==actual[key])


def validate_slices(rows,probs,report,name):
    labels=np.array([LABEL2ID[r['label']] for r in rows])
    check(name+':probability_alignment',len(probs)==len(rows) and np.isfinite(probs).all())
    for key,expected in report['slices'].items():
        lang,source=key.split(':',1)
        mask=np.array([r['language']==lang and r['evaluation_slice']==source for r in rows])
        actual=classification_metrics(labels[mask],probs[mask],report['thresholds'][lang])
        compare(expected,actual,name+':'+key)


def main():
    baseline=load(Path('outputs/stream_v4_20261025/protocol.json'))
    for filename,digest in baseline['initial_controls'].items():check('original_control:'+filename,file_hash(Path(filename))==digest)
    data_names=['data_v5/canonical_20261025','data_v6/turn_context_20261025','data_v6/pho_text_20261025']
    datasets={name:{split:list(read_jsonl(Path(name)/(split+'.jsonl'))) for split in ['train','dev','sealed']} for name in data_names}
    for name,splits in datasets.items():
        manifest=load(Path(name)/'manifest.json')
        for split in splits:
            check('data_hash:'+name+':'+split,file_hash(Path(name)/(split+'.jsonl'))==manifest['files'][split]['sha256'])
        groups={split:{r['session_group'] for r in rows} for split,rows in splits.items()}
        for a,b in [('train','dev'),('train','sealed'),('dev','sealed')]:check('whole_call_isolation:'+name+':'+a+':'+b,not groups[a].intersection(groups[b]))
    base=datasets[data_names[0]]
    for name in data_names[1:]:
        for split,rows in datasets[name].items():
            before={r['case_id']:r for r in base[split]};after={r['case_id']:r for r in rows}
            check('same_case_ids:'+name+':'+split,before.keys()==after.keys())
            check('same_reference_labels_and_call_ids:'+name+':'+split,all(r['label']==before[k]['label'] and r['session_group']==before[k]['session_group'] for k,r in after.items()))
            check('causal_current_end:'+name+':'+split,all(abs(r['v6_current_audio_end']-r['absolute_decision'])<1e-6 for r in rows))
            check('causal_partner_end:'+name+':'+split,all(r['v6_partner_audio_end'] is None or r['v6_partner_audio_end']<=r['v6_turn_start']+1e-6<=r['absolute_decision']+1e-6 for r in rows))
    latest=datasets[data_names[-1]]
    check('held_384_retained',len(latest['sealed'])==384)
    check('no_human_gold_claim',all(r['is_human_gold'] is False for split in latest.values() for r in split))
    for split in ['train','dev','sealed']:
        check('EN_ZH_ASR_unchanged:'+split,all(r['text']==old['text'] and r['context']==old['context'] for r,old in zip(datasets[data_names[1]][split],latest[split],strict=True) if r['language']!='vi'))
    legacy=[{**r,'evaluation_slice':'legacy_vi'} for r in read_jsonl(Path('data_v4/causal_20261025/dev.jsonl')) if r['language']=='vi']
    for folder,data_name in [('outputs/natural_v5_20261025/runs/natural_v5_train_20261025',data_names[0]),
        ('outputs/natural_v6_20261025/runs/natural_v6_minilm_finetune_20261025',data_names[1]),
        ('outputs/natural_v7_20261025/runs/natural_v7_text_20261025',data_names[2])]:
        folder=Path(folder);report=load(folder/'report.json');record=report['best_dev']
        rows=[{**r,'evaluation_slice':'natural'} for r in datasets[data_name]['dev']]+legacy
        if 'best_dev_predictions_path' in report:
            p=ROOT/report['best_dev_predictions_path']
        else:
            # All these selected best paths are recorded as project-relative.
            parent=Path(report['best_path']).parent
            evidence=folder.parents[1]
            p=evidence/parent/'dev_predictions.npz'
        prediction=np.load(p)
        check(str(folder)+':stored_labels',np.array_equal(prediction['labels'],np.array([LABEL2ID[r['label']] for r in rows])))
        validate_slices(rows,prediction['prob_EOT'],record,str(folder))
        check(str(folder)+':no_held_quality_claim',not report.get('real_target_achieved',report.get('independent_real_target_achieved',False)))
        # Exactly the local source used for the run is still available.
        identity=load(folder/'identity.json')
        module='natural_v5_train.py' if 'natural_v5' in str(folder) else 'natural_v6_finetune.py' if 'natural_v6' in str(folder) else 'natural_v7_train.py'
        check('executed_source_hash:'+module,file_hash(Path('turn_v2')/module)==identity['source_sha256'])
    for encoder in ['qwen','minilm']:
        root=Path('outputs/natural_v6_20261025');features=np.load(root/(encoder+'_features/features.npz'))
        predictions=np.load(root/(encoder+'_heads/predictions.npz'));report=load(root/(encoder+'_heads/report.json'))
        check(encoder+':feature_hash',file_hash(root/(encoder+'_features/features.npz'))==report['features_sha256'])
        check(encoder+':train_dev_only',set(features['split'])=={'train','dev'})
        for lang,winner in report['winners'].items():
            mask=(features['language']==lang)&(features['split']=='dev')
            compare(winner['dev'],classification_metrics(features['labels'][mask],predictions['prob_EOT'][mask],winner['dev']['threshold']),encoder+':'+lang)
            check(encoder+':head_hash:'+lang,file_hash(Path(winner['path']))==winner['head_sha256'])
    asr=load(ROOT/'matched_ASR_dev/report.json');p=np.load(ROOT/'matched_ASR_dev/predictions.npz');t=asr['identity']['threshold_fixed_from_old_dev']
    compare(asr['old_whisper_small'],classification_metrics(p['labels'],p['old_prob_EOT'],t),'ASR_old_matched')
    compare(asr['new_phowhisper_same_threshold'],classification_metrics(p['labels'],p['new_prob_EOT'],t),'ASR_new_matched')
    v8root=Path('outputs/natural_v8_20261025');v8run=v8root/'runs/natural_v8_lora_20261025_r3'
    lora=load(v8run/'report.json');lora_identity=load(v8run/'identity.json');probe=load(v8run/'backward_probe.json')
    check('LoRA_source_matches',file_hash(Path('turn_v2/natural_v8_lora_r3.py'))==lora_identity['source_sha256'])
    check('LoRA_real_backward',probe['adapter_tensors_with_nonzero_gradients']>0 and not probe['optimizer_step_performed'])
    check('LoRA_optimizer_steps_completed',lora['steps_completed']>0)
    check('LoRA_is_not_a_quantized_deployment',lora_identity['quantization'] is None and lora_identity['not_a_small_deployment_model'])
    common_dev=[{**r,'evaluation_slice':'natural'} for r in latest['dev']]+legacy
    if lora['best_path']:
        prediction=np.load(v8run/'best_dev_predictions.npz')
        check('LoRA_dev_case_alignment',np.array_equal(prediction['labels'],np.array([LABEL2ID[r['label']] for r in common_dev])))
        validate_slices(common_dev,prediction['prob_EOT'],lora['best_dev'],'LoRA_raw_grid_dev')
        check('LoRA_adapter_exact',file_hash(v8root/lora['best_path']/'adapter_model.safetensors')==lora['adapter_sha256'])
    calibration=load(ROOT/'exact_dev_calibration.json')
    from turn_v2.natural_calibration import summarize_exact
    for candidate in calibration['candidates']:
        evidence=ROOT if candidate['kind']=='mdeberta' else v8root
        prediction_path=evidence/candidate['dev_predictions_path'];prediction=np.load(prediction_path)
        check('calibration_prediction_hash:'+candidate['path'],file_hash(prediction_path)==candidate['dev_predictions_sha256'])
        check('exact_dev_calibration_recomputed:'+candidate['path'],summarize_exact(common_dev,prediction['prob_EOT'])==candidate['dev'])
    frozen=load(ROOT/'selected_before_held.json');final=load(ROOT/'final_eval/report.json');test_frozen=load(ROOT/'final_eval/frozen_before_test.json')
    check('model_selected_before_held_hash',file_hash(ROOT/'selected_before_held.json')==test_frozen['selection_sha256'])
    check('exact_calibration_hash',file_hash(ROOT/'exact_dev_calibration.json')==frozen['exact_calibration_sha256'])
    check('held_data_hash_frozen',file_hash(Path(data_names[-1])/'sealed.jsonl')==test_frozen['sealed_sha256'])
    for name,candidate in [('selected',frozen['selected']),('V4_control',test_frozen['controls'][0]),('V5_control',test_frozen['controls'][1])]:
        p=np.load(ROOT/'final_eval'/(name+'_predictions.npz'));held=latest['sealed']
        labels=np.array([LABEL2ID[r['label']] for r in held]);langs=np.array([r['language'] for r in held])
        check('held_case_alignment:'+name,np.array_equal(p['labels'],labels) and np.array_equal(p['case_id'],np.array([r['case_id'] for r in held])))
        for lang in ['vi','en','zh']:
            index=langs==lang
            compare(final['metrics'][name][lang],classification_metrics(labels[index],p['prob_EOT'][index],candidate['dev']['thresholds'][lang]),'held:'+name+':'+lang)
    numeric_target=all(m['accuracy']>.9 and m['eot_f1']>.9 for m in final['metrics']['selected'].values())
    check('held_numeric_target_not_overclaimed',numeric_target==final['automatic_held_numeric_target_met'])
    check('held_marker_now_regression',load(Path(data_names[-1])/'sealed_consumed.json')['status']=='SEEN_NOW_REGRESSION_ONLY')
    crawl=list(read_jsonl(ROOT/'final_eval/crawl_predictions.jsonl'))
    check('crawl_204_unlabeled_no_fake_accuracy',len(crawl)==204 and all(r['label'] is None and r['is_gold'] is False for r in crawl) and final['crawl_accuracy'] is None and final['crawl_eot_f1'] is None)
    crawl_inputs=list(read_jsonl(Path('data_v4/fresh_crawl_20261025/review_cases.jsonl')))
    check('crawl_original_observed_prefixes',len(crawl)==len(crawl_inputs) and all(r['candidate_id']==old['candidate_id'] and r['text']==old['text'] and r['context']==old['context'] for r,old in zip(crawl,crawl_inputs,strict=True)))
    check('crawl_frozen_threshold_decisions',all(r['threshold']==final['thresholds'][r['language']] and r['prediction']==('EOT' if r['prob_EOT']>=r['threshold'] else 'NEOT') for r in crawl))
    player=json.loads(re.search(r'<script[^>]*id="data"[^>]*>(.*?)</script>',(ROOT/'crawl_review.html').read_text(encoding='utf-8'),re.S).group(1))
    check('crawl_player_actual_saved_predictions',player['rows']==crawl and player['model']['identity_sha256']==final['selected_identity'])
    audio=load(ROOT/'asr_review/manifest.json')
    check('listening20_train_dev_only',len(audio['items'])==20 and not audio['held_cases_included'] and all(r['split'] in ['train','dev'] for r in audio['items']))
    for item in audio['items']:check('listening_hash:'+item['case_id'],file_hash(ROOT/'asr_review'/item['audio'])==item['audio_sha256'])
    notebook=load(Path('a.ipynb'));backup=load(ROOT/'a_before_campaign_status.ipynb')
    check('previous_notebook_cells_unchanged',notebook['cells'][:len(backup['cells'])]==backup['cells'])
    cell=next(c for c in notebook['cells'] if c['cell_type']=='code' and 'natural_campaign_status' in c.get('metadata',{}).get('tags',[]))
    receipt=load(ROOT/'notebook_colab_receipt.json')
    check('notebook_code_actually_executed_in_colab',receipt['status']=='ok' and hashlib.sha256(''.join(cell['source']).encode()).hexdigest()==receipt['source_sha256'])
    check('notebook_actual_saved_outputs',cell['outputs']==receipt['outputs'])
    for relative in ['review.html','asr_review/review.html','crawl_review.html']:
        path=ROOT/relative;text=path.read_text(encoding='utf-8');scripts=re.findall(r'<script>(.*?)</script>',text,re.S)
        js=path.with_suffix('.js');js.write_text(scripts[-1],encoding='utf-8')
        r=subprocess.run(['node','--check',str(js)],capture_output=True,text=True)
        check('JS_syntax:'+relative,r.returncode==0)
        check('template_resolved:'+relative,'__DATA__' not in text)
    check('unit_tests_passed',load(ROOT/'unit_tests_final.json')['exit_code']==0)
    check('review_behaviour_passed',load(ROOT/'review_behaviour_validation.json')['status']=='PASS')
    sources=sorted([*Path('turn_v2').glob('natural_*.py'),*Path('tools').glob('natural_*.py'),*Path('tests').glob('test_natural*.py'),
        Path('tools/templates/natural_campaign.html'),Path('tools/templates/natural_crawl_player.html'),Path('tools/templates/natural_status_cell.py'),Path('tools/check_natural_review.js'),Path('turn_v2/common.py'),Path('turn_v2/metrics.py'),Path('turn_v2/stream_v4_train.py'),
        Path('tools/colab_runtime.py'),Path('tools/distill_job.py'),Path('tools/validate_natural_campaign.py'),Path('README_NATURAL_CAMPAIGN.md'),Path('a.ipynb')])
    items=[]
    for path in sources:
        destination=ROOT/'source_final'/path;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():check('snapshot_immutable:'+str(path),destination.read_bytes()==path.read_bytes())
        else:shutil.copy2(path,destination)
        items.append({'path':str(path),'sha256':file_hash(path)})
    write_json(ROOT/'source_final_manifest.json',{'files':items})
    failed=[c for c in CHECKS if not c['passed']]
    outcome={'technical_status':'PASS' if not failed else 'FAIL','passed':len(CHECKS)-len(failed),'total':len(CHECKS),'checks':CHECKS,
        'user_target_achieved':False,'gold_quality_not_certified':True,'source_final_manifest_sha256':file_hash(ROOT/'source_final_manifest.json'),
        'UI_scope':'JavaScript syntax checked; no visual browser QA claim','old_holdout_evaluated':False}
    write_json(ROOT/'completion_validation.json',outcome)
    print(json.dumps({k:v for k,v in outcome.items() if k!='checks'}),flush=True)
    if failed:print(json.dumps({'failed':failed}),flush=True);raise SystemExit(1)


if __name__=='__main__':main()
