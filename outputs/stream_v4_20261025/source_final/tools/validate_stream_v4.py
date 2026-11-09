"""Audit immutable controls, causal inputs, recomputed scores and receipts.

Technical success is reported separately from the unmet quality objective.
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.common import LABEL2ID, file_hash, input_id, read_jsonl, write_json
from turn_v2.metrics import classification_metrics

ROOT=Path('outputs/stream_v4_20261025')
CHECKS=[]


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def check(name,condition):
    CHECKS.append({'check':name,'passed':bool(condition)})


def compare_metrics(expected,actual,prefix):
    for key in ['samples','accuracy','eot_f1','false_eot','missed_eot','confusion_matrix_NEOT_EOT']:
        if isinstance(expected[key],float):
            valid=math.isclose(expected[key],actual[key],rel_tol=0,abs_tol=1e-12)
        else:valid=expected[key]==actual[key]
        check(prefix+':'+key,valid)


def main():
    protocol=load(ROOT/'protocol.json')
    for name,digest in protocol['initial_controls'].items():
        check('unchanged_control:'+name,file_hash(Path(name))==digest)
    for package in ['turn_v2_distilled_qat4_20261024','turn_v3_trilingual_pilot_20261024']:
        folder=Path('models')/package
        manifest=load(folder/'model_manifest.json')
        for name,identity in manifest['files'].items():
            path=folder/'onnx'/name
            check('unchanged_package:'+package+':'+name,path.stat().st_size==identity['bytes'] and file_hash(path)==identity['sha256'])
    data=Path('data_v4/causal_20261025');manifest=load(data/'manifest.json')
    check('frozen_data_manifest',file_hash(data/'manifest.json')==protocol['data_manifest_sha256'])
    splits={name:list(read_jsonl(data/(name+'.jsonl'))) for name in ['train','dev','sealed']}
    for name in splits:
        check('split_hash:'+name,file_hash(data/(name+'.jsonl'))==manifest['split_sha256'][name])
    for left,right in [('train','dev'),('train','sealed'),('dev','sealed')]:
        a={r['language']+input_id(r) for r in splits[left]};b={r['language']+input_id(r) for r in splits[right]}
        check('normalized_input_disjoint:'+left+':'+right,not a.intersection(b))
    sessions={name:{r['session_group'] for r in rows if r['source'].startswith('livekit')} for name,rows in splits.items()}
    check('conversation_group_isolation',not sessions['train'].intersection(sessions['dev']))
    raw={}
    for path in sorted(Path('data_v4/metadata_frozen_20261025').glob('??_*.json')):
        for row in load(path):raw[row['id']]=row
    causal_count=0;causal_valid=True
    for rows in splits.values():
        for row in rows:
            if row['source']!='scicom_read_speech':continue
            original=raw[row['source_id']];stamp=row['timestamp_seconds']
            words=[w['word'] for w in original['words'] if float(w['end'])<=stamp+1e-8]
            text=('' if row['language']=='zh' else ' ').join(words)
            causal_valid &= text==row['text'] and row['context']=='' and row['current_turn_full_message_excluded']
            causal_count+=1
    check('all_read_speech_prefixes_causal_no_full_current_message',causal_valid and causal_count>10000)
    report=load(ROOT/'colab_final/report.json');frozen=load(ROOT/'colab_final/frozen_before_test.json')
    summary=load(ROOT/'summary.json');checkpoint=ROOT/summary['selected_checkpoint']
    check('selected_weights_exact',file_hash(checkpoint/'model.safetensors')==frozen['weights_sha256']==summary['weights_sha256'])
    check('selected_weights_bytes',(checkpoint/'model.safetensors').stat().st_size==summary['weights_bytes'])
    selected=load(checkpoint.parent/'best_dev.json')
    check('selected_thresholds_from_dev',frozen['thresholds']==selected['thresholds'])
    check('selected_dev_receipt_hash',file_hash(checkpoint.parent/'best_dev.json')==frozen['selected_dev_sha256'])
    follow=load(ROOT/'runs/stream_v4_balanced_20261025/report.json')
    check('followup_control_retained',follow['best_path']==summary['selected_checkpoint'] and follow['stop_reason']=='DEV_PLATEAU')
    check('quality_not_claimed',not report['all_real_languages_over_90'] and not summary['user_target_achieved'])
    check('seen_test_marked_regression','SEEN_TEST_NOW_REGRESSION_ONLY' in frozen['test_status'])
    fresh_vi=list(read_jsonl(Path('data_v4/vi_natural_pause_20261025/cases.jsonl')))
    fresh_en_zh=list(read_jsonl(Path('data_v4/en_zh_natural_pause_20261025/cases.jsonl')))
    check('fresh_vi_hash',file_hash(Path('data_v4/vi_natural_pause_20261025/cases.jsonl'))==frozen['fresh_vi_sha256'])
    check('fresh_en_zh_hash',file_hash(Path('data_v4/en_zh_natural_pause_20261025/cases.jsonl'))==frozen['fresh_en_zh_sha256'])
    train_sources={r.get('source_id') for r in splits['train']+splits['dev']}
    check('fresh_audio_source_ids_outside_train_dev',not {r['source_id'] for r in fresh_vi+fresh_en_zh}.intersection(train_sources))
    check('no_human_gold_claim_for_audio',all(r['is_human_gold'] is False for r in fresh_vi+fresh_en_zh))
    scopes={'dev':splits['dev'],'sealed_read_speech':splits['sealed'],
        'vi_full_legacy_dev':[{**r,'language':'vi'} for r in read_jsonl(Path('data_v2/frozen/dev.jsonl'))],
        'fresh_vi_natural_auto_labels':fresh_vi,'fresh_en_zh_natural_auto_labels':fresh_en_zh,
        'original_teacher_matched_fresh_auto_labels':fresh_vi+fresh_en_zh}
    for name,rows in scopes.items():
        filename='original_teacher_fresh_predictions.npz' if name.startswith('original_teacher') else name+'_predictions.npz'
        prediction=np.load(ROOT/'colab_final'/filename)
        labels=np.array([LABEL2ID[r['label']] for r in rows]);langs=np.array([r['language'] for r in rows]);probs=prediction['prob_EOT']
        check('case_alignment:'+name,np.array_equal(labels,prediction['labels']) and np.array_equal(langs,prediction['language']))
        check('finite_probabilities:'+name,np.all(np.isfinite(probs)) and np.all((probs>=0)&(probs<=1)))
        thresholds=frozen['baseline_dev_thresholds'] if name.startswith('original_teacher') else frozen['thresholds']
        for lang in sorted(set(langs)):
            mask=langs==lang;actual=classification_metrics(labels[mask],probs[mask],thresholds[lang])
            compare_metrics(report['metrics'][name][lang],actual,name+':'+lang)
        if name=='dev':
            for key,expected in selected['slices'].items():
                lang,source=key.split(':',1);mask=np.array([r['language']==lang and r['source']==source for r in rows])
                compare_metrics(expected,classification_metrics(labels[mask],probs[mask],thresholds[lang]),'dev_slice:'+key)
    crawl=list(read_jsonl(ROOT/'colab_final/fresh_crawl_predictions.jsonl'))
    check('crawl_204_unlabeled',len(crawl)==204 and all(r['label'] is None and r['is_gold'] is False for r in crawl))
    check('crawl_metrics_unavailable',report['fresh_crawl_accuracy'] is None and report['fresh_crawl_eot_f1'] is None)
    backup=load(ROOT/'a_before_stream_v4.ipynb');notebook=load(Path('a.ipynb'))
    check('previous_notebook_cells_preserved',notebook['cells'][:len(backup['cells'])]==backup['cells'])
    cells=[c for c in notebook['cells'] if c['cell_type']=='code' and 'stream_v4_result' in c.get('metadata',{}).get('tags',[])]
    receipt=load(ROOT/'notebook_colab_receipt.json')
    check('notebook_exact_code_executed_on_colab',len(cells)==1 and receipt['status']=='ok' and file_hash(ROOT/'notebook_result_cell.py')==receipt['source_sha256'])
    check('notebook_saved_actual_outputs',cells[0]['outputs']==receipt['outputs'])
    check('unit_tests_45',load(ROOT/'unit_tests_final.json')['passed']==45)
    check('portable_downloaded_checkpoint',load(ROOT/'checkpoint_portability.json')['status']=='PASS')
    for clip in load(ROOT/'vi_review_audio_manifest.json')['audio_clips']:
        check('review_audio_hash:'+clip['path'],file_hash(Path(clip['path']))==clip['sha256'])
    for name in ['review.html','fresh_crawl_review.html','vi_natural_review.html']:
        text=(ROOT/name).read_text(encoding='utf-8')
        scripts=re.findall(r'<script(?:\s[^>]*)?>(.*?)</script>',text,re.S)
        javascript=next(s for s in scripts if s.lstrip().startswith('const '))
        script=ROOT/(name+'.js');script.write_text(javascript,encoding='utf-8')
        result=subprocess.run(['node','--check',str(script)],capture_output=True,text=True)
        check('javascript_syntax:'+name,result.returncode==0)
        check('no_html_template_placeholders:'+name,not any(p in text for p in ['__DATA__','__IDENTITY__','__NATURAL__','__BASELINE__']))
    for upload in ['campaign_upload.zip','final_evaluation_upload.zip']:
        import zipfile
        with zipfile.ZipFile(ROOT/upload) as archive:
            check('current_sources_match_uploaded:'+upload,all(Path(item.filename).read_bytes()==archive.read(item) for item in archive.infolist()))
    snapshots=[]
    paths=[Path('a.ipynb'),Path('README_STREAM_V4.md'),Path('EXPERIMENT_STREAM_V4_20261025.md'),Path('tests/test_stream_v4.py')]
    paths+=sorted(Path('turn_v2').glob('stream_v4_*.py'))
    paths+=[Path('tools')/name for name in ['stream_v4_colab.py','stream_v4_followup_colab.py','stream_v4_final_colab.py','backup_stream_v4.py',
        'prepare_vi_real_v4.py','prepare_en_zh_real_v4.py','crawl_stream_v4.py','review_stream_v4.py','review_vi_natural_v4.py',
        'evaluate_local_stream_v4.py','check_stream_v4_portability.py','report_stream_v4.py','validate_stream_v4.py','colab_runtime.py']]
    for path in paths:
        destination=ROOT/'source_final'/path
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():check('source_snapshot_immutable:'+str(path),destination.read_bytes()==path.read_bytes())
        else:shutil.copy2(path,destination)
        snapshots.append({'path':str(path),'sha256':file_hash(path)})
    write_json(ROOT/'source_final_manifest.json',{'files':snapshots})
    write_json(data/'sealed_consumed_v4.json',{'status':'SEEN_REGRESSION_ONLY','training_selection_used_test':False,
        'candidates':['local_recovery_r2','selected_colab_teacher','original_teacher_control'],
        'model_and_protocol_selected_before_final_pilot_scores':True,'held_sha256':file_hash(data/'sealed.jsonl'),
        'fresh_vi_sha256':frozen['fresh_vi_sha256'],'fresh_en_zh_sha256':frozen['fresh_en_zh_sha256'],
        'report_sha256':file_hash(ROOT/'colab_final/report.json')})
    failed=[c for c in CHECKS if not c['passed']]
    result={'technical_status':'PASS' if not failed else 'FAIL','passed':len(CHECKS)-len(failed),'total':len(CHECKS),
        'user_quality_objective_achieved':False,'fresh_audio_human_gold':False,'old_holdout_evaluated':False,
        'causal_read_speech_cases_checked':causal_count,'checks':CHECKS,'source_manifest_sha256':file_hash(ROOT/'source_final_manifest.json'),
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'ui_scope':'JavaScript syntax only; no claim of visual browser QA'}
    write_json(ROOT/'completion_validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
    if failed:print(json.dumps({'failed':failed}),flush=True);raise SystemExit(1)


if __name__=='__main__':main()
