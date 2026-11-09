"""Verify the corrected architecture, receipts and reported diagnostics."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.common import LABEL2ID,file_hash,read_jsonl,write_json
from turn_v2.metrics import classification_metrics

ROOT=Path('outputs/monolingual_20261025')


def main():
    checks=[]
    def check(name,value):checks.append({'check':name,'passed':bool(value)})
    report=json.loads((ROOT/'report.json').read_text());registry=json.loads((ROOT/'registry.json').read_text())
    check('technical_finalization_checks',report['all_technical_checks_passed'])
    check('explicit_separate_paths',len({v['path'] for v in registry['models'].values()})==3)
    seen=list(read_jsonl(Path('data_v6/pho_text_20261025/sealed.jsonl')))
    for lang in ['vi','en','zh']:
        rows=[r for r in seen if r['language']==lang];p=np.load(ROOT/(lang+'_regression.npz'));labels=np.array([LABEL2ID[r['label']] for r in rows])
        contract=json.loads((Path(registry['models'][lang]['path'])/'turn_detector_config.json').read_text())
        check(lang+':regression_case_alignment',np.array_equal(p['labels'],labels) and np.array_equal(p['case_id'],np.array([r['case_id'] for r in rows])))
        check(lang+':regression_recomputed',classification_metrics(labels,p['prob_EOT'],contract['threshold'])==report['regression_metrics'][lang])
        check(lang+':weights_exact',file_hash(Path(registry['models'][lang]['path'])/'model.onnx')==registry['models'][lang]['model_sha256'])
    with zipfile.ZipFile(ROOT/'upload.zip') as source:
        for lang in ['en','zh']:
            teacher=ROOT/f'runs/mono_{lang}_teacher_20261025';identity=json.loads((teacher/'identity.json').read_text())
            for name in ['monolingual_teacher.py','monolingual_student.py','monolingual_data.py']:
                path=Path('turn_v2')/name;check(lang+':uploaded_source_matches:'+name,source.read(path.as_posix())==path.read_bytes())
            check(lang+':executed_teacher_source',identity['source_sha256']==file_hash(Path('turn_v2/monolingual_teacher.py')))
            job=json.loads((ROOT/f'jobs/mono_{lang}_teacher_20261025/status.json').read_text());check(lang+':teacher_job_completed',job['status']=='COMPLETED' and job['exit_code']==0)
            job=json.loads((ROOT/f'jobs/mono_{lang}_student_20261025/status.json').read_text());check(lang+':student_job_completed',job['status']=='COMPLETED' and job['exit_code']==0)
            data=Path('data_v9/monolingual_20261025')/lang
            for split in ['train','dev']:check(lang+':language_isolation:'+split,{r['language'] for r in read_jsonl(data/(split+'.jsonl'))}=={lang})
            backup=json.loads((ROOT/('backup_'+lang)/'receipt.json').read_text());check(lang+':teacher_saved_locally',file_hash(teacher/'best/model.safetensors')==backup['sha256'])
    check('VI_accepted_weights_unchanged',registry['models']['vi']['model_sha256']=='9aeae3868139d5c2faa19bee34d7128a90c41ae5b0161a8f87fc039d96188412')
    crawl=list(read_jsonl(ROOT/'crawl_predictions.jsonl'));check('crawl_not_fake_gold',len(crawl)==204 and all(r['label'] is None and r['is_gold'] is False for r in crawl) and report['crawl_accuracy'] is None)
    html=(ROOT/'crawl_review.html').read_text(encoding='utf-8');payload=json.loads(re.search(r'<script[^>]*id="data"[^>]*>(.*?)</script>',html,re.S).group(1))
    check('player_actual_private_model_predictions',payload['rows']==crawl and payload['model']['registry']==registry['models'])
    scripts=re.findall(r'<script>(.*?)</script>',html,re.S);js=ROOT/'crawl_review.js';js.write_text(scripts[-1],encoding='utf-8')
    check('player_JS_syntax',subprocess.run(['node','--check',str(js)],capture_output=True).returncode==0)
    old=json.loads((ROOT/'a_before_monolingual.ipynb').read_text(encoding='utf-8'));nb=json.loads(Path('a.ipynb').read_text(encoding='utf-8'))
    check('previous39_notebook_cells_preserved',nb['cells'][:len(old['cells'])]==old['cells'])
    cell=next(c for c in nb['cells'] if c['cell_type']=='code' and 'monolingual_status' in c.get('metadata',{}).get('tags',[]));receipt=json.loads((ROOT/'notebook_final_receipt.json').read_text(encoding='utf-8'))
    check('notebook_real_Colab_execution',receipt['status']=='ok' and hashlib.sha256(''.join(cell['source']).encode()).hexdigest()==receipt['source_sha256'] and cell['outputs']==receipt['outputs'])
    check('59_tests_passed',json.loads((ROOT/'unit_tests.json').read_text())['exit_code']==0 and '59 passed' in (ROOT/'unit_tests.log').read_text())
    check('no_unseen_quality_overclaim',report['real_quality_target_certified'] is False and report['regression_is_fresh_test'] is False)
    sources=[*Path('turn_v2').glob('monolingual*.py'),*Path('tools').glob('*monolingual*.py'),Path('tools/templates/monolingual_status_cell.py'),Path('tests/test_monolingual.py'),Path('README_MONOLINGUAL.md'),Path('a.ipynb')]
    items=[]
    for path in sources:
        target=ROOT/'source_final'/path;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():check('source_snapshot_immutable:'+str(path),target.read_bytes()==path.read_bytes())
        else:shutil.copy2(path,target)
        items.append({'path':str(path),'sha256':file_hash(path)})
    write_json(ROOT/'source_manifest.json',{'files':items})
    result={'status':'PASS' if all(r['passed'] for r in checks) else 'FAIL','passed':sum(r['passed'] for r in checks),'total':len(checks),'checks':checks,
        'quality_target_certified':False,'scope':'Technical architecture/provenance/metrics checks; not a model quality certificate'}
    write_json(ROOT/'completion_validation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
    if result['status']!='PASS':print(json.dumps([r for r in checks if not r['passed']]));raise SystemExit(1)


if __name__=='__main__':main()
