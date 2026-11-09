"""Recompute dev metrics and verify the data/CE/reward comparison artifacts."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.common import LABEL2ID,file_hash,input_id,read_jsonl,write_json
from turn_v2.metrics import classification_metrics

ROOT=Path('outputs/reward_v10_20261025')


def main():
    checks=[]
    def check(name,value):checks.append({'check':name,'passed':bool(value)})
    with zipfile.ZipFile(ROOT/'upload.zip') as z:
        for relative in ['turn_v2/reward_v10_train.py','turn_v2/reward_v10_qwen.py','turn_v2/reward_v10_objective.py','turn_v2/reward_v10_data.py']:
            check('executed_source_unchanged:'+relative,z.read(relative)==Path(relative).read_bytes())
    for lang in ['vi','en','zh']:
        data=Path('data_v10/reward_20261025')/lang;manifest=json.loads((data/'manifest.json').read_text());train=list(read_jsonl(data/'train.jsonl'));dev=list(read_jsonl(data/'dev.jsonl'))
        for split in ['train','dev']:check(lang+':data_hash:'+split,file_hash(data/(split+'.jsonl'))==manifest['split_sha256'][split])
        check(lang+':own_language_only',{r['language'] for r in train+dev}=={lang})
        check(lang+':no_exact_train_dev_overlap',not {input_id(r) for r in train}&{input_id(r) for r in dev})
        p=ROOT/lang;report=json.loads((p/'report.json').read_text());identity=json.loads((p/'identity.json').read_text());labels=np.array([LABEL2ID[r['label']] for r in dev])
        check(lang+':executed_train_source',identity['source_sha256']==file_hash(Path('turn_v2/reward_v10_train.py')))
        for phase,record,path in [('baseline',report['baseline'],p/'baseline_predictions.npz')]+[(v['phase'],v['dev'],p/v['phase']/'dev_predictions.npz') for v in report['phases']]:
            predictions=np.load(path);check(lang+':'+phase+':labels',np.array_equal(labels,predictions['labels']))
            for name,expected in record['slices'].items():
                mask=np.array([r['evaluation_slice']==name for r in dev]);check(lang+':'+phase+':'+name,classification_metrics(labels[mask],predictions['prob_EOT'][mask],record['threshold'])==expected)
            check(lang+':'+phase+':minimum_score',record['selection_score']==min(m['joint_score'] for m in record['slices'].values()))
        for phase in report['phases']:
            check(lang+':'+phase['phase']+':student_hash',file_hash(p/phase['phase']/'student.pt')==phase['student_sha256'])
            check(lang+':'+phase['phase']+':backprop',phase['gradient_verified'])
        values={r['phase']:r['dev']['selection_score'] for r in report['phases']}
        check(lang+':RL_vs_CE_claim',report['RL_beats_matched_CE_on_selection_score']==(values['reinforce']>values['ce_control']))
        check(lang+':RL_vs_cost_CE_claim',report['RL_beats_cost_CE_on_selection_score']==(values['reinforce']>values['cost_ce']))
        state=json.loads((ROOT/f'jobs/reward_v10_{lang}_20261025_r2/status.json').read_text());check(lang+':job_completed',state['status']=='COMPLETED' and state['exit_code']==0)
    qwen=json.loads((ROOT/'qwen/report.json').read_text());audit=list(read_jsonl(ROOT/'qwen/audit.jsonl'));silver=list(read_jsonl(ROOT/'qwen/silver.jsonl'))
    check('Qwen_silver_count',len(silver)==qwen['accepted']==sum(r['accepted'] for r in audit))
    check('Qwen_source_matches',qwen['source_sha256']==file_hash(Path('turn_v2/reward_v10_qwen.py')))
    check('Qwen_not_gold',qwen['gold'] is False and all(r['is_gold'] is False and r['sample_weight']==.25 for r in silver))
    check('Qwen_pending_crawl_not_eval_gold',json.loads(Path('data_v10/reward_20261025/crawl/manifest.json').read_text())['labels_created'] is False)
    old=json.loads((ROOT/'a_before_v10.ipynb').read_text(encoding='utf-8'));nb=json.loads(Path('a.ipynb').read_text(encoding='utf-8'))
    check('notebook_earlier_cells_preserved',nb['cells'][:len(old['cells'])]==old['cells'])
    cell=next(c for c in nb['cells'] if c['cell_type']=='code' and 'reward_v10_status' in c.get('metadata',{}).get('tags',[]));receipt=json.loads((ROOT/'notebook_final_receipt.json').read_text(encoding='utf-8'))
    check('notebook_actual_Colab_execution',receipt['status']=='ok' and cell['outputs']==receipt['outputs'] and hashlib.sha256(''.join(cell['source']).encode()).hexdigest()==receipt['source_sha256'])
    check('unit_tests_passed',json.loads((ROOT/'unit_tests.json').read_text())['exit_code']==0)
    sources=[*Path('turn_v2').glob('reward_v10*.py'),*Path('tools').glob('*reward_v10*.py'),Path('tools/templates/reward_v10_status_cell.py'),Path('tests/test_reward_v10.py'),Path('README_REWARD_V10.md'),Path('a.ipynb')]
    for source in sources:
        target=ROOT/'source_final'/source;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():check('source_snapshot_unchanged:'+str(source),target.read_bytes()==source.read_bytes())
        else:shutil.copy2(source,target)
    result={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','passed':sum(c['passed'] for c in checks),'total':len(checks),'checks':checks,'quality_target_certified':False}
    write_json(ROOT/'completion_validation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
    if result['status']!='PASS':print(json.dumps([c for c in checks if not c['passed']]));raise SystemExit(1)


if __name__=='__main__':main()
