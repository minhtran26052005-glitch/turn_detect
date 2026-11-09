"""Verify real teacher artifacts on Colab without downloading their weights."""
import hashlib
import json
from pathlib import Path

from teacher_first_colab import campaign_runtime

ROOT = Path('outputs/teacher_first_v14_20261026')


def main():
    runtime = campaign_runtime(); runtime.connect()
    code = '''from pathlib import Path
import json,hashlib,sqlite3,math
from safetensors import safe_open
from turn_v2.teacher_first_protocol import distillation_gate
root=Path('/content/turn_detect_v2');folder=root/'outputs/teacher_first_v14_20261026'
checks=[];evidence={}
def check(name,value):checks.append({'check':name,'passed':bool(value)})
config=json.loads((folder/'config.json').read_text());check('unbounded_teacher_campaign',config['max_cycles']==0)
check('student_training_disabled',config['student_training_enabled'] is False)
check('pinned_pretrained_revision',config['revision']=='a0484667b22365f84929a935b5e50a51f71f159d')
old=json.loads((root/'outputs/continuous_v13_20261026/status.json').read_text());check('old_GRU_stopped_after_user_handoff',old['status']=='STOPPED_BY_REQUEST')
c=sqlite3.connect('file:'+str(folder/'state.sqlite')+'?mode=ro',uri=True);champions={}
hashes=[]
for lang in ['vi','en','zh']:
 path=folder/'teachers'/lang
 identity=json.loads((path/'identity.json').read_text());resume=json.loads((path/'resume_receipt.json').read_text())
 report=json.loads((path/'best_report.json').read_text());champions[lang]=report
 check(lang+':full_pretrained_independent_teacher',identity['parameters']==278810882 and identity['trainable_parameters']==identity['parameters'] and identity['train_languages']==[lang] and not identity['shared_trainable_weights'])
 weights=path/'best/model.safetensors';digest=hashlib.file_digest(weights.open('rb'),'sha256').hexdigest();hashes.append(digest)
 check(lang+':selected_weights_exact',digest==report['weights_sha256'])
 with safe_open(str(weights),framework='pt',device='cpu') as tensors:
  count=sum(math.prod(tensors.get_slice(key).get_shape()) for key in tensors.keys())
 check(lang+':actual_checkpoint_parameter_count',count==identity['parameters'])
 checkpoint=root/resume['checkpoint'];check(lang+':optimizer_checkpoint_exists',checkpoint.stat().st_size==resume['bytes'] and resume['total_steps']>0)
 train=[json.loads(row[0]) for row in c.execute("SELECT record FROM samples WHERE language=? AND split='train'",(lang,))]
 dev=[json.loads(row[0]) for row in c.execute("SELECT record FROM samples WHERE language=? AND split='semantic_dev'",(lang,))]
 group=lambda r:r.get('session_group') or r.get('source_id') or r.get('group_id')
 check(lang+':source_group_isolation',not {group(r) for r in train}&{group(r) for r in dev})
 check(lang+':protected_inputs_not_in_train',not c.execute("SELECT 1 FROM samples s JOIN protected p ON s.key=p.key WHERE s.language=? AND s.split='train' LIMIT 1",(lang,)).fetchone())
 check(lang+':protected_sources_not_in_train',not c.execute("SELECT 1 FROM samples s JOIN protected_sources p ON s.source_id=p.source_id WHERE s.language=? AND s.split='train' LIMIT 1",(lang,)).fetchone())
 check(lang+':train_not_raw_proxy_labels',all(r.get('label_source')=='QWEN8B_SEMANTIC_TWO_VIEW_SILVER' and r.get('is_gold') is False for r in train))
 evidence[lang]={'parameters':count,'selected_step':report['total_steps'],'optimizer_total_steps':resume['total_steps'],'checkpoint_SHA256':digest,'optimizer_bytes':resume['bytes'],'semantic_dev':report['semantic_dev'],'natural_floor_dev':report['natural_floor_dev']}
c.close();check('three_distinct_teacher_weights',len(set(hashes))==3)
check('silver_only_cannot_trigger_distillation',distillation_gate(champions)['distillation_allowed'] is False)
print(json.dumps({'checks':checks,'evidence':evidence,'quality_target_certified':False,'scope':'Technical integrity and task separation, not independent quality acceptance'}))
'''
    result = runtime.execute(code, timeout=120, echo=False); runtime.close()
    (ROOT / 'teacher_artifact_validation_receipt.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    assert result['status'] == 'ok', 'Inspect validation receipt; wait until all three teachers have trained'
    value = json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type'] == 'stream'))
    nb = json.loads(Path('a.ipynb').read_text(encoding='utf-8'))
    before = json.loads((ROOT / 'a_before_teacher_first.ipynb').read_text(encoding='utf-8'))
    # Executing cells in VS Code legitimately changes outputs/execution_count.
    # Protect every original cell's identity, order, type and source instead.
    preserved = len(nb['cells']) >= len(before['cells']) and all(
        all(current.get(key) == old.get(key) for key in ['id', 'cell_type', 'source'])
        for current, old in zip(nb['cells'], before['cells']))
    value['checks'].append({'check': 'original_notebook_sources_and_cell_order_preserved', 'passed': preserved})
    last = next(c for c in nb['cells'] if c['cell_type'] == 'code' and 'teacher_first_v14_status' in c.get('metadata', {}).get('tags', []))
    proof = json.loads(Path(last['metadata']['verification_receipt']).read_text(encoding='utf-8'))
    value['checks'].append({'check': 'status_cell_actually_executed_on_Colab', 'passed': proof['status'] == 'ok' and last['outputs'] == proof['outputs'] and hashlib.sha256(''.join(last['source']).encode()).hexdigest() == proof['source_sha256']})
    value.update(passed=sum(c['passed'] for c in value['checks']), total=len(value['checks']))
    value['status'] = 'PASS' if value['passed'] == value['total'] else 'FAIL'
    (ROOT / 'teacher_artifact_validation.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in value.items() if k not in ['checks', 'evidence']}))
    assert value['status'] == 'PASS', [c for c in value['checks'] if not c['passed']]


if __name__ == '__main__': main()
