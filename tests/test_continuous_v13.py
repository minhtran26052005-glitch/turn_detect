from pathlib import Path
import json
from turn_v2.continuous_v13_state import Store,atomic_json


def sample(text='new',source='wiki:1',label='EOT'):
    return {'text':text,'context':'','source_id':source,'language':'vi','judge_label':label,'label':label}


def test_frozen_dev_protects_source_and_input_after_restart(tmp_path):
    db=tmp_path/'state.sqlite';store=Store(db)
    assert store.add('visible-1',sample(),'semantic_dev',0)
    assert not store.add('visible-2',sample('other words'),'train',1)
    assert not store.add('visible-1',sample('new','wiki:2'),'train',1)
    store.db.close();restored=Store(db)
    assert not restored.allowed('visible-3','wiki:1')
    assert restored.add('visible-4',sample('other','wiki:2'),'train',1)


def test_accepted_data_accumulates_idempotently(tmp_path):
    store=Store(tmp_path/'s.sqlite');a=sample()
    assert store.add('one',a,'train',1)
    assert not store.add('one',a,'train',2)
    assert store.add('two',sample('other','wiki:2','NEOT'),'train',2)
    assert len(store.rows('vi','train'))==2
    store.put('cycle',3);assert store.get('cycle')==3
    store.backup(tmp_path/'snapshot.sqlite');restored=Store(tmp_path/'snapshot.sqlite')
    assert restored.get('cycle')==3 and len(restored.rows('vi','train'))==2


def test_atomic_status_retains_explicit_unknown_quality(tmp_path):
    p=tmp_path/'status.json';atomic_json(p,{'stage':'TRAINING','quality_target_certified':False})
    assert json.loads(p.read_text())['quality_target_certified'] is False
    assert not p.with_suffix('.json.tmp').exists()


def test_kl_backward_accepts_frozen_inference_tensor_without_teacher_gradient():
    import torch
    from turn_v2.continuous_v13_objective import supervised_with_anchor
    with torch.inference_mode():reference=torch.tensor([[.7,.3],[.2,.8]])
    logits=torch.tensor([[0.,0.],[0.,0.]],requires_grad=True)
    loss=supervised_with_anchor(logits,torch.tensor([0,1]),reference)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
    assert logits.grad[0,0]<0 and logits.grad[1,1]<0 and reference.grad is None
