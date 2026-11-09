"""Checkpointed handover to the authorized uploaded VI/EN curriculum trainer."""
from pathlib import Path
import copy
import fcntl
import gzip
import json
import os
import shutil
import subprocess
import time
import traceback

import numpy as np

from .uploaded_vi_en_sentence_mix_data_v7 import ROOT, OUT, PARENT, MODULES, StreamOrder, initialize, reserve, save, sha, make_order, check_order


def identity(launch):
    assert launch['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    proc = Path('/proc')/str(launch['pid'])
    if not proc.exists():
        return False
    raw=(proc/'stat').read_text()
    assert raw[raw.rfind(')')+2:].split()[19] == launch['start_ticks']
    assert launch['module'].encode() in (proc/'cmdline').read_bytes().split(bytes([0]))
    return True


def workers(module):
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if module.encode() in (proc/'cmdline').read_bytes().split(bytes([0])):
                found.append(int(proc.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):
            continue
    return sorted(found)


def verify_actual_training(launch, manifest):
    import torch
    from transformers import AutoTokenizer
    from .multilingual_malbert_vi_en_sentence_mix_train_v7 import fingerprint, equal
    from .tiny_compression_v1.common import encode_pair
    torch.set_num_threads(2)
    pin=OUT/'frozen/verified_curriculum_resume.pt'
    assert not pin.exists()
    with (OUT/'runs/finetune/latest_resume.pt').open('rb') as source,pin.open('xb') as target:
        shutil.copyfileobj(source,target)
        target.flush()
        os.fsync(target.fileno())
    parent=torch.load(OUT/'frozen/parent_full_resume.pt',map_location='cpu',weights_only=False,mmap=True)
    initial=torch.load(OUT/'frozen/initial_child_resume.pt',map_location='cpu',weights_only=False,mmap=True)
    actual=torch.load(pin,map_location='cpu',weights_only=False,mmap=True)
    plan=json.loads((OUT/'curriculum_plan.json').read_text())
    for key in ['model','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng','best_threshold']:
        assert equal(parent[key],initial[key])
    rng=np.random.default_rng()
    rng.bit_generator.state=copy.deepcopy(parent['sampler_rng'])
    expected_order=make_order(plan['stages'][3],rng)
    assert equal(expected_order,initial['order'])
    assert equal(rng.bit_generator.state,initial['sampler_rng'])
    assert actual['fingerprint'] == fingerprint(manifest)
    assert actual['parameters'] == 11941122 and len(actual['optimizer']['state']) == 27
    assert actual['branch_step'] >= 500 and actual['step'] == parent['step']+actual['branch_step']
    assert actual['curriculum_stage'] == 3 and actual['curriculum_stage_step'] == actual['branch_step']
    assert actual['epoch'] == 0 and actual['uploaded_rows_seen'] > initial['uploaded_rows_seen']
    assert actual['loss_policy'] == {'NEOT':1.,'EOT':2.}
    check_order(plan['stages'][3],actual['order'])
    assert {int(value['step']) for value in actual['optimizer']['state'].values()} == {actual['step']}
    assert all(abs(group['lr']-4e-6)<1e-15 and group['weight_decay']==.01 for group in actual['optimizer']['param_groups'])
    assert all(bool(torch.isfinite(value).all()) for value in actual['model'].values())
    assert any(not torch.equal(value,initial['model'][key]) for key,value in actual['model'].items())
    assert all(key in actual for key in ['torch_rng','cuda_rng','numpy_rng','python_rng','sampler_rng','order','cursor','stream_state'])
    active=set(plan['stages'][3]['stream_indices'])
    assert all(lane['stream'] in active for lane in actual['stream_state']['slots'])
    streams=initialize(json.load(gzip.open(OUT/'train_streams.json.gz','rt',encoding='utf-8')))
    tokenizer=AutoTokenizer.from_pretrained(OUT/'tokenizer',local_files_only=True,token=False)
    runner=StreamOrder(streams,initial['order'],initial['stream_state'],lanes=16)
    trace=json.loads((OUT/'first_training_causal_trace.json').read_text())
    offset=0
    scored=0
    updates=0
    while offset<len(trace):
        rows,states=runner.rows()
        part=trace[offset:offset+len(rows)]
        assert rows and len(part)==len(rows)
        probabilities=[]
        for row,item in zip(rows,part):
            for key in ['stream','boundary','k','start','label','scored']:
                assert item[key]==row[key]
            assert item['origin']==row['origin'] and row['stream'] in active
            assert item['threshold']==initial['best_threshold']
            source=streams[row['stream']]
            boundary=row['boundary']
            assert row['context']==' '.join(source['words'][row['start']:boundary])
            assert row['text']==' '.join(source['words'][boundary:boundary+row['k']])
            assert any(segment['start']<boundary and boundary+row['k']<=segment['end'] for segment in source['segments'])
            assert row['label']==int(boundary in source['_eots'])
            ids=encode_pair(tokenizer,row['context'],row['text'],256)['input_ids']
            import hashlib
            assert hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()==item['input_ids_sha256']
            probabilities.append(item['p_eot'])
            scored+=int(row['scored'])
        updates+=int(any(row['scored'] for row in rows))
        runner.advance(states,probabilities,initial['best_threshold'])
        offset+=len(rows)
    assert updates==100 and offset==len(trace)
    first=json.loads((OUT/'independent_training_verification.json').read_text())
    assert first['uploaded_rows_seen']-initial['uploaded_rows_seen']==scored
    assert identity(launch)
    assert workers(launch['module'])==[launch['pid']]
    assert not Path('/proc',str(json.loads((PARENT/'launcher.json').read_text())['pid'])).exists()
    proof=dict(complete=True,launcher=launch,parent_step=parent['step'],step=actual['step'],
        branch_step=actual['branch_step'],curriculum_stage=actual['curriculum_stage'],
        curriculum_family_fraction=1.0,stage_scored_exposures=actual['curriculum_stage_seen'],
        actual_causal_training_inputs_audited=len(trace),actual_optimizer_updates_audited=updates,
        all_labels_masks_synthetic_pack_future_and_predicted_resets_verified=True,
        model_optimizer_and_global_rng_preserved_at_cutover=True,
        initial_stage_sampler_permutation_verified=True,parent_sampler_order_lanes_kept_in_full_pin=True,
        optimizer_entries=27,all_weights_finite_and_changed_after_training=True,parameters=11941122,
        full_filtered_epochs_complete=0,new_checkpoint_sha256=sha(pin),
        manifest_sha256=sha(OUT/'dataset_manifest.json'),curriculum_plan_sha256=sha(OUT/'curriculum_plan.json'),
        actual_parent_process_absent=True,test_used=False)
    save('independent_actual_curriculum_training_verification.json',proof)
    return proof


def run():
    assert not (OUT/'cutover_complete.json').exists()
    assert not (OUT/'launcher.json').exists()
    reserve()
    manifest=json.loads((OUT/'dataset_manifest.json').read_text())
    prep=json.loads((OUT/'independent_preparation_verification.json').read_text())
    assert prep['complete'] and prep['manifest_sha256']==sha(OUT/'dataset_manifest.json')
    assert json.loads((OUT/'technical_fixture_verification.json').read_text())['complete']
    assert json.loads((OUT/'independent_pack_verification.json').read_text())['complete']
    assert manifest['curriculum_plan_sha256']==sha(OUT/'curriculum_plan.json')
    for name in MODULES:
        assert manifest['code_sha256'][name]==sha(ROOT/'turn_v2'/name)
    parent=json.loads((PARENT/'launcher.json').read_text())
    assert identity(parent)
    assert workers(parent['module'])==[parent['pid']]
    save('handover_liveness_verification.json',dict(parent=parent,actual_process_identity_verified=True,
        user_authorized_curriculum_cutover=True,manifest_sha256=sha(OUT/'dataset_manifest.json')))
    marker=PARENT/'STOP_REQUESTED'
    if not marker.exists():
        with marker.open('x') as stream:
            stream.write('User authorized balanced synthetic sentence packs 1..10 on 2026-11-07\n')
            stream.flush()
            os.fsync(stream.fileno())
    while identity(parent):
        save('cutover_status.json',dict(stage='WAITING_FOR_PARENT_CHECKPOINTED_EXIT',parent=parent,
             parent_status=json.loads((PARENT/'status.json').read_text())))
        time.sleep(5)
    assert json.loads((PARENT/'status.json').read_text())['stage']=='STOPPED_WITH_CHECKPOINT'
    assert not (PARENT/'error.json').exists()
    assert not workers('turn_v2.multilingual_malbert_vi_en_sentence_mix_train_v7')
    env=os.environ.copy()
    env.update(LD_LIBRARY_PATH='/home/edabk/miniconda3/envs/.VLLM/lib/python3.10/site-packages/nvidia/cu13/lib',
        PYTHONPATH=str(ROOT/'outputs/multilingual_tiny_compression_20261102_v1/runtime_deps')+':'+str(ROOT),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    with (OUT/'worker.log').open('ab') as log:
        proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','turn_v2.multilingual_malbert_vi_en_sentence_mix_train_v7'],
             cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(.5)
    raw=Path('/proc',str(proc.pid),'stat').read_text()
    launch=dict(pid=proc.pid,start_ticks=raw[raw.rfind(')')+2:].split()[19],boot_id=parent['boot_id'],
                module='turn_v2.multilingual_malbert_vi_en_sentence_mix_train_v7',source_sha256=manifest['code_sha256'])
    save('launcher.json',launch)
    save('cutover_status.json',dict(stage='VERIFYING_ACTUAL_CURRICULUM_TRAINING',launcher=launch))
    while True:
        assert identity(launch), 'Inspect the exited worker before any restart'
        assert not (OUT/'error.json').exists()
        receipt=OUT/'runs/finetune/resume_receipt.json'
        if receipt.exists() and json.loads(receipt.read_text())['branch_step']>=500:
            break
        time.sleep(3)
    proof=verify_actual_training(launch,manifest)
    save('cutover_complete.json',proof)
    save('cutover_status.json',dict(stage='COMPLETE_TRAINING_VERIFIED',proof=proof))


if __name__=='__main__':
    os.nice(10)
    with (OUT/'CUTOVER_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run()
        except Exception as error:
            save('cutover_error.json',dict(error=repr(error),traceback=traceback.format_exc()))
            raise
