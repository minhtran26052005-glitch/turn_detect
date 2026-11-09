"""CPU regression fixture exercising the real curriculum fit loop and full resume."""
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import patch
import copy
import gzip
import json
import os

import numpy as np
import torch


def run():
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    utils.is_torchvision_available = imports.is_torchvision_available = lambda: False
    utils.is_librosa_available = imports.is_librosa_available = lambda: False
    from . import multilingual_malbert_vi_en_curriculum_train_v6 as trainer
    from . import uploaded_vi_en_curriculum_data_v6 as data
    torch.set_num_threads(2)
    actual_out = data.OUT
    assert not (actual_out/'technical_fixture_verification.json').exists()
    manifest = data.prepare()
    fixture = actual_out/'technical_fixture'
    fixture.mkdir(exist_ok=True)
    pin = fixture/'warm_parent.pt'
    if not pin.exists():
        with (data.PARENT/'runs/finetune/latest_resume.pt').open('rb') as source, pin.open('xb') as target:
            import shutil
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
    parent = torch.load(pin, map_location='cpu', weights_only=False, mmap=True)
    sources = []
    for index in range(4):
        sources.append(dict(key='fixture-'+str(index), family='fixture-family-'+str(index),
            domain='fixture', source_split='train', words=['we','can','start','now','then','check','again'],
            segments=[dict(start=0,end=7,language='vi' if index%2 else 'en')],
            eots=[3], unknown=[2], conflict_mask={'1':[],'2':[]}, source_rows=1))
    stages = []
    for index, members in enumerate([[0,1],[0,1,2],[0,1,2,3],[0,1,2,3]]):
        stages.append(dict(index=index, stream_indices=members, family_count=len(members),
            fraction_of_families=[.1,.25,.5,1.][index], update_budget=[1,2,3,None][index],
            eligible_points_per_pass=9*len(members)))
    plan = dict(stages=stages)
    original_save = trainer.atomic_torch
    results = []
    for name in ['uninterrupted', 'resumed']:
        out = fixture/name
        assert not out.exists(), 'Inspect an existing fixture before rerunning it'
        out.mkdir()
        data.OUT = trainer.OUT = out
        for asset in ['pretrained','tokenizer']:
            os.symlink(data.PARENT/asset, out/asset, target_is_directory=True)
        data.save('curriculum_plan.json', plan)
        with gzip.open(out/'train_streams.json.gz', 'wt', encoding='utf-8') as stream:
            json.dump(sources, stream)
        fake_manifest = dict(manifest)
        fake_manifest['files'] = {'train_streams.json.gz': data.sha(out/'train_streams.json.gz')}
        data.save('dataset_manifest.json', fake_manifest)
        data.save('migration_receipt.json', dict(technical_fixture=True))
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(parent['sampler_rng'])
        order = data.make_order(stages[0], rng)
        state = dict(parent)
        state.update(branch_step=0,parent_step=parent['step'],epoch=0,epoch_scored_seen=0,
            order=order,cursor=0,stream_state=None,sampler_rng=rng.bit_generator.state,
            curriculum_stage=0,curriculum_stage_step=0,curriculum_stage_seen=0,curriculum_stage_pass=0,
            cutover_partial_epoch=False,parent_uploaded_rows_seen=parent['uploaded_rows_seen'],
            best_key=(-1.,-1.,-1.),fingerprint=trainer.fingerprint(fake_manifest))
        folder=out/'runs/technical_fixture'
        original_save(folder/'latest_resume.pt', state)
        data.save('runs/technical_fixture/best_report.json', dict(technical_fixture=True))
        calls = 0
        def interruption():
            nonlocal calls
            calls += 1
            return name == 'resumed' and calls == 6
        with ExitStack() as stack:
            stack.enter_context(patch.object(torch.nn.Module, 'cuda', lambda self,*args,**kwargs:self))
            stack.enter_context(patch.object(torch.cuda, 'get_rng_state', lambda:parent['cuda_rng'].clone()))
            stack.enter_context(patch.object(torch.cuda, 'set_rng_state', lambda value:None))
            stack.enter_context(patch.object(torch.cuda, 'empty_cache', lambda:None))
            stack.enter_context(patch.object(torch, 'autocast', lambda *args,**kwargs:nullcontext()))
            for factory_name in ['zeros', 'as_tensor', 'tensor']:
                factory = getattr(torch, factory_name)
                def cpu_factory(*args, _factory=factory, **kwargs):
                    if kwargs.get('device') == 'cuda':
                        kwargs['device'] = 'cpu'
                    return _factory(*args, **kwargs)
                stack.enter_context(patch.object(torch, factory_name, cpu_factory))
            stack.enter_context(patch.object(trainer, 'stopped', interruption))
            try:
                trainer.fit('technical_fixture', fake_manifest, data.initialize(copy.deepcopy(sources)), [], [], {}, min_epochs=1)
            except InterruptedError:
                assert name == 'resumed'
                interrupted = torch.load(folder/'latest_resume.pt', map_location='cpu', weights_only=False, mmap=True)
                assert interrupted['curriculum_stage'] == 2 and interrupted['curriculum_stage_step'] == 1
                assert interrupted['stream_state']['slots']
                with patch.object(trainer, 'stopped', lambda:False):
                    trainer.fit('technical_fixture', fake_manifest, data.initialize(copy.deepcopy(sources)), [], [], {}, min_epochs=1)
            else:
                assert name == 'uninterrupted'
        result = torch.load(folder/'latest_resume.pt', map_location='cpu', weights_only=False, mmap=True)
        assert result['curriculum_stage'] == 3 and result['epoch'] == 1
        assert len(result['optimizer']['state']) == 27
        for index, expected in enumerate([1,2,3]):
            receipt=json.loads((folder/('curriculum_stage_%02d_complete.json'%index)).read_text())
            assert receipt['optimizer_updates'] == expected and not receipt['full_uploaded_epoch']
        coverage=json.loads((folder/'epoch_0001_coverage.json').read_text())
        assert coverage['scored_points'] == coverage['expected_eligible_points'] == 36
        results.append(result)
    keys=['model','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng','sampler_rng',
          'order','cursor','stream_state','step','branch_step','epoch','epoch_scored_seen',
          'curriculum_stage','curriculum_stage_step','curriculum_stage_seen','curriculum_stage_pass']
    assert all(trainer.equal(results[0][key],results[1][key]) for key in keys)
    data.OUT = trainer.OUT = actual_out
    proof=dict(complete=True, technical_fixture_only=True, cpu_fp32_execution=True,
               actual_fit_function_exercised=True, staged_budgets_verified=[1,2,3],
               masked_no_update_rows_exercised=True, active_lane_mid_stage_resume=True,
               full_epoch_eligible_gap_count_verified=36, optimizer_entries=27,
               uninterrupted_and_resumed_fields_bitwise_equal=keys,
               not_a_model_quality_or_deployment_result=True)
    data.save('technical_fixture_verification.json', proof)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    os.nice(10)
    run()
