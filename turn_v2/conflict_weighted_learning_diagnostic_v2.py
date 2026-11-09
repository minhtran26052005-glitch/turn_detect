"""Bounded memorization sanity check and paired learning-rate experiment."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import copy
import fcntl
import gc
import gzip
import hashlib
import json
import os
import random
import shutil
import time
import traceback

import numpy as np

from .uploaded_vi_en_data_v5 import OUT as MAIN, StreamOrder, initialize, sha
from .malbert_eval_zip_v2 import kind, metric
from .tiny_compression_v1.common import encode_pair

ROOT = MAIN.parents[1]
OUT = ROOT / 'outputs/malbert_learning_diagnostic_20261106_v1'
ARM_STEPS = 3000
SANITY_STEPS = 2000
PRECISION_TARGET = .50


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def status(stage, **values):
    raw = Path('/proc/self/stat').read_text()
    save('status.json', dict(stage=stage, checked_utc=datetime.now(timezone.utc).isoformat(),
         pid=os.getpid(), start_ticks=raw[raw.rfind(')')+2:].split()[19], **values))


def reserve():
    mount = next(row.split() for row in Path('/proc/mounts').read_text().splitlines()
                 if row.split()[1] == '/mnt/hdd500')
    assert mount[0] == '/dev/sda1' and 'rw' in mount[3].split(',')
    assert min(shutil.disk_usage(OUT).free, shutil.disk_usage('/mnt/hdd500').free) >= 13*2**30


def atomic_torch(path, value):
    import torch
    reserve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    with tmp.open('rb') as stream:
        os.fsync(stream.fileno())
    tmp.replace(path)


def parent_identity():
    launch = json.loads((MAIN / 'launcher.json').read_text())
    proc = Path('/proc') / str(launch['pid'])
    raw = (proc / 'stat').read_text()
    assert raw[raw.rfind(')')+2:].split()[19] == launch['start_ticks']
    assert launch['module'].encode() in (proc / 'cmdline').read_bytes().split(bytes([0]))
    assert launch['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    return launch


def rng_state():
    import torch
    return dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                numpy_rng=np.random.get_state(), python_rng=random.getstate())


def restore_rng(value):
    import torch
    torch.set_rng_state(value['torch_rng'])
    torch.cuda.set_rng_state(value['cuda_rng'])
    np.random.set_state(value['numpy_rng'])
    random.setstate(value['python_rng'])


def new_model():
    from transformers import AlbertConfig
    from .malbert_turn_student_v1 import MalbertTurnStudent
    model = MalbertTurnStudent(AlbertConfig.from_pretrained(MAIN / 'pretrained', local_files_only=True, token=False), sep_token_id=3)
    assert model.parameter_count() == 11941122
    return model.cuda()


def batch(sequences, labels=None):
    import torch
    width = max(map(len, sequences))
    ids = torch.zeros((len(sequences), width), dtype=torch.long, device='cuda')
    mask = torch.zeros_like(ids)
    for index, seq in enumerate(sequences):
        ids[index, :len(seq)] = torch.as_tensor(seq, device='cuda')
        mask[index, :len(seq)] = 1
    target = torch.tensor(labels, device='cuda') if labels is not None else None
    return ids, mask, target


def pin():
    import torch
    path = OUT / 'frozen/full_resume.pt'
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    path.parent.mkdir(exist_ok=True)
    parent_identity()
    with (MAIN / 'runs/finetune/latest_resume.pt').open('rb') as source, path.open('xb') as target:
        shutil.copyfileobj(source, target)
        target.flush()
        os.fsync(target.fileno())
    value = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    assert value['parameters'] == 11941122 and len(value['optimizer']['state']) == 27
    assert {int(v['step']) for v in value['optimizer']['state'].values()} == {value['step']}
    assert value['loss_policy'] == {'NEOT': 1., 'EOT': 2.}
    assert sha(MAIN / 'dataset_manifest.json') == value['fingerprint']['dataset_manifest_sha256']
    assert sha(MAIN / 'train_streams.json.gz') == value['fingerprint']['train_streams_sha256']
    for name, expected in value['fingerprint']['code_sha256'].items():
        assert sha(ROOT / 'turn_v2' / name) == expected
    save('checkpoint_receipt.json', dict(step=value['step'], branch_step=value['branch_step'],
         full_resume_sha256=sha(path), optimizer_entries=27, fingerprint=value['fingerprint'],
         preserved_source_order_cursor_lanes_and_all_rng=True, parameter_count=11941122,
         main_identity=parent_identity(), loss_weights=[1., 2.],
         paired_arm_steps=ARM_STEPS, paired_rates=[4e-6, 1e-5],
         sanity_rate=3e-5, sanity_max_steps=SANITY_STEPS,
         sanity_fresh_optimizer=True, precision_target_calibration=PRECISION_TARGET,
         selection_scope='Calibration DEV only. Monitoring families reported separately. TEST unused.'))
    return value


def select_fixed_sample(parent, streams, tokenizer):
    import torch
    from .uploaded_vi_en_evaluation_v5 import predictor
    path = OUT / 'fixed_train_sample.json'
    if path.exists():
        return json.loads(path.read_text())
    status('COLLECTING_FIXED_TRAIN_INPUTS')
    model = new_model()
    model.load_state_dict(parent['model'], strict=True)
    model.eval()
    predict = predictor(model, tokenizer)
    source_order = sorted(range(len(streams)), key=lambda i: hashlib.sha256(('sanity-v1:' + streams[i]['key']).encode()).digest())
    runner = StreamOrder(streams, np.asarray([2*i+1 for i in source_order]), lanes=32)
    pools = defaultdict(dict)
    labels_by_key = {}
    key_ownership = {}
    conflict_keys = set()
    target_counts = {(lang, label): 125 if label else 375 for lang in ['vi', 'en'] for label in [0, 1]}
    inferred = 0
    while any(len(pools[key]) < count+20 for key, count in target_counts.items()):
        rows, states = runner.rows()
        assert rows, 'Insufficient eligible TRAIN sample'
        probs, sequences = predict(rows)
        for row, seq in zip(rows, sequences):
            if not row['scored']:
                continue
            digest = hashlib.sha256(np.asarray(seq, dtype='<u4').tobytes()).hexdigest()
            label = row['label']
            previous = labels_by_key.setdefault(digest, label)
            if previous != label:
                conflict_keys.add(digest)
                for pool in pools.values():
                    pool.pop(digest, None)
                continue
            if digest in conflict_keys:
                continue
            key = row['language'], label
            if key_ownership.setdefault(digest, key) != key:
                continue
            if len(pools[key]) < target_counts[key]+20:
                pools[key].setdefault(digest, dict(input_ids=seq, label=label, language=row['language'],
                    stream=row['stream'], boundary=row['boundary'], context_start=row['start'],
                    future=row['text'], context=row['context'], encoded_sha256=digest,
                    origin=row['origin']))
        runner.advance(states, probs, parent['best_threshold'])
        inferred += len(rows)
        if inferred % 1000 < len(rows):
            status('COLLECTING_FIXED_TRAIN_INPUTS', inferences=inferred,
                   buckets={str(k):len(v) for k,v in pools.items()})
    sample = []
    for key, count in target_counts.items():
        sample.extend(sorted(pools[key].values(), key=lambda r:r['encoded_sha256'])[:count])
    assert len(sample) == 1000 and len({r['encoded_sha256'] for r in sample}) == 1000
    for row in sample:
        stream = streams[row['stream']]
        boundary = row['boundary']
        assert boundary not in stream['_unknown'] and boundary not in stream['conflict_mask']['2']
        assert row['label'] == int(boundary in stream['_eots'])
        seq = encode_pair(tokenizer, row['context'], row['future'], 256)['input_ids']
        assert seq == row['input_ids']
    save('fixed_train_sample.json', sample)
    save('fixed_sample_receipt.json', dict(complete=True, rows=1000, EOT_per_language=125,
         NEOT_per_language=375, conflicts_encountered_and_removed=len(conflict_keys),
         original_causal_collection_inferences=inferred, labels_and_encoding_verified=True,
         sample_is_purposefully_class_stratified_not_natural_distribution=True,
         oracle_reference_reset=False, fixed_inputs_preserved_for_memorization=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return sample


def fixed_metrics(model, sample):
    import torch
    from torch.nn import functional as F
    model.eval()
    counts = defaultdict(Counter)
    correct = 0
    losses = []
    for start in range(0, len(sample), 32):
        part = sample[start:start+32]
        ids, mask, target = batch([r['input_ids'] for r in part], [r['label'] for r in part])
        with torch.inference_mode():
            logits = model(ids, mask).float()
            predictions = logits.argmax(-1).cpu().tolist()
            losses.extend(F.cross_entropy(logits, target, reduction='none').cpu().tolist())
        for row, prediction in zip(part, predictions):
            counts[row['language']][kind(bool(row['label']), bool(prediction))] += 1
            correct += int(prediction == row['label'])
    metrics = {lang:metric(c) for lang,c in counts.items()}
    passed = correct/len(sample) >= .98 and all(m['eot_f1'] >= .95 and m['eot_recall'] >= .95 for m in metrics.values())
    return dict(accuracy=correct/len(sample), mean_unweighted_CE=float(np.mean(losses)),
                by_language=metrics, memorization_passed=passed, decision_threshold=.5)


def sanity(parent, sample):
    import torch
    from torch.nn import functional as F
    if (OUT / 'sanity/report.json').exists():
        return json.loads((OUT / 'sanity/report.json').read_text())
    model = new_model()
    model.load_state_dict(parent['model'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=.01)
    torch.manual_seed(4731)
    torch.cuda.manual_seed_all(4731)
    rng = np.random.default_rng(4731)
    order = rng.permutation(len(sample))
    cursor = 0
    initial = fixed_metrics(model, sample)
    history = [dict(step=0, **initial)]
    first_gradient_check = {}
    path = OUT / 'sanity/latest_resume.pt'
    start_step = 0
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state)
        rng.bit_generator.state = state['sample_rng']
        order, cursor, start_step = state['order'], state['cursor'], state['step']
        history = state['history']
    result = initial
    for step in range(start_step+1, SANITY_STEPS+1):
        if cursor+32 > len(order):
            order = rng.permutation(len(sample))
            cursor = 0
        part = [sample[int(i)] for i in order[cursor:cursor+32]]
        cursor += 32
        model.train()
        optimizer.zero_grad(set_to_none=True)
        denominator = sum(2. if row['label'] else 1. for row in part)
        loss_value = 0.
        for offset in range(0, len(part), 4):
            micro = part[offset:offset+4]
            ids, mask, labels = batch([r['input_ids'] for r in micro], [r['label'] for r in micro])
            logits = model(ids, mask).float()
            loss = F.cross_entropy(logits, labels, weight=logits.new_tensor([1.,2.]), reduction='sum') / denominator
            assert bool(torch.isfinite(loss))
            loss.backward()
            loss_value += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert bool(torch.isfinite(norm))
        if step == 1:
            first_gradient_check = dict(classifier_nonzero=bool(model.classifier.weight.grad.abs().sum()>0),
                backbone_nonzero=any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.backbone.parameters()),
                finite_grad_norm=float(norm))
            assert first_gradient_check['classifier_nonzero'] and first_gradient_check['backbone_nonzero']
        optimizer.step()
        if step % 50 == 0:
            status('SANITY_MEMORIZATION_TRAINING', step=step, budget=SANITY_STEPS, loss=loss_value)
        if step % 200 == 0:
            result = fixed_metrics(model, sample)
            history.append(dict(step=step, **result))
            atomic_torch(path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                step=step, order=order, cursor=cursor, sample_rng=rng.bit_generator.state,
                history=history, **rng_state()))
            save('sanity/progress.json', dict(step=step, initial=initial, current=result, first_gradient_check=first_gradient_check))
            if result['memorization_passed']:
                break
    report = dict(complete=True, initial=initial, final=result, steps=step, history=history,
                  first_gradient_check=first_gradient_check, fresh_optimizer=True, lr=3e-5,
                  fixed_sample_rows=1000, effective_batch=32, gradient_microbatch=4, dropout_enabled_during_fit=True,
                  threshold=.5, weighted_CE=[1.,2.], memorization_passed=result['memorization_passed'])
    save('sanity/report.json', report)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return report


def train_arm(name, rate, parent, streams, tokenizer):
    import torch
    from torch.nn import functional as F
    folder = OUT / name
    if (folder / 'training_complete.json').exists():
        return folder
    model = new_model()
    model.load_state_dict(parent['model'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=.01)
    optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
    for group in optimizer.param_groups:
        group['lr'] = rate
    runner = StreamOrder(streams, parent['order'], copy.deepcopy(parent['stream_state']), lanes=16)
    restore_rng(parent)
    sample_rng = np.random.default_rng()
    sample_rng.bit_generator.state = copy.deepcopy(parent['sampler_rng'])
    step = 0
    scored_seen = 0
    losses = []
    path = folder / 'latest_resume.pt'
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        runner = StreamOrder(streams, state['order'], state['stream_state'], lanes=16)
        restore_rng(state)
        sample_rng.bit_generator.state = state['sampler_rng']
        step, scored_seen = state['step'], state['scored_seen']
    initial_lr = optimizer.param_groups[0]['lr']
    assert initial_lr == rate
    while step < ARM_STEPS:
        model.train()
        rows, states = runner.rows()
        if not rows:
            runner = StreamOrder(streams, sample_rng.permutation(len(streams)*2), lanes=16)
            continue
        seqs = [encode_pair(tokenizer, r['context'], r['text'], 256)['input_ids'] for r in rows]
        ids, mask, labels = batch(seqs, [r['label'] for r in rows])
        scored = torch.tensor([r['scored'] for r in rows], dtype=torch.bool, device='cuda')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(ids, mask)
        logits = logits.float()
        probs = logits.detach().softmax(-1)[:,1].cpu().tolist()
        if bool(scored.any()):
            loss = F.cross_entropy(logits[scored], labels[scored], weight=logits.new_tensor([1.,2.]))
            assert bool(torch.isfinite(loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            step += 1
            scored_seen += int(scored.sum())
            losses.append(float(loss.detach()))
        runner.advance(states, probs, parent['best_threshold'])
        if step % 100 == 0:
            reserve()
            status('PAIRED_LR_TRAINING', arm=name, lr=rate, step=step, budget=ARM_STEPS, scored_seen=scored_seen)
        if step % 500 == 0:
            atomic_torch(path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                scored_seen=scored_seen, order=runner.order, stream_state=runner.state(),
                sampler_rng=sample_rng.bit_generator.state, parent_step=parent['step'],
                parent_fingerprint=parent['fingerprint'], lr=rate, **rng_state()))
    atomic_torch(folder / 'weights.pt', model.state_dict())
    assert {int(v['step']) for v in optimizer.state.values()} == {parent['step']+ARM_STEPS}
    save(name+'/training_complete.json', dict(complete=True, lr=rate, updates=step, scored_seen=scored_seen,
         parent_step=parent['step'], global_optimizer_step=parent['step']+step, optimizer_entries=len(optimizer.state),
         mean_training_CE=float(np.mean(losses)), weights_sha256=sha(folder/'weights.pt'),
         inherited_model_optimizer_order_cursor_lanes_all_rng=True,
         comparison_scope='Same initial checkpoint, optimizer, RNG and source order. LR changes predictions, so later predicted-reset contexts may diverge.',
         source_threshold_during_training=parent['best_threshold']))
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return folder


def evaluate_arm(name, weights, calibration, monitor):
    import torch
    from .uploaded_vi_en_evaluation_v5 import predictor, rollout
    path = OUT / name / 'dev_report.json'
    if path.exists():
        return json.loads(path.read_text())
    model = new_model()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    predict = predictor(model, tokenizer_global)
    curves = []
    for threshold in np.linspace(.1,.9,9):
        cache = OUT / name / ('calibration_%.1f.json'%threshold)
        if cache.exists():
            result = json.loads(cache.read_text())
        else:
            status('CALIBRATING_PAIRED_DEV', arm=name, threshold=float(threshold))
            result = rollout(predict, calibration, float(threshold))
            save(name+'/'+cache.name, result)
        curves.append(result)
    eligible = [r for r in curves if all((m['eot_precision'] or 0) >= PRECISION_TARGET for m in r['by_language'].values())]
    if eligible:
        selected = max(eligible, key=lambda r:min(m['eot_recall'] or 0 for m in r['by_language'].values()))
        selection_status = 'CALIBRATION_PRECISION_TARGET_MET'
    else:
        selected = max(curves, key=lambda r:min(m['eot_precision'] or 0 for m in r['by_language'].values()))
        selection_status = 'NO_THRESHOLD_MEETS_CALIBRATION_TARGET_DIAGNOSTIC_FALLBACK'
    status('SCORING_SEPARATE_DEV_MONITOR', arm=name, threshold=selected['threshold'])
    result = rollout(predict, monitor, selected['threshold'])
    report = dict(complete=True, arm=name, precision_target=PRECISION_TARGET,
        selection_status=selection_status, selected_threshold=selected['threshold'],
        calibration_selected=selected['by_language'], calibration_curve=[dict(threshold=r['threshold'],by_language=r['by_language']) for r in curves],
        monitor=result, weights_sha256=sha(weights), target_chosen_before_any_arm_scoring=True,
        monitor_used_to_select_threshold=False, test_used=False,
        precision_matching_scope='Same 50% minimum per-language CALIBRATION precision. Actual monitoring precision may differ and is reported, not assumed equal.')
    save(name+'/dev_report.json', report)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run():
    import torch
    import transformers.utils as u
    import transformers.utils.import_utils as im
    u.is_torchvision_available = im.is_torchvision_available = lambda: False
    u.is_librosa_available = im.is_librosa_available = lambda: False
    from transformers import AutoTokenizer
    global tokenizer_global
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.25)
    assert not (OUT / 'report.json').exists(), 'Inspect the completed diagnostic instead of rerunning'
    reserve()
    parent = pin()
    tokenizer_global = AutoTokenizer.from_pretrained(MAIN / 'tokenizer', local_files_only=True, token=False)
    streams = initialize(json.load(gzip.open(MAIN / 'train_streams.json.gz', 'rt', encoding='utf-8')))
    sample = select_fixed_sample(parent, streams, tokenizer_global)
    sanity_result = sanity(parent, sample)
    if not sanity_result['memorization_passed']:
        save('report.json', dict(complete=True, stage='SANITY_FAILED_PAIRED_LR_NOT_STARTED', sanity=sanity_result,
             diagnosis='Fixed-input memorization criterion was not reached within the predeclared budget. Inspect label/encoding/gradient/optimization diagnostics before LR comparison.'))
        status('COMPLETE_SANITY_FAILED')
        return
    base = OUT / 'baseline/weights.pt'
    if not base.exists():
        atomic_torch(base, parent['model'])
    folders = []
    for name, rate in [('lr4e6',4e-6),('lr1e5',1e-5)]:
        folders.append((name, train_arm(name, rate, parent, streams, tokenizer_global)))
    calibration = initialize(json.load(gzip.open(MAIN/'dev_calibration.json.gz','rt',encoding='utf-8')))
    monitor = initialize(json.load(gzip.open(MAIN/'dev_monitor.json.gz','rt',encoding='utf-8')))
    reports = [evaluate_arm('baseline',base,calibration,monitor)]
    reports.extend(evaluate_arm(name,folder/'weights.pt',calibration,monitor) for name,folder in folders)
    assert parent_identity()['pid'] == json.loads((OUT/'checkpoint_receipt.json').read_text())['main_identity']['pid']
    assert sha(MAIN/'dataset_manifest.json') == parent['fingerprint']['dataset_manifest_sha256']
    save('report.json', dict(complete=True, sanity=sanity_result, parent_step=parent['step'],
         parent_sha256=sha(OUT/'frozen/full_resume.pt'), paired_updates=ARM_STEPS,
         loss_weights=[1.,2.], precision_target_on_calibration=PRECISION_TARGET,
         reports=reports, main_training_continues=True, auto_applied_to_main=False,
         conclusion_scope='Bounded LR experiment. No TEST was scored. No production/default model replacement.'))
    status('COMPLETE')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    os.nice(10)
    with (OUT/'EXPERIMENT_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run()
        except Exception as error:
            save('error.json',dict(error=repr(error),traceback=traceback.format_exc()))
            status('ERROR',error=repr(error))
            raise
