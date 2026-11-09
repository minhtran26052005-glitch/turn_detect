"""Continuous, separate pretrained teachers. No student training before the gate."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
import zipfile

from .common import LABEL2ID, file_hash, input_id, read_jsonl, write_jsonl
from .continuous_v13_state import Store, atomic_json, singleton
from .teacher_first_protocol import LANGUAGES, source_identity, training_fingerprint, require_train_classes, distillation_gate

RUN = Path('outputs/teacher_first_v14_20261026')
OLD = Path('outputs/continuous_v13_20261026')
DATA = Path('data_v10/reward_20261025')
BASE = 'microsoft/mdeberta-v3-base'
REV = 'a0484667b22365f84929a935b5e50a51f71f159d'
STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def status(stage, **values):
    atomic_json(RUN / 'status.json', {'status': 'RUNNING', 'stage': stage, 'pid': os.getpid(),
                'heartbeat_utc': datetime.now(timezone.utc).isoformat(), 'teacher_first': True,
                'quality_target_certified': False, 'student_training_enabled': False, **values})


def bootstrap():
    if (RUN / 'state.sqlite').exists():
        return
    # Handoff is requested by the user; refuse to race a still-running GRU worker.
    launcher = json.loads((OLD / 'launcher.json').read_text())
    proc = Path('/proc') / str(launcher['pid']) / 'cmdline'
    assert not proc.exists() or b'turn_v2.continuous_v13' not in proc.read_bytes(), 'Old worker must finish its checkpoint first'
    source = sqlite3.connect('file:' + str(OLD / 'state.sqlite') + '?mode=ro', uri=True)
    destination = sqlite3.connect(RUN / 'state.sqlite')
    source.backup(destination); destination.close(); source.close()
    atomic_json(RUN / 'handoff.json', {'from': str(OLD), 'old_status': json.loads((OLD / 'status.json').read_text()),
                'reason': 'USER_REQUEST_PRETRAINED_LARGE_TEACHER_BEFORE_DISTILLATION',
                'old_models_and_logs_preserved': True, 'inherited_state_sha256': file_hash(RUN / 'state.sqlite')})


def collect(cycle, candidate_limit):
    """Qwen runs in this child and releases all GPU memory on child exit."""
    import torch
    from .auto_rl_v12 import GeneratedJudge
    from .auto_rl_v11 import PROBE, prefixes
    from .continuous_v13_r3 import Crawler, judged, visible_key
    store = Store(RUN / 'state.sqlite')
    if store.get('teacher_collect_done:' + str(cycle)):
        return
    status('LOADING_QWEN_FOR_DATA', cycle=cycle)
    judge = GeneratedJudge()
    probe = judge.score(PROBE)
    passed = all(r['judge_accepted'] and r['judge_label'] == r['expected'] for r in probe)
    atomic_json(RUN / 'cycles' / f'{cycle:06d}' / 'judge_probe.json', {'passed': passed, 'cases': probe})
    assert passed, 'Qwen probes failed; do not use its labels'
    crawler = Crawler(store)
    for lang in LANGUAGES:
        key = f'teacher_collect_done:{cycle}:{lang}'
        if store.get(key):
            continue
        status('COLLECTING_TEACHER_DATA', cycle=cycle, language=lang, pool_counts=store.counts())
        pages, errors = crawler.pages(lang, cycle, pages=6)
        candidates = [r for page in pages for r in prefixes(page) if store.allowed(visible_key(r), r['source_id'])]
        # Public transcript/replay text is rejudged. Old proxy labels are never
        # used as truth, and all inherited development/source exclusions apply.
        pool = list(read_jsonl(DATA / lang / 'train.jsonl'))
        cursor = store.get('teacher_text_cursor:' + lang, 0)
        start = cursor
        while len(candidates) < candidate_limit and cursor - start < len(pool):
            old = pool[cursor % len(pool)]; cursor += 1
            row = {**old, 'language': lang, 'source_id': source_identity(old)}
            if store.allowed(visible_key(row), row['source_id']):
                candidates.append(row)
        # Bound the observed window before judging so distant text is never
        # silently removed from a label that the student is asked to reproduce.
        bounded = []
        for row in candidates:
            row = {**row, 'context': row.get('context', '')[-600:], 'text': row['text'][-1200:]}
            # Prompt includes a long fixed rubric; leave explicit room for it.
            while len(judge.tokenizer.encode(json.dumps({'context': row['context'], 'text': row['text']}, ensure_ascii=False))) > 850:
                key = 'context' if len(row['context']) > len(row['text']) // 2 else 'text'
                row[key] = row[key][len(row[key]) // 2:]
            if store.allowed(visible_key(row), row['source_id']): bounded.append(row)
        candidates = bounded
        audit, accepted = judged(candidates, judge, store, 'train', cycle)
        folder = RUN / 'cycles' / f'{cycle:06d}'
        write_jsonl(folder / (lang + '_judge_audit.jsonl'), audit)
        atomic_json(folder / (lang + '_crawl.json'), {'sources': len(pages), 'candidates': len(candidates),
                    'accepted': len(accepted), 'errors': errors, 'labels': 'QWEN_SEMANTIC_SILVER_NOT_GOLD',
                    'upstream_proxy_labels_used_for_training': False})
        store.put('teacher_text_cursor:' + lang, cursor); store.put(key, True)
    store.put('teacher_collect_done:' + str(cycle), True)
    store.db.close(); del judge; gc.collect(); torch.cuda.empty_cache()


def encode(rows, tokenizer):
    import numpy as np
    from .monolingual_data import encode_tail
    ids = np.full((len(rows), 256), tokenizer.pad_token_id, np.int32)
    lengths = np.zeros(len(rows), np.int64)
    for i, row in enumerate(rows):
        value = encode_tail(tokenizer, row.get('context', ''), row['text'])['input_ids']
        ids[i, :len(value)] = value; lengths[i] = len(value)
    return ids, lengths, np.array([LABEL2ID[r['label']] for r in rows], np.int64)


def predict(model, encoded, batch_size=16):
    import numpy as np
    import torch
    from .metrics import probabilities
    ids, lengths, _ = encoded
    logits = np.zeros((len(ids), 2), np.float32); model.eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, len(ids), batch_size):
            stop = min(start + batch_size, len(ids)); width = int(lengths[start:stop].max())
            x = torch.from_numpy(ids[start:stop, :width].astype(np.int64)).cuda()
            mask = torch.arange(width, device='cuda')[None] < torch.from_numpy(lengths[start:stop]).cuda()[:, None]
            logits[start:stop] = model(input_ids=x, attention_mask=mask.long()).logits.float().cpu().numpy()
    return probabilities(logits)


def train_teacher(language, cycle, steps, microbatch):
    import numpy as np
    import torch
    import transformers.utils as hf_utils
    import transformers.utils.import_utils as hf_imports
    hf_utils.is_torchvision_available = hf_imports.is_torchvision_available = lambda: False
    hf_utils.is_librosa_available = hf_imports.is_librosa_available = lambda: False
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .metrics import classification_metrics, calibrate_on_dev
    torch.set_num_threads(2); torch.manual_seed(20261026 + LANGUAGES.index(language))
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'Use connected Colab GPU'
    store = Store(RUN / 'state.sqlite'); done = f'teacher_train_done:{cycle}:{language}'
    if store.get(done):
        return
    train = store.rows(language, 'train'); counts = require_train_classes(train)
    dev = store.rows(language, 'semantic_dev')
    floor = [r for r in read_jsonl(DATA / language / 'dev.jsonl') if r['evaluation_slice'] == 'natural']
    folder = RUN / 'teachers' / language; folder.mkdir(parents=True, exist_ok=True)
    status('LOADING_PRETRAINED_TEACHER', cycle=cycle, language=language, base=BASE, revision=REV)
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REV)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, revision=REV, num_labels=2,
             label2id=LABEL2ID, id2label={0: 'NEOT', 1: 'EOT'}, dtype=torch.float32).cuda()
    model.gradient_checkpointing_enable()
    params = list(model.named_parameters())
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in params if not n.startswith(('classifier.', 'pooler.'))], 'lr': 1e-5},
        {'params': [p for n, p in params if n.startswith(('classifier.', 'pooler.'))], 'lr': 5e-5}], weight_decay=.01)
    x, lengths, labels = encode(train, tokenizer)
    dev_encoded = encode(dev, tokenizer); floor_encoded = encode(floor, tokenizer)
    rng = np.random.default_rng(20261026 + cycle); total_steps = 0; start_step = 0
    fingerprint = training_fingerprint(train); checkpoint = folder / 'latest_resume.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        total_steps = saved['total_steps']
        if saved['cycle'] == cycle:
            assert saved['training_fingerprint'] == fingerprint, 'Training pool changed during an interrupted teacher phase'
            start_step = saved['cycle_step']; rng.bit_generator.state = saved['numpy_rng']
            torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
        del saved; gc.collect()
    identity = {'language': language, 'base': BASE, 'revision': REV, 'architecture': type(model).__name__,
                'parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'layers': model.config.num_hidden_layers, 'hidden_size': model.config.hidden_size,
                'training': 'FULL_PRETRAINED_FINE_TUNING_FP32_MASTER_BF16_AUTOCAST',
                'train_languages': [language], 'shared_trainable_weights': False,
                'dev_definition': 'INHERITED_FROZEN_SEMANTIC_QWEN_SILVER_SMALL_NOT_GOLD',
                'dev_cases': len(dev), 'natural_floor_cases': len(floor),
                'source_sha256': file_hash(Path(__file__)), 'input_codec': 'monolingual_data.encode_tail:256'}
    if not (folder / 'identity.json').exists():
        atomic_json(folder / 'identity.json', identity)

    def evaluate(stage):
        p = predict(model, dev_encoded); semantic = calibrate_on_dev(dev_encoded[2], p)
        observed = classification_metrics(floor_encoded[2], predict(model, floor_encoded), semantic['threshold'])
        value = {'language': language, 'cycle': cycle, 'stage': stage, 'total_steps': total_steps,
                 'semantic_dev': semantic, 'natural_floor_dev': observed, 'identity': identity,
                 'quality_target_certified': False, 'student_training_enabled': False}
        np.savez_compressed(RUN / 'cycles' / f'{cycle:06d}' / (language + '_' + stage + '_predictions.npz'),
                            semantic_labels=dev_encoded[2], semantic_prob=p)
        old = store.get('large_teacher:' + language)
        if old is None or semantic['joint_score'] > old['semantic_dev']['joint_score']:
            model.save_pretrained(folder / 'best', safe_serialization=True); tokenizer.save_pretrained(folder / 'best')
            value['path'] = str(folder / 'best'); value['weights_sha256'] = file_hash(folder / 'best/model.safetensors')
            store.put('large_teacher:' + language, value); atomic_json(folder / 'best_report.json', value)
        atomic_json(RUN / 'cycles' / f'{cycle:06d}' / (language + '_' + stage + '.json'), value)
        return value

    if not (folder / 'pretrained_baseline.json').exists():
        initial = evaluate('pretrained_baseline')
        initial['classification_head'] = 'NEW_RANDOM_EOT_HEAD_ON_PRETRAINED_ENCODER; NOT_TASK_PRETRAINED'
        atomic_json(folder / 'pretrained_baseline.json', initial)
    pools = {label: np.flatnonzero(labels == label) for label in [0, 1]}

    def save_resume(step):
        temporary = folder / 'latest_resume.tmp'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'cycle': cycle,
                    'cycle_step': step, 'total_steps': total_steps, 'training_fingerprint': fingerprint,
                    'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state()}, temporary)
        temporary.replace(checkpoint)
        atomic_json(folder / 'resume_receipt.json', {'cycle': cycle, 'cycle_step': step, 'total_steps': total_steps,
                    'training_fingerprint': fingerprint, 'bytes': checkpoint.stat().st_size,
                    'weights_and_optimizer_remote_only': True, 'checkpoint': str(checkpoint)})

    accumulation = 2
    torch.cuda.reset_peak_memory_stats()
    for step in range(start_step + 1, steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for _ in range(accumulation):
            index = np.concatenate([rng.choice(pools[label], microbatch // 2, replace=True) for label in [0, 1]])
            rng.shuffle(index); width = int(lengths[index].max())
            ids = torch.from_numpy(x[index, :width].astype(np.int64)).cuda()
            mask = torch.arange(width, device='cuda')[None] < torch.from_numpy(lengths[index]).cuda()[:, None]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask.long()).logits.float()
                loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(labels[index]).cuda(), label_smoothing=.02) / accumulation
            assert torch.isfinite(loss); loss.backward(); losses.append(float(loss.detach()) * accumulation)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); assert torch.isfinite(gradient)
        warmup = min(1., (total_steps + 1) / 20)
        for group, base in zip(optimizer.param_groups, [1e-5, 5e-5]):
            group['lr'] = base * warmup
        optimizer.step(); total_steps += 1
        if step % 5 == 0 or step == steps:
            record = {'cycle': cycle, 'language': language, 'step': step, 'steps': steps, 'total_steps': total_steps,
                      'loss': float(np.mean(losses)), 'train_cases': len(train), 'label_counts': counts,
                      'parameters': identity['parameters'], 'trainable_parameters': identity['trainable_parameters'],
                      'GPU_peak_GiB': torch.cuda.max_memory_allocated() / 2**30}
            atomic_json(folder / 'progress.json', record); status('TRAINING_LARGE_TEACHER', **record)
            print(json.dumps(record), flush=True)
        if step % 25 == 0 or step == steps or STOP:
            save_resume(step)
        if STOP:
            store.db.close(); return
    report = evaluate('trained'); store.put(done, True)
    atomic_json(folder / 'last_training_report.json', report)
    store.db.close(); del model, optimizer; gc.collect(); torch.cuda.empty_cache()


def metadata_snapshot(store, cycle):
    """Small recovery metadata; full teacher/optimizer stay on Colab unless Drive is mounted."""
    folder = RUN / 'recovery'; folder.mkdir(exist_ok=True); store.backup(folder / 'state_snapshot.sqlite')
    paths = [folder / 'state_snapshot.sqlite', RUN / 'config.json', RUN / 'status.json', RUN / 'handoff.json']
    paths += list((RUN / 'teachers').glob('*/*.json')) + list((RUN / 'teachers').glob('*/best/*.json'))
    paths += [Path(__file__), Path('turn_v2/teacher_first_protocol.py')]
    archive = folder / 'metadata_snapshot.zip'; temporary = folder / 'metadata_snapshot.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            if path.is_file(): z.write(path, str(path))
    temporary.replace(archive)
    atomic_json(folder / 'latest_snapshot.json', {'cycle': cycle, 'path': str(archive), 'bytes': archive.stat().st_size,
                'sha256': file_hash(archive), 'scope': 'METADATA_ONLY_NOT_A_FULL_TEACHER_RECOVERY',
                'teacher_weights_backed_up_externally': False})


def run_child(phase, cycle, language, args):
    command = [sys.executable, '-u', '-m', 'turn_v2.teacher_first_v14', '--phase', phase,
               '--cycle', str(cycle), '--steps', str(args.steps), '--microbatch', str(args.microbatch),
               '--candidates', str(args.candidates)]
    if language: command += ['--language', language]
    log = RUN / 'cycles' / f'{cycle:06d}' / ((language or 'data') + '.log')
    with log.open('a') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        atomic_json(RUN / 'child.json', {'pid': process.pid, 'command': command,
                    'start_ticks': (Path('/proc') / str(process.pid) / 'stat').read_text().split()[21], 'log': str(log)})
        sent_stop = False
        while process.poll() is None:
            if (RUN / 'STOP_REQUESTED').exists(): request_stop()
            if STOP and not sent_stop: process.send_signal(signal.SIGTERM); sent_stop = True
            time.sleep(2)
        if process.returncode and not STOP:
            raise RuntimeError(f'Child {phase}/{language} failed ({process.returncode}); see {log}')


def orchestrate(args):
    import shutil
    RUN.mkdir(parents=True, exist_ok=True)
    with singleton(RUN / 'worker.lock'):
        bootstrap(); store = Store(RUN / 'state.sqlite')
        config = {'base': BASE, 'revision': REV, 'languages': list(LANGUAGES), 'independent_checkpoints': True,
                  'parameters_expected': 'about278M each, exact count in teacher identity', 'max_cycles': 0,
                  'steps_per_language_per_cycle': args.steps, 'microbatch': args.microbatch, 'accumulation': 2,
                  'candidates_per_language': args.candidates, 'student_training_enabled': False,
                  'selection': 'FROZEN_SEMANTIC_SILVER_DEV', 'natural_floor_is_separate': True,
                  'qualification': 'Independent sealed EOT gold >=90 accuracy AND F1 required before distillation',
                  'weights_policy': 'Full teachers and optimizers remain remote; Windows receives metadata only',
                  'source_sha256': file_hash(Path(__file__))}
        if not (RUN / 'config.json').exists(): atomic_json(RUN / 'config.json', config)
        else: assert json.loads((RUN / 'config.json').read_text()) == config
        failures = 0
        while not STOP:
            try:
                if (RUN / 'STOP_REQUESTED').exists(): request_stop(); break
                if shutil.disk_usage(RUN).free < 25 * 2**30:
                    status('WAITING_FOR_REMOTE_DISK', free_GiB=shutil.disk_usage(RUN).free / 2**30); time.sleep(30); continue
                cycle = store.get('teacher_cycle', 1); (RUN / 'cycles' / f'{cycle:06d}').mkdir(parents=True, exist_ok=True)
                run_child('collect', cycle, None, args)
                for lang in LANGUAGES:
                    if STOP: break
                    run_child('train', cycle, lang, args)
                if STOP: break
                champions = {lang: store.get('large_teacher:' + lang, {}) for lang in LANGUAGES}
                qualifications = {}
                for lang in LANGUAGES:
                    path = RUN / 'qualifications' / (lang + '.json')
                    if path.exists(): qualifications[lang] = json.loads(path.read_text())
                gate = distillation_gate(champions, qualifications)
                atomic_json(RUN / 'distillation_gate.json', gate)
                summary = {'cycle': cycle, 'champions': champions, 'pool_counts': store.counts(),
                           'distillation_gate': gate, 'quality_target_certified': False, 'next_cycle': cycle + 1}
                atomic_json(RUN / 'cycles' / f'{cycle:06d}' / 'summary.json', summary)
                store.put('teacher_cycle', cycle + 1); status('TEACHER_CYCLE_COMPLETE_CONTINUING', **summary)
                metadata_snapshot(store, cycle); failures = 0
            except Exception as error:
                failures += 1; trace = traceback.format_exc()
                atomic_json(RUN / 'latest_error.json', {'at': datetime.now(timezone.utc).isoformat(),
                            'message': str(error), 'traceback': trace})
                status('RETRYING_TEACHER_ERROR', error=str(error), backoff_seconds=min(600, 30 * 2**min(failures, 4)))
                deadline = time.time() + min(600, 30 * 2**min(failures, 4))
                while time.time() < deadline and not STOP:
                    if (RUN / 'STOP_REQUESTED').exists(): request_stop()
                    time.sleep(2)
        atomic_json(RUN / 'status.json', {'status': 'STOPPED_BY_REQUEST', 'pid': os.getpid(), 'teacher_first': True})
        store.db.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['run', 'collect', 'train'], default='run')
    parser.add_argument('--language', choices=LANGUAGES); parser.add_argument('--cycle', type=int, default=1)
    parser.add_argument('--steps', type=int, default=100); parser.add_argument('--microbatch', type=int, default=16)
    parser.add_argument('--candidates', type=int, default=384); args = parser.parse_args()
    assert args.steps > 0 and args.microbatch > 0 and args.microbatch % 2 == 0
    for signum in [signal.SIGTERM, signal.SIGINT]: signal.signal(signum, request_stop)
    if args.phase == 'run': orchestrate(args)
    elif args.phase == 'collect': collect(args.cycle, args.candidates)
    else:
        assert args.language
        train_teacher(args.language, args.cycle, args.steps, args.microbatch)


if __name__ == '__main__': main()
