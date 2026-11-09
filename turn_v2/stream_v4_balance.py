"""Predeclared class-balanced follow-up, selected on development only.

The preceding campaign and all held/fresh data stay untouched. The previous
winner remains a candidate; a follow-up is retained only on improved dev score.
"""
import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import LABEL2ID, file_hash, read_jsonl, write_json
from .stream_v4_train import encode_rows, infer, summarize


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(20261026)
    rng = np.random.default_rng(20261026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    campaign = json.loads((args.previous / 'campaign.json').read_text())
    base = Path(campaign['best_path'])
    assert base.is_dir()
    manifest = json.loads((args.data / 'manifest.json').read_text())
    for split in ['train', 'dev']:
        assert file_hash(args.data / (split + '.jsonl')) == manifest['split_sha256'][split]
    train = list(read_jsonl(args.data / 'train.jsonl'))
    dev = list(read_jsonl(args.data / 'dev.jsonl'))
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, dtype=torch.float32).cuda()
    x, lengths = encode_rows(train, tokenizer)
    dx, dl = encode_rows(dev, tokenizer)
    labels = np.array([LABEL2ID[r['label']] for r in train], dtype=np.int64)
    groups = defaultdict(list)
    for i, row in enumerate(train):
        groups[(row['language'], row['source'], row['label'])].append(i)
    groups = {key: np.array(value) for key, value in groups.items()}
    initial_probs = infer(model, dx, dl)
    initial = summarize(dev, initial_probs)
    write_json(args.output / 'initial_dev.json', initial)
    best = initial['minimum_slice_joint_score']
    best_path = str(base)
    best_record = initial
    history = []
    write_json(args.output / 'identity.json', {
        'initial_path': str(base), 'initial_sha256': file_hash(base / 'model.safetensors'),
        'data_manifest_sha256': file_hash(args.data / 'manifest.json'),
        'source_sha256': file_hash(Path(__file__)), 'rounds_max': args.rounds,
        'steps_per_round': args.steps, 'microbatch': 16, 'accumulation': 2,
        'sampling': 'equal languages; EN/ZH 80% conversational train, 20% read speech; equal EOT/NEOT within source',
        'selection': 'same minimum development slice accuracy/F1; preceding control remains eligible',
        'held_or_fresh_inputs_loaded': False, 'old_holdout_loaded': False,
        'early_stop': 'three consecutive dev evaluations without strict improvement',
    })
    model.gradient_checkpointing_enable()
    stale = 0
    reason = 'MAX_ROUNDS'
    for round_id in range(1, args.rounds + 1):
        folder = args.output / f'round_{round_id:02d}'
        folder.mkdir()
        lr = [2e-6, 1e-6][round_id - 1]
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
        for step in range(1, args.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro in range(2):
                lang = ['vi', 'en', 'zh'][(2 * (step - 1) + micro) % 3]
                source = 'vi_legacy_rehearsal' if lang == 'vi' else (
                    'livekit_PREVIOUSLY_EXPOSED_development' if rng.random() < .8 else 'scicom_read_speech')
                index = np.concatenate([rng.choice(groups[(lang, source, label)], 8,
                    replace=len(groups[(lang, source, label)]) < 8) for label in ['EOT', 'NEOT']])
                rng.shuffle(index)
                width = int(lengths[index].max())
                ids = torch.from_numpy(x[index, :width].astype(np.int64)).cuda()
                mask = torch.arange(width, device='cuda')[None] < torch.from_numpy(lengths[index].astype(np.int64)).cuda()[:, None]
                logits = model(input_ids=ids, attention_mask=mask.long()).logits.float()
                loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(labels[index]).cuda(), label_smoothing=.02) / 2
                assert torch.isfinite(loss)
                loss.backward()
                losses.append(float(loss.detach()) * 2)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            factor = min(1., step / 30) * (.2 + .8 * .5 * (1 + math.cos(math.pi * step / args.steps)))
            for group in optimizer.param_groups:
                group['lr'] = lr * factor
            optimizer.step()
            if step % 50 == 0:
                progress = {'stage': 'BALANCED_TRAINING', 'round': round_id, 'step': step,
                    'steps_per_round': args.steps, 'loss': float(np.mean(losses)), 'elapsed_seconds': time.monotonic() - started}
                write_json(args.output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
            if step % 100 == 0 or step == args.steps:
                probs = infer(model, dx, dl)
                record = {'round': round_id, 'step': step, **summarize(dev, probs)}
                history.append(record)
                write_json(args.output / 'history.json', history)
                if record['minimum_slice_joint_score'] > best:
                    best = record['minimum_slice_joint_score']
                    best_path = str(folder / 'best')
                    best_record = record
                    model.save_pretrained(best_path, safe_serialization=True)
                    tokenizer.save_pretrained(best_path)
                    write_json(folder / 'best_dev.json', record)
                    np.savez_compressed(folder / 'dev_predictions.npz', labels=np.array([LABEL2ID[r['label']] for r in dev]), prob_EOT=probs)
                    stale = 0
                else:
                    stale += 1
                print(json.dumps({'stage': 'BALANCED_DEV', **record}), flush=True)
                if record['development_all_slices_over_90'] or stale >= 3:
                    reason = 'DEV_TARGET' if record['development_all_slices_over_90'] else 'DEV_PLATEAU'
                    break
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        if reason != 'MAX_ROUNDS':
            break
        model = AutoModelForSequenceClassification.from_pretrained(best_path, dtype=torch.float32).cuda()
        model.gradient_checkpointing_enable()
    write_json(args.output / 'report.json', {'status': 'FOLLOWUP_FINISHED_REAL_TARGET_NOT_CERTIFIED',
        'stop_reason': reason, 'best_path': best_path, 'best_minimum_dev_score': best,
        'best_record': best_record, 'history': history,
        'checkpoint_sha256': file_hash(Path(best_path) / 'model.safetensors'),
        'elapsed_seconds': time.monotonic() - started, 'held_or_fresh_inputs_loaded': False,
        'all_real_languages_over_90': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data_v4/causal_20261025'))
    p.add_argument('--previous', type=Path, default=Path('runs/stream_v4_campaign_20261025'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int, choices=[1, 2], default=2)
    p.add_argument('--steps', type=int, default=400)
    run(p.parse_args())
