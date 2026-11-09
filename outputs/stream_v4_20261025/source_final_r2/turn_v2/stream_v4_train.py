"""Iterative causal-text teacher adaptation with honest development gates.

Never reports an independent >90% result when real VI gold is unavailable.
Each round persists a checkpoint, per-case predictions and slice metrics.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import LABEL2ID, file_hash, normalize, read_jsonl, write_json
from .metrics import probabilities, classification_metrics, calibrate_on_dev


def encode_stream(tokenizer, row, maximum=256):
    # Keep newest current text, not the oldest tokens of a growing stream.
    text = tokenizer.encode(normalize(row["text"]), add_special_tokens=False)
    current = text[-(maximum - 3):]
    context = tokenizer.encode(normalize(row.get("context", "")), add_special_tokens=False)
    budget = maximum - 3 - len(current)
    context = context[-budget:] if budget else []
    ids = [tokenizer.cls_token_id] + context + [tokenizer.sep_token_id] + current + [tokenizer.sep_token_id]
    return ids


def encode_rows(rows, tokenizer):
    ids = np.zeros((len(rows), 256), np.int32); lengths = np.zeros(len(rows), np.int16)
    for i, row in enumerate(rows):
        values = encode_stream(tokenizer, row); ids[i, :len(values)] = values; lengths[i] = len(values)
    return ids, lengths


def infer(model, ids, lengths, batch_size=64):
    result = np.zeros((len(ids), 2), np.float32)
    model.eval()
    order = np.argsort(lengths, kind="stable")
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            index = order[start:start + batch_size]; width = int(lengths[index].max())
            x = torch.from_numpy(ids[index, :width].astype(np.int64)).cuda()
            mask = torch.arange(width, device="cuda")[None] < torch.from_numpy(lengths[index].astype(np.int64)).cuda()[:, None]
            logits = model(input_ids=x, attention_mask=mask.long()).logits.float()
            if not torch.isfinite(logits).all():raise ValueError("Non-finite evaluation")
            result[index] = logits.cpu().numpy()
    return probabilities(result)


def summarize(rows, probs):
    labels = np.asarray([LABEL2ID[r["label"]] for r in rows])
    thresholds = {}; languages = {}; slices = {}
    for lang in ["vi", "en", "zh"]:
        index = np.array([r["language"] == lang for r in rows])
        primary = np.array([r["language"] == lang and (r["source"].startswith("livekit") if lang != "vi" else True) for r in rows])
        selected = calibrate_on_dev(labels[primary], probs[primary])
        thresholds[lang] = selected["threshold"]
        languages[lang] = classification_metrics(labels[index], probs[index], selected["threshold"])
        for source in sorted({r["source"] for r in rows if r["language"] == lang}):
            sub = np.array([r["language"] == lang and r["source"] == source for r in rows])
            slices[lang + ":" + source] = classification_metrics(labels[sub], probs[sub], selected["threshold"])
    score = min(m["joint_score"] for m in slices.values())
    return {"languages": languages, "slices": slices, "thresholds": thresholds, "minimum_slice_joint_score": score,
        "development_all_slices_over_90": all(m["accuracy"] > .9 and m["eot_f1"] > .9 for m in slices.values()),
        "independent_real_three_language_success": False, "acceptance_blocker": "No independent real-pause Vietnamese gold; legacy VI is a text-prefix proxy"}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(20261025); np.random.seed(20261025)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    start = time.monotonic()
    data_manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    for split in ["train", "dev"]:assert file_hash(args.data / (split + ".jsonl")) == data_manifest["split_sha256"][split]
    rows = list(read_jsonl(args.data / "train.jsonl")); dev = list(read_jsonl(args.data / "dev.jsonl"))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, dtype=torch.float32).cuda()
    x, lengths = encode_rows(rows, tok); dx, dl = encode_rows(dev, tok)
    labels = np.array([LABEL2ID[r["label"]] for r in rows])
    train_groups = defaultdict(list)
    for i, row in enumerate(rows):train_groups[(row["language"], row["source"])].append(i)
    train_groups = {k: np.array(v) for k, v in train_groups.items()}
    write_json(args.output / "identity.json", {"initial_checkpoint_sha256": file_hash(args.model / "model.safetensors"),
        "data_manifest_sha256": file_hash(args.data / "manifest.json"), "source_sha256": file_hash(Path(__file__)),
        "train_samples": len(rows), "dev_samples": len(dev), "rounds_max": args.rounds, "steps_per_round": args.steps,
        "input_contract": "tail of current text + budgeted context; no future text; 256tokens", "sealed_test_loaded": False})
    initial_probs = infer(model, dx, dl)
    initial = summarize(dev, initial_probs)
    write_json(args.output / "initial_dev.json", initial)
    np.savez_compressed(args.output / "initial_dev.npz", labels=np.array([LABEL2ID[r["label"]] for r in dev]), prob_EOT=initial_probs)
    print(json.dumps({"stage": "INITIAL_DEV", **initial}), flush=True)
    model.gradient_checkpointing_enable()
    best = initial["minimum_slice_joint_score"]; best_path = None; reports = []
    rng = np.random.default_rng(20261025)
    for round_id in range(1, args.rounds + 1):
        folder = args.output / f"round_{round_id:02d}"; folder.mkdir()
        lr = [1e-5, 5e-6, 2e-6][min(round_id - 1, 2)]
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
        losses = []
        # Every round has balanced language exposure. EN/ZH focus on real pause
        # development training rows; read-speech remains a separate slice.
        real_fraction = [.5, .8, .9][min(round_id - 1, 2)]
        round_best = -1.; evaluations = []
        for step in range(1, args.steps + 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                lang = ["vi", "en", "zh"][(2 * (step - 1) + micro) % 3]
                source = "vi_legacy_rehearsal" if lang == "vi" else ("livekit_PREVIOUSLY_EXPOSED_development" if rng.random() < real_fraction else "scicom_read_speech")
                pool = train_groups[(lang, source)]; index = rng.choice(pool, 16, replace=len(pool) < 16)
                width = int(lengths[index].max())
                ids = torch.from_numpy(x[index, :width].astype(np.int64)).cuda()
                mask = torch.arange(width, device="cuda")[None] < torch.from_numpy(lengths[index].astype(np.int64)).cuda()[:, None]
                y = torch.from_numpy(labels[index]).cuda()
                logits = model(input_ids=ids, attention_mask=mask.long()).logits.float()
                loss = F.cross_entropy(logits, y, label_smoothing=.01) / 2
                if not torch.isfinite(loss):raise ValueError("Non-finite training loss")
                loss.backward();losses.append(float(loss.detach()) * 2)
            factor = min(1., step / 40) * (.2 + .8 * .5 * (1 + math.cos(math.pi * step / args.steps)))
            for group in optimizer.param_groups:group["lr"] = lr * factor
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            if step % 50 == 0:
                progress = {"stage": "TRAINING", "round": round_id, "step": step, "steps_per_round": args.steps, "loss": float(np.mean(losses[-100:])), "elapsed_seconds": time.monotonic() - start}
                write_json(args.output / "progress.json", progress);print(json.dumps(progress), flush=True)
            if step % 200 == 0 or step == args.steps:
                probs = infer(model, dx, dl); metrics = summarize(dev, probs)
                record = {"round": round_id, "step": step, **metrics}; evaluations.append(record)
                write_json(folder / "evaluations.json", evaluations)
                print(json.dumps({"stage": "DEVELOPMENT_EVAL", **record}), flush=True)
                if metrics["minimum_slice_joint_score"] > round_best:
                    round_best = metrics["minimum_slice_joint_score"]
                    model.save_pretrained(folder / "best", safe_serialization=True);tok.save_pretrained(folder / "best")
                    write_json(folder / "best_dev.json", record)
                    np.savez_compressed(folder / "dev_predictions.npz", labels=np.array([LABEL2ID[r["label"]] for r in dev]), prob_EOT=probs)
        del optimizer
        record = json.loads((folder / "best_dev.json").read_text(encoding="utf-8"))
        reports.append(record)
        if round_best > best:
            best = round_best;best_path = str(folder / "best")
        write_json(args.output / "campaign.json", {"status": "ROUNDS_IN_PROGRESS", "rounds": reports, "best_path": best_path, "best_minimum_dev_score": best,
            "independent_real_three_language_success": False, "sealed_test_loaded": False})
        del model;gc.collect();torch.cuda.empty_cache()
        # Carry the best checkpoint of the just-completed round forward.
        model = AutoModelForSequenceClassification.from_pretrained(folder / "best", dtype=torch.float32).cuda()
        model.gradient_checkpointing_enable()
        if record["development_all_slices_over_90"]:break
    write_json(args.output / "campaign.json", {"status": "BOUNDED_ROUNDS_FINISHED_REAL_TARGET_NOT_CERTIFIED", "rounds": reports, "best_path": best_path,
        "best_minimum_dev_score": best, "independent_real_three_language_success": False, "sealed_test_loaded": False,
        "elapsed_seconds": time.monotonic() - start, "acceptance_blocker": "Independent real VI EOT labels unavailable; development plateau/quality below target may remain"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data_v4/causal_20261025"))
    p.add_argument("--model", type=Path, default=Path("runs/tri_teacher_20261024/best"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--steps", type=int, default=600)
    run(p.parse_args())
