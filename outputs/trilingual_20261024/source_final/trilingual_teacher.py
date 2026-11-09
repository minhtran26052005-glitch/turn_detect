"""Train a bounded 12-layer multilingual teacher, then cache train-only KD targets."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import LABEL2ID, encode_pair, file_hash, read_jsonl, write_json
from .distill_teacher import restore_teacher, padded
from .metrics import calibrate_on_dev, classification_metrics, probabilities


def metrics_by_language(labels, probs, languages, thresholds=None):
    result = {}
    for language in sorted(set(languages.tolist())):
        selected = languages == language
        result[language] = (classification_metrics(labels[selected], probs[selected], thresholds[language]) if thresholds
                            else calibrate_on_dev(labels[selected], probs[selected]))
    return result


def encoded(rows, tokenizer):
    ids = np.zeros((len(rows), 256), np.int32)
    lengths = np.zeros(len(rows), np.int16)
    for i, row in enumerate(rows):
        value = encode_pair(tokenizer, row.get("context", ""), row["text"], 256)["input_ids"]
        ids[i, :len(value)] = value
        lengths[i] = len(value)
    return ids, lengths, np.array([LABEL2ID[r["label"]] for r in rows]), np.array([r["language"] for r in rows])


def predict(model, ids, lengths, device, batch_size=64):
    logits = np.zeros((len(ids), 2), np.float32)
    order = np.argsort(lengths, kind="stable")
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            index = order[start:start + batch_size]
            length = int(lengths[index].max())
            x = torch.from_numpy(ids[index, :length].astype(np.int64)).to(device)
            mask = torch.arange(length, device=device)[None] < torch.from_numpy(lengths[index].astype(np.int64)).to(device)[:, None]
            logits[index] = model(input_ids=x, attention_mask=mask.long()).logits.float().cpu().numpy()
    return probabilities(logits)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(20261024)
    np.random.seed(20261024)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Teacher pilot requires the connected Colab GPU"
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    for split in ["train", "dev"]:
        assert file_hash(args.data / (split + ".jsonl")) == manifest["split_sha256"][split]
    from huggingface_hub import model_info
    revision = model_info("microsoft/mdeberta-v3-base", token=False).sha
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", revision=revision)
    print(json.dumps({"stage": "loading_full_multilingual_base", "revision": revision}), flush=True)
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/mdeberta-v3-base", revision=revision,
        num_labels=2, id2label={0: "NEOT", 1: "EOT"}, label2id=LABEL2ID, dtype=torch.float32)
    compact = restore_teacher(args.control)
    state = compact.state_dict()
    embedding_name = "deberta.embeddings.word_embeddings.weight"
    missing, unexpected = model.load_state_dict({k: v for k, v in state.items() if k != embedding_name}, strict=False)
    assert missing == [embedding_name] and not unexpected
    compact_tokenizer = AutoTokenizer.from_pretrained(args.control)
    vocabulary = compact_tokenizer.get_vocab()
    full_vocab = tokenizer.get_vocab()
    remap = [None] * compact.config.vocab_size
    for piece, index in vocabulary.items():
        remap[index] = full_vocab[piece]
    assert all(v is not None for v in remap)
    with torch.no_grad():
        model.deberta.embeddings.word_embeddings.weight[torch.tensor(remap)] = compact.deberta.embeddings.word_embeddings.weight
    # Same compact IDs mapped back to full vocabulary must retain all VI weights.
    checks = []
    initial_rows = list(read_jsonl(args.data / "dev.jsonl"))[:8]
    compact.eval(); model.eval()
    with torch.inference_mode():
        for row in initial_rows:
            arrays = padded([encode_pair(compact_tokenizer, row.get("context", ""), row["text"], 256)])
            a = compact(**{k: torch.from_numpy(v) for k, v in arrays.items()}).logits.numpy()
            mapped = {k: torch.from_numpy(np.asarray(remap, np.int64)[v] if k == "input_ids" else v) for k, v in arrays.items()}
            b = model(**mapped).logits.numpy()
            checks.append(float(np.abs(probabilities(a) - probabilities(b)).max()))
    assert max(checks) < 1e-5
    write_json(args.output / "initialization.json", {"base_revision": revision, "control_sha256": file_hash(args.control / "model.onnx"),
        "full_vocab_size": model.config.vocab_size, "restored_control_rows": len(remap), "mapped_input_probability_delta": max(checks),
        "initialization": "Restore VI 12-layer encoder/head and known embedding rows; unused full-vocab rows from pretrained mDeBERTa"})
    del compact, state
    gc.collect()
    train_rows = list(read_jsonl(args.data / "train.jsonl"))
    dev_rows = list(read_jsonl(args.data / "dev.jsonl"))
    train_ids, train_len, labels, languages = encoded(train_rows, tokenizer)
    dev_ids, dev_len, dev_labels, dev_languages = encoded(dev_rows, tokenizer)
    model = model.to(device)
    initial = metrics_by_language(dev_labels, predict(model, dev_ids, dev_len, device), dev_languages)
    write_json(args.output / "initial_dev.json", initial)
    params = [{"params": [p for n, p in model.named_parameters() if not n.startswith(("classifier.", "pooler."))], "lr": args.lr},
              {"params": [p for n, p in model.named_parameters() if n.startswith(("classifier.", "pooler."))], "lr": args.lr * 5}]
    optimizer = torch.optim.AdamW(params, weight_decay=.01)
    model.gradient_checkpointing_enable()
    rng = np.random.default_rng(20261024)
    orders = {lang: np.flatnonzero(languages == lang) for lang in ["vi", "en", "zh"]}
    best = -1.
    history = []
    losses = []
    # Choose language per microbatch to keep scripts similarly padded, balanced
    # across the three languages independently of their available row counts.
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for micro in range(2):
            lang = ["vi", "en", "zh"][(2 * (step - 1) + micro) % 3]
            index = rng.choice(orders[lang], 16, replace=False)
            length = int(train_len[index].max())
            x = torch.from_numpy(train_ids[index, :length].astype(np.int64)).to(device)
            mask = torch.arange(length, device=device)[None] < torch.from_numpy(train_len[index].astype(np.int64)).to(device)[:, None]
            y = torch.from_numpy(labels[index]).to(device)
            logits = model(input_ids=x, attention_mask=mask.long()).logits
            loss = F.cross_entropy(logits.float(), y, label_smoothing=.02) / 2
            assert torch.isfinite(loss)
            loss.backward()
            losses.append(float(loss.detach()) * 2)
        factor = min(1., step / 50) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        optimizer.param_groups[0]["lr"] = args.lr * factor
        optimizer.param_groups[1]["lr"] = args.lr * 5 * factor
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 25 == 0:
            progress = {"stage": "TEACHER_TRAINING", "step": step, "total_steps": args.steps, "loss": float(np.mean(losses[-50:])), "elapsed_seconds": time.monotonic() - started}
            write_json(args.output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        if step % 200 == 0 or step == args.steps:
            probs = predict(model, dev_ids, dev_len, device)
            metrics = metrics_by_language(dev_labels, probs, dev_languages)
            score = min(m["joint_score"] for m in metrics.values())
            history.append({"step": step, "dev": metrics, "minimum_language_joint_score": score})
            write_json(args.output / "history.json", history)
            print(json.dumps(history[-1]), flush=True)
            if score > best:
                best = score
                model.save_pretrained(args.output / "best", safe_serialization=True)
                tokenizer.save_pretrained(args.output / "best")
                np.savez_compressed(args.output / "dev_predictions.npz", labels=dev_labels, prob_EOT=probs, language=dev_languages)
                write_json(args.output / "best_dev.json", history[-1])
    del optimizer, model
    gc.collect(); torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(args.output / "best", dtype=torch.float32).to(device).eval()
    best_dev = json.loads((args.output / "best_dev.json").read_text(encoding="utf-8"))
    contract = {"schema": 1, "max_length": 256, "threshold": .5,
        "language_thresholds": {lang: m["threshold"] for lang, m in best_dev["dev"].items()},
        "normalization": "NFKC lowercase punctuation-to-space collapse-whitespace", "label2id": LABEL2ID,
        "input_format": "[CLS] context [SEP] text [SEP]; preserve text first", "smoke_only": False,
        "quality_scope": manifest["scope"], "quality_status": "STRUCTURAL_PROXY_PILOT_ONLY"}
    write_json(args.output / "best/turn_detector_config.json", contract)
    cache = args.output / "cache"
    cache.mkdir()
    np.save(cache / "train_ids.npy", train_ids); np.save(cache / "train_lengths.npy", train_len)
    np.save(cache / "train_labels.npy", labels); np.save(cache / "train_language.npy", languages)
    np.save(cache / "dev_ids.npy", dev_ids); np.save(cache / "dev_lengths.npy", dev_len)
    np.save(cache / "dev_labels.npy", dev_labels); np.save(cache / "dev_language.npy", dev_languages)
    captured = []
    hook = model.pooler.register_forward_hook(lambda module, inputs, output: captured.append(output.detach()))
    teacher_logits = np.lib.format.open_memmap(cache / "teacher_logits.npy", mode="w+", dtype=np.float32, shape=(len(labels), 2))
    teacher_hidden = np.lib.format.open_memmap(cache / "teacher_hidden.npy", mode="w+", dtype=np.float16, shape=(len(labels), 768))
    order = np.argsort(train_len, kind="stable")
    with torch.inference_mode():
        for start in range(0, len(labels), 64):
            index = order[start:start + 64]
            length = int(train_len[index].max())
            x = torch.from_numpy(train_ids[index, :length].astype(np.int64)).to(device)
            mask = torch.arange(length, device=device)[None] < torch.from_numpy(train_len[index].astype(np.int64)).to(device)[:, None]
            captured.clear()
            logits = model(input_ids=x, attention_mask=mask.long()).logits.float()
            teacher_logits[index] = logits.cpu().numpy()
            teacher_hidden[index] = captured[-1].cpu().numpy().astype(np.float16)
            if start % 4096 == 0:
                write_json(args.output / "progress.json", {"stage": "CACHING_TRAIN_KD_TARGETS", "samples": start, "total": len(labels), "elapsed_seconds": time.monotonic() - started})
    hook.remove(); teacher_logits.flush(); teacher_hidden.flush()
    torch.save(model.deberta.embeddings.word_embeddings.weight.detach().cpu(), cache / "teacher_embeddings.pt")
    result = {"status": "TEACHER_AND_TRAIN_CACHE_COMPLETE", "steps": args.steps, "best_step": best_dev["step"],
        "dev": best_dev["dev"], "train_samples": len(labels), "dev_samples": len(dev_labels),
        "data_manifest_sha256": file_hash(args.data / "manifest.json"), "checkpoint_sha256": file_hash(args.output / "best/model.safetensors"),
        "test_evaluated": False, "old_holdout_evaluated": False, "elapsed_seconds": time.monotonic() - started,
        "scope": manifest["scope"]}
    write_json(args.output / "training_report.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data_v3/pilot_trilingual_20261024"))
    p.add_argument("--control", type=Path, default=Path("models/turn_v2_compact_20261024/onnx"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=.00001)
    run(p.parse_args())
