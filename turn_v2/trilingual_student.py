"""Distill the trained trilingual teacher, then adapt all matrices to INT4."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import time
import unicodedata

import numpy as np
from tokenizers import Tokenizer
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from .common import encode_pair, file_hash, normalize, read_jsonl, write_json
from .distill import evaluate
from .qat_student import latent_state, prepare_qat
from .tiny_student import TinyTurnGRU, load_student
from .trilingual_teacher import metrics_by_language


def vocabulary(teacher, data, output):
    output.mkdir()
    tok = Tokenizer.from_file(str(teacher / "tokenizer.json"))
    source = json.loads((teacher / "tokenizer.json").read_text(encoding="utf-8"))
    observed = set()
    pending = []
    for row in read_jsonl(data / "train.jsonl"):
        pending.extend([normalize(row.get("context", "")), normalize(row["text"])])
        if len(pending) >= 2048:
            for value in tok.encode_batch(pending, add_special_tokens=False):observed.update(value.ids)
            pending = []
    for value in tok.encode_batch(pending, add_special_tokens=False):observed.update(value.ids)
    values = source["model"]["vocab"]
    for entry in sorted(source.get("added_tokens", []), key=lambda t: t["id"]):
        if entry["id"] == len(values):values.append([entry["content"], 0.])
    def fallback(piece):
        value = piece.replace("▁", "")
        return piece.startswith("<0x") or len(value) == 0 or (len(value) == 1 and
            (ord(value) < 128 or "LATIN" in unicodedata.name(value, "") or 0x3400 <= ord(value) <= 0x9fff))
    kept = sorted(observed | {i for i, (piece, _) in enumerate(values) if fallback(piece)} | {t["id"] for t in source["added_tokens"]} | {0, 1, 2, 3})
    remap = {old: new for new, old in enumerate(kept)}
    compact = copy.deepcopy(source)
    compact["model"]["vocab"] = [values[i] for i in kept]
    compact["model"]["unk_id"] = remap[source["model"]["unk_id"]]
    for token in compact["added_tokens"]:token["id"] = remap[token["id"]]
    for token in compact.get("post_processor", {}).get("special_tokens", {}).values():token["ids"] = [remap[i] for i in token["ids"]]
    Tokenizer.from_str(json.dumps(compact)).save(str(output / "tokenizer.json"))
    settings = json.loads((teacher / "tokenizer_config.json").read_text(encoding="utf-8"))
    if settings.get("added_tokens_decoder"):settings["added_tokens_decoder"] = {str(remap[int(k)]): v for k, v in settings["added_tokens_decoder"].items()}
    write_json(output / "tokenizer_config.json", settings)
    write_json(output / "config.json", {"model_type": "deberta-v2", "vocab_size": len(kept), "pad_token_id": 0})
    write_json(output / "kept_teacher_ids.json", kept)
    write_json(output / "report.json", {"vocab_size": len(kept), "source": "TRAIN observed pieces plus ASCII/Latin/all available single CJK/bytes/specials", "dev_or_test_vocab_fit": False})
    return kept


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2); torch.manual_seed(43); np.random.seed(43)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    teacher_report = json.loads((args.teacher / "training_report.json").read_text(encoding="utf-8"))
    assert teacher_report["status"] == "TEACHER_AND_TRAIN_CACHE_COMPLETE"
    assert teacher_report["data_manifest_sha256"] == file_hash(args.data / "manifest.json")
    cache = args.teacher / "cache"
    kept = vocabulary(args.teacher / "best", args.data, args.output / "vocabulary")
    tok = AutoTokenizer.from_pretrained(args.output / "vocabulary")
    original = np.load(cache / "train_ids.npy", mmap_mode="r")
    mapper = np.full(251000, -1, np.int32); mapper[kept] = np.arange(len(kept), dtype=np.int32)
    train_ids = mapper[original]; assert train_ids.min() >= 0
    lengths = np.load(cache / "train_lengths.npy"); labels = np.load(cache / "train_labels.npy")
    train_languages = np.load(cache / "train_language.npy")
    dev_rows = list(read_jsonl(args.data / "dev.jsonl"))
    dev_ids = np.zeros((len(dev_rows), 256), np.int32); dev_lengths = np.zeros(len(dev_rows), np.int16)
    for i, row in enumerate(dev_rows):
        values = encode_pair(tok, row.get("context", ""), row["text"], 256)["input_ids"]
        dev_ids[i, :len(values)] = values; dev_lengths[i] = len(values)
    dev_labels = np.load(cache / "dev_labels.npy"); dev_languages = np.load(cache / "dev_language.npy")
    teacher_logits = np.load(cache / "teacher_logits.npy", mmap_mode="r")
    teacher_hidden = np.load(cache / "teacher_hidden.npy", mmap_mode="r")
    changed = 0
    for i, row in enumerate(read_jsonl(args.data / "train.jsonl")):
        current = encode_pair(tok, row.get("context", ""), row["text"], 256)["input_ids"]
        changed += current != train_ids[i, :int(lengths[i])].tolist()
    write_json(args.output / "vocabulary/train_segmentation_audit.json", {"rows": len(labels), "changed": changed})
    assert changed == 0, "Pruned tokenizer changed training segmentation"
    config = {"model_type": "deberta-v2", "architecture": "tiny_turn_gru", "vocab_size": len(kept), "embedding_size": 64, "hidden_size": 128, "layers": 1, "teacher_size": 768, "pad_token_id": 0}
    model = TinyTurnGRU(**config).to(device)
    embedding = torch.load(cache / "teacher_embeddings.pt", map_location="cpu", weights_only=True)[kept].float().to(device)
    mean = embedding.mean(0); _, _, basis = torch.pca_lowrank(embedding - mean, q=64, center=False, niter=5)
    with torch.no_grad():model.embedding.weight.copy_((embedding - mean) @ basis[:, :64])
    del embedding, mean, basis
    architecture_report = {"config": config, "deployment_parameters": sum(p.numel() for n, p in model.named_parameters() if not n.startswith("teacher_projection.")),
        "data_manifest_sha256": file_hash(args.data / "manifest.json"), "teacher_checkpoint_sha256": teacher_report["checkpoint_sha256"], "train_samples": len(labels), "dev_samples": len(dev_labels), "test_evaluated": False}
    write_json(args.output / "run_identity.json", architecture_report)
    results = []
    for phase, max_epochs, lr in [("distill", 16, .001), ("qat", 6, .0001)]:
        folder = args.output / (phase + "_best"); folder.mkdir()
        write_json(folder / "config.json", config)
        for filename in ["tokenizer.json", "tokenizer_config.json"]:shutil.copy2(args.output / "vocabulary" / filename, folder / filename)
        if phase == "qat":
            model = load_student(args.output / "distill_best", device).train()
            quantizers = prepare_qat(model, 64)
            torch.save({"group_size": 64, "scales": {n: q.scale.cpu() for n, q in quantizers.items()}, "weight_names": list(quantizers)}, folder / "quant_params.pt")
            torch.backends.cuda.matmul.allow_tf32 = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
        best = -1.; bad = 0; history = []; step = 0
        total_steps = math.ceil(len(labels) / 256) * max_epochs
        rng = np.random.default_rng(43)
        for epoch in range(1, max_epochs + 1):
            model.train(); losses = []
            order = rng.permutation(len(labels))
            for start in range(0, len(labels), 256):
                index = order[start:start + 256]; length = int(lengths[index].max())
                x = torch.from_numpy(train_ids[index, :length].astype(np.int64)).to(device)
                mask = torch.arange(length, device=device)[None] < torch.from_numpy(lengths[index].astype(np.int64)).to(device)[:, None]
                y = torch.from_numpy(labels[index]).to(device)
                target = torch.from_numpy(np.array(teacher_logits[index], copy=True)).to(device)
                hidden_target = torch.from_numpy(np.array(teacher_hidden[index], dtype=np.float32, copy=True)).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, hidden = model(x, mask.long(), return_features=True)
                loss = .5 * F.cross_entropy(logits, y, label_smoothing=.02) + .5 * 4 * F.kl_div(F.log_softmax(logits / 2, -1), F.softmax(target / 2, -1), reduction="batchmean") + .05 * F.mse_loss(F.layer_norm(hidden, (768,)), F.layer_norm(hidden_target, (768,)))
                assert torch.isfinite(loss)
                loss.backward()
                if phase == "qat" and step == 0:
                    gradients = {}
                    for name in quantizers:
                        parent, key = name.rsplit(".", 1)
                        grad = getattr(model.get_submodule(parent).parametrizations, key).original.grad
                        gradients[name] = bool(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0)
                    write_json(folder / "gradient_checks.json", gradients)
                    assert all(gradients.values())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                factor = min(1., (step + 1) / 50) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / total_steps)))
                for group in optimizer.param_groups:group["lr"] = lr * factor
                optimizer.step(); step += 1; losses.append(float(loss.detach()))
            probs = evaluate(model, dev_ids, dev_lengths, dev_labels, 256, device)
            metrics = metrics_by_language(dev_labels, probs, dev_languages)
            score = min(m["joint_score"] for m in metrics.values())
            record = {"phase": phase, "epoch": epoch, "step": step, "loss": float(np.mean(losses)), "dev": metrics,
                "minimum_language_joint_score": score, "elapsed_seconds": time.monotonic() - started}
            history.append(record); write_json(folder / "history.json", history); write_json(args.output / "progress.json", record)
            print(json.dumps(record), flush=True)
            if score > best:
                best = score; bad = 0
                torch.save(latent_state(model) if phase == "qat" else model.state_dict(), folder / "student.pt")
                np.savez_compressed(folder / "dev_predictions.npz", labels=dev_labels, prob_EOT=probs, language=dev_languages)
                write_json(folder / "best_epoch.json", record)
            else:bad += 1
            if epoch >= (6 if phase == "distill" else 4) and bad >= 3:break
        selected = json.loads((folder / "best_epoch.json").read_text(encoding="utf-8"))
        contract = {"schema": 1, "architecture": "tiny_turn_gru", "max_length": 256, "threshold": .5,
            "language_thresholds": {lang: value["threshold"] for lang, value in selected["dev"].items()},
            "padding_to_multiple_of": 8, "intra_op_threads": 2, "normalization": "NFKC lowercase punctuation-to-space collapse-whitespace",
            "label2id": {"NEOT": 0, "EOT": 1}, "smoke_only": False, "training_method": "TRILINGUAL_DISTILL_QAT" if phase == "qat" else "TRILINGUAL_DISTILL",
            "quality_scope": "STRUCTURAL_PROXY_PILOT; not human-adjudicated semantic EOT benchmark", "quality_status": "PILOT_ONLY"}
        write_json(folder / "turn_detector_config.json", contract)
        results.append({"phase": phase, "optimizer_steps": step, "epochs": len(history), "best_epoch": selected["epoch"], "dev": selected["dev"]})
    result = {"status": "TRILINGUAL_STUDENT_DISTILL_QAT_COMPLETE", "phases": results, **architecture_report,
        "elapsed_seconds": time.monotonic() - started, "old_holdout_evaluated": False, "test_evaluated": False}
    write_json(args.output / "training_report.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data_v3/pilot_trilingual_20261024"))
    p.add_argument("--teacher", type=Path, default=Path("runs/tri_teacher_20261024"))
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
