"""Weight-only INT4 QAT for TinyTurnGRU, with frozen deployment group scales.

Five matrix weights are fake-quantized on EVERY forward with clipped STE.
The FP16 scale and dequantized-weight rounding match ONNX DQ21 -> Cast.
Activations and GRU arithmetic remain FP32. No holdout is scored.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from .common import LABEL2ID, encode_pair, file_hash, read_jsonl, write_json
from .distill import evaluate
from .metrics import calibrate_on_dev
from .quantize_student4 import quantize_groups
from .tiny_student import load_student


WEIGHTS = ["embedding.weight", "gru.weight_ih_l0", "gru.weight_hh_l0", "head.0.weight", "head.3.weight"]


class GroupInt4FakeQuant(nn.Module):
    def __init__(self, scale, group_size):
        super().__init__()
        self.group_size = group_size
        self.register_buffer("scale", scale.detach().to(torch.float16))
        self.register_buffer("zero", torch.zeros(scale.numel(), dtype=torch.int32, device=scale.device))

    def forward(self, weight):
        width = weight.shape[-1]
        padded_width = self.scale.shape[-1] * self.group_size
        blocks = F.pad(weight, (0, padded_width - width)).reshape(-1, self.group_size)
        fake = torch.fake_quantize_per_channel_affine(blocks, self.scale.flatten().float(), self.zero, 0, -8, 7)
        # Match FP16 output of ONNX DQ; preserve the clipped STE from fake_quant.
        rounded = fake.to(torch.float16).float()
        fake = fake + (rounded - fake).detach()
        # Partial final groups have padded row strides; cuDNN GRU needs compact
        # matrices when rebuilding its flattened weight buffer each forward.
        return fake.reshape(*weight.shape[:-1], padded_width)[..., :width].contiguous()


def prepare_qat(model, group_size, scales=None):
    quantizers = {}
    for name in WEIGHTS:
        parent, key = name.rsplit(".", 1)
        module = model.get_submodule(parent)
        weight = getattr(module, key)
        if scales is None:
            _, initial_scale, _ = quantize_groups(weight.detach().cpu().numpy(), group_size)
            scale = torch.from_numpy(initial_scale).to(weight.device)
        else:
            scale = scales[name].to(weight.device)
        quantizer = GroupInt4FakeQuant(scale, group_size)
        parametrize.register_parametrization(module, key, quantizer)
        quantizers[name] = quantizer
    return quantizers


def latent_state(model):
    state = {name: value.detach().cpu() for name, value in model.state_dict().items() if ".parametrizations." not in name}
    for name in WEIGHTS:
        parent, key = name.rsplit(".", 1)
        state[name] = getattr(model.get_submodule(parent).parametrizations, key).original.detach().cpu()
    return state


def pack_parameter(weight, scale, group_size):
    """Exact deployment codes and reconstructed FP32 weights, no scale refit."""
    values = np.asarray(weight, dtype=np.float32)
    scales = np.asarray(scale, dtype=np.float16)
    expanded = np.repeat(scales.astype(np.float32), group_size, axis=-1)[..., :values.shape[-1]]
    codes = np.clip(np.rint(values / expanded), -8, 7).astype(np.int8)
    restored = (codes.astype(np.float32) * expanded).astype(np.float16).astype(np.float32)
    return codes, scales, restored


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = json.loads((args.cache / "cache_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "TRAIN_TARGETS_COMPLETE"
    assert file_hash(args.data / "train.jsonl") == report["train_sha256"]
    assert file_hash(args.data / "dev.jsonl") == report["dev_sha256"]
    marker = file_hash(args.data / "holdout_consumed.json")
    model = load_student(args.student, device).train()
    assert model.gru.num_layers == 1
    quantizers = prepare_qat(model, args.group_size)
    scales = {name: q.scale.detach().cpu() for name, q in quantizers.items()}
    torch.save({"group_size": args.group_size, "scales": scales, "weight_names": WEIGHTS,
                "scheme": "signed INT4 [-8,7], fixed FP16 group scales, FP16 DQ rounding, clipped STE"}, args.output / "quant_params.pt")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.student)
    kept = json.loads((args.vocabulary / "kept_teacher_ids.json").read_text(encoding="utf-8"))
    original = np.load(args.cache / "train_ids.npy", mmap_mode="r")
    mapper = np.full(23078, -1, np.int32)
    mapper[kept] = np.arange(len(kept), dtype=np.int32)
    train_ids = mapper[original]
    assert train_ids.min() >= 0
    lengths = np.load(args.cache / "train_lengths.npy")
    labels = np.load(args.cache / "train_labels.npy")
    teacher_logits = np.load(args.cache / "teacher_logits.npy", mmap_mode="r")
    teacher_hidden = np.load(args.cache / "teacher_hidden.npy", mmap_mode="r")
    rows = list(read_jsonl(args.data / "dev.jsonl"))
    dev_ids = np.zeros((len(rows), 256), np.int32)
    dev_lengths = np.zeros(len(rows), np.int16)
    for i, row in enumerate(rows):
        ids = encode_pair(tok, row.get("context", ""), row["text"], 256)["input_ids"]
        dev_ids[i, :len(ids)] = ids
        dev_lengths[i] = len(ids)
    dev_labels = np.asarray([LABEL2ID[r["label"]] for r in rows])
    config = json.loads((args.student / "config.json").read_text(encoding="utf-8"))
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "turn_detector_config.json"]:
        shutil.copy2(args.student / name, args.output / name)
    identity = {"initial_student_sha256": file_hash(args.student / "student.pt"), "train_sha256": report["train_sha256"],
        "dev_sha256": report["dev_sha256"], "holdout_marker_sha256": marker, "teacher_sha256": report["teacher_sha256"],
        "quant_params_sha256": file_hash(args.output / "quant_params.pt"), "seed": args.seed,
        "group_size": args.group_size, "epochs_max": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "loss": "0.5 CE(smoothed0.02) + 0.5 T2-KL(T=2) + 0.05 normalized teacher-feature MSE",
        "source_sha256": {name: file_hash(Path(__file__).parent / name) for name in ["qat_student.py", "tiny_student.py", "distill.py", "quantize_student4.py"]},
        "train_samples": len(labels), "dev_samples": len(dev_labels), "architecture": config,
        "activation_precision": "FP32", "weight_precision": "INT4 fake-quantized every forward", "device": device}
    write_json(args.output / "run_identity.json", identity)
    started = time.monotonic()
    initial = calibrate_on_dev(dev_labels, evaluate(model, dev_ids, dev_lengths, dev_labels, args.batch_size, device))
    write_json(args.output / "initial_fake_quant_dev.json", initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    steps = math.ceil(len(labels) / args.batch_size) * args.epochs
    step = 0
    best = -1.
    bad = 0
    history = []
    rng = np.random.default_rng(args.seed)
    gradient_checked = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        order = rng.permutation(len(labels))
        for start in range(0, len(labels), args.batch_size):
            index = order[start:start + args.batch_size]
            length = int(lengths[index].max())
            x = torch.from_numpy(train_ids[index, :length].astype(np.int64)).to(device)
            mask = torch.arange(length, device=device)[None, :] < torch.from_numpy(lengths[index].astype(np.int64)).to(device)[:, None]
            y = torch.from_numpy(labels[index]).to(device)
            target = torch.from_numpy(np.array(teacher_logits[index], copy=True)).to(device)
            hidden_target = torch.from_numpy(np.array(teacher_hidden[index], dtype=np.float32, copy=True)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, hidden = model(x, mask.long(), return_features=True)
            ce = F.cross_entropy(logits, y, label_smoothing=.02)
            kd = F.kl_div(F.log_softmax(logits / 2., -1), F.softmax(target / 2., -1), reduction="batchmean") * 4
            representation = F.mse_loss(F.layer_norm(hidden, (768,)), F.layer_norm(hidden_target, (768,)))
            loss = .5 * ce + .5 * kd + .05 * representation
            if not torch.isfinite(loss):
                raise ValueError("Non-finite QAT loss")
            loss.backward()
            if not gradient_checked:
                gradients = {}
                for name in WEIGHTS:
                    parent, key = name.rsplit(".", 1)
                    grad = getattr(model.get_submodule(parent).parametrizations, key).original.grad
                    gradients[name] = bool(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0)
                write_json(args.output / "gradient_checks.json", gradients)
                assert all(gradients.values()), "QAT gradient did not reach every original matrix"
                gradient_checked = True
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            factor = min(1., (step + 1) / 100) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / steps)))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * factor
            optimizer.step()
            step += 1
            losses.append(float(loss.detach()))
            if step % 200 == 0:
                progress = {"stage": "QAT_TRAINING", "epoch": epoch, "step": step, "loss": float(np.mean(losses[-200:])), "elapsed_seconds": time.monotonic() - started}
                write_json(args.output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        probabilities = evaluate(model, dev_ids, dev_lengths, dev_labels, args.batch_size, device)
        metrics = calibrate_on_dev(dev_labels, probabilities)
        record = {"epoch": epoch, "step": step, "loss": float(np.mean(losses)), "dev": metrics, "elapsed_seconds": time.monotonic() - started}
        history.append(record)
        write_json(args.output / "history.json", history)
        print(json.dumps(record), flush=True)
        if metrics["joint_score"] > best:
            best = metrics["joint_score"]
            bad = 0
            torch.save(latent_state(model), args.output / "student.pt")
            np.savez_compressed(args.output / "dev_predictions.npz", labels=dev_labels, prob_EOT=probabilities, indices=np.arange(len(dev_labels)))
            write_json(args.output / "best_epoch.json", record)
        else:
            bad += 1
        if epoch >= 4 and bad >= 3:
            break
    selected = json.loads((args.output / "best_epoch.json").read_text(encoding="utf-8"))
    contract = json.loads((args.output / "turn_detector_config.json").read_text(encoding="utf-8"))
    contract.update(threshold=selected["dev"]["threshold"], training_method="INT4_QAT_WITH_DISTILLATION", quantization_group_size=args.group_size,
                    quality_status="QAT_FAKE_QUANT_DEV_ONLY; packed ONNX evaluation pending", quality_scope="Dev selection only; consumed holdout not evaluated")
    write_json(args.output / "turn_detector_config.json", contract)
    result = {"status": "QAT_TRAINING_COMPLETED", "train_samples": len(labels), "dev_samples": len(dev_labels), "optimizer_steps": step,
        "best_epoch": selected["epoch"], "dev_selected_threshold": selected["dev"], "initial_fake_quant_dev": initial,
        "group_size": args.group_size, "gradient_checks_passed": gradient_checked,
        "quant_params_sha256": file_hash(args.output / "quant_params.pt"), "student_sha256": file_hash(args.output / "student.pt"),
        "holdout_evaluated": False, "holdout_marker_unchanged": file_hash(args.data / "holdout_consumed.json") == marker,
        "elapsed_seconds": time.monotonic() - started}
    write_json(args.output / "training_report.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data_v2/frozen"))
    p.add_argument("--cache", type=Path, default=Path("runs/distill_cache_20261024"))
    p.add_argument("--vocabulary", type=Path, default=Path("runs/distill_vocabulary_20261024"))
    p.add_argument("--group-size", type=int, choices=[32, 64], default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=.0001)
    p.add_argument("--seed", type=int, default=43)
    run(p.parse_args())
