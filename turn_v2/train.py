"""Fine tune on frozen train/dev; keep holdout sealed until an explicit final evaluation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments, set_seed)

from .common import (ID2LABEL, LABEL2ID, encode_pair, file_hash, read_jsonl, write_json)
from .metrics import calibrate_on_dev, classification_metrics, probabilities
from .prepare import augmentation_filter


class ProgressReceipt(TrainerCallback):
    """Persist progress independently of the notebook/WebSocket connection."""
    def __init__(self, output):
        self.path = output / "progress.json"
        self.last_metrics = {}
        self.started = time.time()

    def publish(self, state, event, metrics=None):
        if metrics:
            self.last_metrics.update({key: value for key, value in metrics.items()
                                      if isinstance(value, (str, int, float, bool))})
        receipt = {"event": event, "updated_at_unix": time.time(),
                   "elapsed_seconds": time.time() - self.started,
                   "step": state.global_step, "max_steps": state.max_steps,
                   "epoch": state.epoch, "best_model_checkpoint": state.best_model_checkpoint,
                   "best_metric": state.best_metric, "metrics": self.last_metrics}
        temporary = self.path.with_suffix(".tmp")
        write_json(temporary, receipt)
        temporary.replace(self.path)

    def on_train_begin(self, args, state, control, **kwargs):
        self.publish(state, "training")

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.publish(state, "training", logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.publish(state, "evaluated", metrics)

    def on_save(self, args, state, control, **kwargs):
        self.publish(state, "checkpoint_saved")

    def on_train_end(self, args, state, control, **kwargs):
        self.publish(state, "training_finished_finalizing_artifacts")


class WeightedCollator:
    def __init__(self, tokenizer):
        self.padding = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    def __call__(self, features):
        weights = [feature.get("sample_weight", 1.0) for feature in features]
        clean = [{key: value for key, value in feature.items() if key != "sample_weight"}
                 for feature in features]
        batch = self.padding(clean)
        batch["sample_weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch


class WeightedTrainer(Trainer):
    def __init__(self, *args, smoothing=0.02, **kwargs):
        super().__init__(*args, **kwargs)
        self.smoothing = smoothing
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        weights = inputs.pop("sample_weight")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        losses = torch.nn.functional.cross_entropy(outputs.logits.float(), labels,
            reduction="none", label_smoothing=self.smoothing)
        loss = (losses * weights.to(losses.device)).mean()
        return (loss, outputs) if return_outputs else loss


def encoded_dataset(records, tokenizer, max_length):
    return Dataset.from_list([
        {**encode_pair(tokenizer, row.get("context", ""), row["text"], max_length),
         "labels": LABEL2ID[row["label"]], "sample_weight": float(row.get("sample_weight", 1.0))}
        for row in records])


def verify_frozen(data_dir):
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["split_sha256"].items():
        if file_hash(data_dir / f"{name}.jsonl") != expected:
            raise ValueError(f"Frozen split {name} has changed")
    return manifest


def run(args):
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError("Output directory is not empty. Choose a new run name or explicitly --resume CHECKPOINT")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = verify_frozen(args.data)
    set_seed(args.seed)
    train_rows = list(read_jsonl(args.data / "train.jsonl"))
    dev_rows = list(read_jsonl(args.data / "dev.jsonl"))
    silver_count = 0
    if args.silver:
        silver, rejected = augmentation_filter(list(read_jsonl(args.silver)), args.data, args.legacy_test)
        train_rows.extend(silver)
        silver_count = len(silver)
        write_json(args.output / "silver_filter.json", {"accepted": len(silver), "rejected": len(rejected)})
    if args.smoke:
        rng = random.Random(args.seed)
        # Select a balanced small subset; this is not a quality evaluation.
        def subset(rows, per_label):
            selected = []
            for label in ("NEOT", "EOT"):
                candidates = [row for row in rows if row["label"] == label]
                selected.extend(rng.sample(candidates, min(per_label, len(candidates))))
            rng.shuffle(selected)
            return selected
        teacher_rows = [row for row in train_rows if row.get("label_source") == "qwen_agreement_silver"][:8]
        provided = [row for row in train_rows if row.get("label_source") != "qwen_agreement_silver"]
        train_rows, dev_rows = subset(provided, 16) + teacher_rows, subset(dev_rows, 8)
    print(f"Preparing tokenizer/model {args.model}; train={len(train_rows)}, dev={len(dev_rows)}", flush=True)
    revision = args.revision
    if not Path(args.model).is_dir() and revision is None:
        from huggingface_hub import model_info
        previous_identity = args.output / "run_identity.json"
        if args.resume and previous_identity.exists():
            revision = json.loads(previous_identity.read_text(encoding="utf-8")).get("base_model_revision")
        revision = revision or model_info(args.model, token=False).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2,
        id2label=ID2LABEL, label2id=LABEL2ID, dtype=torch.float32, revision=revision)
    train_dataset = encoded_dataset(train_rows, tokenizer, args.max_length)
    dev_dataset = encoded_dataset(dev_rows, tokenizer, args.max_length)
    use_cuda = torch.cuda.is_available()
    bf16 = args.precision == "bf16"
    if bf16 and not (use_cuda and torch.cuda.is_bf16_supported()):
        raise ValueError("Requested bf16 is not supported on this device")
    steps = args.max_steps if args.max_steps > 0 else (3 if args.smoke else -1)
    interval = 1 if args.smoke else args.eval_steps
    accumulation = 1 if args.smoke else args.accumulation
    total_steps = steps if steps > 0 else math.ceil(len(train_rows) / (args.batch_size * accumulation)) * args.epochs
    training_args = TrainingArguments(output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs, max_steps=steps, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=accumulation,
        warmup_steps=max(1, math.ceil(total_steps * 0.06)), lr_scheduler_type="cosine", weight_decay=0.01, max_grad_norm=1.0,
        eval_strategy="steps", save_strategy="no" if args.smoke else "steps", eval_steps=interval, save_steps=interval,
        logging_steps=1 if args.smoke else 50, save_total_limit=2,
        disable_tqdm=True,
        load_best_model_at_end=not args.smoke, metric_for_best_model="joint_score", greater_is_better=True,
        bf16=bf16, fp16=False, tf32=use_cuda and torch.cuda.get_device_capability(0)[0] >= 8,
        gradient_checkpointing=True,
        remove_unused_columns=False, report_to=[], seed=args.seed, data_seed=args.seed,
        dataloader_num_workers=0, dataloader_pin_memory=use_cuda)

    def compute_metrics(prediction):
        logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        if not np.isfinite(logits).all():
            raise ValueError(f"Non-finite dev logits: shape={logits.shape}, precision={args.precision}")
        values = classification_metrics(prediction.label_ids, probabilities(logits))
        return {key: value for key, value in values.items() if key != "confusion_matrix_NEOT_EOT"}

    optimizer_groups = []
    for head in (False, True):
        for decay in (False, True):
            params = [parameter for name, parameter in model.named_parameters()
                      if parameter.requires_grad
                      and (name.startswith(("classifier.", "pooler."))) == head
                      and (not ("bias" in name or "layernorm" in name.lower() or "layer_norm" in name.lower())) == decay]
            if params:
                optimizer_groups.append({"params": params, "lr": args.lr * (5 if head else 1),
                                         "weight_decay": 0.01 if decay else 0.0})
    optimizer = torch.optim.AdamW(optimizer_groups, eps=1e-8)
    trainer = WeightedTrainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=dev_dataset, processing_class=tokenizer, data_collator=WeightedCollator(tokenizer),
        compute_metrics=compute_metrics, smoothing=0.02,
        callbacks=([ProgressReceipt(args.output)] if args.smoke else
                   [ProgressReceipt(args.output), EarlyStoppingCallback(early_stopping_patience=3)]),
        optimizers=(optimizer, None))
    identity = {"manifest_sha256": file_hash(args.data / "manifest.json"), "seed": args.seed,
                "model": args.model, "base_model_revision": revision, "max_length": args.max_length,
                "silver_sha256": file_hash(args.silver) if args.silver else None,
                "training_settings": {"lr": args.lr, "epochs": args.epochs, "max_steps": steps,
                                      "batch_size": args.batch_size, "accumulation": accumulation,
                                      "precision": args.precision},
                "code_sha256": {name: file_hash(Path(__file__).parent / name)
                                for name in ("train.py", "common.py", "metrics.py", "prepare.py")}}
    identity_path = args.output / "run_identity.json"
    if args.resume:
        if not identity_path.exists() or json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Resume refused: data/model/seed do not match this run")
    write_json(identity_path, identity)
    start = time.monotonic()
    trained = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    best = args.output / "best"
    trainer.save_model(str(best))
    tokenizer.save_pretrained(best)
    predicted = trainer.predict(dev_dataset)
    logits = predicted.predictions[0] if isinstance(predicted.predictions, tuple) else predicted.predictions
    probs = probabilities(logits)
    fixed = classification_metrics(predicted.label_ids, probs)
    calibrated = calibrate_on_dev(predicted.label_ids, probs)
    report = {"mode": "SMOKE_NOT_A_QUALITY_RESULT" if args.smoke else "FULL_DEV_SELECTION",
        "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
        "elapsed_seconds": time.monotonic() - start, "optimizer_steps": trainer.state.global_step,
        "train_samples": len(train_rows), "dev_samples": len(dev_rows), "silver_samples_available": silver_count,
        "silver_samples_in_training_dataset": sum(row.get("label_source") == "qwen_agreement_silver" for row in train_rows),
        "train_metrics": trained.metrics, "dev_argmax": fixed, "dev_selected_threshold": calibrated,
        "target": {"accuracy": 0.9, "eot_f1": 0.9}, "holdout_evaluated": False,
        "target_achieved": False,
        "versions": {package: importlib.metadata.version(package)
                     for package in ("torch", "transformers", "datasets", "accelerate")},
        "data_identity": identity, "base_model_revision": revision}
    write_json(args.output / "training_report.json", report)
    write_json(best / "turn_detector_config.json", {"schema": 1, "max_length": args.max_length,
        "threshold": calibrated["threshold"], "normalization": "NFKC lowercase punctuation-to-space collapse-whitespace",
        "input_format": "[CLS] context [SEP] text [SEP]; context left-truncated; pad right",
        "label2id": LABEL2ID, "smoke_only": args.smoke, **identity})
    np.savez_compressed(args.output / "dev_predictions.npz", labels=predicted.label_ids, prob_EOT=probs)
    if args.export:
        from .export import export_model
        report["export"] = export_model(best, args.output / "onnx", dev_rows[:8])
        write_json(args.output / "training_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data_v2/frozen"))
    parser.add_argument("--model", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--revision", help="Pin the Hugging Face model commit; resolved and recorded by default")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--silver", type=Path)
    parser.add_argument("--legacy-test", type=Path, default=Path("test.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.accumulation < 1 or args.max_length < 3:
        parser.error("Invalid batch/length settings")
    run(args)


if __name__ == "__main__":
    main()
