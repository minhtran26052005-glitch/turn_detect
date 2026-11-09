"""Evaluate a compression candidate on dev only; never reopen the consumed holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from .common import LABEL2ID, encode_pair, file_hash, read_jsonl, write_json
from .metrics import calibrate_on_dev, classification_metrics, probabilities


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    contract = json.loads((args.model / "turn_detector_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    if file_hash(args.data / "dev.jsonl") != manifest["split_sha256"]["dev"]:
        raise ValueError("Frozen dev changed")
    marker = args.data / "holdout_consumed.json"
    marker_before = file_hash(marker) if marker.exists() else None
    rows = list(read_jsonl(args.data / "dev.jsonl"))
    indices = np.arange(len(rows))
    if args.limit:
        indices = np.sort(np.random.default_rng(20261024).choice(len(rows), min(args.limit, len(rows)), replace=False))
    selected = [rows[index] for index in indices]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    options = ort.SessionOptions()
    for key, value in contract.get("onnx_session_options", {}).items():
        options.add_session_config_entry(key, str(value))
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    started = time.monotonic()
    session = ort.InferenceSession(str(args.model / "model.onnx"), sess_options=options,
                                  providers=["CPUExecutionProvider"])
    load_seconds = time.monotonic() - started
    print(json.dumps({"stage": "model_loaded", "seconds": load_seconds}), flush=True)
    encoded = [encode_pair(tokenizer, row.get("context", ""), row["text"], contract["max_length"])
               for row in selected]
    reference_tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    kept = json.loads(args.kept_ids.read_text(encoding="utf-8")) if args.kept_ids else None
    changed = 0
    if kept:
        for row, value in zip(selected, encoded, strict=True):
            original = encode_pair(reference_tokenizer, row.get("context", ""), row["text"], contract["max_length"])
            changed += [kept[index] for index in value["input_ids"]] != original["input_ids"]
    print(json.dumps({"stage": "tokenization_audited", "samples": len(selected), "changed": int(changed)}), flush=True)
    latencies, probs, total_tokens = [], [], 0
    started = time.monotonic()
    for start in range(0, len(encoded), args.batch_size):
        values = encoded[start:start + args.batch_size]
        longest = max(len(value["input_ids"]) for value in values)
        multiple = contract.get("padding_to_multiple_of", 8)
        length = contract["max_length"] if args.fixed_padding else min(contract["max_length"], ((longest + multiple - 1) // multiple) * multiple)
        arrays = {"input_ids": np.asarray([value["input_ids"] + [tokenizer.pad_token_id] * (length-len(value["input_ids"])) for value in values], dtype=np.int64),
                  "attention_mask": np.asarray([value["attention_mask"] + [0] * (length-len(value["attention_mask"])) for value in values], dtype=np.int64)}
        sample_start = time.perf_counter()
        logits = session.run(["logits"], arrays)[0]
        latencies.append((time.perf_counter() - sample_start) * 1000)
        probs.extend(probabilities(logits).tolist())
        total_tokens += sum(len(value["input_ids"]) for value in values)
        if len(probs) % 1000 < args.batch_size or len(probs) == len(selected):
            write_json(args.output / "progress.json", {"samples": len(probs), "total": len(selected),
                "elapsed_seconds": time.monotonic() - started})
            print(json.dumps({"samples": len(probs), "total": len(selected), "elapsed_seconds": time.monotonic()-started}), flush=True)
    elapsed = time.monotonic() - started
    labels = np.asarray([LABEL2ID[row["label"]] for row in selected])
    probs = np.asarray(probs)
    fixed = classification_metrics(labels, probs, contract["threshold"])
    calibrated = calibrate_on_dev(labels, probs)
    baseline = np.load(args.reference_run / "dev_predictions.npz")
    assert np.array_equal(baseline["labels"][indices], labels)
    teacher_probs = baseline["prob_EOT"][indices]
    teacher_metrics = classification_metrics(labels, teacher_probs, 0.5)
    reference_bytes = (args.teacher / "model.onnx").stat().st_size
    model_bytes = (args.model / "model.onnx").stat().st_size
    deployment_bytes = sum(path.stat().st_size for path in args.model.iterdir() if path.is_file())
    report = {"scope": "DEV_ONLY_COMPRESSION_SELECTION; consumed holdout is not evaluated",
        "model": str(args.model.resolve()), "model_sha256": file_hash(args.model / "model.onnx"),
        "contract_sha256_before": file_hash(args.model / "turn_detector_config.json"),
        "dev_sha256": file_hash(args.data / "dev.jsonl"), "samples": len(selected),
        "dev_at_teacher_threshold": fixed, "dev_selected_threshold": calibrated,
        "teacher_dev_matched": teacher_metrics,
        "agreement_with_teacher_at_0_5": float(((probs >= .5) == (teacher_probs >= .5)).mean()),
        "mean_abs_probability_delta": float(np.abs(probs-teacher_probs).mean()),
        "max_abs_probability_delta": float(np.abs(probs-teacher_probs).max()),
        "segmentation_changed_rows": int(changed), "encoded_tokens": total_tokens,
        "reference_model_bytes": reference_bytes, "model_bytes": model_bytes,
        "deployment_bytes": deployment_bytes, "model_size_ratio": model_bytes/reference_bytes,
        "deployment_size_ratio": deployment_bytes/reference_bytes,
        "size_target_model_met": model_bytes <= reference_bytes/10,
        "size_target_deployment_met": deployment_bytes <= reference_bytes/10,
        "dev_quality_target_met": calibrated["accuracy"] >= .9 and calibrated["eot_f1"] >= .9,
        "cpu": {"threads": args.threads, "batch_size": args.batch_size,
            "session_options": contract.get("onnx_session_options", {}),
            "padding": "fixed_max_length" if args.fixed_padding else "dynamic_multiple_8",
            "load_seconds": load_seconds, "inference_seconds": elapsed,
            "samples_per_second": len(selected)/elapsed,
            "batch_latency_ms_after_first": {"mean": float(np.mean(latencies[1:])),
                "p50": float(np.percentile(latencies[1:],50)), "p95": float(np.percentile(latencies[1:],95))}},
        "holdout_evaluated": False, "holdout_marker_unchanged": (file_hash(marker) if marker.exists() else None) == marker_before}
    np.savez_compressed(args.output / "dev_predictions.npz", indices=indices, labels=labels, prob_EOT=probs)
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if args.select_threshold:
        if args.limit or len(selected) != len(rows):
            raise ValueError("Threshold can only be selected with full dev")
        contract["threshold"] = calibrated["threshold"]
        contract["quality_status"] = "DEV_ONLY_QUALITY_TARGET_MET" if report["dev_quality_target_met"] else "DEV_QUALITY_TARGET_NOT_MET"
        contract["compression_dev_report"] = str((args.output / "report.json").resolve())
        write_json(args.model / "turn_detector_config.json", contract)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data_v2/frozen"))
    parser.add_argument("--teacher", type=Path, default=Path("models/turn_v2_trial_20261024/onnx"))
    parser.add_argument("--reference-run", type=Path, default=Path("outputs/training_trial_20261024"))
    parser.add_argument("--kept-ids", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fixed-padding", action="store_true")
    parser.add_argument("--select-threshold", action="store_true")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
