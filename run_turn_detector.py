"""Run the supplied ONNX turn detector and evaluate JSONL data locally."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

# Only tokenization is needed; avoid importing optional training frameworks.
os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from tqdm import tqdm

from text_based import TurnDetectorONNX


ROOT = Path(__file__).resolve().parent
LABELS = {0: "NEOT", 1: "EOT"}


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"{path}:{number}: expected a string 'text'")
            if not isinstance(record.get("context", ""), str):
                raise ValueError(f"{path}:{number}: expected a string 'context'")
            record.setdefault("context", "")
            if "label" in record and record["label"] not in LABELS.values():
                raise ValueError(f"{path}:{number}: expected label EOT or NEOT")
            records.append(record)
    return records


def input_key(record: dict) -> tuple[str, str]:
    return record.get("context", "").strip().lower(), record["text"].strip().lower()


def score(rows: list[dict]) -> dict:
    counts = collections.Counter((row["label"], row["prediction"]) for row in rows)
    tn, fp = counts["NEOT", "NEOT"], counts["NEOT", "EOT"]
    fn, tp = counts["EOT", "NEOT"], counts["EOT", "EOT"]
    divide = lambda a, b: a / b if b else None
    f1 = divide(2 * tp, 2 * tp + fp + fn)
    negative_f1 = divide(2 * tn, 2 * tn + fp + fn)
    return {
        "samples": len(rows),
        "label_counts": dict(collections.Counter(row["label"] for row in rows)),
        "accuracy": divide(tp + tn, len(rows)),
        "eot_precision": divide(tp, tp + fp),
        "eot_recall": divide(tp, tp + fn),
        "eot_f1": f1,
        "macro_f1": (f1 + negative_f1) / 2
        if f1 is not None and negative_f1 is not None else None,
        "confusion_matrix": {
            "rows_actual": ["NEOT", "EOT"],
            "columns_predicted": ["NEOT", "EOT"],
            "values": [[tn, fp], [fn, tp]],
        },
        "false_eot": fp,
        "missed_eot": fn,
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "checkpoint-23000")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="One utterance to classify")
    source.add_argument("--input", type=Path, help="JSONL with text, context and optional label")
    parser.add_argument("--context", default="", help="Previous conversation for --text")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Default 1 matches per-utterance use; quantized outputs can depend on batching")
    parser.add_argument("--limit", type=int, help="Only process the first N input records")
    parser.add_argument("--train", type=Path, help="Report exact input overlap and non-overlapping scores")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "latest")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 3 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size and limit must be positive; max-length must be >= 3")
    if not args.model.is_dir():
        parser.error(f"Model directory missing: {args.model}. Extract checkpoint-23000.zip first.")

    records = ([{"text": args.text, "context": args.context}]
               if args.text is not None else load_records(args.input))
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        parser.error("Input contains no records")

    import onnxruntime as ort
    if args.device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
        parser.error("CUDAExecutionProvider is unavailable in this Python environment; use --device cpu")
    load_start = time.perf_counter()
    model = TurnDetectorONNX(str(args.model.resolve()), args.max_length, args.device)
    load_seconds = time.perf_counter() - load_start
    providers = model.session.get_providers()
    if args.device == "cuda" and "CUDAExecutionProvider" not in providers:
        raise RuntimeError("CUDA session initialization fell back to CPU")

    # Warmup is kept out of evaluation timing.
    model.predict(records[0]["context"], records[0]["text"])
    results = []
    latencies = []
    run_start = time.perf_counter()
    if args.batch_size == 1:
        for index, record in enumerate(tqdm(records, desc="Predicting", unit="sample"), 1):
            start = time.perf_counter()
            prediction = model.predict(record["context"], record["text"])
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
            results.append({"row": index, **record,
                            "predicted_class": prediction["predicted_class"],
                            "prediction": LABELS[prediction["predicted_class"]],
                            "prob_EOT": prediction["prob_EOT"], "latency_ms": elapsed})
    else:
        predictions = model.predict_batch(records, batch_size=args.batch_size)
        for index, (record, prediction) in enumerate(zip(records, predictions, strict=True), 1):
            results.append({"row": index, **record,
                            "predicted_class": prediction["predicted"],
                            "prediction": LABELS[prediction["predicted"]],
                            "prob_EOT": prediction["prob"]})
    inference_seconds = time.perf_counter() - run_start

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_keys = None
    if args.train:
        train_keys = set()
        with args.train.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    train_keys.add(input_key(json.loads(line)))
        for row in results:
            row["exact_input_in_train"] = input_key(row) in train_keys
    labeled = [row for row in results if "label" in row]
    for row in labeled:
        row["correct"] = row["prediction"] == row["label"]
    for filename, rows in [("predictions.jsonl", results),
                           ("errors.jsonl", [row for row in labeled if not row["correct"]])]:
        with (args.output_dir / filename).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    versions = {}
    for package in ["numpy", "onnxruntime", "onnxruntime-gpu", "transformers", "tokenizers", "tqdm"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    hashes = {}
    files = [ROOT / "text_based.py", Path(__file__).resolve(), *sorted(args.model.glob("*"))]
    files += [path for path in [args.input, args.train] if path is not None]
    for path in files:
        if path.is_file():
            with path.open("rb") as stream:
                hashes[str(path.resolve())] = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": sys.argv,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "package_versions": versions,
        "available_providers": ort.get_available_providers(),
        "session_providers": providers,
        "settings": {"max_length": args.max_length, "batch_size": args.batch_size,
                     "decision": "argmax of two logits; class 1 is EOT per supplied inference code",
                     "context_truncation": "left", "text_truncation": "right only beyond max_length - 3",
                     "lowercase": True},
        "samples": len(results),
        "nonempty_context_samples": sum(bool(row["context"].strip()) for row in results),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "samples_per_second": len(results) / inference_seconds,
        "sha256": hashes,
        "metrics": score(labeled) if labeled else None,
    }
    if latencies:
        report["latency_ms_including_tokenization_after_warmup"] = {
            "mean": float(np.mean(latencies)), "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)), "max": float(max(latencies)),
        }
    if train_keys is not None:
        report["overlap_definition"] = "exact (context.strip().lower(), text.strip().lower())"
        report["exact_input_overlap_train"] = sum(row["exact_input_in_train"] for row in results)
        report["metrics_without_exact_train_overlap"] = score(
            [row for row in labeled if not row["exact_input_in_train"]])
        report["metrics_on_exact_train_overlap"] = score(
            [row for row in labeled if row["exact_input_in_train"]])
    write_json(args.output_dir / "metrics.json", report)
    if args.text is not None:
        print(json.dumps(results[0], ensure_ascii=False, indent=2))
    if labeled:
        print(json.dumps({key: report[key] for key in ["metrics", "inference_seconds", "samples_per_second"]},
                         ensure_ascii=False, indent=2))
    print(f"Saved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
