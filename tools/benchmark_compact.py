"""Compare original/compact ONNX CPU latency with matched samples and thread counts."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turn_v2.common import file_hash, read_jsonl, write_json
from turn_v2.predict import TurnDetectorV2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=Path("models/turn_v2_trial_20261024/onnx"))
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    rows = list(read_jsonl(Path("data_v2/frozen/dev.jsonl")))
    indices = np.sort(np.random.default_rng(20261024).choice(len(rows), args.samples, replace=False))
    records = [rows[index] for index in indices]
    result = {"selection": "fixed uniform dev sample; seed 20261024", "indices": indices.tolist(),
        "threads": args.threads, "batch_size": 1, "warmup_samples": 8,
        "timing_includes": "normalization, tokenization, ONNX inference, probability and threshold",
        "results": {}}
    for name, path in [("teacher", args.teacher), ("compact", args.compact)]:
        started = time.perf_counter()
        model = TurnDetectorV2(path, threads=args.threads)
        current = {"path": str(path.resolve()), "model_sha256": file_hash(path/"model.onnx"),
                   "load_seconds": time.perf_counter()-started, "profiles": {}}
        original_multiple = model.contract.get("padding_to_multiple_of")
        for profile in ("matched_dynamic8", "deployment_padding"):
            if profile == "matched_dynamic8":
                model.contract["padding_to_multiple_of"] = 8
            elif original_multiple:
                model.contract["padding_to_multiple_of"] = original_multiple
            else:
                model.contract.pop("padding_to_multiple_of", None)
            for row in records[:8]:
                model.predict(row["text"], row.get("context", ""))
            timings, predictions = [], []
            for row in records:
                started = time.perf_counter()
                prediction = model.predict(row["text"], row.get("context", ""))
                timings.append(1000*(time.perf_counter()-started))
                predictions.append(prediction)
            current["profiles"][profile] = {"mean_ms": float(np.mean(timings)),
                "p50_ms": float(np.percentile(timings,50)), "p95_ms": float(np.percentile(timings,95)),
                "latencies_ms": timings, "predictions": predictions,
                "padding_multiple": model.contract.get("padding_to_multiple_of"),
                "max_length": model.contract["max_length"]}
        result["results"][name] = current
        del model
        gc.collect()
        print(json.dumps({"model": name, "profiles": {key: {field: value for field,value in item.items()
            if field not in ("latencies_ms", "predictions")} for key,item in current["profiles"].items()}}), flush=True)
    for profile in ("matched_dynamic8", "deployment_padding"):
        before = result["results"]["teacher"]["profiles"][profile]
        after = result["results"]["compact"]["profiles"][profile]
        result[profile+"_p50_speedup"] = before["p50_ms"] / after["p50_ms"]
    write_json(args.output, result)
    print(json.dumps({key:value for key,value in result.items() if key.endswith("speedup")}), flush=True)


if __name__ == "__main__":
    main()
