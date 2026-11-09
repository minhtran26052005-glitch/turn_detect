"""One final matched EN/ZH proxy-test evaluation after dev thresholds freeze."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import file_hash, read_jsonl, write_json
from .trilingual_teacher import encoded, metrics_by_language, predict


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    marker = args.output / "test_consumed.json"
    contract_path = args.model / "turn_detector_config.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    teacher_contract = json.loads((args.teacher / "turn_detector_config.json").read_text(encoding="utf-8"))
    data_manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    assert file_hash(args.data / "test.jsonl") == data_manifest["split_sha256"]["test"]
    frozen = {"student_model_sha256": file_hash(args.model / "model.onnx"), "student_contract_sha256": file_hash(contract_path),
        "student_thresholds": contract["language_thresholds"], "teacher_thresholds": teacher_contract["language_thresholds"],
        "teacher_checkpoint_sha256": file_hash(args.teacher / "model.safetensors"), "test_sha256": data_manifest["split_sha256"]["test"],
        "status": "FINAL_TEST_STARTED; NO_FURTHER_SELECTION", "started_unix": time.time(), "scope": "EN/ZH official-test dialogue-derived structural proxy, not human semantic gold"}
    write_json(marker, frozen)
    spec = importlib.util.spec_from_file_location("final_native", args.model / "run_model.py")
    native = importlib.util.module_from_spec(spec)
    prior = sys.dont_write_bytecode; sys.dont_write_bytecode = True
    try:spec.loader.exec_module(native)
    finally:sys.dont_write_bytecode = prior
    detector = native.TinyOnnxDetector(args.model)
    rows = list(read_jsonl(args.data / "test.jsonl"))
    student_probs = np.asarray([detector.predict(r["text"], r.get("context", ""), language=r["language"])["prob_EOT"] for r in rows])
    labels = np.asarray([int(r["label"] == "EOT") for r in rows]); languages = np.asarray([r["language"] for r in rows])
    student_metrics = metrics_by_language(labels, student_probs, languages, contract["language_thresholds"])
    np.savez_compressed(args.output / "student_test_predictions.npz", labels=labels, prob_EOT=student_probs, language=languages)
    torch.set_num_threads(2)
    model = AutoModelForSequenceClassification.from_pretrained(args.teacher, dtype=torch.float32).cuda().eval()
    tok = AutoTokenizer.from_pretrained(args.teacher)
    ids, lengths, labels, languages = encoded(rows, tok)
    teacher_probs = predict(model, ids, lengths, "cuda")
    teacher_metrics = metrics_by_language(labels, teacher_probs, languages, teacher_contract["language_thresholds"])
    np.savez_compressed(args.output / "teacher_test_predictions.npz", labels=labels, prob_EOT=teacher_probs, language=languages)
    vi_rows = [{**r, "language": "vi"} for r in read_jsonl(Path("data_v2/frozen/dev.jsonl"))]
    vi_ids, vi_lengths, vi_labels, vi_languages = encoded(vi_rows, tok)
    teacher_vi_probs = predict(model, vi_ids, vi_lengths, "cuda")
    teacher_vi = metrics_by_language(vi_labels, teacher_vi_probs, vi_languages, teacher_contract["language_thresholds"])["vi"]
    np.savez_compressed(args.output / "teacher_vi_full_dev.npz", labels=vi_labels, prob_EOT=teacher_vi_probs)
    # Matched dev CPU verification gives platform parity without retuning.
    dev = list(read_jsonl(args.data / "dev.jsonl"))
    dev_probs = np.asarray([detector.predict(r["text"], r.get("context", ""), language=r["language"])["prob_EOT"] for r in dev])
    np.savez_compressed(args.output / "student_dev_colab.npz", prob_EOT=dev_probs)
    assert file_hash(contract_path) == frozen["student_contract_sha256"]
    result = {"scope": frozen["scope"], "samples": len(rows), "teacher": teacher_metrics, "student_INT4": student_metrics,
        "teacher_vi_full_dev_regression": teacher_vi,
        "frozen_before_test": frozen, "thresholds_unchanged": True, "old_vi_holdout_evaluated": False,
        "test_used_for_selection": False, "student_dev_samples": len(dev), "elapsed_seconds": time.time() - frozen["started_unix"]}
    write_json(args.output / "report.json", result)
    frozen.update(status="FINAL_TEST_COMPLETED", finished_unix=time.time())
    write_json(marker, frozen)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("models/turn_v3_trilingual_pilot_20261024/onnx"))
    p.add_argument("--teacher", type=Path, default=Path("runs/tri_teacher_20261024/best"))
    p.add_argument("--data", type=Path, default=Path("data_v3/pilot_trilingual_20261024"))
    p.add_argument("--output", type=Path, default=Path("runs/tri_final_20261024"))
    run(p.parse_args())
