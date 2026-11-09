"""Frozen text-only checks at real LiveKit pauses; no threshold/policy sweep."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import file_hash, read_jsonl, write_json
from .trilingual_teacher import encoded, predict, metrics_by_language


def native(folder):
    spec = importlib.util.spec_from_file_location("pause_native_" + folder.parent.name, folder / "run_model.py")
    module = importlib.util.module_from_spec(spec)
    before = sys.dont_write_bytecode; sys.dont_write_bytecode = True
    try:spec.loader.exec_module(module)
    finally:sys.dont_write_bytecode = before
    return module.TinyOnnxDetector(folder)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    rows = list(read_jsonl(args.cases))
    models = {"student_INT4": args.model, "vi_QAT_control": Path("models/turn_v2_distilled_qat4_20261024/onnx")}
    freeze = {name: {"weights": file_hash(folder / "model.onnx"), "contract": file_hash(folder / "turn_detector_config.json")} for name, folder in models.items()}
    freeze["teacher"] = {"weights": file_hash(args.teacher / "model.safetensors"), "contract": file_hash(args.teacher / "turn_detector_config.json")}
    write_json(args.output / "frozen_before_evaluation.json", {"models": freeze, "cases_sha256": file_hash(args.cases), "thresholds_or_weights_retuned": False})
    labels = np.asarray([int(r["label"] == "EOT") for r in rows]); languages = np.asarray([r["language"] for r in rows])
    summaries = {}
    for name, folder in list(models.items()) + [("teacher", args.teacher)]:
        if name == "teacher":
            torch.set_num_threads(2)
            model = AutoModelForSequenceClassification.from_pretrained(folder, dtype=torch.float32).cuda().eval()
            tokenizer = AutoTokenizer.from_pretrained(folder)
            ids, lengths, _, _ = encoded(rows, tokenizer)
            probs = predict(model, ids, lengths, "cuda")
            contract = json.loads((folder / "turn_detector_config.json").read_text())
            thresholds = contract["language_thresholds"]
        else:
            detector = native(folder)
            if name == "student_INT4":
                probs = np.asarray([detector.predict(r["text"], r.get("context", ""), language=r["language"])["prob_EOT"] for r in rows])
                thresholds = detector.contract["language_thresholds"]
            else:
                probs = np.asarray([detector.predict(r["text"], r.get("context", ""))["prob_EOT"] for r in rows])
                thresholds = {"en": detector.contract["threshold"], "zh": detector.contract["threshold"]}
        metrics = metrics_by_language(labels, probs, languages, thresholds)
        turn_outcomes = defaultdict(dict)
        for row, probability in zip(rows, probs, strict=True):
            outcome = turn_outcomes[row["language"]].setdefault(row["source_id"], {"premature": False, "final_eot": False})
            fired = probability >= thresholds[row["language"]]
            if row["label"] == "NEOT" and fired:outcome["premature"] = True
            if row["label"] == "EOT":outcome["final_eot"] = bool(fired)
        policy = {lang: {"turns": len(values), "turns_with_premature_EOT": sum(v["premature"] for v in values.values()),
            "premature_turn_fraction": sum(v["premature"] for v in values.values()) / len(values),
            "turns_no_premature_and_final_EOT": sum(not v["premature"] and v["final_eot"] for v in values.values())} for lang, values in turn_outcomes.items()}
        summaries[name] = {"span_metrics": metrics, "fixed_200ms_policy": policy}
        np.savez_compressed(args.output / (name + "_predictions.npz"), labels=labels, prob_EOT=probs, language=languages)
        print(json.dumps({"model": name, **summaries[name]}), flush=True)
    write_json(args.output / "report.json", {"scope": "EXTERNAL_REAL_PAUSE_TEXT_ONLY; fixed 200ms, oracle word timings, zero transcript lag; NOT official audio leaderboard",
        "samples": len(rows), "models": summaries, "frozen_models": freeze, "evaluation_only": True,
        "thresholds_or_weights_retuned": False, "training_data_source": "No LiveKit rows used in teacher/student training"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("models/turn_v3_trilingual_pilot_20261024/onnx"))
    p.add_argument("--teacher", type=Path, default=Path("runs/tri_teacher_20261024/best"))
    p.add_argument("--cases", type=Path, default=Path("data_v3/livekit_eot_20261024/cases.jsonl"))
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
