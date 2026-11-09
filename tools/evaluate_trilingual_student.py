"""Stage a portable trilingual INT4 candidate and freeze dev-only thresholds."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from turn_v2.common import file_hash, read_jsonl, write_json
from turn_v2.metrics import calibrate_on_dev, classification_metrics


def main():
    evidence = Path("outputs/trilingual_20261024")
    student = evidence / "runs/tri_student_20261024/qat_best"
    package = Path("models/turn_v3_trilingual_pilot_20261024")
    model = package / "onnx"
    model.mkdir(parents=True, exist_ok=False)
    for p in (student / "onnx_int4").iterdir():
        if p.is_file():shutil.copy2(p, model / p.name)
    shutil.copy2("turn_v2/trilingual_runtime.py", model / "run_model.py")
    shutil.copy2("models/turn_v2_distilled_qat4_20261024/onnx/requirements.txt", model / "requirements.txt")
    spec = importlib.util.spec_from_file_location("trilingual_native", model / "run_model.py")
    module = importlib.util.module_from_spec(spec)
    before = sys.dont_write_bytecode; sys.dont_write_bytecode = True
    try:spec.loader.exec_module(module)
    finally:sys.dont_write_bytecode = before
    detector = module.TinyOnnxDetector(model)
    rows = list(read_jsonl(Path("data_v3/pilot_trilingual_20261024/dev.jsonl")))
    probs = np.asarray([detector.predict(r["text"], r.get("context", ""), language=r["language"])["prob_EOT"] for r in rows])
    labels = np.asarray([int(r["label"] == "EOT") for r in rows]); languages = np.asarray([r["language"] for r in rows])
    metrics = {lang: calibrate_on_dev(labels[languages == lang], probs[languages == lang]) for lang in ["vi", "en", "zh"]}
    detector.contract["language_thresholds"] = {lang: m["threshold"] for lang, m in metrics.items()}
    detector.contract["quality_status"] = "TRILINGUAL_PILOT_TARGET_MET_ON_PROXY_DEV" if all(m["joint_score"] >= .9 for m in metrics.values()) else "TRILINGUAL_PILOT_TARGET_NOT_MET"
    detector.contract["current_text_truncation"] = "right_keep_first; overlength input explicitly flagged by runner"
    write_json(model / "turn_detector_config.json", detector.contract)
    np.savez_compressed(evidence / "student_dev_predictions.npz", labels=labels, prob_EOT=probs, language=languages)
    expected = np.load(student / "dev_predictions.npz")
    training_delta = float(np.max(np.abs(probs - expected["prob_EOT"])))
    strict = np.load(evidence / "strict_gpu_reference.npz")
    assert np.array_equal(labels, strict["labels"])
    parity = float(np.max(np.abs(probs - strict["prob_EOT"])))
    assert parity < 1e-4
    vi_rows = list(read_jsonl(Path("data_v2/frozen/dev.jsonl")))
    vi_probs = np.asarray([detector.predict(r["text"], r.get("context", ""), language="vi")["prob_EOT"] for r in vi_rows])
    vi_labels = np.asarray([int(r["label"] == "EOT") for r in vi_rows])
    vi_metrics = classification_metrics(vi_labels, vi_probs, detector.contract["language_thresholds"]["vi"])
    np.savez_compressed(evidence / "vi_full_dev_predictions.npz", labels=vi_labels, prob_EOT=vi_probs)
    result = {"scope": "STRUCTURAL_PROXY_DEV_SELECTION; full VI regression is previously exposed dev", "dev": metrics,
        "vi_full_dev": vi_metrics, "strict_gpu_reference_vs_packed_dev_max_delta": parity,
        "original_training_eval_vs_packed_max_delta": training_delta,
        "precision_note": "Training cuDNN TF32 was enabled; verify exact same QAT weights with TF32 disabled. GPU fake-quant weights match packed decode exactly.",
        "quality_target_all_languages_met": all(m["joint_score"] >= .9 for m in metrics.values()),
        "full_vi_target_met": vi_metrics["joint_score"] >= .9, "test_evaluated": False, "old_holdout_evaluated": False,
        "deployed_thresholds": detector.contract["language_thresholds"], "model_sha256": file_hash(model / "model.onnx"),
        "torch_imported": "torch" in sys.modules, "transformers_imported": "transformers" in sys.modules,
        "onnx_bytes": (model / "model.onnx").stat().st_size, "deployment_bytes": sum(p.stat().st_size for p in model.iterdir() if p.is_file())}
    write_json(evidence / "student_dev_report.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__": main()
