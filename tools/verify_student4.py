"""Verify the delivered native-only INT4 package on every frozen dev input."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("outputs/quant4_20261024/native_verification.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    package = root / "models/turn_v2_distilled_int4_20261024"
    model = package / "onnx"
    manifest = json.loads((package / "model_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = model / name
        assert path.stat().st_size == expected["bytes"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected["sha256"]
    spec = importlib.util.spec_from_file_location("native_int4", model / "run_model.py")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    detector = module.TinyOnnxDetector(model)
    rows = [json.loads(line) for line in (root / "data_v2/frozen/dev.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    predictions = [detector.predict(r["text"], r.get("context", ""))["prob_EOT"] for r in rows]
    reference = np.load(root / "outputs/quant4_20261024/int4_g32_quality/dev_predictions.npz")
    probability_delta = float(np.max(np.abs(np.asarray(predictions) - reference["prob_EOT"])))
    labels = np.asarray([int(r["label"] == "EOT") for r in rows])
    assert np.array_equal(labels, reference["labels"])
    chosen = np.asarray(predictions) >= detector.contract["threshold"]
    tp = int(((labels == 1) & chosen).sum())
    fp = int(((labels == 0) & chosen).sum())
    fn = int(((labels == 1) & ~chosen).sum())
    metrics = {"accuracy": float((chosen == labels).mean()), "eot_f1": 2 * tp / (2 * tp + fp + fn), "false_eot": fp, "missed_eot": fn}
    texts = ["hà nội là thủ đô và là một trong những trung tâm văn hóa lớn của việt nam", "tôi muốn đặt vé đi", "cảm ơn bạn"]
    edges = [("", ""), ("XIN CHÀO!!! ＡＢＣ１２３", "Xin chào? Tôi muốn hỏi:"),
             ("cảm ơn bạn đã giúp đỡ ❤", "a " * 1000), ("xin chào " * 300, "ngữ cảnh " * 500)]
    result = {"hashes_verified": True, "model_sha256": manifest["files"]["model.onnx"]["sha256"],
        "samples": len(rows), "full_dev_probability_delta_vs_evaluator": probability_delta,
        "dev_metrics": metrics, "torch_imported": "torch" in sys.modules, "transformers_imported": "transformers" in sys.modules,
        "demos": [{"text": text, **detector.predict(text)} for text in texts],
        "edge_cases": [{"text": text, "context": context, **detector.predict(text, context)} for text, context in edges],
        "holdout_evaluated": False}
    result["passed"] = (len(rows) == 17833 and probability_delta < 1e-6 and metrics["accuracy"] >= .9 and metrics["eot_f1"] >= .9
                        and not result["torch_imported"] and not result["transformers_imported"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "edge_cases"}))
    assert result["passed"]


if __name__ == "__main__":
    main()
