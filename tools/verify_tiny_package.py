"""Native-only full-dev verification for any packaged TinyOnnxDetector model."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="Full dev_predictions.npz")
    parser.add_argument("--dev", type=Path, default=Path("data_v2/frozen/dev.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.package / "model_manifest.json").read_text(encoding="utf-8"))
    folder = args.package / "onnx"
    for name, identity in manifest["files"].items():
        path = folder / name
        assert path.stat().st_size == identity["bytes"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == identity["sha256"]
    spec = importlib.util.spec_from_file_location("portable_tiny_model", folder / "run_model.py")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    detector = module.TinyOnnxDetector(folder)
    rows = [json.loads(line) for line in args.dev.read_text(encoding="utf-8").splitlines() if line.strip()]
    actual = np.array([detector.predict(row["text"], row.get("context", ""))["prob_EOT"] for row in rows])
    reference = np.load(args.reference)
    labels = np.asarray([int(r["label"] == "EOT") for r in rows])
    assert np.array_equal(labels, reference["labels"])
    delta = float(np.max(np.abs(actual - reference["prob_EOT"])))
    predicted = actual >= detector.contract["threshold"]
    tp = int(((labels == 1) & predicted).sum())
    fp = int(((labels == 0) & predicted).sum())
    fn = int(((labels == 1) & ~predicted).sum())
    metrics = {"accuracy": float((predicted == labels).mean()), "eot_f1": 2 * tp / (2 * tp + fp + fn), "false_eot": fp, "missed_eot": fn}
    texts = ["hà nội là thủ đô và là một trong những trung tâm văn hóa lớn của việt nam", "tôi muốn đặt vé đi", "cảm ơn bạn"]
    edges = [("", ""), ("XIN CHÀO!!! ＡＢＣ１２３", "Xin chào? Tôi muốn hỏi:"),
             ("cảm ơn bạn đã giúp đỡ ❤", "a " * 1000), ("xin chào " * 300, "ngữ cảnh " * 500)]
    result = {"model_sha256": manifest["files"]["model.onnx"]["sha256"], "hashes_verified": True, "samples": len(rows),
        "max_probability_delta_vs_reference": delta, "dev_metrics": metrics,
        "torch_imported": "torch" in sys.modules, "transformers_imported": "transformers" in sys.modules,
        "demos": [{"text": text, **detector.predict(text)} for text in texts],
        "edge_cases": [{"text": text, "context": context, **detector.predict(text, context)} for text, context in edges], "holdout_evaluated": False}
    result["passed"] = (len(rows) == 17833 and delta < 1e-6 and metrics["accuracy"] >= .9 and metrics["eot_f1"] >= .9
                        and not result["torch_imported"] and not result["transformers_imported"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "edge_cases"}))
    assert result["passed"]


if __name__ == "__main__":
    main()
