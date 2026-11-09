"""One final holdout evaluation after dev selection passes both 90% targets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from .common import LABEL2ID, encode_pair, file_hash, read_jsonl, write_json, write_jsonl
from .metrics import classification_metrics, probabilities


def evaluate(model_dir: Path, data_dir: Path, output: Path):
    contract = json.loads((model_dir / "turn_detector_config.json").read_text(encoding="utf-8"))
    if contract["smoke_only"]:
        raise ValueError("A smoke model cannot be submitted for the final target claim")
    if contract["manifest_sha256"] != file_hash(data_dir / "manifest.json"):
        raise ValueError("Model and frozen holdout manifest do not match")
    report_path = model_dir.parent / "training_report.json"
    dev_report = json.loads(report_path.read_text(encoding="utf-8"))["dev_selected_threshold"]
    if dev_report["accuracy"] < 0.9 or dev_report["eot_f1"] < 0.9:
        raise ValueError("Dev has not reached both targets. Keep holdout sealed and improve using train/dev.")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if file_hash(data_dir / "holdout.jsonl") != manifest["split_sha256"]["holdout"]:
        raise ValueError("Holdout file has changed")
    weight_files = sorted([*model_dir.glob("*.safetensors"), *model_dir.glob("pytorch_model*.bin")])
    if not weight_files:
        raise ValueError("No PyTorch weights found")
    identity = {"weight_sha256": {path.name: file_hash(path) for path in weight_files},
                "contract_sha256": file_hash(model_dir / "turn_detector_config.json")}
    marker = data_dir / "holdout_consumed.json"
    if marker.exists():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old["model_identity"] == identity and Path(old["report"]).exists():
            return json.loads(Path(old["report"]).read_text(encoding="utf-8"))
        raise ValueError("This holdout was already opened. Do not tune another candidate against it; create a new independent evaluation set.")
    output.mkdir(parents=True, exist_ok=True)
    final_report = (output / "holdout_report.json").resolve()
    receipt = {"opened_at": datetime.now(timezone.utc).isoformat(), "model_identity": identity,
               "report": str(final_report), "threshold_selected_on": "dev"}
    # Atomic reservation prevents two candidates from opening it concurrently.
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    rows = list(read_jsonl(data_dir / "holdout.jsonl"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    collate = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    probs = []
    for start in range(0, len(rows), 16):
        encoded = [encode_pair(tokenizer, row.get("context", ""), row["text"], contract["max_length"])
                   for row in rows[start:start + 16]]
        batch = collate(encoded).to(device)
        with torch.inference_mode():
            probs.extend(probabilities(model(**batch).logits.float().cpu().numpy()).tolist())
    metrics = classification_metrics([LABEL2ID[row["label"]] for row in rows], probs, contract["threshold"])
    result = {"status": "TARGET_MET" if metrics["accuracy"] >= 0.9 and metrics["eot_f1"] >= 0.9 else "TARGET_NOT_MET",
        "metrics": metrics, "threshold_selected_on": "dev", "threshold_frozen": contract["threshold"],
        "target": {"accuracy": 0.9, "eot_f1": 0.9}, "model_identity": identity,
        "manifest_sha256": contract["manifest_sha256"],
        "scope": "Provided-label holdout with heuristic source grouping; not a human-reviewed real-dialogue evaluation"}
    write_json(final_report, result)
    write_jsonl(output / "holdout_predictions.jsonl", [{**row, "prob_EOT": prob,
        "prediction": "EOT" if prob >= contract["threshold"] else "NEOT"}
        for row, prob in zip(rows, probs, strict=True)])
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data_v2/frozen"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.model, args.data, args.output)
