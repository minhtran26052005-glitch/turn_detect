"""Summarize saved C75 predictions without running inference."""
import argparse
import gzip
import json
from pathlib import Path

from .assets import sha256


def summarize(path, expected):
    if sha256(path) != expected["sha256"]:
        raise ValueError(f"trace SHA-256 mismatch: {path}")
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    threshold = None
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not row["included_in_clean_test"]:
                continue
            if threshold is None:
                threshold = row["threshold_from_dev"]
            if row["threshold_from_dev"] != threshold:
                raise ValueError("mixed thresholds in frozen trace")
            expected_prediction = "EOT" if row["p_eot"] >= threshold else "NEOT"
            if row["predicted"] != expected_prediction:
                raise ValueError("stored prediction disagrees with threshold")
            label, predicted = row["label"], row["predicted"]
            if label not in ("EOT", "NEOT") or predicted not in ("EOT", "NEOT"):
                raise ValueError("unknown label")
            key = {("EOT", "EOT"): "tp", ("NEOT", "NEOT"): "tn",
                   ("NEOT", "EOT"): "fp", ("EOT", "NEOT"): "fn"}[(label, predicted)]
            counts[key] += 1
    tp, tn, fp, fn = (counts[k] for k in ("tp", "tn", "fp", "fn"))
    n = tp + tn + fp + fn
    if not n:
        raise ValueError("empty clean test")
    return {"samples": n, "accuracy": (tp + tn) / n, "eot_f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "threshold": threshold, **counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.input / "manifest.json"
    raw = not manifest_path.exists()
    if raw:
        manifest_path = Path(__file__).resolve().parents[1] / "data/test/trace_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = {language: summarize(args.input / (f"test_predictions_{language}.jsonl.gz" if raw else entry["name"]), entry)
               for language, entry in manifest["files"].items()}
    expected = manifest["clean_full_test_rows"] if raw else sum(entry["rows"] for entry in manifest["files"].values())
    if sum(value["samples"] for value in results.values()) != expected:
        raise ValueError("clean row count differs from published test")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
