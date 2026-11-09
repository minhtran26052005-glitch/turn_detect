import gzip
import hashlib
import json
from turn_detection.summarize_test import summarize


def test_summarize_uses_only_clean_rows_and_frozen_predictions(tmp_path):
    path = tmp_path / "trace.jsonl.gz"
    rows = [
        {"included_in_clean_test": True, "label": "EOT", "predicted": "EOT", "p_eot": 0.8, "threshold_from_dev": 0.6},
        {"included_in_clean_test": True, "label": "NEOT", "predicted": "EOT", "p_eot": 0.7, "threshold_from_dev": 0.6},
        {"included_in_clean_test": False, "label": "NEOT", "predicted": "NEOT", "p_eot": 0.1, "threshold_from_dev": 0.6},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    result = summarize(path, {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert result["samples"] == 2 and result["accuracy"] == 0.5
    assert result["tp"] == 1 and result["fp"] == 1
