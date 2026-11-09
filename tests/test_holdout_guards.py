import json

import pytest

from turn_v2.common import file_hash, write_json
from turn_v2.evaluate import evaluate


def test_smoke_model_cannot_open_holdout(tmp_path):
    model = tmp_path / "best"
    write_json(model / "turn_detector_config.json", {"smoke_only": True})
    with pytest.raises(ValueError, match="smoke model"):
        evaluate(model, tmp_path / "data", tmp_path / "report")
    assert not (tmp_path / "data/holdout_consumed.json").exists()


def test_both_dev_targets_required_before_holdout(tmp_path):
    model, data = tmp_path / "best", tmp_path / "data"
    write_json(data / "manifest.json", {"schema": 1})
    write_json(model / "turn_detector_config.json", {"smoke_only": False,
        "manifest_sha256": file_hash(data / "manifest.json")})
    write_json(tmp_path / "training_report.json", {"dev_selected_threshold": {"accuracy": 0.95, "eot_f1": 0.8}})
    with pytest.raises(ValueError, match="Dev has not reached both targets"):
        evaluate(model, data, tmp_path / "report")
    assert not (data / "holdout_consumed.json").exists()


def test_changed_holdout_rejected_before_model_loading(tmp_path):
    model, data = tmp_path / "best", tmp_path / "data"
    write_json(data / "manifest.json", {"split_sha256": {"holdout": "different-hash"}})
    (data / "holdout.jsonl").write_text('{"text":"changed"}\n', encoding="utf-8")
    write_json(model / "turn_detector_config.json", {"smoke_only": False,
        "manifest_sha256": file_hash(data / "manifest.json")})
    write_json(tmp_path / "training_report.json", {"dev_selected_threshold": {"accuracy": 0.95, "eot_f1": 0.95}})
    with pytest.raises(ValueError, match="Holdout file has changed"):
        evaluate(model, data, tmp_path / "report")
    assert not (data / "holdout_consumed.json").exists()
