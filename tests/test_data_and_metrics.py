import json
from pathlib import Path

import numpy as np
import pytest

from turn_v2.common import encode_pair, input_id, normalize, read_jsonl, write_jsonl
from turn_v2.crawl import candidates_from_extract
from turn_v2.label_qwen import acceptance, parse_label
from turn_v2.metrics import calibrate_on_dev, classification_metrics
from turn_v2.prepare import augmentation_filter, infer_groups, prepare


@pytest.fixture
def frozen(tmp_path):
    rows = []
    for index in range(80):
        rows.extend([
            {"text": f"nhóm {index} cần hỏi đường đến địa điểm nào", "context": "", "label": "EOT"},
            {"text": f"nhóm {index} cần hỏi đường đến", "context": "", "label": "NEOT"}])
    train, legacy = tmp_path / "train.jsonl", tmp_path / "legacy.jsonl"
    write_jsonl(train, rows)
    write_jsonl(legacy, [{"text": "nội dung cũ ngoài tập mới", "context": "", "label": "EOT"}])
    output = tmp_path / "frozen"
    manifest = prepare(train, legacy, output, dev_fraction=0.2, holdout_fraction=0.2)
    return train, legacy, output, manifest


def test_family_siblings_stay_together():
    rows = [{"text": "tôi muốn đặt vé đi hà nội", "label": "EOT"},
            {"text": "tôi muốn đặt vé đi", "label": "NEOT"},
            {"text": "tôi muốn", "label": "NEOT"}]
    grouped, orphans = infer_groups(rows)
    assert len({row["group_id"] for row in grouped}) == 1
    assert orphans == 0


def test_groups_and_inputs_do_not_cross_splits(frozen):
    _, _, root, _ = frozen
    rows = {name: list(read_jsonl(root / f"{name}.jsonl")) for name in ("train", "dev", "holdout")}
    for first, second in [("train", "dev"), ("train", "holdout"), ("dev", "holdout")]:
        assert not {r["group_id"] for r in rows[first]} & {r["group_id"] for r in rows[second]}
        assert not {input_id(r) for r in rows[first]} & {input_id(r) for r in rows[second]}


def test_frozen_splits_are_idempotent_and_detect_tampering(frozen):
    train, legacy, root, manifest = frozen
    assert prepare(train, legacy, root, dev_fraction=0.2, holdout_fraction=0.2) == manifest
    with (root / "dev.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="has changed"):
        prepare(train, legacy, root, dev_fraction=0.2, holdout_fraction=0.2)


def test_legacy_overlap_removes_whole_family(frozen, tmp_path):
    train, _, _, _ = frozen
    rows = list(read_jsonl(train))
    legacy = tmp_path / "other_legacy.jsonl"
    write_jsonl(legacy, [rows[1]])
    root = tmp_path / "other_split"
    prepare(train, legacy, root, dev_fraction=0.2, holdout_fraction=0.2)
    forbidden = {input_id(row) for row in rows[:2]}
    for name in ("train", "dev", "holdout"):
        assert forbidden.isdisjoint(input_id(row) for row in read_jsonl(root / f"{name}.jsonl"))


def test_conflicting_labels_are_quarantined(frozen, tmp_path):
    train, legacy, _, _ = frozen
    rows = list(read_jsonl(train))
    rows.append({**rows[0], "label": "NEOT"})
    alternate = tmp_path / "conflicting_train.jsonl"
    write_jsonl(alternate, rows)
    manifest = prepare(alternate, legacy, tmp_path / "conflicts", dev_fraction=0.2, holdout_fraction=0.2)
    assert manifest["quarantine_reasons"]["conflicting_labels_for_same_input"] == 2


def test_silver_cannot_enter_protected_inputs_or_prefixes(frozen):
    _, legacy, root, _ = frozen
    held = next(read_jsonl(root / "holdout.jsonl"))
    while len(held["text"].split()) < 8:
        held = next(row for row in read_jsonl(root / "holdout.jsonl") if len(row["text"].split()) >= 8)
    common = {"context": "", "label": "EOT", "label_source": "qwen_agreement_silver", "source_id": "crawl:1"}
    silver = [{**common, "text": held["text"]},
              {**common, "text": held["text"] + " ngay bây giờ"},
              {**common, "text": "xin cảm ơn bạn đã hỗ trợ tôi hôm nay"}]
    accepted, rejected = augmentation_filter(silver, root, legacy)
    assert len(accepted) == 1 and len(rejected) == 2
    assert accepted[0]["sample_weight"] == 0.5


@pytest.mark.parametrize("raw", [
    '{"label":"YES","confidence":0.99,"reason":"x"}',
    '{"label":"EOT","confidence":true,"reason":"x"}',
    '{"label":"EOT","confidence":NaN,"reason":"x"}',
    'not json'])
def test_malformed_teacher_output_is_rejected(raw):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_label(raw)


def test_teacher_disagreement_and_ambiguity_are_not_training_labels():
    good = {"label": "EOT", "confidence": 0.95, "reason": "đủ ý"}
    assert acceptance([good, good]) == "EOT"
    assert acceptance([good, {**good, "label": "NEOT"}]) is None
    assert acceptance([{**good, "label": "AMBIGUOUS"}] * 2) is None
    assert acceptance([good, {**good, "confidence": 0.5}]) is None


def test_crawl_does_not_assign_labels_or_leak_terminal_punctuation():
    rows = candidates_from_extract("demo", 12, "Tôi muốn đặt vé máy bay đi Hà Nội vào ngày mai.")
    assert len(rows) == 3
    assert all("label" not in row for row in rows)
    assert all(row["source_url"] == "https://vi.wikipedia.org/?curid=12" for row in rows)
    assert all(row["text"] == normalize(row["text"]) for row in rows)


def test_metrics_capture_premature_end_and_missed_end():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8])
    assert result["confusion_matrix_NEOT_EOT"] == [[1, 1], [1, 1]]
    assert result["accuracy"] == result["eot_f1"] == 0.5


def test_dev_calibration_is_not_permitted_to_mutate_inputs():
    labels, probs = np.array([0, 0, 1, 1]), np.array([0.2, 0.6, 0.8, 0.9])
    before = probs.copy()
    selected = calibrate_on_dev(labels, probs)
    assert selected["accuracy"] == selected["eot_f1"] == 1
    assert 0.6 < selected["threshold"] <= 0.8
    assert np.array_equal(probs, before)


def test_current_utterance_survives_long_history():
    class Tokenizer:
        cls_token_id, sep_token_id = 101, 102
        def encode(self, text, add_special_tokens=False):
            return [int(word) for word in text.split()]
    encoded = encode_pair(Tokenizer(), "1 2 3 4 5 6", "7 8 9", max_length=8)
    assert encoded["input_ids"] == [101, 5, 6, 102, 7, 8, 9, 102]
    assert sum(encoded["attention_mask"]) == 8
