import json

import pytest

from turn_v2.common import digest, input_id
from turn_v2.crawl import iter_candidates
from turn_v2.crawl_predict import CandidateGuard, load_detector, render_report, score_stream, select_review


def candidate(text, identifier="1", **extra):
    return {"candidate_id": identifier, "text": text, "context": "", "source_id": "wiki:1",
            "source_url": "https://vi.wikipedia.org/?curid=1", **extra}


class Detector:
    def predict(self, text, context=""):
        return {"prediction": "EOT", "prob_EOT": 0.8, "threshold": 0.75, "smoke_only": True}


def test_prediction_is_flushed_before_next_candidate_is_requested(tmp_path):
    path = tmp_path / "predictions.jsonl"
    def stream():
        yield candidate("mẫu đầu đã đầy đủ")
        first = json.loads(path.read_text(encoding="utf-8"))
        assert first["prediction"] == "EOT"
        yield candidate("mẫu thứ hai", "2")
    rows, _ = score_stream(stream(), Detector(), path, CandidateGuard())
    assert len(rows) == 2


def test_prediction_never_becomes_a_training_label(tmp_path):
    rows, skipped = score_stream([candidate("xin chào", label="NEOT"),
                                  candidate("Xin chào!", "2")], Detector(),
                                 tmp_path / "rows.jsonl", CandidateGuard())
    assert len(rows) == 1 and "label" not in rows[0]
    assert rows[0]["label_source"] == "model_prediction_not_a_training_label"
    assert skipped["duplicate_or_protected_input"] == 1


def test_source_prefix_and_exact_protection():
    exact = candidate("đầu vào đã có")
    guard = CandidateGuard(exact=[input_id(exact)], source_ids=["blocked"],
                           prefixes=[digest("một hai ba")], prefix_words=3)
    assert guard.reject_reason(exact) == "duplicate_or_protected_input"
    assert guard.reject_reason(candidate("khác", source_id="blocked")) == "protected_source"
    assert guard.reject_reason(candidate("một hai ba bốn")) == "protected_long_prefix"


def test_review_queue_uses_saved_threshold_rather_than_half():
    rows = [dict(candidate("a"), decision_margin=abs(0.79 - 0.8)),
            dict(candidate("b", "2"), decision_margin=abs(0.5 - 0.8))]
    assert select_review(rows, 1)[0]["text"] == "a"


def test_review_also_samples_overconfident_eot_prefixes():
    rows = [{**candidate("near", "1"), "decision_margin": 0.01, "prediction": "EOT"},
            {**candidate("near two", "2"), "decision_margin": 0.02, "prediction": "NEOT"},
            {**candidate("unfinished", "3", candidate_kind="prefix"), "decision_margin": 0.49, "prediction": "EOT"},
            {**candidate("other", "4"), "decision_margin": 0.1, "prediction": "NEOT"}]
    selected = select_review(rows, 4)
    assert len(selected) == len({row["candidate_id"] for row in selected}) == 4
    assert next(row for row in selected if row["candidate_id"] == "3")["review_selection_reason"] == "high_probability_eot_on_a_prefix"


def test_smoke_checkpoint_requires_explicit_opt_in(tmp_path):
    (tmp_path / "turn_detector_config.json").write_text(json.dumps({"smoke_only": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="smoke checkpoint"):
        load_detector(tmp_path, "cpu")


def test_invalid_probability_is_not_written_as_a_prediction(tmp_path):
    class Invalid:
        def predict(self, text, context=""):
            return {"prob_EOT": float("nan"), "threshold": 0.5}
    with pytest.raises(ValueError, match="invalid probability"):
        score_stream([candidate("x")], Invalid(), tmp_path / "invalid.jsonl", CandidateGuard())
    assert (tmp_path / "invalid.jsonl").read_text() == ""


def test_html_escapes_crawled_text(tmp_path):
    path = tmp_path / "results.html"
    row = {**candidate("<script>alert(1)</script>"), "prediction": "EOT", "prob_EOT": 0.8, "threshold": 0.75}
    render_report(path, [row], {"model": {"smoke_only": True}})
    text = path.read_text(encoding="utf-8")
    assert "<script>" not in text and "&lt;script&gt;" in text


def test_crawler_stops_before_fetching_the_next_page(monkeypatch):
    requested = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"query": {"pages": [{"title": "demo", "pageid": 12,
                "extract": "Tôi muốn đặt vé máy bay đi Hà Nội vào ngày mai."}]}}
    class Session:
        headers = {}
        def get(self, url, params, timeout):
            requested.append(params["titles"])
            return Response()
    monkeypatch.setattr("turn_v2.crawl.requests.Session", Session)
    rows = list(iter_candidates(["one", "two"], max_candidates=1))
    assert len(rows) == 1 and requested == ["one"]
