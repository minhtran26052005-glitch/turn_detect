"""Crawl -> predict immediately -> optionally ask Qwen to review a small queue.

Predictions are observations, not training labels. This module never updates
model weights or computes an RL reward, and never evaluates holdout quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import html
import json
import math
from pathlib import Path
import time

from .common import digest, file_hash, input_id, normalize, read_jsonl, write_json, write_jsonl
from .crawl import DEFAULT_TITLES, iter_candidates


class CandidateGuard:
    """Filter exact duplicates/protected prefixes using text and provenance only."""
    def __init__(self, exact=(), source_ids=(), prefixes=(), prefix_words=8):
        self.exact = set(exact)
        self.sources = set(source_ids)
        self.prefixes = set(prefixes)
        self.prefix_words = prefix_words

    @classmethod
    def from_frozen(cls, root, legacy_test):
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        exact, sources, prefixes = set(), set(), set()
        words = manifest["split_settings"]["prefix_words"]
        for name in ("train", "dev", "holdout"):
            path = root / f"{name}.jsonl"
            if file_hash(path) != manifest["split_sha256"][name]:
                raise ValueError(f"Frozen {name} split has changed")
            for row in read_jsonl(path):
                exact.add(input_id(row))
                if name != "train":
                    if row.get("source_id"):
                        sources.add(row["source_id"])
                    tokens = normalize(row["text"]).split()
                    if len(tokens) >= words:
                        prefixes.add(digest(" ".join(tokens[:words])))
        if file_hash(legacy_test) != manifest["input_sha256"]["legacy_test"]:
            raise ValueError("Legacy evaluation file has changed")
        for row in read_jsonl(legacy_test):
            exact.add(input_id(row))
            if row.get("source_id"):
                sources.add(row["source_id"])
            tokens = normalize(row["text"]).split()
            if len(tokens) >= words:
                prefixes.add(digest(" ".join(tokens[:words])))
        return cls(exact, sources, prefixes, words)

    def reject_reason(self, row):
        key = input_id(row)
        if key in self.exact:
            return "duplicate_or_protected_input"
        if row.get("source_id") in self.sources:
            return "protected_source"
        tokens = normalize(row["text"]).split()
        if len(tokens) >= self.prefix_words and digest(" ".join(tokens[:self.prefix_words])) in self.prefixes:
            return "protected_long_prefix"
        self.exact.add(key)
        return None


def load_detector(path, device, allow_smoke=False):
    if (path / "turn_detector_config.json").exists():
        from .predict import TurnDetectorV2
        contract = json.loads((path / "turn_detector_config.json").read_text(encoding="utf-8"))
        if contract.get("smoke_only") and not allow_smoke:
            raise ValueError("This is a smoke checkpoint; pass --allow-smoke only for a functional test")
        model = TurnDetectorV2(path, device)
        return model, {"format": "v2", "smoke_only": contract.get("smoke_only", False),
                       "threshold": contract["threshold"]}
    # The supplied legacy checkpoint has a different normalization contract.
    from text_based import TurnDetectorONNX
    old = TurnDetectorONNX(str(path), device=device)

    class Legacy:
        def predict(self, text, context=""):
            value = old.predict(context, text)
            return {"prediction": "EOT" if value["predicted_class"] else "NEOT",
                    "prob_EOT": value["prob_EOT"], "threshold": 0.5, "smoke_only": False}
    return Legacy(), {"format": "legacy", "smoke_only": False, "threshold": 0.5}


def score_stream(candidates, detector, destination, guard, review_margin=0.15):
    """Write and flush each result before requesting the next candidate."""
    predictions, skipped = [], Counter()
    with destination.open("w", encoding="utf-8") as output:
        for row in candidates:
            reason = guard.reject_reason(row)
            if reason:
                skipped[reason] += 1
                continue
            start = time.perf_counter()
            value = detector.predict(row["text"], row.get("context", ""))
            prob, threshold = value["prob_EOT"], value["threshold"]
            if not math.isfinite(prob) or not 0 <= prob <= 1:
                raise ValueError("Model returned an invalid probability")
            margin = abs(prob - threshold)
            result = {**row, "prediction": value["prediction"], "prob_EOT": prob,
                "threshold": threshold, "decision_margin": margin,
                "near_threshold": margin <= review_margin, "smoke_only": value["smoke_only"],
                "prediction_latency_ms": (time.perf_counter() - start) * 1000,
                "label_source": "model_prediction_not_a_training_label"}
            # No 'label' key: this prediction must not silently become reference truth.
            result.pop("label", None)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            predictions.append(result)
            print(json.dumps({key: result[key] for key in ("text", "prediction", "prob_EOT", "near_threshold")},
                             ensure_ascii=False), flush=True)
    return predictions, dict(skipped)


def select_review(rows, budget):
    if budget <= 0:
        return []
    selected, used = [], set()

    def take(candidates, count, reason):
        for row in candidates:
            if row["candidate_id"] not in used:
                selected.append({**row, "review_selection_reason": reason})
                used.add(row["candidate_id"])
                count -= 1
                if count == 0:
                    break

    ranked = sorted(rows, key=lambda row: (row["decision_margin"], row["candidate_id"]))
    take(ranked, min(len(rows), (budget + 1) // 2), "near_decision_threshold")
    remaining = min(budget, len(rows)) - len(selected)
    risky = sorted([row for row in rows if row.get("candidate_kind") == "prefix"
                    and row["prediction"] == "EOT"], key=lambda row: -row["decision_margin"])
    if remaining:
        take(risky, 1, "high_probability_eot_on_a_prefix")
    remaining = min(budget, len(rows)) - len(selected)
    if remaining:
        take(sorted(rows, key=lambda row: digest("coverage-v1:" + row["candidate_id"])), remaining,
             "deterministic_coverage_sample")
    return selected


def render_report(path, rows, report):
    body = []
    for row in rows:
        teacher = row.get("teacher_label") or row.get("teacher_status", "chưa xét")
        body.append("<tr><td>" + html.escape(row.get("candidate_kind", "")) + "</td><td>"
            + html.escape(row["text"]) + "</td><td>" + html.escape(row["prediction"])
            + f"</td><td>{row['prob_EOT']:.4f}</td><td>{row['threshold']:.4f}</td><td>"
            + html.escape(teacher) + '</td><td><a href="' + html.escape(row["source_url"], quote=True)
            + '">nguồn</a></td></tr>')
    notice = "CHECKPOINT SMOKE — chỉ kiểm tra luồng chạy." if report["model"]["smoke_only"] else "Dự đoán trên dữ liệu mới; chưa có nhãn tham chiếu để tính accuracy."
    path.write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><title>Crawl và dự đoán EOT</title>'
        '<style>body{font:16px system-ui;margin:32px;color:#172338;max-width:1300px}table{border-collapse:collapse;width:100%}'
        'td,th{border:1px solid #d7e0ea;padding:10px;text-align:left}th{background:#edf4fb}small{color:#506176}</style>'
        '<h1>Crawl → dự đoán EOT</h1><p><b>' + html.escape(notice) + '</b></p><p>'
        + str(len(rows)) + ' mẫu. Xác suất EOT là đầu ra model, chưa phải độ chính xác. Nhãn Qwen là silver.</p>'
        '<table><thead><tr><th>Loại</th><th>Văn bản ASR</th><th>Dự đoán</th><th>P(EOT)</th><th>Ngưỡng</th><th>Qwen</th><th>Nguồn</th></tr></thead>'
        '<tbody>' + ''.join(body) + '</tbody></table><p><small>Không cập nhật trọng số, không thực hiện RL và không mở holdout để đánh giá.</small></p></html>', encoding="utf-8")


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new run name")
    guard = CandidateGuard.from_frozen(args.protected_data, args.legacy_test)
    detector, model_info = load_detector(args.model, args.device, args.allow_smoke)
    args.output.mkdir(parents=True, exist_ok=True)
    sources = []
    predictions, skipped = score_stream(iter_candidates(args.titles, args.limit, sources), detector,
        args.output / "predictions.jsonl", guard, args.review_margin)
    if not predictions:
        raise RuntimeError("No new usable candidates; inspect titles or duplicate filtering")
    queue = select_review(predictions, args.review_budget)
    write_jsonl(args.output / "review_queue.jsonl", queue)
    del detector
    gc.collect()
    if args.device == "cuda":
        import torch
        torch.cuda.empty_cache()
    teacher_report = None
    if args.qwen_review and queue:
        from .label_qwen import label
        teacher_report = label(argparse.Namespace(input=args.output / "review_queue.jsonl",
            output=args.output / "qwen", model=args.teacher_model, limit=len(queue), min_confidence=0.9))
        audit = {row["candidate_id"]: row for row in read_jsonl(args.output / "qwen/audit.jsonl")}
        silver = {row["candidate_id"]: row for row in read_jsonl(args.output / "qwen/silver.jsonl")}
        for row in predictions:
            key = row["candidate_id"]
            if key in silver:
                row["teacher_label"] = silver[key]["label"]
                row["teacher_status"] = "accepted_silver"
                row["model_teacher_disagree"] = row["prediction"] != silver[key]["label"]
            elif key in audit:
                row["teacher_status"] = "ambiguous_disagreed_or_invalid"
        write_jsonl(args.output / "reviewed_predictions.jsonl", predictions)
    files = [*args.model.glob("*.onnx"), *args.model.glob("*.safetensors"),
             *args.model.glob("config.json"), *args.model.glob("turn_detector_config.json")]
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SMOKE_INFERENCE_ONLY" if model_info["smoke_only"] else "INFERENCE_COMPLETED",
        "model": {**model_info, "path": str(args.model),
                  "sha256": {path.name: file_hash(path) for path in files}},
        "predictions": len(predictions), "prediction_counts": dict(Counter(row["prediction"] for row in predictions)),
        "skipped": skipped, "review_queue_size": len(queue), "teacher": teacher_report,
        "accepted_teacher_disagreements": sum(row.get("model_teacher_disagree", False) for row in predictions),
        "sources": sources, "training_performed": False, "rl_performed": False,
        "holdout_evaluated": False, "accuracy": None,
        "review_selection": "half near the decision threshold, one high-probability EOT prefix, remaining deterministic coverage; probabilities uncalibrated",
        "limits": "Crawled prose and silver labels do not establish dialogue accuracy or a 90% result."}
    write_json(args.output / "report.json", report)
    render_report(args.output / "results.html", predictions, report)
    print(json.dumps({"output": str(args.output), "predictions": len(predictions), "rl_performed": False}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--titles", nargs="+", default=DEFAULT_TITLES)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--review-budget", type=int, default=4)
    parser.add_argument("--review-margin", type=float, default=0.15)
    parser.add_argument("--protected-data", type=Path, default=Path("data_v2/frozen"))
    parser.add_argument("--legacy-test", type=Path, default=Path("test.jsonl"))
    parser.add_argument("--qwen-review", action="store_true")
    parser.add_argument("--teacher-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or args.review_budget < 0 or not 0 <= args.review_margin <= 1:
        parser.error("Invalid limit, review budget or margin")
    run(args)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
