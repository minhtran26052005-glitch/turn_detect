"""Replay crawled prose word/character by word/character, including punctuation."""
import argparse
from collections import defaultdict
import importlib.util
import inspect
import json
from pathlib import Path
import re
import sys

import numpy as np

from turn_v2.common import normalize, read_jsonl, write_json, write_jsonl


def events(text, language):
    pattern = r"[\u3400-\u9fff]|[A-Za-z0-9]+|[^\s]" if language == "zh" else r"\w+(?:['’\-]\w+)*|[^\w\s]"
    return list(re.finditer(pattern, text))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, default=Path("models/turn_v2_distilled_qat4_20261024"))
    p.add_argument("--input", type=Path, default=Path("outputs/trilingual_stream_20261024/crawl/paragraphs.jsonl"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--paragraphs-per-language", type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    folder = args.package / "onnx"
    spec = importlib.util.spec_from_file_location("stream_native", folder / "run_model.py")
    module = importlib.util.module_from_spec(spec)
    prior = sys.dont_write_bytecode; sys.dont_write_bytecode = True
    try: spec.loader.exec_module(module)
    finally: sys.dont_write_bytecode = prior
    detector = module.TinyOnnxDetector(folder)
    language_argument = "language" in inspect.signature(detector.predict).parameters
    rows = list(read_jsonl(args.input))
    selected = []
    for lang in ["vi", "en", "zh"]:
        selected.extend([r for r in rows if r["language"] == lang][:args.paragraphs_per_language])
    trace, paragraph_reports = [], []
    summaries = defaultdict(list)
    punctuation_deltas = []
    for paragraph_id, row in enumerate(selected):
        language = row["language"]
        detector.contract["threshold"] = detector.contract.get("language_thresholds", {}).get(language, detector.contract["threshold"])
        text = row["text"]
        points = events(text, language)
        previous_prob = None
        first_eot = None
        for step, match in enumerate(points):
            prefix = text[:match.end()]
            result = detector.predict(prefix, "", language=language) if language_argument else detector.predict(prefix, "")
            mark = match.group()
            is_comma = mark in {",", "，", "、", ";", "；", ":", "："}
            sentence_end = mark in {".", "。", "!", "！", "?", "？"}
            kind = "comma_or_clause" if is_comma else "sentence_punctuation" if sentence_end else "content"
            original_length = len(detector.tokenizer.encode(normalize(prefix), add_special_tokens=False).ids)
            entry = {"paragraph_id": paragraph_id, "step": step, "language": language, "source_url": row["source_url"],
                "title": row["title"], "raw_prefix": prefix, "emitted_piece": mark, "boundary_kind": kind,
                "has_observed_continuation": bool(normalize(text[match.end():])), "source_fraction": match.end() / len(text),
                "input_tokens_before_truncation": original_length, "input_truncated": original_length > detector.contract["max_length"] - 3,
                "annotation_scope": "UNLABELED_PROSE; comma/period is a syntactic diagnostic, not true EOT gold", **result}
            if (is_comma or sentence_end) and previous_prob is not None:
                entry["punctuation_only_probability_delta"] = abs(previous_prob - result["prob_EOT"])
                punctuation_deltas.append(entry["punctuation_only_probability_delta"])
            if result["prediction"] == "EOT" and first_eot is None:
                first_eot = {"step": step, "source_fraction": entry["source_fraction"], "raw_prefix": prefix}
            trace.append(entry)
            if kind != "content": summaries[language + ":" + kind].append(entry)
            previous_prob = result["prob_EOT"]
        paragraph_reports.append({"paragraph_id": paragraph_id, "language": language, "source_url": row["source_url"],
            "title": row["title"], "text": text, "events": len(points), "first_eot": first_eot,
            "final_prediction": result, "scope": "First EOT in prose is not a validated speech cut-off"})
    report = {"model": str(args.package), "scope": "STREAMING_CRAWLED_PROSE_DIAGNOSTIC_ONLY; no accuracy/F1 claim",
        "paragraphs": len(selected), "prefix_events": len(trace), "punctuation_only_max_probability_delta": max(punctuation_deltas, default=0.),
        "boundary_summary": {key: {"events": len(values), "predicted_EOT": sum(v["prediction"] == "EOT" for v in values),
            "mean_prob_EOT": float(np.mean([v["prob_EOT"] for v in values])), "truncated_events": sum(v["input_truncated"] for v in values)} for key, values in summaries.items()},
        "input_truncated_events": sum(v["input_truncated"] for v in trace),
        "limitations": ["Prose is not conversation gold", "Comma does not imply turn end", "Model drops punctuation before inference", "No VAD/pause-duration signal in this text-only replay"]}
    write_jsonl(args.output / "trace.jsonl", trace)
    write_json(args.output / "paragraphs.json", paragraph_reports)
    write_json(args.output / "report.json", report)
    print(json.dumps(report))


if __name__ == "__main__": main()
