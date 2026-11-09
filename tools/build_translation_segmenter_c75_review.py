"""Render bounded, already-scored reports. No tokenizer/model execution."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys

import markdown

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turn_v2.common import normalize

JOB = ROOT / "outputs/translation_segmenter_v15d_20261026/diagnostic_test_c75_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = json.loads((JOB / "report.json").read_text(encoding="utf-8"))
    audit = json.loads((JOB / "test_provenance_audit_v1.json").read_text(encoding="utf-8"))
    report_sha = sha(JOB / "report.json")
    assert report_sha == audit["raw_report_sha256"]
    assert report["complete"] and not report["test_used_for_selection"]
    for lang, value in report["languages"].items():
        assert value["metrics"]["samples"] == audit["languages"][lang]["counts"]["rows"]
        assert sum(x["rows"] for x in audit["languages"][lang]["by_source"].values()) == value["metrics"]["samples"]
    plan = json.loads((JOB / "crawl_plan_v2.json").read_text(encoding="utf-8"))
    plan.update(schema=3, reviewed_analysis="error_analysis_reviewed_v3.md", test_provenance="test_provenance.md",
                test_provenance_audit="test_provenance_audit_v1.json")
    plan["source_url_metadata_repair"] = {
        "status": "PREPARED_NOT_ACTIVATED",
        "wikipedia_invalid_fallback": "https://huggingface.co/datasets/Wikipedia API",
        "future_wikipedia_url": "https://{language}.wikipedia.org/?curid={original_page_id}",
        "preserve_original_url_and_add_resolution_provenance": True,
        "baai_original_article_url_not_available_for_every_record": True,
        "frozen_test_records_not_rewritten": True,
    }
    (JOB / "crawl_plan_v3.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    analysis = (JOB / "error_analysis_reviewed_v2.md").read_text(encoding="utf-8")
    provenance = (JOB / "test_provenance.md").read_text(encoding="utf-8")
    assert sum(ord(x) > 127 for x in analysis) > 500
    assert sum(ord(x) > 127 for x in provenance) > 500
    (JOB / "error_analysis_reviewed_v3.md").write_text(analysis, encoding="utf-8")
    render = lambda value: markdown.markdown(value, extensions=["tables", "fenced_code"])
    origins = []
    for lang, value in audit["languages"].items():
        for source, group in value["by_source"].items():
            origin = group["sample_origin"]
            origins.append("<details><summary>" + html.escape(f"{lang} / {source}: {group['clean_source_pages']} pages, {group['rows']} rows") +
                           "</summary><pre>" + html.escape(json.dumps(origin, ensure_ascii=False, indent=2)) + "</pre></details>")
    samples = [json.loads(line) for line in (JOB / "review_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(samples) == 360
    cases = []
    for row in samples:
        head = f"{row['language']} | target {row['label']} | predicted {row['predicted']} | k={row['lookahead_words']} | correct={row['correct']} | clean={row['included_in_clean_test']}"
        visible = normalize(row["context"]) + " | " + normalize(row["text"])
        origin = {k: row[k] for k in ("source_id", "source_title", "source_url", "example_id", "paragraph_index", "sentence_index", "target_punctuation")}
        cases.append("<details><summary>" + html.escape(head) + "</summary><p>Normalized text before the 256-token truncation:</p><pre>" + html.escape(visible) +
                     "</pre><p>Source metadata (URL can be a fallback):</p><pre>" + html.escape(json.dumps(origin, ensure_ascii=False, indent=2)) +
                     "</pre><p>Original paragraph for post-hoc label audit:</p><pre>" + html.escape(row.get("original_source_paragraph_for_posthoc_label_audit", "")) + "</pre></details>")
    page = """<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cycle75: test, source provenance and error analysis</title><style>
body{font:16px system-ui;max-width:1200px;margin:24px auto;padding:16px;color:#16232d;line-height:1.65;background:#fafcfe}
nav{position:sticky;top:0;background:#e5eff9;padding:12px;border-radius:8px}nav a{margin-right:16px}
table{border-collapse:collapse;display:block;overflow-x:auto;margin:16px 0}td,th{border:1px solid #bdcdda;padding:7px 12px}th{background:#e8f0f8}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf2f6;padding:12px;border-radius:6px}code{overflow-wrap:anywhere}
details{border:1px solid #bdcdda;border-radius:7px;padding:10px;margin:10px 0}summary{cursor:pointer}h1,h2,h3{scroll-margin-top:80px}
</style></head><body><nav><a href="#results">Results and causes</a><a href="#test-provenance">Test provenance</a><a href="#origins">Source records</a><a href="#cases">360 review samples</a></nav>
"""
    page += '<section id="results">' + render(analysis) + '</section><section id="test-provenance">' + render(provenance) + "</section>"
    page += '<section id="origins"><h2>Source origin records</h2>' + "".join(origins) + "</section>"
    page += '<section id="cases"><h2>360 review samples</h2>' + "".join(cases) + "</section></body></html>"
    (JOB / "review_complete_v3.html").write_text(page, encoding="utf-8")
    assert "119.193" in page and 'id="test-provenance"' in page and len(page.encode()) < 5 * 1024**2
    receipt = {"schema": 3, "at_utc": datetime.now(timezone.utc).isoformat(), "raw_report_sha256": report_sha,
               "model_or_threshold_tuned_on_test": False, "crawl_recipe_activated": False,
               "local_render_from_verified_bounded_reports": True,
               "files_sha256": {name: sha(JOB / name) for name in ("error_analysis_reviewed_v3.md", "test_provenance.md", "test_provenance_audit_v1.json", "crawl_plan_v3.json", "review_complete_v3.html", "README.md")}}
    (JOB / "reviewed_analysis_receipt_v3.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report_sha256": report_sha, "review_bytes": len(page.encode()), "samples": len(samples), "source_groups": sum(d["clean_source_groups"] for d in audit["languages"].values())}))


if __name__ == "__main__":
    main()
