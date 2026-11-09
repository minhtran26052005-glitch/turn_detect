"""Build a compact review from the latest verified local v15d SQLite snapshot."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone

ROOT = Path("outputs/translation_segmenter_v15d_20261026")
DB_PATH = ROOT / "state_snapshot.sqlite"


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "v15d_review_renderer", Path("tools/export_translation_segmenter_v15d_review.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    db = sqlite3.connect(f"file:{DB_PATH.resolve()}?mode=ro", uri=True)
    payload = {
        "cycle": None,
        "samples": [],
        "pages_by_language": [],
        "example_totals": {},
        "source_dispositions": [],
        "teachers": {},
    }
    status_path = ROOT / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        payload["cycle"] = status.get("cycle")
        payload["parent_verified"] = status.get("status") == "RUNNING"
        payload["child_verified"] = True
        payload["gpu"] = "See status receipt"
    payload["pages_by_language"] = [
        dict(zip(("language", "pages"), row))
        for row in db.execute("SELECT language,COUNT(*) FROM sources GROUP BY 1 ORDER BY 1")
    ]
    payload["example_totals"] = dict(
        db.execute("SELECT language,COUNT(*) FROM examples GROUP BY 1")
    )
    payload["source_dispositions"] = [
        dict(zip(("language", "source_key", "disposition", "documents", "examples"), row))
        for row in db.execute(
            "SELECT language,source_key,disposition,COUNT(*),SUM(examples) "
            "FROM crawl_observations GROUP BY 1,2,3 ORDER BY 1,2,3"
        )
    ]
    provenance_exists = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_provenance'"
    ).fetchone() is not None

    receipt_path = ROOT / "status_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
    live_teachers = receipt.get("teachers")
    for lang in ("vi", "en", "zh"):
        folder = ROOT / "teachers" / lang
        state = {}
        for name in ("identity.json", "progress.json", "deferred.json", "best_report.json", "last_training_report.json"):
            p = folder / name
            if p.exists():
                state[name] = json.loads(p.read_text(encoding="utf-8"))
        # A fresh runtime can have no current teacher yet. Do not silently show
        # an old local report as the model available on that runtime.
        if isinstance(live_teachers, dict):
            state = live_teachers.get(lang, {})
        payload["teachers"][lang] = state
        trace = folder / "dev_trace.jsonl"
        language_samples = []
        for label in ("EOT", "NEOT"):
            for horizon in (1, 2):
                rows = db.execute(
                    "SELECT example_id,source_id,record_json FROM examples "
                    "WHERE language=? AND split='dev' AND label=? AND lookahead_words=? "
                    "ORDER BY RANDOM() LIMIT 30",
                    (lang, label, horizon),
                ).fetchall()
                for example_id, source_id, raw in rows:
                    item = json.loads(raw)
                    item.update({"language": lang, "split": "dev", "example_id": example_id})
                    page_id = str(source_id).removeprefix(lang + ":")
                    source = None
                    if provenance_exists:
                        found = db.execute(
                            "SELECT source_key FROM source_provenance WHERE language=? AND page_id=?",
                            (lang, page_id),
                        ).fetchone()
                        if found:
                            source = found[0]
                    item["source"] = source or "carryover v15c / Wikipedia"
                    item["predicted"] = None
                    item["correct"] = None
                    language_samples.append(item)
                    payload["samples"].append(item)
        # Retain predictions only for the sampled rows, rather than loading
        # every dev trace from all languages into Windows memory.
        if trace.exists():
            wanted = {item["example_id"]: item for item in language_samples}
            with trace.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    item = wanted.get(row.get("example_id"))
                    if item is not None:
                        item["predicted"] = row.get("predicted")
                        item["correct"] = row.get("correct")
    db.close()

    payload["parent_verified"] = bool(receipt.get("launcher.json_process_verified"))
    payload["child_verified"] = bool(receipt.get("child.json_process_verified"))
    payload["gpu"] = receipt.get("GPU", "unknown")
    evaluation_receipt = ROOT / "heldout_test_v2" / "status_receipt.json"
    if evaluation_receipt.exists():
        evaluated = json.loads(evaluation_receipt.read_text(encoding="utf-8")).get("result", {})
        payload["heldout_stage"] = evaluated.get("status.json", {}).get("stage", "UNKNOWN")
        payload["heldout_review_ready"] = (ROOT / "heldout_test_v2" / "review.html").exists()
    output = module.render(payload)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cycle": payload.get("cycle"),
        "snapshot": str(DB_PATH),
        "sample_rows": len(payload["samples"]),
        "examples_by_language": payload["example_totals"],
        "parent_verified_at_snapshot": payload["parent_verified"],
        "child_verified_at_snapshot": payload["child_verified"],
        "gpu_at_snapshot": payload["gpu"],
        "review_html": str(output),
    }
    (ROOT / "review_receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
