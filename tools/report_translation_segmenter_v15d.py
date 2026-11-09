"""Build a local review page from a verified v15d data snapshot."""
from __future__ import annotations

import html
import json
from pathlib import Path
import sqlite3


ROOT = Path("outputs/translation_segmenter_v15d_20261026")


def render(root=ROOT):
    root = Path(root)
    db_path = root / "state_snapshot.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    examples = []
    query = """SELECT e.language,e.split,e.record_json FROM examples e
               ORDER BY e.language,e.split,e.example_id"""
    for language, split, raw in db.execute(query):
        item = json.loads(raw)
        page_id = str(item.get("source_id", "")).removeprefix(language + ":")
        source = "carryover from v15c (Wikipedia)"
        if "source_provenance" in tables:
            row = db.execute("SELECT source_key FROM source_provenance WHERE language=? AND page_id=?",
                             (language, page_id)).fetchone()
            if row:
                source = row[0]
        item.update({"language": language, "split": split, "source": source})
        examples.append(item)
    sources = []
    if "crawl_observations" in tables:
        sources = [dict(zip(("language", "source_key", "disposition", "documents", "examples"), row))
            for row in db.execute("""SELECT language,source_key,disposition,COUNT(*),SUM(examples)
                FROM crawl_observations GROUP BY 1,2,3 ORDER BY 1,2,3""")]
    rejected = []
    if "crawl_rejected_text" in tables:
        rejected = [dict(zip(("source_key", "language", "document_id", "title", "url", "text", "reason", "cycle"), row))
            for row in db.execute("SELECT source_key,language,document_id,title,url,text,reason,cycle FROM crawl_rejected_text ORDER BY cycle DESC LIMIT 300")]
    summary = json.loads((root / "data_summary.json").read_text(encoding="utf-8")) if (root / "data_summary.json").exists() else {}
    db.close()

    trace_by_id = {}
    for language in ("vi", "en", "zh"):
        trace_path = root / "teachers" / language / "dev_trace.jsonl"
        if not trace_path.exists():
            trace_path = root / "dev_traces" / f"{language}.jsonl"
        if trace_path.exists():
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line); trace_by_id[row.get("example_id")] = row
    rows = []
    for item in examples:
        trace = trace_by_id.get(item.get("example_id"), {})
        rows.append("<tr data-lang='{lang}' data-source='{source}' data-split='{split}' data-k='{k}' data-label='{label}'>"
            "<td>{lang}</td><td>{source}</td><td>{split}</td><td>{k}</td><td>{context}</td>"
            "<td>{future}</td><td>{mark}</td><td>{label}</td><td>{pred}</td><td>{correct}</td></tr>".format(
                lang=html.escape(item.get("language", "")), source=html.escape(item.get("source", "")),
                split=html.escape(item.get("split", "")), k=html.escape(str(item.get("lookahead_words", ""))),
                context=html.escape(item.get("context", "")), future=html.escape(item.get("text", "")),
                mark=html.escape(item.get("target_punctuation", "")), label=html.escape(item.get("label", "")),
                pred=html.escape(trace.get("predicted", "PENDING")),
                correct=("—" if trace.get("correct") is None else "OK" if trace["correct"] else "MISS")))
    source_rows = ["<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
        *(html.escape(str(row.get(key, ""))) for key in ("language", "source_key", "disposition", "documents", "examples")))
        for row in sources]
    teacher_rows = []
    for language in ("vi", "en", "zh"):
        folder = root / "teachers" / language
        report_path = folder / "last_training_report.json"
        deferred_path = folder / "deferred.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            teacher_rows.append("<tr><td>{}</td><td>cycle {}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                language, report.get("cycle", ""), report.get("stage", ""), report.get("dev_samples", ""),
                report.get("dev_accuracy", ""), report.get("dev_eot_f1", ""),
                json.dumps(report.get("terminal_period_eot_recall_by_lookahead", {}), ensure_ascii=False)))
        elif deferred_path.exists():
            deferred = json.loads(deferred_path.read_text(encoding="utf-8"))
            teacher_rows.append("<tr><td>{}</td><td>cycle {}</td><td>deferred</td><td>{}</td><td colspan='3'>{}</td></tr>".format(
                language, deferred.get("cycle", ""), deferred.get("dev_samples", ""),
                html.escape(deferred.get("reason", ""))))
        else:
            teacher_rows.append(f"<tr><td>{language}</td><td colspan='6'>No training result recorded yet</td></tr>")
    rejected_rows = ["<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td class='text'>{}</td></tr>".format(
        html.escape(str(row.get("language", ""))), html.escape(str(row.get("source_key", ""))),
        html.escape(str(row.get("reason", ""))), html.escape(str(row.get("title", ""))),
        html.escape(str(row.get("text", "")[:1000]))) for row in rejected]

    page = """<!doctype html><html lang='vi'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Translation sentence-boundary crawl review v15d</title>
<style>body{font:14px system-ui;margin:20px;color:#17202a}h1{font-size:22px}p{max-width:1100px;line-height:1.5;color:#455}select{margin:0 8px 12px 0;padding:6px}table{border-collapse:collapse;width:100%;margin:10px 0 28px}th,td{border:1px solid #ccd3d8;padding:7px;vertical-align:top}th{position:sticky;top:0;background:#eef2f5}.text{min-width:220px;max-width:500px;overflow-wrap:anywhere}details{margin:16px 0}tr[data-label='NEOT']{background:#fbfcff}</style>
<h1>Text-only sentence-boundary corpus review</h1>
<p>Snapshot includes punctuation-free model inputs and punctuation-derived target metadata. Sentence-final punctuation maps to EOT; internal punctuation maps to NEOT. These are labels from prose, not human dialogue/audio gold. See per-source crawl dispositions below; short documents rejected by the sample extractor are retained for inspection. Data snapshot excludes model weights and optimizer state.</p>
<p>Latest crawl receipt: <code>__SUMMARY__</code></p>
<details open><summary>Model output (dev only; test is not used for selection)</summary><table><thead><tr><th>Lang</th><th>Cycle</th><th>Stage</th><th>Dev samples</th><th>Accuracy</th><th>EOT F1</th><th>Period EOT recall by lookahead</th></tr></thead><tbody>__TEACHERS__</tbody></table></details>
<label>Language <select id='lang'><option value=''>All</option><option>vi</option><option>en</option><option>zh</option></select></label>
<label>Split <select id='split'><option value=''>All</option><option>train</option><option>dev</option><option>test</option></select></label>
<label>Lookahead <select id='k'><option value=''>All</option><option>1</option><option>2</option></select></label>
<span id='count'></span>
<table><thead><tr><th>Lang</th><th>Source</th><th>Split</th><th>Lookahead</th><th>Text before boundary</th><th>Future text</th><th>Target mark (label only)</th><th>Gold</th><th>Pred</th><th>Match</th></tr></thead><tbody>__ROWS__</tbody></table>
<details open><summary>Crawl accounting — accepted, skipped and rejected documents</summary><table><thead><tr><th>Lang</th><th>Source</th><th>Disposition</th><th>Documents</th><th>Examples</th></tr></thead><tbody>__SOURCES__</tbody></table></details>
<details><summary>Rejected short text samples (up to 300 most recent)</summary><table><thead><tr><th>Lang</th><th>Source</th><th>Reason</th><th>Title</th><th>Text (first 1000 chars)</th></tr></thead><tbody>__REJECTED__</tbody></table></details>
<script>function apply(){let l=document.querySelector('#lang').value,s=document.querySelector('#split').value,k=document.querySelector('#k').value,n=0;for(const r of document.querySelectorAll('tbody tr[data-lang]')){let show=(!l||r.dataset.lang===l)&&(!s||r.dataset.split===s)&&(!k||r.dataset.k===k);r.hidden=!show;if(show)n++}document.querySelector('#count').textContent=n+' examples'}for(const x of document.querySelectorAll('select'))x.addEventListener('change',apply);apply()</script></html>"""
    page = page.replace("__SUMMARY__", html.escape(json.dumps(summary, ensure_ascii=False, separators=(",", ":"))))
    page = page.replace("__ROWS__", "\n".join(rows)).replace("__SOURCES__", "\n".join(source_rows)).replace("__REJECTED__", "\n".join(rejected_rows)).replace("__TEACHERS__", "\n".join(teacher_rows))
    output = root / "review.html"
    output.write_text(page, encoding="utf-8")
    return output, len(rows), len(sources), len(rejected)


if __name__ == "__main__":
    path, cases, counters, rejects = render()
    print(json.dumps({"review_html": str(path), "examples": cases,
                      "source_counters": counters, "rejected_text_rows": rejects}, ensure_ascii=False))
