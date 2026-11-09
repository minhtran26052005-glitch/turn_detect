"""Render a local, filterable review of the text-only sentence-boundary outputs."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sqlite3


ROOT = Path("outputs/translation_segmenter_v15c_20261026")


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render(root=ROOT):
    root = Path(root)
    cases = {}
    db_path = root / "state_snapshot.sqlite"
    if db_path.exists():
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        for lang, split, raw in db.execute("SELECT language,split,record_json FROM examples ORDER BY language,split,example_id"):
            item = json.loads(raw); item["split"] = split
            cases[(lang, item["example_id"])] = item
        db.close()
    for lang in ("vi", "en", "zh"):
        for item in load_jsonl(root / "dev_traces" / f"{lang}.jsonl"):
            item["split"] = "dev"
            cases[(lang, item["example_id"])] = {**cases.get((lang, item["example_id"]), {}), **item}
    rows = sorted(cases.items(), key=lambda pair: (pair[0][0], pair[1].get("split", ""), pair[1].get("example_id", "")))

    body = []
    for (lang, _), item in rows:
        target = item.get("label", "")
        predicted = item.get("predicted", "")
        correct_value = item.get("correct")
        outcome = "pending" if correct_value is None else "true" if correct_value else "false"
        ok_text = "PENDING" if correct_value is None else "OK" if correct_value else "MISS"
        body.append(
            "<tr data-lang='{lang}' data-split='{split}' data-k='{k}' data-label='{label}' data-ok='{outcome}'>"
            "<td>{lang}</td><td>{split}</td><td>{k}</td><td class='text'>{context}</td>"
            "<td class='text'>{future}</td><td>{mark}</td><td>{label}</td>"
            "<td>{pred}</td><td>{prob}</td><td>{ok_text}</td></tr>".format(
                lang=html.escape(lang), split=html.escape(item.get("split", "")),
                k=html.escape(str(item.get("lookahead_words", ""))),
                context=html.escape(item.get("context", "")), future=html.escape(item.get("text", "")),
                mark=html.escape(item.get("target_punctuation", "")), label=html.escape(target),
                pred=html.escape(predicted), prob="" if item.get("p_eot") is None else f"{float(item['p_eot']):.4f}",
                outcome=outcome, ok_text=ok_text))

    page = """<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Text-only sentence-boundary review</title>
<style>body{font:14px system-ui;margin:20px;color:#17202a}h1{font-size:22px}p{max-width:1000px;line-height:1.5;color:#455}select{margin:0 8px 12px 0;padding:6px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd3d8;padding:7px;vertical-align:top}th{position:sticky;top:0;background:#eef2f5}.text{min-width:220px;max-width:430px;white-space:normal;overflow-wrap:anywhere}tr[data-ok='false']{background:#fff0ed}tr[data-ok='true']{background:#f3fff3}.target{font-family:monospace}</style>
<h1>Text-only sentence-boundary dev review</h1>
<p>The model input columns contain punctuation-free text: left context and 1–2 following words. “Target mark” is label metadata from the original punctuation position; it is not part of either input field. EOT targets are sentence-final punctuation; NEOT targets are internal punctuation. These are crawled prose labels, not dialogue or human-audio gold. This page lists corpus rows across train/dev/test; prediction cells remain pending until dev inference is recorded, and test rows are never used for checkpoint selection.</p>
<label>Language <select id='lang'><option value=''>All</option><option>vi</option><option>en</option><option>zh</option></select></label>
<label>Split <select id='split'><option value=''>All</option><option>train</option><option>dev</option><option>test</option></select></label>
<label>Lookahead <select id='k'><option value=''>All</option><option>1</option><option>2</option></select></label>
<label>Prediction <select id='ok'><option value=''>All</option><option value='true'>Correct</option><option value='false'>Miss</option><option value='pending'>Not scored</option></select></label>
<span id='count'></span><table><thead><tr><th>Lang</th><th>Split</th><th>Lookahead</th><th>Text before boundary</th><th>Future text</th><th>Target mark (label only)</th><th>Gold</th><th>Pred</th><th>P(EOT)</th><th>Match</th></tr></thead><tbody>__ROWS__</tbody></table>
<script>function apply(){let l=document.querySelector('#lang').value,s=document.querySelector('#split').value,k=document.querySelector('#k').value,o=document.querySelector('#ok').value,n=0;for(const r of document.querySelectorAll('tbody tr')){let show=(!l||r.dataset.lang===l)&&(!s||r.dataset.split===s)&&(!k||r.dataset.k===k)&&(!o||r.dataset.ok===o);r.hidden=!show;if(show)n++}document.querySelector('#count').textContent=n+' cases'}for(const x of document.querySelectorAll('select'))x.addEventListener('change',apply);apply()</script></html>""".replace("__ROWS__", "\n".join(body))
    output = root / "review.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output, len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    path, count = render(args.root)
    print(json.dumps({"review_html": str(path), "review_rows": count}, ensure_ascii=False))
