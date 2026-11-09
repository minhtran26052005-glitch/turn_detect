"""Export a compact, stratified dev review from the live Colab SQLite database."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import html
import json
from pathlib import Path
import shutil
import sqlite3

from colab_runtime import ColabRuntime


LOCAL = Path("outputs/translation_segmenter_v15d_20261026")
REMOTE_ROOT = "/content/turn_detect_v2"
REMOTE_RUN = "outputs/translation_segmenter_v15d_20261026"
PER_GROUP = 30


def render(payload):
    samples = payload["samples"]
    body = []
    for row in samples:
        if row.get("predicted") is None:
            outcome = "PENDING"
        elif not row.get("prediction_current", False):
            outcome = f"prior model c{row.get('prediction_cycle', '?')}"
        else:
            outcome = "OK" if row.get("correct") else "MISS"
        body.append("<tr data-lang='{lang}' data-label='{label}' data-k='{k}'>"
            "<td>{lang}</td><td>{source}</td><td>{split}</td><td>{k}</td>"
            "<td class='text'>{context}</td><td class='text'>{future}</td>"
            "<td>{mark}</td><td>{label}</td><td>{pred}</td><td>{outcome}</td></tr>".format(
                lang=html.escape(row["language"]), source=html.escape(row.get("source", "")),
                split=html.escape(row["split"]), k=html.escape(str(row["lookahead_words"])),
                context=html.escape(row.get("context", "")), future=html.escape(row.get("text", "")),
                mark=html.escape(row.get("target_punctuation", "")), label=html.escape(row["label"]),
                pred=html.escape(row.get("predicted") or "PENDING"), outcome=outcome))
    counts = []
    for row in payload["pages_by_language"]:
        lang = row["language"]
        counts.append(f"<tr><td>{html.escape(lang)}</td><td>{row['pages']}</td><td>{payload['example_totals'].get(lang,0)}</td></tr>")
    source_rows = ["<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
        html.escape(x["language"]), html.escape(x["source_key"]), html.escape(x["disposition"]),
        x["documents"], x["examples"]) for x in payload["source_dispositions"]]
    model_rows = []
    for lang in ("vi", "en", "zh"):
        state = payload["teachers"].get(lang, {})
        report = state.get("last_training_report.json") or state.get("best_report.json")
        progress = state.get("progress.json", {})
        if report:
            per = report.get("terminal_period_eot_recall_by_lookahead", {})
            cycle_label = str(report.get("cycle", ""))
            if report.get("cycle") != payload.get("cycle"):
                cycle_label += " (prior data)"
            model_rows.append("<tr><td>{}</td><td>{}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td><td>{}</td></tr>".format(
                lang, cycle_label, report.get("dev_accuracy", 0), report.get("dev_eot_f1", 0),
                report.get("terminal_period_eot_recall", 0), html.escape(json.dumps(per, ensure_ascii=False))))
        elif progress:
            if progress.get("cycle") == payload.get("cycle"):
                progress_text = f"Training {progress.get('step',0)}/{progress.get('steps',0)}, loss {progress.get('loss',0):.4f}"
            else:
                progress_text = f"Progress is stale from cycle {progress.get('cycle','?')}; no current-cycle report yet"
            model_rows.append(f"<tr><td>{lang}</td><td>{progress.get('cycle','')}</td><td colspan='4'>{html.escape(progress_text)}</td></tr>")
        elif state.get("deferred.json"):
            deferred = state["deferred.json"]
            model_rows.append(f"<tr><td>{lang}</td><td>{deferred.get('cycle','')}</td><td colspan='4'>Deferred: {html.escape(deferred.get('reason',''))}</td></tr>")
        else:
            model_rows.append(f"<tr><td>{lang}</td><td colspan='5'>No training result yet</td></tr>")
    page = """<!doctype html><html lang='vi'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>v15d crawl and dev review</title><style>body{font:14px system-ui;margin:20px;color:#17202a}h1{font-size:22px}p{max-width:1100px;line-height:1.5;color:#455}table{border-collapse:collapse;width:100%;margin:12px 0 28px}th,td{border:1px solid #ccd3d8;padding:7px;vertical-align:top}th{position:sticky;top:0;background:#eef2f5}.text{min-width:220px;max-width:460px;overflow-wrap:anywhere}select{margin:8px;padding:6px}</style>
<h1>Text-only sentence-boundary crawl review</h1>
<p>Corpus examples sampled from the dev split; see the review receipt for their source and cycle. Input fields are punctuation-free; original punctuation is target metadata only. Final punctuation => EOT; internal punctuation => NEOT. These are prose-derived labels, not human conversational/audio gold. This page contains a bounded review; model weights and optimizer are backed up directly from Colab to Drive.</p>
<p>Cycle __CYCLE__; remote parent verified: __PARENT__; child: __CHILD__; GPU: __GPU__</p>
<h2>Corpus totals</h2><table><thead><tr><th>Lang</th><th>Source pages</th><th>Examples</th></tr></thead><tbody>__COUNTS__</tbody></table>
<h2>Source accounting</h2><table><thead><tr><th>Lang</th><th>Source</th><th>Disposition</th><th>Documents</th><th>Examples</th></tr></thead><tbody>__SOURCES__</tbody></table>
<h2>Model dev output (not test)</h2><p>Metrics from older cycles are labeled prior data. Predictions from an older checkpoint are marked as prior model in the sample table.</p><table><thead><tr><th>Lang</th><th>Cycle</th><th>Accuracy</th><th>EOT F1</th><th>Period EOT recall</th><th>By lookahead</th></tr></thead><tbody>__MODELS__</tbody></table>
<label>Language<select id='lang'><option value=''>All</option><option>vi</option><option>en</option><option>zh</option></select></label><label>Label<select id='label'><option value=''>All</option><option>EOT</option><option>NEOT</option></select></label><label>Lookahead<select id='k'><option value=''>All</option><option>1</option><option>2</option></select></label><span id='n'></span>
<table><thead><tr><th>Lang</th><th>Source</th><th>Split</th><th>Lookahead</th><th>Context before boundary</th><th>Future words</th><th>Target mark</th><th>Gold</th><th>Prediction</th><th>Match</th></tr></thead><tbody>__ROWS__</tbody></table>
<script>function f(){let l=document.querySelector('#lang').value,y=document.querySelector('#label').value,k=document.querySelector('#k').value,n=0;for(let r of document.querySelectorAll('tbody tr[data-lang]')){let v=(!l||r.dataset.lang===l)&&(!y||r.dataset.label===y)&&(!k||r.dataset.k===k);r.hidden=!v;if(v)n++}document.querySelector('#n').textContent=n+' sampled dev cases'}for(let e of document.querySelectorAll('select'))e.addEventListener('change',f);f()</script></html>"""
    values = {"__CYCLE__": str(payload.get("cycle", "?")),
        "__PARENT__": str(payload.get("parent_verified", False)), "__CHILD__": str(payload.get("child_verified", False)),
        "__GPU__": html.escape(payload.get("gpu", "unknown")), "__COUNTS__": "\n".join(counts),
        "__SOURCES__": "\n".join(source_rows), "__MODELS__": "\n".join(model_rows),
        "__ROWS__": "\n".join(body)}
    for key, value in values.items(): page = page.replace(key, value)
    if payload.get("heldout_stage"):
        test_notice = "Held-out test: " + html.escape(str(payload["heldout_stage"]))
        if payload.get("heldout_review_ready"):
            test_notice += " &middot; <a href='heldout_test_v2/review.html'>Open frozen test results</a>"
        else:
            test_notice += ". No held-out test score is available yet."
        page = page.replace("<h2>Model dev output (not test)</h2>",
                            "<p style='background:#eaf0ff;padding:12px'>" + test_notice + "</p><h2>Model dev output (not test)</h2>")
    if any(row.get("source_key") == "hf:industrycorpus2_news_media:en" and
           row.get("disposition") == "rejected_language" for row in payload["source_dispositions"]):
        notice = "v15d3 repair is reflected in this snapshot; BAAI English rows overlapping Chinese were quarantined."
        style = "background:#e8f7ee;border:1px solid #84c99b;padding:12px;margin:12px 0"
    else:
        notice = "PRE-REPAIR SNAPSHOT: English BAAI rows are contaminated by a Chinese stream. Do not use the English model scores here; see the v15d3 repair receipt."
        style = "background:#fff1d6;border:1px solid #e0a634;padding:12px;margin:12px 0"
    page = page.replace("<h1>Text-only sentence-boundary crawl review</h1>",
        f"<h1>Text-only sentence-boundary crawl review</h1><div style='{style}'><b>Data audit:</b> {html.escape(notice)}</div>")
    output = LOCAL / "review.html"; output.write_text(page, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-path")
    args = parser.parse_args()
    LOCAL.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(LOCAL).free
    reserve_bytes = 10 * 1024**3
    if free_bytes < reserve_bytes + 5 * 1024**2:
        receipt = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "skipped": True, "reason": "local_free_space_below_review_reserve",
            "free_bytes": free_bytes, "reserve_bytes": reserve_bytes,
            "sample_rows": 0, "review_updated": False, "full_database_downloaded": False}
        (LOCAL / "review_transfer_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, ensure_ascii=False))
        return
    runtime = ColabRuntime(0, session_path=args.session_path)
    connection = runtime.connect()
    if connection.get("execution_state") != "idle":
        runtime.close()
        raise RuntimeError("Kernel busy; no review export cell was queued")
    code = f'''import json,sqlite3
from pathlib import Path
root=Path({REMOTE_ROOT!r});run=root/{REMOTE_RUN!r};db=sqlite3.connect('file:'+str(run/'corpus.sqlite')+'?mode=ro',uri=True)
def verified(record):
 try:
  argv=record['command'];module=argv[argv.index('-m')+1];p=Path('/proc')/str(record['pid'])
  return module.encode() in (p/'cmdline').read_bytes().split(bytes([0])) and (p/'stat').read_text().split()[21]==record['start_ticks']
 except (OSError,KeyError,ValueError,IndexError):return False
out={{'cycle':None,'samples':[],'pages_by_language':[],'example_totals':{{}},'source_dispositions':[],'teachers':{{}}}}
status_path=run/'status.json'
if status_path.exists():
 s=json.loads(status_path.read_text());out['cycle']=s.get('cycle')
 launcher=json.loads((run/'launcher.json').read_text())
 out['parent_verified']=verified(launcher)
 child_path=run/'child.json';child=json.loads(child_path.read_text()) if child_path.exists() else {{}};cp=Path('/proc')/str(child.get('pid'))
 out['child_verified']=verified(child)
 import subprocess;out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
out['pages_by_language']=[dict(zip(['language','pages'],r)) for r in db.execute('SELECT language,COUNT(*) FROM sources GROUP BY 1 ORDER BY 1')]
out['example_totals']={{lang:n for lang,n in db.execute('SELECT language,COUNT(*) FROM examples GROUP BY 1')}}
if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='crawl_observations'").fetchone():
 out['source_dispositions']=[dict(zip(['language','source_key','disposition','documents','examples'],r)) for r in db.execute('SELECT language,source_key,disposition,COUNT(*),SUM(examples) FROM crawl_observations GROUP BY 1,2,3 ORDER BY 1,2,3')]
for lang in ['vi','en','zh']:
 folder=run/'teachers'/lang;state={{}}
 for name in ['identity.json','progress.json','deferred.json','best_report.json','last_training_report.json']:
  p=folder/name
  if p.exists():state[name]=json.loads(p.read_text())
 trace=folder/'dev_trace.jsonl'
 out['teachers'][lang]=state
 lang_samples=[]
 for label in ['EOT','NEOT']:
  for horizon in [1,2]:
   rows=[json.loads(r[0]) for r in db.execute("SELECT record_json FROM examples WHERE language=? AND split='dev' AND label=? AND lookahead_words=? ORDER BY RANDOM() LIMIT ?", (lang,label,horizon,{PER_GROUP}))]
   for row in rows:
    row['language']=lang;row['split']='dev'
    pid=row.get('source_id','').removeprefix(lang+':')
    src=db.execute('SELECT source_key FROM source_provenance WHERE language=? AND page_id=?',(lang,pid)).fetchone() if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_provenance'").fetchone() else None
    row['source']=src[0] if src else 'carryover v15c / Wikipedia'
    row['predicted']=None;row['correct']=None
    model_state=out['teachers'][lang];report=model_state.get('last_training_report.json') or model_state.get('best_report.json') or {{}}
    row['prediction_cycle']=report.get('cycle');row['prediction_current']=report.get('cycle')==out.get('cycle')
    lang_samples.append(row)
    out['samples'].append(row)
 if trace.exists():
  wanted={{r['example_id']:r for r in lang_samples}}
  with trace.open() as stream:
   for line in stream:
    if not line.strip():continue
    value=json.loads(line);row=wanted.get(value.get('example_id'))
    if row is not None:row['predicted']=value.get('predicted');row['correct']=value.get('correct')
evaluation=run/'heldout_test_v2/status.json'
if evaluation.exists():out['heldout_stage']=json.loads(evaluation.read_text()).get('stage')
db.close();print(json.dumps(out,ensure_ascii=False,separators=(',',':')))
'''
    result = runtime.execute(code, timeout=180, echo=False)
    runtime.close()
    if result.get("status") != "ok": raise RuntimeError(json.dumps(result, ensure_ascii=False))
    raw = "".join(x["text"] for x in result.get("outputs", []) if x.get("output_type") == "stream")
    payload = json.loads(raw)
    payload['heldout_review_ready'] = (LOCAL/'heldout_test_v2/review.html').exists()
    if len(raw.encode('utf-8')) > 5 * 1024**2:
        raise RuntimeError("Review exceeded the bounded local report size")
    (LOCAL/'remote_review_payload.json').write_text(raw,encoding='utf-8')
    path = render(payload)
    receipt = {"exported_at_utc": datetime.now(timezone.utc).isoformat(), "cycle": payload.get("cycle"),
        "remote_parent_verified": payload.get("parent_verified"), "remote_child_verified": payload.get("child_verified"),
        "remote_counts": payload.get("example_totals"), "sample_rows": len(payload.get("samples", [])),
        "full_database_downloaded": False, "review_html": str(path),
        "skipped": False, "review_updated": True}
    (LOCAL / "review_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (LOCAL / "review_transfer_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__": main()
