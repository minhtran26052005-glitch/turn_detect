"""Launch/inspect/fetch the one-shot v15d Colab held-out evaluation."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

from colab_runtime import ColabRuntime

LOCAL = Path("outputs/translation_segmenter_v15d_20261026/heldout_test_v2")
REMOTE = "outputs/translation_segmenter_v15d_20261026/heldout_test_v2"
SOURCE = Path("turn_v2/translation_segmenter_v15d_test_v2.py")


def cell(runtime, code):
    result = runtime.execute(code, timeout=60, echo=False)
    if result.get("status") != "ok":
        errors = [x for x in result.get("outputs", []) if x.get("output_type") == "error"]
        raise RuntimeError(json.dumps(errors, ensure_ascii=False))
    text = "".join(x.get("text", "") for x in result.get("outputs", []) if x.get("output_type") == "stream")
    return json.loads(text.strip())


def launch(runtime):
    data = SOURCE.read_bytes()
    source_sha = hashlib.sha256(data).hexdigest()
    runtime.request("PUT", "api/contents/content/turn_detect_v2/colab/" + SOURCE.name,
                    json={"type": "file", "format": "base64", "content": base64.b64encode(data).decode("ascii")})
    code = f'''import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path('/content/turn_detect_v2');run=root/'outputs/translation_segmenter_v15d_20261026';job=root/{REMOTE!r}
def live(record):
 try:
  argv=record['command'];module=argv[argv.index('-m')+1];p=Path('/proc')/str(record['pid'])
  return module.encode() in (p/'cmdline').read_bytes().split(bytes([0])) and (p/'stat').read_text().split()[21]==record['start_ticks']
 except (OSError,KeyError,ValueError,IndexError):return False
parent=json.loads((run/'launcher.json').read_text());assert live(parent),'Campaign parent identity unavailable'
source=root/'colab'/{SOURCE.name!r};raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()=={source_sha!r}
target=root/'turn_v2'/{SOURCE.name!r}
if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()=={source_sha!r},'Preserve source; use a new evaluator module version'
else:target.write_bytes(raw)
job.mkdir(exist_ok=True);lp=job/'launcher.json'
if lp.exists() and live(json.loads(lp.read_text())):
 print(json.dumps({{'reused_live_evaluator':True,'launcher':json.loads(lp.read_text())}}))
elif (job/'report.json').exists():
 print(json.dumps({{'already_complete':True,'report':str(job/'report.json')}}))
else:
 assert not (job/'frozen').exists() and not (job/'error.json').exists(),'Preserve prior evaluation state; inspect failure before resubmitting'
 assert shutil.disk_usage(root).free>33*1024**3,'Insufficient Colab space above reserve for frozen checkpoints'
 command=[sys.executable,'-u','-m','turn_v2.translation_segmenter_v15d_test_v2','--minimum-cycle','45','--wait-seconds','7200']
 log=job/'worker.log'
 with log.open('a') as stream:p=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}})
 record={{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':command,'source_sha256':{source_sha!r},'log':str(log.relative_to(root))}}
 lp.write_text(json.dumps(record,indent=2));print(json.dumps({{'launched':True,'launcher':record}}))'''
    return cell(runtime, code)


def status(runtime):
    return cell(runtime, f'''import json,shutil
from pathlib import Path
root=Path('/content/turn_detect_v2');job=root/{REMOTE!r};out={{}}
for name in ('launcher.json','status.json','error.json','manifest.json','report.json','partial_report.json'):
 p=job/name
 if p.exists():out[name]=json.loads(p.read_text())
r=out.get('launcher.json',{{}});live=False
try:
 p=Path('/proc')/str(r['pid']);argv=r['command'];module=argv[argv.index('-m')+1]
 live=module.encode() in (p/'cmdline').read_bytes().split(bytes([0])) and (p/'stat').read_text().split()[21]==r['start_ticks']
except (OSError,KeyError,ValueError,IndexError):pass
out['process_verified']=live;out['remote_free_bytes']=shutil.disk_usage(root).free
p=job/'worker.log'
if p.exists():
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2000));out['log_tail']=f.read().decode(errors='replace')
print(json.dumps(out,ensure_ascii=False))''')


def render_review():
    report = json.loads((LOCAL / "report.json").read_text(encoding="utf-8"))
    samples = [json.loads(line) for line in (LOCAL / "review_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    table = []
    for lang, value in report["languages"].items():
        m = value["metrics"]
        dot = value.get("terminal_period_eot_recall")
        dot_text = f"{dot:.2%}" if dot is not None else "n/a"
        table.append(f'<tr><td>{html.escape(lang)}</td><td>{m["samples"]:,}</td><td>{m["accuracy"]:.2%}</td><td>{m["eot_f1"]:.2%}</td><td>{dot_text}</td><td>{value["leakage_audit"]["removed_rows"]:,}</td></tr>')
    payload = json.dumps(samples, ensure_ascii=False).replace("<", "\\u003c")
    page = '''<!doctype html><meta charset="utf-8"><title>Frozen held-out prose test</title>
<style>body{font:16px system-ui;max-width:1250px;margin:35px auto;padding:0 20px;background:#f4f6fb;color:#1b2540}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border-bottom:1px solid #dfe5ef;text-align:left}.case{background:white;padding:18px;margin:14px 0;border-left:5px solid #c44747;border-radius:6px;white-space:pre-wrap}.ok{border-color:#26925e}select,input{padding:9px;margin:12px 8px 12px 0}small{color:#52617c}</style>
<h1>Held-out test: lowercase text, no input punctuation</h1><p>Frozen data and model weights; thresholds selected on dev. Overlapping model inputs/source text removed before scoring. Source-page bootstrap intervals and raw metrics remain in report.json. These targets come from prose punctuation, not human/audio gold.</p>
<table><thead><tr><th>Language</th><th>Test rows</th><th>Accuracy</th><th>EOT F1</th><th>Period EOT recall</th><th>Overlap removed</th></tr></thead><tbody>TABLE</tbody></table>
<p><select id="lang"><option value="">All languages</option><option>vi</option><option>en</option><option>zh</option></select><select id="verdict"><option value="">All cases</option><option value="bad">Errors</option><option value="ok">Correct</option></select><input id="search" placeholder="Search text"></p><small id="count"></small><div id="cases"></div>
<script>const rows=PAYLOAD;function draw(){const lang=document.getElementById('lang').value,v=document.getElementById('verdict').value,q=document.getElementById('search').value.toLowerCase();const chosen=rows.filter(r=>(!lang||r.language===lang)&&(!v||(v==='ok')===r.correct)&&(!q||(r.context+' '+r.text).includes(q)));document.getElementById('count').textContent=chosen.length+' review cases (selected examples; percentages above use the full clean test)';const parent=document.getElementById('cases');parent.replaceChildren();for(const r of chosen){const box=document.createElement('div');box.className='case'+(r.correct?' ok':'');const head=document.createElement('strong');head.textContent=r.language+' · target '+r.label+' · predicted '+r.predicted+' · P(EOT) '+r.p_eot.toFixed(4)+' · lookahead '+r.lookahead_words+' · target punctuation '+r.target_punctuation;const text=document.createElement('p');text.textContent=r.context+' ▏ '+r.text;const source=document.createElement('small');source.textContent=r.source_title+' · '+r.example_id;box.append(head,text,source);parent.append(box)}}for(const id of ['lang','verdict','search'])document.getElementById(id).addEventListener('input',draw);draw();</script>'''.replace("TABLE", "".join(table)).replace("PAYLOAD", payload)
    (LOCAL / "review.html").write_text(page, encoding="utf-8")


def fetch(runtime):
    state = status(runtime)
    if not state.get("report.json", {}).get("complete"):
        raise RuntimeError("Held-out evaluation is not complete; preserve its partial state")
    for name in ("report.json", "manifest.json", "status.json", "review_samples.jsonl"):
        response = runtime.request("GET", "api/contents/content/turn_detect_v2/" + REMOTE + "/" + name + "?format=text")
        value = response.json()
        if value.get("type") != "file" or int(value.get("size") or 0) > 5 * 1024**2:
            raise RuntimeError("Unexpected type/size for compact held-out artifact: " + name)
        content = value.get("content", "")
        raw = base64.b64decode(content) if value.get("format") == "base64" else content.encode("utf-8")
        if len(raw) > 5 * 1024**2:
            raise RuntimeError("Held-out artifact exceeds local size cap")
        (LOCAL / name).write_bytes(raw)
    render_review()
    return {"complete": True, "report": str(LOCAL / "report.json"), "review": str(LOCAL / "review.html"),
            "languages": state["report.json"]["languages"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("launch", "status", "fetch"))
    parser.add_argument("--session-path")
    args = parser.parse_args()
    LOCAL.mkdir(parents=True, exist_ok=True)
    runtime = ColabRuntime(0, session_path=args.session_path)
    try:
        connection = runtime.connect()
        if connection.get("execution_state") != "idle":
            raise RuntimeError("Kernel busy; do not queue an evaluation control cell")
        value = {"launch": launch, "status": status, "fetch": fetch}[args.action](runtime)
        (LOCAL / (args.action + "_receipt.json")).write_text(
            json.dumps({"connection": connection, "result": value}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        task_path = LOCAL.parent / "task_state.json"
        if args.action == "status" and task_path.exists():
            task = json.loads(task_path.read_text(encoding="utf-8"))
            task["test_evaluation"] = {"state": value.get("status.json", {}),
                "process_verified_at_check": value.get("process_verified", False),
                "launcher": value.get("launcher.json", {}),
                "control_script": "tools/translation_segmenter_heldout_colab.py",
                "test_scores_available": bool(value.get("report.json", {}).get("complete")),
                "partial_languages": list(value.get("partial_report.json", {}).get("languages", {}))}
            task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"connection": connection, "result": value}, ensure_ascii=False))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
