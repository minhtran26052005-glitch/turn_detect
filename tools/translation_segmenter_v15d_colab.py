"""Launch, inspect, stop, or fetch the data-only snapshot of v15d on Colab."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from urllib.parse import quote
import zipfile

from colab_runtime import ColabRuntime
from storage_guard import ensure_free_space
from teacher_first_colab import campaign_runtime


LOCAL = Path("outputs/translation_segmenter_v15d_20261026")
REMOTE_ROOT = "/content/turn_detect_v2"
REMOTE_RUN = "outputs/translation_segmenter_v15d_20261026"
SOURCE = Path("turn_v2/translation_segmenter_v15d2.py")
LANGUAGES = ("vi", "en", "zh")
DOCS_PER_SOURCE = 100
STEPS = 100
MICROBATCH = 32
SOURCES = [
    {"key": "hf:fineweb2:vie_Latn", "language": "vi", "dataset": "HuggingFaceFW/fineweb-2", "config": "vie_Latn", "split": "train", "license": "ODC-By-1.0", "kind": "web"},
    {"key": "hf:fineweb:en", "language": "en", "dataset": "HuggingFaceFW/fineweb", "config": None, "split": "train", "license": "ODC-By-1.0", "kind": "web"},
    {"key": "hf:industrycorpus2_news_media:en", "language": "en", "dataset": "BAAI/IndustryCorpus2_news_media", "config": None, "split": "train", "license": "Apache-2.0", "kind": "news_media"},
    {"key": "hf:fineweb2:cmn_Hani", "language": "zh", "dataset": "HuggingFaceFW/fineweb-2", "config": "cmn_Hani", "split": "train", "license": "ODC-By-1.0", "kind": "web"},
    {"key": "hf:industrycorpus2_news_media:zh", "language": "zh", "dataset": "BAAI/IndustryCorpus2_news_media", "config": None, "split": "train", "license": "Apache-2.0", "kind": "news_media"},
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def execute(runtime, action, code, timeout=120):
    result = runtime.execute(code, timeout=timeout, echo=False)
    proof = LOCAL / f"{action}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    proof.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if result.get("status") != "ok":
        raise RuntimeError(f"Colab {action} failed; inspect {proof}")
    chunks = [item["text"] for item in result.get("outputs", []) if item.get("output_type") == "stream"]
    return json.loads("".join(chunks))


def config_for(source_hash):
    return {"campaign": "translation_segmenter_v15d", "languages": list(LANGUAGES),
        "lookahead_words": [1, 2], "punctuation_in_model_input": False, "qwen_judge_enabled": False,
        "docs_per_hf_source_cycle": DOCS_PER_SOURCE, "wiki_pages_per_language_cycle": 10,
        "datasets": SOURCES,
        "gated_dataset_skipped": "fsnaix/vietnamese-corpus-large; public-card probe returned gated",
        "base_model": "microsoft/mdeberta-v3-base",
        "base_revision": "a0484667b22365f84929a935b5e50a51f71f159d",
        "steps": STEPS, "microbatch": MICROBATCH, "source_sha256": source_hash}


def launch(runtime):
    source_bytes = SOURCE.read_bytes(); source_hash = sha(source_bytes)
    config = config_for(source_hash)
    runtime.request("PUT", "api/contents/content/turn_detect_v2/colab/translation_segmenter_v15d2.py",
        json={"type": "file", "format": "base64", "content": base64.b64encode(source_bytes).decode("ascii")})
    code = f'''import hashlib,json,os,shutil,sqlite3,subprocess,sys
from pathlib import Path
root=Path({REMOTE_ROOT!r});run=root/{REMOTE_RUN!r};run.mkdir(parents=True,exist_ok=True)
if (run/'STOP_REQUESTED').exists():
 old=run/('STOP_REQUESTED_ARCHIVED_'+__import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
 (run/'STOP_REQUESTED').replace(old)
for name in ['outputs/teacher_first_v14_20261026','outputs/translation_segmenter_v15_20261026','outputs/translation_segmenter_v15b_20261026','outputs/translation_segmenter_v15c_20261026']:
 p=root/name/'status.json'
 if p.exists() and 'teacher_first_v14' in name:
  assert json.loads(p.read_text()).get('status')=='STOPPED_BY_REQUEST','v14 must be stopped'
 if p.exists() and ('v15b' in name or 'v15c' in name):
  assert json.loads(p.read_text()).get('status')=='STOPPED_BY_REQUEST','Previous crawler must be stopped'
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:cmd=(proc/'cmdline').read_bytes()
 except OSError:continue
 for module in [b'turn_v2.teacher_first_v14',b'turn_v2.continuous_v13',b'turn_v2.translation_segmenter_v15',b'turn_v2.translation_segmenter_v15b',b'turn_v2.translation_segmenter_v15c',b'turn_v2.translation_segmenter_v15d']:
  assert module+b'\\x00' not in cmd,'Another campaign process is active: '+proc.name
uploaded=root/'colab/translation_segmenter_v15d2.py';data=uploaded.read_bytes()
assert hashlib.sha256(data).hexdigest()=={source_hash!r}
target=root/'turn_v2/translation_segmenter_v15d2.py'
if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()=={source_hash!r},'Preserve remote code; use a new module for fixes'
else:target.write_bytes(data)
db=run/'corpus.sqlite';old=root/'outputs/translation_segmenter_v15c_20261026/corpus.sqlite';copied=False
if not db.exists():
 assert old.exists(),'v15c corpus missing; refusing to start without prior data'
 src=sqlite3.connect('file:'+str(old)+'?mode=ro',uri=True);dst=sqlite3.connect(db);src.backup(dst);dst.close();src.close();copied=True
source_hash_db=hashlib.sha256(db.read_bytes()).hexdigest()
con=sqlite3.connect(db);con.executescript("""
CREATE TABLE IF NOT EXISTS crawl_observations(source_key TEXT NOT NULL,language TEXT NOT NULL,document_id TEXT NOT NULL,content_sha256 TEXT NOT NULL,disposition TEXT NOT NULL,reason TEXT NOT NULL,title TEXT NOT NULL,url TEXT NOT NULL,text_chars INTEGER NOT NULL,examples INTEGER NOT NULL,first_cycle INTEGER NOT NULL,last_cycle INTEGER NOT NULL,seen_count INTEGER NOT NULL DEFAULT 1,PRIMARY KEY(source_key,document_id));
CREATE INDEX IF NOT EXISTS crawl_observations_source ON crawl_observations(language,source_key,disposition);
CREATE TABLE IF NOT EXISTS crawl_rejected_text(source_key TEXT NOT NULL,language TEXT NOT NULL,document_id TEXT NOT NULL,title TEXT NOT NULL,url TEXT NOT NULL,text TEXT NOT NULL,reason TEXT NOT NULL,cycle INTEGER NOT NULL,PRIMARY KEY(source_key,document_id));
CREATE TABLE IF NOT EXISTS source_provenance(language TEXT NOT NULL,page_id TEXT NOT NULL,source_key TEXT NOT NULL,dataset TEXT NOT NULL,license TEXT NOT NULL,kind TEXT NOT NULL,PRIMARY KEY(language,page_id));
""")
summary_path=root/'outputs/translation_segmenter_v15c_20261026/data_summary.json'
previous_summary=json.loads(summary_path.read_text()) if summary_path.exists() else {{}}
row=con.execute("SELECT value FROM meta WHERE key='next_cycle'").fetchone();saved=row[0] if row else None
next_cycle=max(int(saved or 1),int(previous_summary.get('cycle',0))+1)
con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('next_cycle',?)",(str(next_cycle),))
con.execute("INSERT OR IGNORE INTO source_provenance(language,page_id,source_key,dataset,license,kind) SELECT language,page_id,'wiki:carryover-v15c','Wikipedia API','CC-BY-SA-4.0','encyclopedia' FROM sources")
con.commit();counts={{'pages':con.execute('SELECT COUNT(*) FROM sources').fetchone()[0],'examples':con.execute('SELECT COUNT(*) FROM examples').fetchone()[0]}};con.close()
receipt={{'copied_from':'outputs/translation_segmenter_v15c_20261026/corpus.sqlite','sqlite_backup_api':'Connection.backup','copied_new_database':copied,'source_cycle':previous_summary.get('cycle'),'source_database_sha256':source_hash_db,'counts':counts,'next_cycle':next_cycle,'teacher_weights_copied':False,'qwen_downloaded':False}}
(run/'carryover_receipt.json').write_text(json.dumps(receipt,indent=2))
config={config!r};config_path=run/'config.json'
if config_path.exists():
 old_config=json.loads(config_path.read_text())
 if old_config!=config:
  old_core=dict(old_config);new_core=dict(config);old_hash=old_core.pop('source_sha256',None);new_hash=new_core.pop('source_sha256',None)
  assert old_core==new_core,'Config mismatch; refusing to overwrite existing v15d run'
  hist_path=run/'source_revision_history.json';history=json.loads(hist_path.read_text()) if hist_path.exists() else []
  history.append({{'at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'previous_sha256':old_hash,'new_sha256':new_hash,'reason':'v15d2 fixes snapshot packaging; corpus schema and HF stream state unchanged'}})
  hist_path.write_text(json.dumps(history,indent=2));config_path.write_text(json.dumps(config,indent=2,ensure_ascii=False))
else:config_path.write_text(json.dumps(config,indent=2,ensure_ascii=False))
if shutil.disk_usage(root).free<25*1024**3:raise RuntimeError('Less than 25 GiB free on Colab; refusing to launch')
cmd=[sys.executable,'-u','-m','turn_v2.translation_segmenter_v15d2','--phase','run','--docs-per-source',str({DOCS_PER_SOURCE}),'--steps',str({STEPS}),'--microbatch',str({MICROBATCH})]
log=run/('worker_'+str(len(list(run.glob('worker_*.log')))+1)+'.log')
with log.open('w') as stream:p=subprocess.Popen(cmd,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1'}})
launcher={{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':cmd,'module':'turn_v2.translation_segmenter_v15d2','log':str(log.relative_to(root)),'source_sha256':{source_hash!r},'campaign':'translation_segmenter_v15d','max_cycles':0}}
(run/'launcher.json').write_text(json.dumps(launcher,indent=2));print(json.dumps({{'launcher':launcher,'carryover':receipt}}))
'''
    value = execute(runtime, "launch", code)
    (LOCAL / "launcher.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    carry = value.get("carryover", {})
    if not carry.get("copied_new_database", False):
        resume = {"event": "resumed existing v15d database",
            "previous_worker_module": "turn_v2.translation_segmenter_v15d",
            "new_worker_module": "turn_v2.translation_segmenter_v15d2",
            "database_copied": False, "database_sha256_at_resume": carry.get("source_database_sha256"),
            "pages_at_resume": carry.get("counts", {}).get("pages"),
            "examples_at_resume": carry.get("counts", {}).get("examples"),
            "previous_cycle": carry.get("source_cycle"), "next_cycle": carry.get("next_cycle"),
            "weights_or_optimizer_copied": False, "stream_state_kept_in_existing_sqlite": True}
        (LOCAL / "resume_receipt.json").write_text(json.dumps(resume, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return value


def status(runtime):
    code = f'''import json,sqlite3,subprocess,shutil
from pathlib import Path
root=Path({REMOTE_ROOT!r});run=root/{REMOTE_RUN!r};out={{}}
for n in ['launcher.json','child.json','status.json','latest_error.json','config.json','data_summary.json','carryover_receipt.json','recovery/latest_snapshot.json']:
 p=run/n
 if p.exists():out[n]=json.loads(p.read_text())
for n in ['launcher.json','child.json']:
 r=out.get(n,{{}});pid=r.get('pid');p=Path('/proc')/str(pid);ok=False
 if pid and (p/'cmdline').exists():
  try:
   cmd=(p/'cmdline').read_bytes();ticks=(p/'stat').read_text().split()[21];argv=r.get('command',[]);expected=argv[argv.index('-m')+1] if '-m' in argv else r.get('module','turn_v2.translation_segmenter_v15d');ok=expected.encode()+b'\\x00' in cmd and ticks==r.get('start_ticks');out[n+'_current_start_ticks']=ticks
  except (OSError,IndexError):pass
 out[n+'_process_verified']=ok
 if r.get('log'):
  log=root/r['log']
  if log.exists():out[n+'_log_tail']=log.read_text(errors='replace')[-1800:]
db=run/'corpus.sqlite'
if db.exists():
 c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
 out['pages_by_language']=[dict(zip(['language','pages'],x)) for x in c.execute('SELECT language,COUNT(*) FROM sources GROUP BY 1 ORDER BY 1')]
 out['examples_by_language_split']=[dict(zip(['language','split','examples'],x)) for x in c.execute('SELECT language,split,COUNT(*) FROM examples GROUP BY 1,2 ORDER BY 1,2')]
 if c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='crawl_observations'").fetchone():
  out['source_dispositions']=[dict(zip(['language','source_key','disposition','documents','examples'],x)) for x in c.execute('SELECT language,source_key,disposition,COUNT(*),SUM(examples) FROM crawl_observations GROUP BY 1,2,3 ORDER BY 1,2,3')]
 c.close()
out['teachers']={{}}
for lang in ['vi','en','zh']:
 out['teachers'][lang]={{}}
 for name in ['identity.json','progress.json','deferred.json','best_report.json','last_training_report.json']:
  p=run/'teachers'/lang/name
  if p.exists():out['teachers'][lang][name]=json.loads(p.read_text())
out['GPU']=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
out['remote_disk_free_bytes']=shutil.disk_usage('/content').free
print(json.dumps(out,ensure_ascii=False))
'''
    value = execute(runtime, "status", code)
    value["queried_at_utc"] = datetime.now(timezone.utc).isoformat()
    (LOCAL / "status_receipt.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    launcher = value.get("launcher.json", {})
    live = value.get("status.json", {})
    task = {
        "campaign": "translation_segmenter_v15d", "updated_at_utc": value["queried_at_utc"],
        "status": live.get("status", "UNKNOWN"),
        "objective": {"model_input": "punctuation-free text with one or two future words",
            "label_source": "sentence-final punctuation EOT; internal punctuation NEOT",
            "punctuation_in_model_input": False, "qwen_used": False,
            "languages": ["vi", "en", "zh"], "quality_target_certified": False,
            "prose_labels_are_human_gold": False},
        "runtime": {"parent_pid": launcher.get("pid"),
            "parent_start_ticks": launcher.get("start_ticks"),
            "parent_verified": value.get("launcher.json_process_verified", False),
            "child_pid": value.get("child.json", {}).get("pid"),
            "child_verified": value.get("child.json_process_verified", False),
            "stage": live.get("stage"), "cycle": live.get("cycle", value.get("data_summary.json", {}).get("cycle", 0) + 1),
            "last_completed_crawl_cycle": value.get("data_summary.json", {}).get("cycle"),
            "source": live.get("source"), "gpu": value.get("GPU"),
            "remote_disk_free_bytes": value.get("remote_disk_free_bytes")},
        "carryover": value.get("carryover_receipt.json", {}),
        "data": {"pages_by_language": value.get("pages_by_language", []),
            "examples_by_language_split": value.get("examples_by_language_split", []),
            "source_dispositions": value.get("source_dispositions", []),
            "latest_snapshot": value.get("recovery/latest_snapshot.json")},
        "teachers": value.get("teachers", {}),
        "snapshot_scope": "data/metadata only; no model weights or optimizer"}
    (LOCAL / "task_state.json").write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return value


def stop(runtime):
    code = f'''import json
from pathlib import Path
root=Path({REMOTE_ROOT!r});run=root/{REMOTE_RUN!r};r=json.loads((run/'launcher.json').read_text());pid=r['pid'];p=Path('/proc')/str(pid)
assert (p/'cmdline').exists(),'v15d parent already absent'
cmd=(p/'cmdline').read_bytes();ticks=(p/'stat').read_text().split()[21]
argv=r.get('command',[]);expected=argv[argv.index('-m')+1] if '-m' in argv else r.get('module','turn_v2.translation_segmenter_v15d')
assert expected.encode()+b'\\x00' in cmd and ticks==r['start_ticks'],'Parent identity mismatch; refusing stop'
child_path=run/'child.json';child=json.loads(child_path.read_text()) if child_path.exists() else {{}};cpid=child.get('pid');cp=Path('/proc')/str(cpid);verified=False
if cpid and (cp/'cmdline').exists():
 cc=(cp/'cmdline').read_bytes();ct=(cp/'stat').read_text().split()[21];child_argv=child.get('command',[]);child_expected=child_argv[child_argv.index('-m')+1] if '-m' in child_argv else child.get('module','turn_v2.translation_segmenter_v15d');verified=child_expected.encode()+b'\\x00' in cc and ct==child.get('start_ticks')
 assert verified,'Training child identity mismatch; refusing stop'
marker=run/'STOP_REQUESTED';marker.write_text('Requested graceful stop; retain SQLite, HF stream states and model checkpoints.\\n')
print(json.dumps({{'marker':str(marker),'parent_pid':pid,'parent_verified':True,'child_pid':cpid,'child_verified':verified}}))
'''
    return execute(runtime, "stop", code)


def snapshot(runtime):
    code = f'''import json,hashlib
from pathlib import Path
root=Path({REMOTE_ROOT!r});run=root/{REMOTE_RUN!r};p=run/'recovery/latest_snapshot.json'
meta=json.loads(p.read_text());archive=root/meta['path'];assert archive.exists()
assert archive.stat().st_size==meta['bytes'] and hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()==meta['sha256']
print(json.dumps(meta))
'''
    meta = execute(runtime, "snapshot", code)
    if meta.get("scope") != "DATA_AND_METADATA_ONLY_NO_MODEL_WEIGHTS":
        raise RuntimeError(f"Unexpected snapshot scope: {meta.get('scope')}")
    ensure_free_space(LOCAL, meta["bytes"] * 3 + 10 * 1024**3)
    remote_path = "content/turn_detect_v2/" + meta["path"]
    payload = runtime.request("GET", "api/contents/" + quote(remote_path, safe="/") + "?format=base64").json()
    if payload.get("type") != "file" or payload.get("format") != "base64":
        raise RuntimeError("Colab Contents API did not return a base64 file")
    raw = base64.b64decode(payload["content"])
    digest = hashlib.sha256(raw).hexdigest(); size = len(raw)
    if size != meta["bytes"] or digest != meta["sha256"]:
        raise RuntimeError("Snapshot SHA/size mismatch")
    part = LOCAL / "data_snapshot.partial.zip"
    part.write_bytes(raw)
    target = LOCAL / "data_snapshot.zip"
    previous = LOCAL / f"data_snapshot_cycle{meta['cycle'] - 1}.zip"
    if target.exists() and not previous.exists(): shutil.copy2(target, previous)
    part.replace(target)
    with zipfile.ZipFile(target) as archive:
        db_bytes = archive.read("state_snapshot.sqlite")
        for filename in ("status.json", "data_summary.json", "config.json"):
            if filename in archive.namelist():
                (LOCAL / filename).write_bytes(archive.read(filename))
    temp_db = LOCAL / "state_snapshot.partial.sqlite"; temp_db.write_bytes(db_bytes)
    con = sqlite3.connect(f"file:{temp_db}?mode=ro", uri=True)
    integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    counts = {"pages": con.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
              "examples": con.execute("SELECT COUNT(*) FROM examples").fetchone()[0]}
    con.close()
    if integrity != "ok": temp_db.unlink(missing_ok=True); raise RuntimeError(f"SQLite integrity: {integrity}")
    local_db = LOCAL / "state_snapshot.sqlite"
    prev_db = LOCAL / f"state_snapshot_cycle{meta['cycle'] - 1}.sqlite"
    if local_db.exists() and not prev_db.exists(): shutil.copy2(local_db, prev_db)
    temp_db.replace(local_db)
    # Download only reports and per-case dev traces, never checkpoints or model weights.
    for language in LANGUAGES:
        for filename in ("identity.json", "progress.json", "deferred.json", "best_report.json",
                         "last_training_report.json", "dev_trace.jsonl"):
            relative = f"{REMOTE_RUN}/teachers/{language}/{filename}"
            url = runtime.base_url + "files/" + quote("content/turn_detect_v2/" + relative, safe="/")
            response = runtime.http.get(url, stream=True, timeout=60, allow_redirects=False)
            if response.status_code != 200 or "html" in response.headers.get("content-type", ""):
                response.close()
                continue
            target_file = LOCAL / "teachers" / language / filename
            partial_file = target_file.with_suffix(target_file.suffix + ".partial")
            target_file.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with partial_file.open("wb") as stream:
                for block in response.iter_content(1024 * 1024):
                    if block:
                        written += len(block)
                        if written > 30 * 1024**2:
                            stream.close(); partial_file.unlink(missing_ok=True); break
                        stream.write(block)
            response.close()
            if partial_file.exists() and written <= 30 * 1024**2:
                partial_file.replace(target_file)
    meta.update({"local_sha256": digest.hexdigest(), "database_integrity": integrity, "local_counts": counts})
    (LOCAL / "snapshot_receipt.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = subprocess.run([sys.executable, "tools/report_translation_segmenter_v15d.py"],
        check=True, capture_output=True, text=True, encoding="utf-8")
    (LOCAL / "review_build_receipt.json").write_text(report.stdout, encoding="utf-8")
    return {"snapshot": meta, "review": json.loads(report.stdout)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "status", "snapshot", "stop"))
    parser.add_argument("--session-path", help="Exact a.ipynb notebook session if more than one exists")
    args = parser.parse_args(); LOCAL.mkdir(parents=True, exist_ok=True)
    if args.action == "launch": ensure_free_space(LOCAL, 10 * 1024**3)
    runtime = campaign_runtime(args.session_path)
    try:
        connection = runtime.connect()
        if args.action == "launch" and connection.get("execution_state") != "idle":
            raise RuntimeError("Colab kernel is busy; refusing to queue a crawler launch into another notebook cell")
        if args.action == "launch": value = launch(runtime)
        elif args.action == "status": value = status(runtime)
        elif args.action == "stop": value = stop(runtime)
        else: value = snapshot(runtime)
        print(json.dumps({"connection": connection, "result": value}, ensure_ascii=False))
    finally:
        runtime.close()


if __name__ == "__main__": main()
