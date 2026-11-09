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
from storage_guard import ensure_free_space, download_checkpoint as download_verified_file


LOCAL = Path("outputs/translation_segmenter_v15d_20261026")
REMOTE_ROOT = "/content/turn_detect_v2"
REMOTE_RUN = "outputs/translation_segmenter_v15d_20261026"
SOURCE = Path("turn_v2/translation_segmenter_v15d3.py")
LANGUAGES = ("vi", "en", "zh")
DOCS_PER_SOURCE = 200
STEPS = 100
MICROBATCH = 32
SOURCES = [
    {"key": "hf:fineweb2:vie_Latn", "language": "vi", "dataset": "HuggingFaceFW/fineweb-2", "config": "vie_Latn", "split": "train", "license": "ODC-By-1.0", "kind": "web"},
    {"key": "hf:fineweb:en", "language": "en", "dataset": "HuggingFaceFW/fineweb", "config": None, "split": "train", "license": "ODC-By-1.0", "kind": "web"},
    {"key": "hf:fineweb2:cmn_Hani", "language": "zh", "dataset": "HuggingFaceFW/fineweb-2", "config": "cmn_Hani", "split": "train", "license": "ODC-By-1.0", "kind": "web"},
    {"key": "hf:industrycorpus2_news_media:zh", "language": "zh", "dataset": "BAAI/IndustryCorpus2_news_media", "config": None, "split": "train", "license": "Apache-2.0", "kind": "news_media"},
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
    runtime.request("PUT", "api/contents/content/turn_detect_v2/colab/translation_segmenter_v15d3.py",
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
for module in [b'turn_v2.teacher_first_v14',b'turn_v2.continuous_v13',b'turn_v2.translation_segmenter_v15',b'turn_v2.translation_segmenter_v15b',b'turn_v2.translation_segmenter_v15c',b'turn_v2.translation_segmenter_v15d',b'turn_v2.translation_segmenter_v15d2',b'turn_v2.translation_segmenter_v15d3']:
  assert module+b'\\x00' not in cmd,'Another campaign process is active: '+proc.name
uploaded=root/'colab/translation_segmenter_v15d3.py';data=uploaded.read_bytes()
assert hashlib.sha256(data).hexdigest()=={source_hash!r}
target=root/'turn_v2/translation_segmenter_v15d3.py'
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
  previous_sources=list(new_core['datasets']);previous_sources.insert(2,{{'key':'hf:industrycorpus2_news_media:en','language':'en','dataset':'BAAI/IndustryCorpus2_news_media','config':None,'split':'train','license':'Apache-2.0','kind':'news_media'}})
  expected_old=dict(new_core);expected_old['datasets']=previous_sources;expected_old['docs_per_hf_source_cycle']=100
  assert old_core==expected_old,'Unexpected config drift; refusing to overwrite existing v15d run'
  hist_path=run/'source_revision_history.json';history=json.loads(hist_path.read_text()) if hist_path.exists() else []
  history.append({{'at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'previous_sha256':old_hash,'new_sha256':new_hash,'reason':'v15d3 quarantines the BAAI EN stream after a cross-language hash audit and adds Chinese ASCII-period extraction'}})
  hist_path.write_text(json.dumps(history,indent=2));config_path.write_text(json.dumps(config,indent=2,ensure_ascii=False))
else:config_path.write_text(json.dumps(config,indent=2,ensure_ascii=False))
if shutil.disk_usage(root).free<25*1024**3:raise RuntimeError('Less than 25 GiB free on Colab; refusing to launch')
cmd=[sys.executable,'-u','-m','turn_v2.translation_segmenter_v15d3','--phase','run','--docs-per-source',str({DOCS_PER_SOURCE}),'--steps',str({STEPS}),'--microbatch',str({MICROBATCH})]
log=run/('worker_'+str(len(list(run.glob('worker_*.log')))+1)+'.log')
with log.open('w') as stream:p=subprocess.Popen(cmd,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1'}})
launcher={{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':cmd,'module':'turn_v2.translation_segmenter_v15d3','log':str(log.relative_to(root)),'source_sha256':{source_hash!r},'campaign':'translation_segmenter_v15d','max_cycles':0}}
(run/'launcher.json').write_text(json.dumps(launcher,indent=2));print(json.dumps({{'launcher':launcher,'carryover':receipt}}))
'''
    value = execute(runtime, "launch", code)
    (LOCAL / "launcher.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    carry = value.get("carryover", {})
    if not carry.get("copied_new_database", False):
        resume = {"event": "resumed existing v15d database",
            "previous_worker_module": "turn_v2.translation_segmenter_v15d2",
            "new_worker_module": "turn_v2.translation_segmenter_v15d3",
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
for n in ['launcher.json','child.json','status.json','latest_error.json','config.json','data_summary.json','carryover_receipt.json','recovery/latest_snapshot.json','recovery/drive_restore_receipt_v1.json']:
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
 for name in ['identity.json','progress.json','deferred.json','best_report.json','last_training_report.json','resume_receipt.json']:
  p=run/'teachers'/lang/name
  if p.exists():out['teachers'][lang][name]=json.loads(p.read_text())
out['GPU']=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
out['remote_disk_free_bytes']=shutil.disk_usage('/content').free
full=run/'full_epoch_v1';out['full_epoch_campaign']={{}}
for name in ('config.json','data_status.json','data_error_v2.json','current_dataset.json','campaign_status.json','campaign_error.json','training_status.json','dataset_backup_ready.json','latest_round_receipt.json','transition_v2/status.json','transition_v2/error.json'):
 p=full/name
 if p.exists():out['full_epoch_campaign'][name]=json.loads(p.read_text())
job=run/'diagnostic_test_c75_v1'
if job.exists():
 d={{}}
 for name in ['status.json','analysis_v2/status.json','continuation_v2_error.json','analysis_v2/error.json','review_publication_v3/status.json','review_publication_v3/error.json','review_publication_v4/status.json','review_publication_v4/error.json','review_publication_v5/status.json','review_publication_v5/error.json','review_publication_v6/status.json','review_publication_v6/error.json','review_publication_v7/status.json','review_publication_v7/error.json','legacy_comparison_validation_v1.json','old_dev_c75_vi_report.json']:
  p=job/name
  if p.exists():d[name]=json.loads(p.read_text())
 p=job/'launcher.json';record=json.loads(p.read_text()) if p.exists() else {{}};verified=False
 try:
  argv=record['command'];module=argv[argv.index('-m')+1];proc=Path('/proc')/str(record['pid'])
  verified=module.encode() in (proc/'cmdline').read_bytes().split(bytes([0])) and (proc/'stat').read_text().split()[21]==record['start_ticks']
 except (OSError,KeyError,ValueError,IndexError):pass
 d['process_verified']=verified
 p=job/'report.json';q=job/'partial_report.json'
 report=json.loads(p.read_text()) if p.exists() else json.loads(q.read_text()) if q.exists() else {{}}
 d['report_complete']=report.get('complete',False)
 d['languages']={{lang:{{'metrics':value.get('metrics',{{}}),'leakage_audit':value.get('leakage_audit',{{}})}} for lang,value in report.get('languages',{{}}).items()}}
 d['analysis_complete']=(job/'error_analysis_v1.json').exists();d['parser_audit_complete']=(job/'parser_audit_v1.json').exists()
 d['reserve_exposure_audit_complete']=(job/'reserve_exposure_audit_v1.json').exists()
 for name in ['priority_window_v1/status.json','priority_window_v1/pause_receipt.json','priority_window_v1/resume_receipt.json','priority_window_v1/error.json','finalizer_v1/status.json']:
  p=job/name
  if p.exists():d[name]=json.loads(p.read_text())
 record=out.get('launcher.json',{{}})
 try:
  proc=Path('/proc')/str(record['pid'])
  d['campaign_launcher_paused_for_user_test_priority']=((proc/'stat').read_text().split()[2] in ('T','t') and d.get('priority_window_v1/pause_receipt.json',{{}}).get('parent',{{}}).get('start_ticks')==record['start_ticks'])
 except (OSError,KeyError,IndexError):d['campaign_launcher_paused_for_user_test_priority']=False
 p=job/'drive_manifest.json';d['drive_complete']=json.loads(p.read_text()).get('complete',False) if p.exists() else False
 out['diagnostic_c75']=d
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
            "child_start_ticks": value.get("child.json", {}).get("start_ticks"),
            "child_verified": value.get("child.json_process_verified", False),
            "stage": live.get("stage"), "cycle": live.get("cycle", value.get("data_summary.json", {}).get("cycle", 0) + 1),
            "language": live.get("language"), "step": live.get("step"), "steps": live.get("steps"),
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
    # These receipts belong to separate control operations; polling must retain
    # their evidence instead of silently erasing it from task_state.json.
    receipt_path = LOCAL / "snapshot_receipt.json"
    if receipt_path.exists():
        task["local_snapshot"] = json.loads(receipt_path.read_text(encoding="utf-8"))
        local_meta = task["local_snapshot"]
        db_file = LOCAL / "state_snapshot.sqlite"
        db_bytes = local_meta.get("sqlite_uncompressed_bytes", db_file.stat().st_size if db_file.exists() else 0)
        free = shutil.disk_usage(LOCAL).free
        need = int(db_bytes) * 2 + int(local_meta["bytes"]) * 2 + 10 * 1024**3
        task["local_storage"] = {"free_bytes": free, "reserve_bytes": 10 * 1024**3,
            "conservative_next_snapshot_need_bytes": need,
            "estimated_large_sync_allowed": free >= need}
    policy_path = LOCAL / "storage_policy.json"
    if policy_path.exists():
        task["storage_policy"] = json.loads(policy_path.read_text(encoding="utf-8"))
        if not task["storage_policy"].get("local_large_snapshot_downloads_enabled", True):
            task.setdefault("local_storage", {})["estimated_large_sync_allowed"] = False
            task["local_storage"]["large_artifacts_destination"] = "google_drive"
    receipt_path = LOCAL / "restore_latest_receipt.json"
    if receipt_path.exists():
        restored = json.loads(receipt_path.read_text(encoding="utf-8"))
        task["recovery"] = restored.get("remote_restore", {}).get("restore_receipt", {})
    receipt_path = LOCAL / "heldout_test_v2" / "status_receipt.json"
    if receipt_path.exists():
        evaluated = json.loads(receipt_path.read_text(encoding="utf-8")).get("result", {})
        task["test_evaluation"] = {"state": evaluated.get("status.json", {}),
            "process_verified_at_check": evaluated.get("process_verified", False),
            "launcher": evaluated.get("launcher.json", {}),
            "control_script": "tools/translation_segmenter_heldout_colab.py",
            "test_scores_available": bool(evaluated.get("report.json", {}).get("complete"))}
        report_path = LOCAL / "heldout_test_v2" / "report.json"
        summary_path = LOCAL / "heldout_test_v2" / "result_summary.json"
        if report_path.exists() and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("report_sha256") == file_sha(report_path):
                task["test_evaluation"]["summary"] = summary
                task["test_evaluation"]["final_report_local"] = str(report_path)
                task["test_evaluation"]["review_local"] = str(report_path.with_name("review.html"))
    receipt_path = LOCAL / "drive_status_receipt.json"
    if receipt_path.exists():
        drive = json.loads(receipt_path.read_text(encoding="utf-8")).get("result", {})
        manifest = drive.get("latest_verified_manifest.json", {})
        progress = drive.get("progress_manifest.json", {})
        task["drive_backup"] = {"state": drive.get("status.json", {}),
            "process_verified_at_check": drive.get("process_verified", False),
            "launcher": drive.get("launcher.json", {}),
            "full_recovery_available": manifest.get("full_recovery_available", False),
            "latest_verified_data_cycle": manifest.get("files", {}).get("data/data_snapshot.zip", {}).get("source_receipt", {}).get("cycle"),
            "latest_manifest": manifest,
            "recovery_verification_state": drive.get("verification_status_v2.json", {}),
            "recovery_verification": drive.get("recovery_verification.json", {})}
        scored = task.get("test_evaluation", {}).get("summary", {})
        if scored:
            files = manifest.get("files", {})
            report_matches = files.get("heldout_test_v2/report.json", {}).get("sha256") == scored.get("report_sha256")
            evaluated = json.loads((LOCAL / "heldout_test_v2" / "report.json").read_text(encoding="utf-8"))
            trace_matches = {lang: bool(item.get("full_trace_sha256") and files.get(f"heldout_test_v2/test_predictions_{lang}.jsonl.gz", {}).get("sha256") == item["full_trace_sha256"])
                             for lang, item in evaluated.get("languages", {}).items()}
            task["test_evaluation"]["full_drive_backup_verified"] = bool(manifest.get("pass_complete") and report_matches and len(trace_matches) == 3 and all(trace_matches.values()))
            task["test_evaluation"]["full_trace_manifest_matches"] = trace_matches
    restore_path = LOCAL / "drive_restore_status_v1.json"
    if restore_path.exists() or value.get("recovery/drive_restore_receipt_v1.json"):
        proof = value.get("recovery/drive_restore_receipt_v1.json", {})
        validation_finished = bool(value.get("launcher.json_process_verified") and
            (live.get("status") == "RUNNING" or str(live.get("stage", "")).startswith("RESUMING_") or
             live.get("stage") == "RESTORED_VERIFIED_CHECKPOINTS"))
        # Read proof from this VM; a saved Windows control receipt is history.
        task["runtime_recovery"] = {"state": live,
            "restore_process_verified_at_check": value.get("launcher.json_process_verified", False),
            "receipt": proof,
            "control_script": "tools/restore_translation_segmenter_drive_colab_v3.py" if proof.get("restore_module") == "restore_drive_campaign_v3" else "tools/restore_translation_segmenter_drive_colab_v1.py"}
        task["teacher_weights_optimizer_restored_in_current_VM"] = bool(
            validation_finished and proof.get("weights_optimizer_rng_restored") and proof.get("training_fingerprints_match"))
    if value.get("diagnostic_c75"):
        task["diagnostic_test_c75_v1"] = {**value["diagnostic_c75"],
            "control_script": "tools/translation_segmenter_diagnostic_colab.py",
            "old_heldout_not_rescored": True, "unscored_reserve_protected": True,
            "review": "diagnostic_test_c75_v1/bao_cao_ket_qua.html",
            "report_audience": "nontechnical; final results, model inputs/outputs and real examples; no execution diary",
            "technical_review_preserved": "diagnostic_test_c75_v1/review_technical_v3.html",
            "pending_local_report_publication": json.loads((LOCAL / "diagnostic_test_c75_v1/pending_drive_publication_v5.json").read_text(encoding="utf-8")) if (LOCAL / "diagnostic_test_c75_v1/pending_drive_publication_v5.json").exists() else None,
            "test_provenance": "diagnostic_test_c75_v1/test_provenance.md",
            "crawl_plan": "diagnostic_test_c75_v1/crawl_plan_v3.json",
            "reported_to_user": (LOCAL / "diagnostic_test_c75_v1/notified_receipt.json").exists(),
            "required_followup": "preserve completed test and traces; prepared changes are not activated; do not relaunch or rescore; continue campaign and verified Drive backups"}
    task["full_epoch_campaign"] = value.get("full_epoch_campaign", {})
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
    policy_path = LOCAL / "storage_policy.json"
    if policy_path.exists() and not json.loads(policy_path.read_text(encoding="utf-8")).get("local_large_snapshot_downloads_enabled", True):
        raise RuntimeError("Local data snapshot download disabled by the user. Use Drive backup and the bounded remote review export.")
    code = f'''import json,hashlib,zipfile
from pathlib import Path
root=Path({REMOTE_ROOT!r});run=root/{REMOTE_RUN!r};p=run/'recovery/latest_snapshot.json'
meta=json.loads(p.read_text());archive=root/meta['path'];assert archive.exists()
assert archive.stat().st_size==meta['bytes'] and hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()==meta['sha256']
with zipfile.ZipFile(archive) as z:meta['sqlite_uncompressed_bytes']=z.getinfo('state_snapshot.sqlite').file_size
print(json.dumps(meta))
'''
    meta = execute(runtime, "snapshot", code)
    if meta.get("scope") != "DATA_AND_METADATA_ONLY_NO_MODEL_WEIGHTS":
        raise RuntimeError(f"Unexpected snapshot scope: {meta.get('scope')}")
    # ZIP bytes alone underestimate the peak: extraction and retaining the
    # previous SQLite snapshot can both require full uncompressed databases.
    sqlite_bytes = int(meta["sqlite_uncompressed_bytes"])
    ensure_free_space(LOCAL, sqlite_bytes * 2 + meta["bytes"] * 2,
                      reserve_bytes=10 * 1024**3)
    target = LOCAL / "data_snapshot.zip"
    reuse_local = (target.exists() and target.stat().st_size == meta["bytes"]
                   and file_sha(target) == meta["sha256"])
    if reuse_local:
        digest = meta["sha256"]
        size = target.stat().st_size
    else:
        remote_path = "content/turn_detect_v2/" + meta["path"]
        part = LOCAL / f"data_snapshot_{meta['sha256'][:12]}.incoming.zip"
        download_verified_file(runtime, remote_path, part, meta["bytes"], meta["sha256"])
        digest = meta["sha256"]; size = part.stat().st_size
        # Recheck after a potentially long transfer, before full SQLite copies.
        ensure_free_space(LOCAL, sqlite_bytes * 2 + meta["bytes"], reserve_bytes=10 * 1024**3)
    if target.exists() and not reuse_local:
        with zipfile.ZipFile(target) as old_archive:
            if "data_summary.json" in old_archive.namelist():
                previous_cycle = json.loads(old_archive.read("data_summary.json")).get("cycle")
            else:
                previous_cycle = json.loads(old_archive.read("status.json")).get("cycle")
        if previous_cycle is None:
            old_receipt = LOCAL / "snapshot_receipt.json"
            previous_cycle = json.loads(old_receipt.read_text(encoding="utf-8")).get("cycle") if old_receipt.exists() else None
        if previous_cycle is not None:
            previous = LOCAL / f"data_snapshot_cycle{previous_cycle}.zip"
            target_sha = file_sha(target)
            if previous.exists() and file_sha(previous) != target_sha:
                previous = LOCAL / f"data_snapshot_cycle{previous_cycle}_{target_sha[:8]}.zip"
            if not previous.exists(): shutil.copy2(target, previous)
    if not reuse_local:
        part.replace(target)
    temp_db = LOCAL / "state_snapshot.partial.sqlite"
    with zipfile.ZipFile(target) as archive:
        with archive.open("state_snapshot.sqlite") as source, temp_db.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        for filename in ("status.json", "data_summary.json", "config.json"):
            if filename in archive.namelist():
                (LOCAL / filename).write_bytes(archive.read(filename))
    con = sqlite3.connect(f"file:{temp_db}?mode=ro", uri=True)
    integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    counts = {"pages": con.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
              "examples": con.execute("SELECT COUNT(*) FROM examples").fetchone()[0]}
    con.close()
    if integrity != "ok": temp_db.unlink(missing_ok=True); raise RuntimeError(f"SQLite integrity: {integrity}")
    local_db = LOCAL / "state_snapshot.sqlite"
    if local_db.exists():
        previous_cycle = None
        try:
            old_db = sqlite3.connect(f"file:{local_db.resolve()}?mode=ro", uri=True)
            row = old_db.execute("SELECT value FROM meta WHERE key='next_cycle'").fetchone()
            previous_cycle = int(row[0]) - 1 if row and row[0].isdigit() else None
            old_db.close()
        except sqlite3.Error:
            previous_cycle = None
        if previous_cycle is not None:
            prev_db = LOCAL / f"state_snapshot_cycle{previous_cycle}.sqlite"
            local_db_sha = file_sha(local_db)
            if prev_db.exists() and file_sha(prev_db) != local_db_sha:
                prev_db = LOCAL / f"state_snapshot_cycle{previous_cycle}_{local_db_sha[:8]}.sqlite"
            if not prev_db.exists():
                ensure_free_space(LOCAL, local_db.stat().st_size, reserve_bytes=10 * 1024**3)
                shutil.copy2(local_db, prev_db)
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
    meta.update({"local_sha256": digest, "database_integrity": integrity, "local_counts": counts})
    (LOCAL / "snapshot_receipt.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = subprocess.run([sys.executable, "tools/render_translation_segmenter_v15d_local_review.py"],
        check=True, capture_output=True, text=True, encoding="utf-8")
    (LOCAL / "review_build_receipt.json").write_text(report.stdout, encoding="utf-8")
    return {"snapshot": meta, "review": json.loads(report.stdout)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "status", "snapshot", "stop"))
    parser.add_argument("--session-path", help="Exact a.ipynb notebook session if more than one exists")
    args = parser.parse_args(); LOCAL.mkdir(parents=True, exist_ok=True)
    if args.action == "launch": ensure_free_space(LOCAL, 10 * 1024**3)
    runtime = ColabRuntime(0, session_path=args.session_path)
    try:
        connection = runtime.connect()
        if connection.get("execution_state") != "idle":
            if args.action == "status":
                from translation_segmenter_full_epoch_colab import status as full_status, status_http
                try: value = full_status(runtime, via_terminal=True)
                except Exception: value = status_http(runtime)
                (LOCAL / "busy_kernel_metadata_receipt.json").write_text(json.dumps({"connection": connection, "result": value}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({"connection": connection, "result": value}, ensure_ascii=False))
                return
            raise RuntimeError("Colab kernel is busy; refusing to queue a crawler launch into another notebook cell")
        if args.action == "launch": value = launch(runtime)
        elif args.action == "status": value = status(runtime)
        elif args.action == "stop": value = stop(runtime)
        else: value = snapshot(runtime)
        print(json.dumps({"connection": connection, "result": value}, ensure_ascii=False))
    finally:
        runtime.close()


if __name__ == "__main__": main()
