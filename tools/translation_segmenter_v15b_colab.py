"""Launch/read the no-Qwen, 1/2-word lookahead campaign on the pinned Colab."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import subprocess
import sys
from urllib.parse import quote
import zipfile

from colab_runtime import ColabRuntime
from storage_guard import ensure_free_space
from teacher_first_colab import campaign_runtime


ROOT = Path("outputs/translation_segmenter_v15b_20261026")
SOURCE = Path("turn_v2/translation_segmenter_v15b.py")
REMOTE_ROOT = "/content/turn_detect_v2"
REMOTE_RUN = "outputs/translation_segmenter_v15b_20261026"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "status", "snapshot", "stop"))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    runtime = campaign_runtime()
    try:
        runtime.connect()
        if args.action == "launch":
            source = SOURCE.read_bytes(); source_hash = sha(source)
            runtime.request("PUT", "api/contents/content/turn_detect_v2/colab/translation_segmenter_v15b.py",
                json={"type":"file","format":"base64","content":base64.b64encode(source).decode("ascii")})
            code = f'''import base64,hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path({REMOTE_ROOT!r});run=root/{REMOTE_RUN!r};run.mkdir(parents=True,exist_ok=True)
old= root/'outputs/teacher_first_v14_20261026'
old_status=json.loads((old/'status.json').read_text())
assert old_status.get('status')=='STOPPED_BY_REQUEST','v14 must be stopped before v15 uses the GPU'
previous=root/'outputs/translation_segmenter_v15_20261026'
previous_status=json.loads((previous/'status.json').read_text())
assert previous_status.get('status')=='STOPPED_BY_REQUEST','v15 must be gracefully stopped before versioned resume'
for pid in Path('/proc').iterdir():
 if pid.name.isdigit():
  try:cmd=(pid/'cmdline').read_bytes()
  except OSError:continue
  assert b'turn_v2.teacher_first_v14' not in cmd,'v14 worker still active'
  assert b'turn_v2.translation_segmenter_v15\\x00' not in cmd,'old v15 worker still active'
  if b'turn_v2.translation_segmenter_v15b' in cmd: raise RuntimeError('v15b already active; refusing duplicate')
uploaded=root/'colab/translation_segmenter_v15b.py';data=uploaded.read_bytes()
assert hashlib.sha256(data).hexdigest()=={source_hash!r}
target=root/'turn_v2/translation_segmenter_v15b.py'
if target.exists(): assert hashlib.sha256(target.read_bytes()).hexdigest()=={source_hash!r},'preserve remote source; use a versioned repair module'
else: target.write_bytes(data)
old_db=previous/'corpus.sqlite';new_db=run/'corpus.sqlite'
copied=False
if old_db.exists() and not new_db.exists():
 import sqlite3
 src=sqlite3.connect(old_db);dst=sqlite3.connect(new_db);src.backup(dst);src.close();dst.close()
 copied=True
if new_db.exists():
 import sqlite3
 c=sqlite3.connect(new_db);counts=c.execute('SELECT COUNT(*) FROM examples').fetchone()[0]
 next_cycle=c.execute("SELECT value FROM meta WHERE key='next_cycle'").fetchone()
 c.close()
 (run/'carryover_receipt.json').write_text(json.dumps({{'source_run':str(previous),'source_v14_status':old_status.get('status'),
  'source_v15_status':previous_status.get('status'),'examples_copied':counts,'next_cycle':next_cycle[0] if next_cycle else None,
  'database_available':new_db.exists(),'source_database_exists':old_db.exists(),
  'database_backup_api':'sqlite3.Connection.backup'}},indent=2))
config={{'base':'microsoft/mdeberta-v3-base','revision':'a0484667b22365f84929a935b5e50a51f71f159d',
 'languages':['vi','en','zh'],'qwen_judge_enabled':False,
 'task':'translation sentence boundary with 1/2 future words',
 'target_definition':'EOT at sentence-final punctuation; NEOT at internal punctuation; punctuation is label metadata only',
 'punctuation_in_model_input':False,
 'split':'page-grouped 80/10/10 train/dev/test; test not used for checkpoint selection',
 'lookahead_words':[1,2],'test_used_for_selection':False,
 'steps_per_language_per_cycle':100,'microbatch':32,'queries_per_language':4,
 'pages_per_query':25,'source_sha256':{source_hash!r}}}
config_path=run/'config.json'
if config_path.exists(): assert json.loads(config_path.read_text())==config,'configuration mismatch; use a new run directory'
else: config_path.write_text(json.dumps(config,indent=2))
cmd=[sys.executable,'-u','-m','turn_v2.translation_segmenter_v15b','--steps','100','--microbatch','32','--queries-per-language','4','--pages-per-query','25']
log=run/'worker_1.log'
with log.open('w') as stream:
 p=subprocess.Popen(cmd,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
   env={{**os.environ,'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1'}})
ticks=(Path('/proc')/str(p.pid)/'stat').read_text().split()[21]
receipt={{'pid':p.pid,'start_ticks':ticks,'command':cmd,'log':str(log.relative_to(root)),
 'source_sha256':{source_hash!r},'qwen_judge_enabled':False,'max_cycles':0,'global_timeout':None}}
(run/'launcher.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
            result = runtime.execute(code, timeout=90, echo=False)
            proof = ROOT / ("launch_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + ".json")
            proof.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            if result.get("status") != "ok":
                raise RuntimeError(f"Colab launch failed; inspect {proof}")
            value=json.loads("".join(x["text"] for x in result["outputs"] if x.get("output_type")=="stream"))
            (ROOT/"launcher.json").write_text(json.dumps(value,indent=2),encoding="utf-8")
            print(json.dumps(value,ensure_ascii=False))
            return

        if args.action == "stop":
            code = f'''import json
from pathlib import Path
root=Path({REMOTE_ROOT!r});run=root/{REMOTE_RUN!r}
launcher=json.loads((run/'launcher.json').read_text());pid=launcher['pid'];proc=Path('/proc')/str(pid)
assert (proc/'cmdline').exists(),'v15 parent process is already absent'
cmd=(proc/'cmdline').read_bytes();ticks=(proc/'stat').read_text().split()[21]
assert b'turn_v2.translation_segmenter_v15' in cmd and ticks==launcher['start_ticks'],'v15 parent identity changed'
child_path=run/'child.json';child=json.loads(child_path.read_text()) if child_path.exists() else {{}}
child_pid=child.get('pid');child_proc=Path('/proc')/str(child_pid);child_verified=False
if child_pid and (child_proc/'cmdline').exists():
 child_cmd=(child_proc/'cmdline').read_bytes();child_ticks=(child_proc/'stat').read_text().split()[21]
 child_verified=(b'turn_v2.translation_segmenter_v15' in child_cmd and child_ticks==child.get('start_ticks'))
 assert child_verified,'v15 child identity changed; refusing to stop a different process'
marker=run/'STOP_REQUESTED';marker.write_text('Graceful versioned-crawler repair; preserve SQLite and receipts, stop before next training phase.\\n')
print(json.dumps({{'marker':str(marker),'parent_pid':pid,'parent_verified':True,'child_pid':child_pid,'child_verified':child_verified}}))
'''
            result = runtime.execute(code, timeout=60, echo=False)
            proof = ROOT / ("stop_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + ".json")
            proof.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            if result.get("status") != "ok":
                raise RuntimeError(f"Colab stop request failed; inspect {proof}")
            value=json.loads("".join(x["text"] for x in result["outputs"] if x.get("output_type")=="stream"))
            print(json.dumps(value,ensure_ascii=False))
            return

        code = f'''import json,sqlite3,subprocess,shutil
from pathlib import Path
root=Path({REMOTE_ROOT!r});run=root/{REMOTE_RUN!r};values={{}}
for name in ['launcher.json','child.json','status.json','latest_error.json','data_summary.json','config.json','recovery/latest_snapshot.json']:
 p=run/name
 if p.exists():values[name]=json.loads(p.read_text())
for name in ['launcher.json','child.json']:
 r=values.get(name,{{}});pid=r.get('pid');proc=Path('/proc')/str(pid);ok=False
 if pid and (proc/'cmdline').exists():
  cmd=(proc/'cmdline').read_bytes();ticks=(proc/'stat').read_text().split()[21]
  ok=b'turn_v2.translation_segmenter_v15b' in cmd and ticks==r.get('start_ticks')
 values[name+'_process_verified']=ok
 if r.get('log'):
  p=root/r['log']
  if p.exists():values[name+'_log_tail']=p.read_text(errors='replace')[-2200:]
values['language_state']={{}}
for language in ['vi','en','zh']:
 folder=run/'teachers'/language;state={{}}
 for filename in ['deferred.json','progress.json','resume_receipt.json','best_report.json','last_training_report.json','identity.json']:
  p=folder/filename
  if p.exists():state[filename]=json.loads(p.read_text())
 values['language_state'][language]=state
db=run/'corpus.sqlite'
if db.exists():
 c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
 values['sample_counts']=[dict(zip(['language','split','label','lookahead_words','samples'],x)) for x in c.execute('SELECT language,split,label,lookahead_words,COUNT(*) FROM examples GROUP BY 1,2,3,4 ORDER BY 1,2,3,4')]
 values['source_page_counts']=[dict(zip(['language','split','pages'],x)) for x in c.execute('SELECT language,split,COUNT(*) FROM sources GROUP BY 1,2 ORDER BY 1,2')]
 c.close()
values['GPU']=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
values['remote_disk_free_bytes']=shutil.disk_usage('/content').free
print(json.dumps(values,ensure_ascii=False))
'''
        result = runtime.execute(code, timeout=120, echo=False)
        proof = ROOT / ((args.action + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")) + ".json")
        proof.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if result.get("status") != "ok":
            raise RuntimeError(f"Colab {args.action} failed; inspect {proof}")
        value=json.loads("".join(x["text"] for x in result["outputs"] if x.get("output_type")=="stream"))
        if args.action == "snapshot":
            meta=value.get("recovery/latest_snapshot.json")
            if not meta:
                raise RuntimeError("No v15 metadata snapshot is available yet")
            assert meta["scope"]=="METADATA_ONLY_NOT_A_FULL_TEACHER_RECOVERY" and meta["bytes"]<30*1024**2
            ensure_free_space(ROOT,meta["bytes"]*2)
            response=runtime.http.get(runtime.base_url+"files/"+quote("content/turn_detect_v2/"+meta["path"],safe="/"),stream=True,timeout=180,allow_redirects=False)
            if response.status_code!=200 or "html" in response.headers.get("content-type",""):
                raise RuntimeError(f"Snapshot download HTTP {response.status_code}")
            target=ROOT/"metadata_snapshot.zip";partial=ROOT/"metadata_snapshot.partial"
            with partial.open("wb") as stream:
                for block in response.iter_content(1024*1024):stream.write(block)
            assert partial.stat().st_size==meta["bytes"] and sha(partial.read_bytes())==meta["sha256"]
            partial.replace(target)
            (ROOT/"snapshot_receipt.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
            trace_root=ROOT/"dev_traces";trace_root.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(target) as archive:
                for member in archive.namelist():
                    path=PurePosixPath(member)
                    if path.name=="state_snapshot.sqlite":
                        info=archive.getinfo(member)
                        ensure_free_space(ROOT,info.file_size*2)
                        (ROOT/path.name).write_bytes(archive.read(member))
                    if path.name=="dev_trace.jsonl" and path.parent.parent.name=="teachers":
                        language=path.parent.name
                        (trace_root/f"{language}.jsonl").write_bytes(archive.read(member))
            report=subprocess.run([sys.executable,"tools/report_translation_segmenter_v15b.py"],
                                  check=True,capture_output=True,text=True)
            (ROOT/"review_build_receipt.json").write_text(report.stdout,encoding="utf-8")
        else:
            value["queried_at_utc"]=datetime.now(timezone.utc).isoformat()
            (ROOT/"status_receipt.json").write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(value,ensure_ascii=False))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
