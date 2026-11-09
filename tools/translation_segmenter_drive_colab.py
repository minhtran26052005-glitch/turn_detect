"""Control the Colab-to-mounted-Drive backup; only small receipts reach Windows."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from colab_runtime import ColabRuntime

LOCAL = Path("outputs/translation_segmenter_v15d_20261026")
SOURCE = Path("turn_v2/translation_segmenter_drive_backup_v1.py")


def execute(runtime, code):
    out = runtime.execute(code, timeout=60, echo=False)
    if out.get("status") != "ok":
        raise RuntimeError(json.dumps([x for x in out.get("outputs", []) if x.get("output_type") == "error"], ensure_ascii=False))
    raw = "".join(x.get("text", "") for x in out.get("outputs", []) if x.get("output_type") == "stream")
    return json.loads(raw)


def launch(runtime):
    source = SOURCE.read_bytes()
    checksum = hashlib.sha256(source).hexdigest()
    config = json.loads((LOCAL / "drive_backup_config.json").read_text(encoding="utf-8"))
    runtime.request("PUT", "api/contents/content/turn_detect_v2/colab/" + SOURCE.name,
                    json={"type": "file", "format": "base64", "content": base64.b64encode(source).decode("ascii")})
    return execute(runtime, f'''import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path('/content/turn_detect_v2');run=root/'outputs/translation_segmenter_v15d_20261026';job=run/'drive_backup_v1'
def live(record):
 try:
  p=Path('/proc')/str(record['pid']);argv=record['command'];module=argv[argv.index('-m')+1]
  return module.encode() in (p/'cmdline').read_bytes().split(bytes([0])) and (p/'stat').read_text().split()[21]==record['start_ticks']
 except (OSError,KeyError,ValueError,IndexError):return False
assert live(json.loads((run/'launcher.json').read_text())),'Campaign parent identity unavailable'
config={config!r};destination=Path(config['drive_path'])
assert os.path.ismount('/content/drive'),'Drive is not mounted'
assert json.loads((destination/'destination_identity.json').read_text())==config['identity'],'Drive identity mismatch'
stage=root/'colab'/{SOURCE.name!r};data=stage.read_bytes();assert hashlib.sha256(data).hexdigest()=={checksum!r}
target=root/'turn_v2'/{SOURCE.name!r}
if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()=={checksum!r},'Preserve source; use a new backup module version'
else:target.write_bytes(data)
job.mkdir(exist_ok=True);cp=job/'config.json'
if cp.exists():assert json.loads(cp.read_text())==config,'Existing backup config differs; preserve it'
else:cp.write_text(json.dumps(config,indent=2))
lp=job/'launcher.json'
if lp.exists() and live(json.loads(lp.read_text())):
 print(json.dumps({{'reused_live_backup':True,'launcher':json.loads(lp.read_text())}}))
else:
 command=[sys.executable,'-u','-m','turn_v2.translation_segmenter_drive_backup_v1'];log=job/'worker.log'
 with log.open('a') as stream:p=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'1'}})
 record={{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':command,'source_sha256':{checksum!r},'drive_path':str(destination),'log':str(log.relative_to(root))}}
 lp.write_text(json.dumps(record,indent=2));print(json.dumps({{'launched':True,'launcher':record}}))''')


def status(runtime):
    return execute(runtime, '''import json,os,shutil
from pathlib import Path
root=Path('/content/turn_detect_v2');job=root/'outputs/translation_segmenter_v15d_20261026/drive_backup_v1';out={}
for name in ('launcher.json','status.json','latest_error.json','latest_verified_manifest.json','progress_manifest.json','verification_status_v2.json','verification_error_v2.json','recovery_verification.json','recovery_verify_v2_launcher.json'):
 p=job/name
 if p.exists():out[name]=json.loads(p.read_text())
record=out.get('launcher.json',{});ok=False
try:
 argv=record['command'];module=argv[argv.index('-m')+1];p=Path('/proc')/str(record['pid'])
 ok=module.encode() in (p/'cmdline').read_bytes().split(bytes([0])) and (p/'stat').read_text().split()[21]==record['start_ticks']
except (OSError,KeyError,ValueError,IndexError):pass
out['process_verified']=ok;out['drive_mounted']=os.path.ismount('/content/drive');out['colab_free_bytes']=shutil.disk_usage('/content').free
print(json.dumps(out,ensure_ascii=False))''')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("launch", "status"))
    parser.add_argument("--session-path")
    args = parser.parse_args()
    runtime = ColabRuntime(0, session_path=args.session_path)
    try:
        connection = runtime.connect()
        if connection.get("execution_state") != "idle":
            raise RuntimeError("Kernel busy; no Drive control cell was queued")
        result = {"launch": launch, "status": status}[args.action](runtime)
        path = LOCAL / ("drive_" + args.action + "_receipt.json")
        path.write_text(json.dumps({"connection": connection, "result": result}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.action == "status":
            task_path = LOCAL / "task_state.json"
            task = json.loads(task_path.read_text(encoding="utf-8")) if task_path.exists() else {}
            verified = result.get("latest_verified_manifest.json", {})
            task["drive_backup"] = {"state": result.get("status.json", {}),
                "process_verified_at_check": result.get("process_verified", False),
                "launcher": result.get("launcher.json", {}),
                "full_recovery_available": verified.get("full_recovery_available", False),
                "latest_manifest": verified,
                "recovery_verification_state": result.get("verification_status_v2.json", {}),
                "recovery_verification": result.get("recovery_verification.json", {})}
            task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"connection": connection, "result": result}, ensure_ascii=False))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
