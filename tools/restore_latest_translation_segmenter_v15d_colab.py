"""Restore the newest verified v15d data snapshot to a fresh connected Colab VM.

Only data, metadata and source code are transferred. Teacher checkpoints and
optimizer state are not included in the snapshot; training resumes from the
pinned pretrained base on the recovered corpus.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import zipfile

from teacher_first_colab import campaign_runtime
from storage_guard import ensure_free_space


LOCAL = Path("outputs/translation_segmenter_v15d_20261026")
SESSION = "a-jvsc-595d62b4-0421-4af5-885c-0d72e4aa1d1002f68001-a8cf-4ec7-9f7b-a3b56cdd3208.ipynb"
SNAPSHOT_CYCLE = 42
SNAPSHOT_SHA256 = "0f0a3181b8b0cd67ee38f39e3a3ea23817a48e2e9df5a6758d1d6e41e7bb6111"
SNAPSHOT_BYTES = 255481878
SOURCE_SHA256 = "cc225968bba652ecdff99e52ea25d3ec7648ed8de954de17bd66b9976767492c"
SOURCES = (
    "turn_v2/__init__.py",
    "turn_v2/translation_segmenter_v15c.py",
    "turn_v2/common.py",
    "turn_v2/monolingual_data.py",
    "turn_v2/metrics.py",
    "turn_v2/translation_segmenter_v15d2.py",
    "turn_v2/translation_segmenter_v15d3.py",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_local():
    archive = LOCAL / "data_snapshot.zip"
    receipt = json.loads((LOCAL / "snapshot_receipt.json").read_text(encoding="utf-8"))
    if archive.stat().st_size != SNAPSHOT_BYTES or digest(archive) != SNAPSHOT_SHA256:
        raise RuntimeError("Latest cycle-42 snapshot SHA/size mismatch; refusing restore")
    if (receipt.get("cycle") != SNAPSHOT_CYCLE or receipt.get("sha256") != SNAPSHOT_SHA256
            or receipt.get("scope") != "DATA_AND_METADATA_ONLY_NO_MODEL_WEIGHTS"
            or receipt.get("database_integrity") != "ok"):
        raise RuntimeError("Local receipt does not authenticate the cycle-42 data snapshot")
    with zipfile.ZipFile(archive) as zf:
        expected_members = {"state_snapshot.sqlite", "config.json", "status.json", "data_summary.json"}
        if set(zf.namelist()) != expected_members:
            raise RuntimeError("Unexpected members in verified snapshot")
        config = json.loads(zf.read("config.json"))
        summary = json.loads(zf.read("data_summary.json"))
        if config.get("source_sha256") != SOURCE_SHA256 or summary.get("cycle") != SNAPSHOT_CYCLE:
            raise RuntimeError("Snapshot cycle/source recipe mismatch")
    for relative in SOURCES:
        if not Path(relative).is_file():
            raise FileNotFoundError(relative)
    if digest(Path(SOURCES[-1])) != SOURCE_SHA256:
        raise RuntimeError("Local v15d3 source does not match the snapshot recipe")
    db = sqlite3.connect(f"file:{(LOCAL/'state_snapshot.sqlite').resolve()}?mode=ro", uri=True)
    try:
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        next_cycle = db.execute("SELECT value FROM meta WHERE key='next_cycle'").fetchone()[0]
        pages = db.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        examples = db.execute("SELECT COUNT(*) FROM examples").fetchone()[0]
    finally:
        db.close()
    if integrity != "ok" or int(next_cycle) != SNAPSHOT_CYCLE + 1:
        raise RuntimeError("Snapshot database integrity or next-cycle marker mismatch")
    if pages != 16045 or examples != 1338209:
        raise RuntimeError("Snapshot database counts differ from the cycle-42 receipt")
    return {"cycle": SNAPSHOT_CYCLE, "next_cycle": int(next_cycle), "sha256": SNAPSHOT_SHA256,
            "bytes": SNAPSHOT_BYTES, "database_integrity": integrity, "pages": pages,
            "examples": examples, "source_sha256": SOURCE_SHA256,
            "weights_or_optimizer_included": False}


def remote_cell(runtime, code: str, timeout=180):
    result = runtime.execute(code, timeout=timeout, echo=False)
    if result.get("status") != "ok":
        errors = [x for x in result.get("outputs", []) if x.get("output_type") == "error"]
        raise RuntimeError("Colab restore cell failed: " + json.dumps(errors, ensure_ascii=False)[:3000])
    text = "".join(x.get("text", "") for x in result.get("outputs", [])
                   if x.get("output_type") == "stream")
    lines = [line for line in text.splitlines() if line.strip().startswith("{")]
    if not lines:
        raise RuntimeError("Restore cell produced no JSON receipt")
    return json.loads(lines[-1])


def put_file(runtime, remote: str, data: bytes) -> str:
    expected = hashlib.sha256(data).hexdigest()
    body = {"type": "file", "format": "base64", "content": base64.b64encode(data).decode("ascii")}
    last_error = None
    for attempt in range(3):
        try:
            response = runtime.request("PUT", "api/contents/content/turn_detect_v2/" + remote, json=body)
            if response.status_code not in (200, 201):
                raise RuntimeError(f"Unexpected upload status {response.status_code} for {remote}")
            return expected
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise last_error


def main():
    ensure_free_space(LOCAL, 10 * 1024**3)
    local_receipt = validate_local()
    runtime = campaign_runtime(SESSION)
    try:
        connection = runtime.connect()
        if connection.get("execution_state") != "idle":
            raise RuntimeError(f"Refusing to restore into a busy kernel: {connection}")

        preflight = remote_cell(runtime, r'''import json, shutil
from pathlib import Path
root=Path('/content/turn_detect_v2'); run=root/'outputs/translation_segmenter_v15d_20261026'
blocked=[b'turn_v2.teacher_first_v14',b'turn_v2.continuous_v13',b'turn_v2.translation_segmenter_v15',b'turn_v2.translation_segmenter_v15b',b'turn_v2.translation_segmenter_v15c',b'turn_v2.translation_segmenter_v15d',b'turn_v2.translation_segmenter_v15d2',b'turn_v2.translation_segmenter_v15d3']
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit(): continue
 try: cmd=(p/'cmdline').read_bytes()
 except OSError: continue
 if any(x in cmd for x in blocked): active.append(p.name)
assert not run.exists(), 'Remote campaign state exists; refusing to overwrite'
assert not active, 'Campaign process already active: '+','.join(active)
free=shutil.disk_usage('/content').free
assert free>=25*1024**3, 'Less than 25 GiB free on Colab'
root.mkdir(parents=True,exist_ok=True);(root/'colab').mkdir(exist_ok=True);(root/'turn_v2').mkdir(exist_ok=True)
print(json.dumps({'run_directory_absent':True,'active_campaign_processes':active,'remote_free_bytes':free,'staging_ready':True}))''')

        uploaded = {}
        archive = (LOCAL / "data_snapshot.zip").read_bytes()
        archive_sha = hashlib.sha256(archive).hexdigest()
        chunk_size = 8 * 1024 * 1024
        chunk_records = []
        total_chunks = (len(archive) + chunk_size - 1) // chunk_size
        for index in range(total_chunks):
            chunk = archive[index * chunk_size:(index + 1) * chunk_size]
            remote = f"colab/v15d_cycle42_chunk_{index:04d}.bin"
            uploaded[remote] = put_file(runtime, remote, chunk)
            chunk_records.append({"path": remote, "sha256": uploaded[remote], "bytes": len(chunk)})
            print(f"Uploaded snapshot chunk {index + 1}/{total_chunks}", flush=True)
        for relative in SOURCES:
            remote = "colab/" + Path(relative).name
            data = Path(relative).read_bytes()
            uploaded[remote] = put_file(runtime, remote, data)

        assemble = f'''import hashlib,json
from pathlib import Path
root=Path('/content/turn_detect_v2');stage=root/'colab';chunks={chunk_records!r}
archive=stage/'v15d_cycle42_snapshot.zip';h=hashlib.sha256();total=0
with archive.open('wb') as output:
 for item in chunks:
  data=(root/item['path']).read_bytes();assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
  output.write(data);h.update(data);total+=len(data)
assert total=={SNAPSHOT_BYTES} and h.hexdigest()=={SNAPSHOT_SHA256!r}
print(json.dumps({{'assembled_bytes':total,'sha256':h.hexdigest(),'chunks':len(chunks)}}))'''
        assembled = remote_cell(runtime, assemble)

        restore = f'''import hashlib,json,shutil,sqlite3,zipfile
from pathlib import Path
root=Path('/content/turn_detect_v2');run=root/'outputs/translation_segmenter_v15d_20261026';stage=root/'colab'
expected={uploaded!r}
for rel,digest in expected.items(): assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest, 'upload hash mismatch '+rel
module_files={list(SOURCES)!r}
for rel in module_files:
 target=root/rel;target.parent.mkdir(parents=True,exist_ok=True);source=stage/Path(rel).name;data=source.read_bytes()
 if target.exists(): assert hashlib.sha256(target.read_bytes()).hexdigest()==hashlib.sha256(data).hexdigest(), 'source conflict '+rel
 else: target.write_bytes(data)
assert not run.exists(), 'remote run appeared after preflight; refusing overwrite'
recovery=run/'recovery';recovery.mkdir(parents=True)
with zipfile.ZipFile(stage/'v15d_cycle42_snapshot.zip') as z:
 assert set(z.namelist())=={{'state_snapshot.sqlite','config.json','status.json','data_summary.json'}}
 z.extract('state_snapshot.sqlite',recovery)
 for name in ('config.json','data_summary.json'):(run/name).write_bytes(z.read(name))
 (recovery/'restored_cycle42_status.json').write_bytes(z.read('status.json'))
shutil.copy2(recovery/'state_snapshot.sqlite',run/'corpus.sqlite')
db=sqlite3.connect(run/'corpus.sqlite');integrity=db.execute('PRAGMA integrity_check').fetchone()[0];next_cycle=int(db.execute("SELECT value FROM meta WHERE key='next_cycle'").fetchone()[0]);pages=db.execute('SELECT COUNT(*) FROM sources').fetchone()[0];examples=db.execute('SELECT COUNT(*) FROM examples').fetchone()[0];db.close()
assert integrity=='ok' and next_cycle==43 and pages==16045 and examples==1338209
config=json.loads((run/'config.json').read_text());assert config['source_sha256']=={SOURCE_SHA256!r}
receipt={{'restored_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'snapshot_cycle':42,'next_cycle':43,'snapshot_sha256':{SNAPSHOT_SHA256!r},'snapshot_bytes':{SNAPSHOT_BYTES},'database_integrity':integrity,'pages':pages,'examples':examples,'teacher_weights_or_optimizer_restored':False}}
(recovery/'restore_receipt.json').write_text(json.dumps(receipt,indent=2))
(run/'status.json').write_text(json.dumps({{'campaign':'translation_segmenter_v15d','status':'RECOVERED_DATA_READY','stage':'RESTORED_CYCLE42_WAITING_FOR_LAUNCH','cycle':42,'next_cycle':43,'qwen_judge_enabled':False,'quality_target_certified':False}},indent=2))
print(json.dumps({{'restore_receipt':receipt,'remote_free_bytes':shutil.disk_usage('/content').free}}))'''
        remote_receipt = remote_cell(runtime, restore)

        imported = remote_cell(runtime, f'''import hashlib,json,os,sys
from pathlib import Path
root=Path('/content/turn_detect_v2');os.chdir(root);sys.path.insert(0,str(root))
import turn_v2.translation_segmenter_v15d3 as module
print(json.dumps({{'module':module.__name__,'source_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'expected':{SOURCE_SHA256!r}}}))''')
        if imported.get("source_sha256") != SOURCE_SHA256:
            raise RuntimeError("Restored Colab worker module failed SHA verification")

        launch = remote_cell(runtime, f'''import json,os,subprocess,sys
from pathlib import Path
root=Path('/content/turn_detect_v2');run=root/'outputs/translation_segmenter_v15d_20261026';rec=json.loads((run/'recovery/restore_receipt.json').read_text())
assert rec['snapshot_sha256']=={SNAPSHOT_SHA256!r} and not (run/'launcher.json').exists()
command=[sys.executable,'-u','-m','turn_v2.translation_segmenter_v15d3','--phase','run','--docs-per-source','200','--steps','100','--microbatch','32']
log=run/('worker_recovery_cycle42_'+__import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.log')
with log.open('w') as stream: p=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1'}})
launcher={{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':command,'module':'turn_v2.translation_segmenter_v15d3','log':str(log.relative_to(root)),'source_sha256':{SOURCE_SHA256!r},'restored_from_cycle':42,'max_cycles':0}}
(run/'launcher.json').write_text(json.dumps(launcher,indent=2));print(json.dumps(launcher))''')

        receipt = {"at_utc": datetime.now(timezone.utc).isoformat(), "connection": connection,
                   "local_snapshot_validation": local_receipt, "remote_preflight": preflight,
                   "remote_assembled_snapshot": assembled,
                   "remote_restore": remote_receipt, "import_verification": imported,
                   "launch": launch, "uploaded_sha256": uploaded,
                   "scope": "data and source only; prior weights and optimizer unavailable"}
        (LOCAL / "restore_latest_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, ensure_ascii=False))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
