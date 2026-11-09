"""Restore the verified v15d data snapshot to a fresh connected Colab runtime."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import zipfile

from colab_runtime import ColabRuntime
from storage_guard import ensure_free_space


LOCAL = Path("outputs/translation_segmenter_v15d_20261026")
REMOTE_ROOT = "/content/turn_detect_v2"
REMOTE_RUN = "outputs/translation_segmenter_v15d_20261026"
SESSION_PATH = (
    "a-jvsc-c56e5860-53dd-42ab-9046-4b7355b137a0c830cfa6-"
    "577a-4f10-ad66-a2ccf59eca47.ipynb"
)
EXPECTED_ARCHIVE_SHA256 = "517f5e47457a916de4640ed3527e9baa94f5d084055791d8a34cda32548e00b3"
EXPECTED_SOURCE_SHA256 = "cc225968bba652ecdff99e52ea25d3ec7648ed8de954de17bd66b9976767492c"
SOURCE_FILES = (
    "turn_v2/__init__.py",
    "turn_v2/translation_segmenter_v15c.py",
    "turn_v2/common.py",
    "turn_v2/monolingual_data.py",
    "turn_v2/metrics.py",
    "turn_v2/translation_segmenter_v15d2.py",
    "turn_v2/translation_segmenter_v15d3.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_cell(runtime, code: str, timeout: int = 120):
    result = runtime.execute(code, timeout=timeout, echo=False)
    if result.get("status") != "ok":
        details = "\n".join(
            item.get("evalue", item.get("text", ""))
            for item in result.get("outputs", [])
            if item.get("output_type") in ("error", "stream")
        )
        raise RuntimeError(f"Colab cell failed: {details[:3000]}")
    text = "".join(
        item.get("text", "")
        for item in result.get("outputs", [])
        if item.get("output_type") == "stream"
    )
    return json.loads(text) if text.strip() else {}


def validate_local_snapshot():
    archive_path = LOCAL / "data_snapshot.zip"
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if archive_path.stat().st_size != 47_441_831:
        raise RuntimeError("Cycle-22 snapshot size changed; refusing restore")
    archive_hash = sha256_file(archive_path)
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("Cycle-22 snapshot SHA-256 mismatch; refusing restore")

    receipt_path = LOCAL / "snapshot_20261026T050430.json"
    wrapper = json.loads(receipt_path.read_text(encoding="utf-8"))
    stdout = "".join(item.get("text", "") for item in wrapper.get("outputs", []))
    receipt = json.loads(stdout.strip())
    if (receipt.get("cycle") != 22 or receipt.get("bytes") != archive_path.stat().st_size
            or receipt.get("sha256") != archive_hash
            or receipt.get("scope") != "DATA_AND_METADATA_ONLY_NO_MODEL_WEIGHTS"):
        raise RuntimeError("Snapshot receipt does not match the verified cycle-22 archive")

    for relative in SOURCE_FILES:
        if not Path(relative).is_file():
            raise FileNotFoundError(relative)
    if sha256_file(Path(SOURCE_FILES[-1])) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Crawler source does not match the archived cycle-22 recipe")

    with tempfile.TemporaryDirectory(prefix="v15d_restore_check_") as temp_dir:
        db_path = Path(temp_dir) / "state_snapshot.sqlite"
        with zipfile.ZipFile(archive_path) as archive:
            if set(archive.namelist()) != {
                "state_snapshot.sqlite", "config.json", "status.json", "data_summary.json"
            }:
                raise RuntimeError("Unexpected files in cycle-22 snapshot")
            db_path.write_bytes(archive.read("state_snapshot.sqlite"))
            config = json.loads(archive.read("config.json"))
            if config.get("source_sha256") != EXPECTED_SOURCE_SHA256:
                raise RuntimeError("Snapshot recipe and crawler source SHA do not match")
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            pages = connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            examples = connection.execute("SELECT COUNT(*) FROM examples").fetchone()[0]
            next_cycle = connection.execute(
                "SELECT value FROM meta WHERE key='next_cycle'"
            ).fetchone()[0]
            pending = connection.execute(
                "SELECT value FROM meta WHERE key='train_after_data_repair_pending'"
            ).fetchone()[0]
            markers = [row[0] for row in connection.execute(
                "SELECT key FROM meta WHERE key LIKE 'migration:%' ORDER BY key"
            )]
            english_baai = connection.execute(
                "SELECT COUNT(*) FROM source_provenance "
                "WHERE language='en' AND source_key='hf:industrycorpus2_news_media:en'"
            ).fetchone()[0]
            per_language = [dict(zip(("language", "pages"), row)) for row in connection.execute(
                "SELECT language,COUNT(*) FROM sources GROUP BY language ORDER BY language"
            )]
            per_language_examples = [dict(zip(("language", "examples"), row))
                for row in connection.execute(
                    "SELECT language,COUNT(*) FROM examples GROUP BY language ORDER BY language"
                )]
        finally:
            connection.close()
        if integrity != "ok" or pages != 2189 or examples != 266552:
            raise RuntimeError("Snapshot SQLite integrity or counts differ from the verified receipt")
        if next_cycle != "23" or pending != "true" or english_baai != 0:
            raise RuntimeError("Cycle, forced-train flag, or English source quarantine check failed")
        if len(markers) < 2:
            raise RuntimeError("Expected language-audit and Chinese re-extraction markers are missing")

    return {
        "cycle": 22,
        "next_cycle": 23,
        "snapshot_bytes": archive_path.stat().st_size,
        "snapshot_sha256": archive_hash,
        "sqlite_integrity": integrity,
        "pages": pages,
        "examples": examples,
        "pages_by_language": per_language,
        "examples_by_language": per_language_examples,
        "migration_markers": markers,
        "train_after_repair_pending": True,
        "english_baai_provenance_rows": english_baai,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "teacher_weights_or_optimizer_in_snapshot": False,
    }


def main():
    ensure_free_space(LOCAL, 10 * 1024**3)
    local = validate_local_snapshot()
    runtime = ColabRuntime(0, session_path=SESSION_PATH)
    try:
        connection = runtime.connect()
        if connection.get("execution_state") != "idle":
            raise RuntimeError("Colab kernel is busy; refusing to queue restore into another cell")

        preflight_code = f'''import json,shutil
from pathlib import Path
root=Path({REMOTE_ROOT!r});run=root/{REMOTE_RUN!r}
needles=[b'turn_v2.translation_segmenter_v15',b'turn_v2.teacher_first_v14',b'turn_v2.continuous_v13']
active=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:cmd=(proc/'cmdline').read_bytes()
 except OSError:continue
 if any(item in cmd for item in needles):active.append(proc.name)
free=shutil.disk_usage('/content').free
assert not root.exists(),'Remote project root exists; preserve it and inspect before restore'
assert not active,'A campaign process is already active: '+','.join(active)
assert free>=25*1024**3,'Less than 25 GiB free on Colab'
root.mkdir(parents=True)
(root/'colab').mkdir()
(root/'turn_v2').mkdir()
print(json.dumps({{'root_created':True,'active_campaign_processes':active,'disk_free_bytes':free}}))'''
        preflight = run_cell(runtime, preflight_code)

        archive_path = LOCAL / "data_snapshot.zip"
        files = [("colab/data_snapshot_cycle22.zip", archive_path.read_bytes())]
        files.extend((f"colab/{Path(relative).name}", Path(relative).read_bytes())
                     for relative in SOURCE_FILES)
        uploaded = {}
        for remote_relative, data in files:
            response = runtime.request(
                "PUT",
                "api/contents/content/turn_detect_v2/" + remote_relative,
                json={"type": "file", "format": "base64",
                      "content": base64.b64encode(data).decode("ascii")},
            )
            if response.status_code not in (200, 201):
                raise RuntimeError(f"Unexpected upload status for {remote_relative}")
            uploaded[remote_relative] = hashlib.sha256(data).hexdigest()

        restore_code = f'''import hashlib,json,shutil,sqlite3,zipfile
from pathlib import Path
root=Path({REMOTE_ROOT!r});run=root/{REMOTE_RUN!r};staging=root/'colab'
expected={uploaded!r}
for relative,digest in expected.items():
 data=(root/relative).read_bytes()
 assert hashlib.sha256(data).hexdigest()==digest,'Upload hash mismatch: '+relative
module_files={list(SOURCE_FILES)!r}
for relative in module_files:
 source=staging/Path(relative).name;target=root/relative
 data=source.read_bytes()
 if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==hashlib.sha256(data).hexdigest(),'Preserve existing source: '+relative
 else:target.write_bytes(data)
archive=staging/'data_snapshot_cycle22.zip'
assert not run.exists(),'Campaign run directory already exists; refusing overwrite'
run.mkdir(parents=True);recovery=run/'recovery';recovery.mkdir()
with zipfile.ZipFile(archive) as z:
 assert set(z.namelist())=={{'state_snapshot.sqlite','config.json','status.json','data_summary.json'}}
 z.extract('state_snapshot.sqlite',recovery)
 for name in ('config.json','data_summary.json'):(run/name).write_bytes(z.read(name))
 (recovery/'restored_cycle22_status.json').write_bytes(z.read('status.json'))
shutil.copy2(recovery/'state_snapshot.sqlite',run/'corpus.sqlite')
config=json.loads((run/'config.json').read_text());assert config['source_sha256']=={EXPECTED_SOURCE_SHA256!r}
con=sqlite3.connect(run/'corpus.sqlite')
integrity=con.execute('PRAGMA integrity_check').fetchone()[0]
pages=con.execute('SELECT COUNT(*) FROM sources').fetchone()[0]
examples=con.execute('SELECT COUNT(*) FROM examples').fetchone()[0]
next_cycle=con.execute("SELECT value FROM meta WHERE key='next_cycle'").fetchone()[0]
pending=con.execute("SELECT value FROM meta WHERE key='train_after_data_repair_pending'").fetchone()[0]
markers=[r[0] for r in con.execute("SELECT key FROM meta WHERE key LIKE 'migration:%' ORDER BY key")]
bad_en=con.execute("SELECT COUNT(*) FROM source_provenance WHERE language='en' AND source_key='hf:industrycorpus2_news_media:en'").fetchone()[0]
con.close()
assert integrity=='ok' and pages==2189 and examples==266552
assert next_cycle=='23' and pending=='true' and bad_en==0 and len(markers)>=2
receipt={{'restored_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'snapshot_cycle':22,'next_cycle':23,'snapshot_sha256':{EXPECTED_ARCHIVE_SHA256!r},'snapshot_bytes':47441831,'database_integrity':integrity,'pages':pages,'examples':examples,'migration_markers':markers,'train_after_data_repair_pending':True,'english_baai_provenance_rows':bad_en,'teacher_weights_or_optimizer_restored':False,'local_source_sha256':{EXPECTED_SOURCE_SHA256!r},'note':'Cycle-22 data and stream state restored; no teacher checkpoint was in the snapshot.'}}
(recovery/'restore_receipt.json').write_text(json.dumps(receipt,indent=2))
(run/'status.json').write_text(json.dumps({{'campaign':'translation_segmenter_v15d','status':'RECOVERED_DATA_READY','stage':'RESTORED_CYCLE22_WAITING_FOR_LAUNCH','cycle':22,'next_cycle':23,'qwen_judge_enabled':False,'quality_target_certified':False,'restore_receipt':receipt}},indent=2))
print(json.dumps({{'restore_receipt':receipt,'remote_disk_free_bytes':shutil.disk_usage('/content').free}}))'''
        restored = run_cell(runtime, restore_code, timeout=180)

        import_code = f'''import json,os,sys
from pathlib import Path
root=Path({REMOTE_ROOT!r});os.chdir(root)
import turn_v2.translation_segmenter_v15d3 as module
print(json.dumps({{'module':module.__name__,'root':str(module.ROOT),'db':str(module.DB_PATH),'source_sha256':module.file_sha(Path(module.__file__))}}))'''
        imported = run_cell(runtime, import_code)
        if imported.get("source_sha256") != EXPECTED_SOURCE_SHA256:
            raise RuntimeError("Remote crawler module import/source hash mismatch")

        launch_code = f'''import json,os,subprocess,sys
from pathlib import Path
root=Path({REMOTE_ROOT!r});run=root/{REMOTE_RUN!r}
receipt=json.loads((run/'recovery/restore_receipt.json').read_text())
assert receipt['snapshot_sha256']=={EXPECTED_ARCHIVE_SHA256!r}
assert not (run/'launcher.json').exists()
command=[sys.executable,'-u','-m','turn_v2.translation_segmenter_v15d3','--phase','run','--docs-per-source','200','--steps','100','--microbatch','32']
log=run/('worker_recovery_cycle22_'+__import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.log')
with log.open('w') as stream:process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1'}})
launcher={{'pid':process.pid,'start_ticks':(Path('/proc')/str(process.pid)/'stat').read_text().split()[21],'command':command,'module':'turn_v2.translation_segmenter_v15d3','log':str(log.relative_to(root)),'source_sha256':{EXPECTED_SOURCE_SHA256!r},'restored_from_cycle':22,'max_cycles':0}}
(run/'launcher.json').write_text(json.dumps(launcher,indent=2))
print(json.dumps({{'launcher':launcher}}))'''
        launched = run_cell(runtime, launch_code)

        receipt = {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "connection": connection,
            "local_snapshot_validation": local,
            "remote_preflight": preflight,
            "remote_restore": restored,
            "remote_module_import": imported,
            "launch": launched,
            "uploaded_files": uploaded,
            "snapshot_scope": "data and metadata only; no model weights or optimizer",
        }
        (LOCAL / "restore_launch_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(receipt, ensure_ascii=False))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
