"""Fetch a fresh, data-only SQLite snapshot from the stopped v15c Colab run."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
from urllib.parse import quote
import zipfile

from storage_guard import ensure_free_space

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teacher_first_colab import campaign_runtime


REMOTE = "/content/turn_detect_v2"
RUN = "outputs/translation_segmenter_v15c_20261026"
LOCAL = Path("outputs/translation_segmenter_v15c_20261026")


def parse_stream(result: dict) -> dict:
    if result.get("status") != "ok":
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    chunks = [item["text"] for item in result.get("outputs", [])
              if item.get("output_type") == "stream"]
    return json.loads("".join(chunks))


def main() -> None:
    LOCAL.mkdir(parents=True, exist_ok=True)
    ensure_free_space(LOCAL, 10 * 1024**3)
    runtime = campaign_runtime()
    try:
        runtime.connect()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        code = f'''import hashlib,json,sqlite3,zipfile
from pathlib import Path
from datetime import datetime,timezone
root=Path({REMOTE!r});run=root/{RUN!r}
status=json.loads((run/'status.json').read_text())
assert status.get('status')=='STOPPED_BY_REQUEST','Refuse snapshot while v15c is not confirmed stopped'
def alive(name):
 p=run/name
 if not p.exists():return False
 rec=json.loads(p.read_text());pid=rec.get('pid');proc=Path('/proc')/str(pid)
 if not pid or not (proc/'cmdline').exists():return False
 try:
  cmd=(proc/'cmdline').read_bytes();ticks=(proc/'stat').read_text().split()[21]
 except (OSError,IndexError):return False
 return b'turn_v2.translation_segmenter_v15c' in cmd and ticks==rec.get('start_ticks')
assert not alive('launcher.json') and not alive('child.json'),'Refuse snapshot while parent or child is live'
summary=json.loads((run/'data_summary.json').read_text());cycle=int(summary['cycle'])
db=run/'corpus.sqlite';assert db.exists(),'Remote corpus database is missing'
recovery=run/'recovery';recovery.mkdir(parents=True,exist_ok=True)
snapshot=recovery/f'v15c_current_cycle_{{cycle:06d}}_{stamp}.sqlite'
src=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True,timeout=60)
dst=sqlite3.connect(snapshot);src.backup(dst);check=dst.execute('PRAGMA integrity_check').fetchone()[0]
counts={{'pages':dst.execute('SELECT COUNT(*) FROM sources').fetchone()[0],
 'examples':dst.execute('SELECT COUNT(*) FROM examples').fetchone()[0]}}
dst.close();src.close();assert check=='ok',check
archive=recovery/f'v15c_current_cycle_{{cycle:06d}}_{stamp}.zip';tmp=archive.with_suffix('.tmp')
with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
 z.write(snapshot,'state_snapshot.sqlite')
 for name in ['status.json','data_summary.json','config.json','carryover_receipt.json']:
  p=run/name
  if p.exists():z.write(p,name)
tmp.replace(archive)
meta={{'cycle':cycle,'path':archive.relative_to(root).as_posix(),'bytes':archive.stat().st_size,
 'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'database_integrity':check,
 'counts':counts,'scope':'DATA_AND_METADATA_ONLY_NO_MODEL_WEIGHTS'}}
(recovery/'latest_snapshot.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta))
'''
        result = runtime.execute(code, timeout=120, echo=False)
        meta = parse_stream(result)
        if meta["scope"] != "DATA_AND_METADATA_ONLY_NO_MODEL_WEIGHTS":
            raise RuntimeError("Unexpected remote snapshot scope")
        if meta["bytes"] >= 30 * 1024**2:
            raise RuntimeError(f"Refusing unexpectedly large snapshot: {meta['bytes']} bytes")
        ensure_free_space(LOCAL, meta["bytes"] * 3 + 10 * 1024**3)
        url = runtime.base_url + "files/" + quote("content/turn_detect_v2/" + meta["path"], safe="/")
        response = runtime.http.get(url, stream=True, timeout=180, allow_redirects=False)
        if response.status_code != 200 or "html" in response.headers.get("content-type", ""):
            raise RuntimeError(f"Snapshot download failed: HTTP {response.status_code}")
        partial = LOCAL / "metadata_snapshot.partial"
        digest = hashlib.sha256(); size = 0
        with partial.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                if block:
                    stream.write(block); digest.update(block); size += len(block)
        if size != meta["bytes"] or digest.hexdigest() != meta["sha256"]:
            partial.unlink(missing_ok=True)
            raise RuntimeError("Downloaded snapshot size/hash does not match Colab receipt")
        archive_path = LOCAL / "metadata_snapshot.zip"
        previous = LOCAL / f"metadata_snapshot_cycle{meta['cycle'] - 1}.zip"
        if archive_path.exists() and not previous.exists():
            shutil.copy2(archive_path, previous)
        partial.replace(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            member = "state_snapshot.sqlite"
            if member not in archive.namelist():
                raise RuntimeError("Snapshot does not contain state_snapshot.sqlite")
            snapshot_bytes = archive.read(member)
            summary = json.loads(archive.read("data_summary.json"))
        local_db = LOCAL / "state_snapshot.sqlite"
        previous_db = LOCAL / f"state_snapshot_cycle{meta['cycle'] - 1}.sqlite"
        if local_db.exists() and not previous_db.exists():
            shutil.copy2(local_db, previous_db)
        tmp_db = LOCAL / "state_snapshot.partial.sqlite"
        tmp_db.write_bytes(snapshot_bytes)
        db = sqlite3.connect(f"file:{tmp_db}?mode=ro", uri=True)
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        local_counts = {
            "pages": db.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
            "examples": db.execute("SELECT COUNT(*) FROM examples").fetchone()[0],
        }
        db.close()
        if integrity != "ok" or local_counts != meta["counts"]:
            tmp_db.unlink(missing_ok=True)
            raise RuntimeError(f"Local DB verification mismatch: {integrity}, {local_counts}")
        tmp_db.replace(local_db)
        meta["local_snapshot_sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        meta["local_database_integrity"] = integrity
        meta["local_counts"] = local_counts
        (LOCAL / "snapshot_receipt.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        runtime.close()
        report = __import__("subprocess").run(
            [sys.executable, "tools/report_translation_segmenter_v15c.py"],
            check=True, capture_output=True, text=True, encoding="utf-8")
        (LOCAL / "review_build_receipt.json").write_text(report.stdout, encoding="utf-8")
        task_path = LOCAL / "task_state.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task["status"] = "STOPPED_BY_REQUEST_DATA_SNAPSHOT_REFRESHED"
        task["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        task["runtime"]["stage"] = "STOPPED_BY_REQUEST"
        task["runtime"]["cycle"] = meta["cycle"]
        task["runtime"]["parent_pid_start_ticks_verified"] = False
        task["runtime"]["child_pid_start_ticks_verified"] = False
        task["runtime"]["gpu"] = "NVIDIA L4; worker stopped; snapshot read only"
        task["data"]["latest_live_crawl_cycle"] = meta["cycle"]
        task["data"]["latest_local_snapshot_cycle"] = meta["cycle"]
        task["data"]["latest_local_snapshot_sha256"] = meta["local_snapshot_sha256"]
        task["data"]["snapshot_examples"] = local_counts["examples"]
        task["data"]["snapshot_pages"] = local_counts["pages"]
        task["data"]["latest_cycle_added_pages"] = summary.get("added_pages", {})
        task["data"]["latest_cycle_added_examples"] = summary.get("added_examples", {})
        task["data"]["latest_cycle_http_errors"] = summary.get("http_errors", [])
        task["data"]["snapshot_is_data_only_not_model_recovery"] = True
        task["quality_target_certified"] = False
        task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"remote": meta, "review": json.loads(report.stdout)}, ensure_ascii=False))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
