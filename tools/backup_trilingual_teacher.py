"""Retrieve the exact trained teacher checkpoint in verified 16MiB pieces."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import threading

from colab_runtime import ColabRuntime
from qat_colab import SESSION
from storage_guard import ensure_free_space,reuse_checkpoint


def main():
    root = Path("outputs/trilingual_20261024")
    if reuse_checkpoint(root,root/'teacher_transfer_manifest.json','runs/tri_teacher_20261024/best/model.safetensors'):return
    ensure_free_space(root,2_500_000_000)
    r = ColabRuntime(0, SESSION); r.connect()
    code = '''from pathlib import Path
import hashlib,json
tri_root=Path('/content/turn_detect_v2');source=tri_root/'runs/tri_teacher_20261024/best/model.safetensors'
parts=tri_root/'outputs/trilingual_20261024/teacher_transfer_parts';parts.mkdir(parents=True,exist_ok=True)
items=[]
with source.open('rb') as stream:
    index=0
    while True:
        data=stream.read(16*1024*1024)
        if not data:break
        path=parts/('part_%03d.bin'%index)
        if path.exists():assert path.read_bytes()==data
        else:path.write_bytes(data)
        items.append({'path':'content/turn_detect_v2/'+str(path.relative_to(tri_root)),'name':path.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});index+=1
print(json.dumps({'bytes':source.stat().st_size,'sha256':hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),'parts':items}))
'''
    result = r.execute(code, timeout=90, echo=False)
    assert result["status"] == "ok"
    metadata = json.loads("".join(o["text"] for o in result["outputs"] if o["output_type"] == "stream"))
    (root / "teacher_transfer_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    r.close()
    assert shutil.disk_usage(Path.cwd()).free > metadata["bytes"] * 2 + 100_000_000, "Insufficient workspace disk for exact checkpoint backup"
    destination = root / "runs/tri_teacher_20261024/best/model.safetensors"
    chunks = root / "teacher_transfer_parts"; chunks.mkdir(exist_ok=True)
    local = threading.local()
    def download(item):
        target = chunks / item["name"]
        if not target.exists() or target.stat().st_size != item["bytes"] or hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
            if not hasattr(local, "runtime"):local.runtime = ColabRuntime(0, SESSION)
            local.runtime.download(item["path"], target)
        assert target.stat().st_size == item["bytes"] and hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]
        return item["name"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for count, name in enumerate(pool.map(download, metadata["parts"]), 1):
            progress = {"parts_verified": count, "total_parts": len(metadata["parts"])}
            (root / "teacher_transfer_progress.json").write_text(json.dumps(progress), encoding="utf-8")
            if count % 10 == 0 or count == len(metadata["parts"]):print(json.dumps(progress), flush=True)
    assert not destination.exists()
    with destination.open("wb") as output:
        for item in metadata["parts"]:
            with (chunks / item["name"]).open("rb") as stream:shutil.copyfileobj(stream, output)
    with destination.open("rb") as stream:actual = hashlib.file_digest(stream, "sha256").hexdigest()
    assert destination.stat().st_size == metadata["bytes"] and actual == metadata["sha256"]
    receipt = {"status": "EXACT_TEACHER_BACKUP_VERIFIED", "path": str(destination), "bytes": metadata["bytes"], "sha256": actual}
    (root / "teacher_checkpoint_backup.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__": main()
