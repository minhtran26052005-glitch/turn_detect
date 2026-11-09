"""Upload a reviewable project bundle into the already connected Colab runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from colab_runtime import ColabRuntime

ROOT = Path(__file__).resolve().parents[1]


def build_bundle(include_data=True):
    destination = ROOT / "colab" / ("turn_detect_v2_bundle.zip" if include_data else "turn_detect_v2_code.zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = [*sorted((ROOT / "turn_v2").glob("*.py")),
             *sorted((ROOT / "tests").glob("*.py")), ROOT / "requirements-colab.txt"]
    if include_data:
        files += [ROOT / "train.jsonl", ROOT / "test.jsonl"]
        files += sorted((ROOT / "data_v2" / "frozen").glob("*.json*"))
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            if path.is_file():
                archive.write(path, path.relative_to(ROOT).as_posix())
    return destination


def sync(include_data=True, server_index=0, session_path=None):
    bundle = build_bundle(include_data)
    runtime = ColabRuntime(server_index, session_path)
    print(json.dumps(runtime.connect()), flush=True)
    remote_archive = "content/" + bundle.name
    runtime.upload(bundle, remote_archive)
    checksum = hashlib.file_digest(bundle.open("rb"), "sha256").hexdigest()
    code = f"""import hashlib, json, pathlib, zipfile
bundle=pathlib.Path({'/' + remote_archive!r})
assert hashlib.file_digest(bundle.open('rb'),'sha256').hexdigest()=={checksum!r}
root=pathlib.Path('/content/turn_detect_v2')
root.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as archive:
    for name in archive.namelist():
        if not (root/name).resolve().is_relative_to(root.resolve()):
            raise ValueError('Unsafe bundle entry')
    archive.extractall(root)
print(json.dumps({{'uploaded':True,'bundle_sha256':{checksum!r},'root':str(root),'files':len(archive.namelist())}}))
"""
    result = runtime.execute(code, timeout=120)
    if result["status"] != "ok":
        raise RuntimeError("Bundle verification/extraction failed")
    receipt = ROOT / "outputs" / "colab_sync.json"
    receipt.write_text(json.dumps({"bundle_sha256": checksum, "bytes": bundle.stat().st_size,
        "includes_data": include_data, **result}, indent=2) + "\n", encoding="utf-8")
    return runtime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--server-index", type=int, default=0)
    parser.add_argument("--session-path")
    args = parser.parse_args()
    sync(not args.code_only, args.server_index, args.session_path)
