"""Download a completed Colab trial and verify its exported model hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
import zipfile

import requests

from colab_runtime import ColabRuntime


def download(runtime, remote, local, checksum, size):
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and local.stat().st_size == size:
        with local.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() == checksum:
                return
    partial = local.with_suffix(local.suffix + ".partial")
    url = runtime.base_url + "files/" + quote(remote.lstrip("/"), safe="/")
    received = partial.stat().st_size if partial.exists() else 0
    if received > size:
        raise RuntimeError("Partial file is larger than its expected size")
    failures = 0
    milestone = received // (64 * 2**20)
    # Short range requests survive the Colab proxy's connection resets.
    while received < size:
        end = min(received + 8 * 2**20, size) - 1
        try:
            with runtime.http.get(url, headers={"Range": f"bytes={received}-{end}"},
                                  stream=True, timeout=(15, 30)) as response:
                response.raise_for_status()
                ranged = response.status_code == 206
                if ranged and not response.headers.get("Content-Range", "").startswith(f"bytes {received}-"):
                    raise RuntimeError("Unexpected range response")
                if not ranged and (received or end + 1 < size):
                    raise RuntimeError("Server did not support the requested resume range")
                with partial.open("ab") as stream:
                    for chunk in response.iter_content(2**20):
                        stream.write(chunk)
                        received += len(chunk)
            failures = 0
            if received // (64 * 2**20) > milestone:
                milestone = received // (64 * 2**20)
                print(json.dumps({"file": local.name, "bytes": received, "total": size}), flush=True)
        except requests.RequestException as error:
            failures += 1
            if failures >= 10:
                raise
            print(json.dumps({"download_retry": type(error).__name__, "bytes_retained": received}), flush=True)
            time.sleep(min(2 ** failures, 20))
    with partial.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if partial.stat().st_size != size or actual != checksum:
        raise RuntimeError(f"Hash/size mismatch for {local.name}; partial retained")
    partial.replace(local)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, default=Path("outputs/training_trial_20261024"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/turn_v2_trial_20261024/onnx"))
    parser.add_argument("--receipts-only", action="store_true")
    parser.add_argument("--include-pytorch", action="store_true", help="Also preserve the selected trainable checkpoint")
    args = parser.parse_args()
    launch = json.loads((args.trial / "launch.json").read_text(encoding="utf-8"))
    name = json.loads("".join(item.get("text", "") for item in launch["outputs"]))["run_name"]
    code = """import json,hashlib,zipfile
from pathlib import Path
root=Path('/content/turn_detect_v2')
name=RUN_NAME
out=root/'runs'/name
job=root/'jobs'/name
status=json.loads((job/'status.json').read_text())
assert status['status']=='COMPLETED',status['status']
packet=job/'receipts.zip'
with zipfile.ZipFile(packet,'w',zipfile.ZIP_DEFLATED) as z:
    for filename in ['training_report.json','run_identity.json','progress.json','dev_predictions.npz']:
        z.write(out/filename,filename)
    last=max((out/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
    z.write(last/'trainer_state.json','trainer_state.json')
    for filename in ['status.json','train.log']:
        z.write(job/filename,filename)
    for path in job.glob('evaluation*'):
        if path.is_file():z.write(path,path.name)
    for path in (out/'best').glob('*.json'):
        z.write(path,'best_metadata/'+path.name)
    for path in (out/'final_holdout').glob('*'):
        if path.is_file():z.write(path,path.name)
    marker=root/'data_v2/frozen/holdout_consumed.json'
    if marker.exists():z.write(marker,'holdout_consumed.json')
    for path in (root/'turn_v2').glob('*.py'):
        z.write(path,'source/'+path.name)
    z.write(root/'requirements-colab.txt','requirements-colab.txt')
def info(path):
    with path.open('rb') as f:checksum=hashlib.file_digest(f,'sha256').hexdigest()
    return {'remote':str(path),'size':path.stat().st_size,'sha256':checksum,'name':path.name}
print(json.dumps({'receipts':info(packet),'onnx':[info(p) for p in sorted((out/'onnx').iterdir()) if p.is_file()],
    'best':[info(p) for p in sorted((out/'best').iterdir()) if p.is_file()]}))
""".replace("RUN_NAME", repr(name))
    runtime = ColabRuntime()
    try:
        response = runtime.execute(code, timeout=60, echo=False)
        if response["status"] != "ok":
            raise RuntimeError(response)
        inventory = json.loads("".join(item.get("text", "") for item in response["outputs"]))
        (args.trial / "remote_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        packet = inventory["receipts"]
        destination = args.trial / "receipts.zip"
        download(runtime, packet["remote"], destination, packet["sha256"], packet["size"])
        with zipfile.ZipFile(destination) as archive:
            for path in archive.namelist():
                if not (args.trial/path).resolve().is_relative_to(args.trial.resolve()):
                    raise ValueError("Unsafe receipt path")
            archive.extractall(args.trial)
        if not args.receipts_only:
            for item in inventory["onnx"]:
                download(runtime, item["remote"], args.model_dir/item["name"], item["sha256"], item["size"])
            if args.include_pytorch:
                for item in inventory["best"]:
                    download(runtime, item["remote"], args.model_dir.parent/"best"/item["name"], item["sha256"], item["size"])
        inventory["local_model_dir"] = str(args.model_dir.resolve()) if not args.receipts_only else None
        inventory["local_pytorch_dir"] = str((args.model_dir.parent/"best").resolve()) if args.include_pytorch and not args.receipts_only else None
        (args.trial / "download_manifest.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        print(json.dumps({"receipts_collected": True, "onnx_collected": not args.receipts_only,
            "pytorch_collected": args.include_pytorch and not args.receipts_only,
            "onnx_files": len(inventory["onnx"]), "model_dir": inventory["local_model_dir"]}), flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
