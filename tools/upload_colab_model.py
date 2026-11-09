"""Upload a model in bounded chunks; assemble only after verifying every byte."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
from urllib.parse import quote

from colab_runtime import ColabRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--server-index", type=int, default=0)
    parser.add_argument("--session-path")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    destination = PurePosixPath(args.destination)
    if not destination.is_relative_to('/content/turn_detect_v2/models') or '..' in destination.parts:
        raise ValueError('Destination must stay within the Colab project models directory')
    files = sorted(p for p in args.model.iterdir() if p.is_file() and p.suffix in ('.json','.onnx'))
    if not files or not any(p.suffix == '.onnx' for p in files):
        raise ValueError('An ONNX deployment folder is required')
    runtime = ColabRuntime(args.server_index, args.session_path)
    connection = runtime.connect()
    chunk_directory = PurePosixPath('/content/turn_detect_v2/uploads') / hashlib.sha256(str(destination).encode()).hexdigest()[:16]
    result = runtime.execute(f"from pathlib import Path; upload_root=Path({str(destination)!r}); upload_root.mkdir(parents=True,exist_ok=True); Path({str(chunk_directory)!r}).mkdir(parents=True,exist_ok=True)", timeout=60)
    if result['status'] != 'ok':
        raise RuntimeError('Cannot prepare the remote deployment directory')
    records = []
    started = time.monotonic()
    for path in files:
        sha = hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        chunks = []
        with path.open('rb') as source:
            index = 0
            while data := source.read(4*1024*1024):
                relative = str(chunk_directory / f'{sha}.{index:05d}').lstrip('/')
                for attempt in range(3):
                    try:
                        runtime.request('PUT','api/contents/'+quote(relative,safe='/'),
                            json={'type':'file','format':'base64','content':base64.b64encode(data).decode('ascii')})
                        break
                    except (RuntimeError, OSError):
                        if attempt == 2:
                            raise
                        time.sleep(2)
                chunks.append({'name':PurePosixPath(relative).name,'sha256':hashlib.sha256(data).hexdigest()})
                index += 1
                print(json.dumps({'file':path.name,'uploaded_bytes':source.tell(),'total_bytes':path.stat().st_size}),flush=True)
        # Idempotent upload retries write identical chunks. Assembly never replaces
        # a completed destination with different contents.
        code = f"""import hashlib,json,os
from pathlib import Path
root=Path({str(destination)!r})
parts=Path({str(chunk_directory)!r})
target=root/{path.name!r}
expected={sha!r}
if target.exists() and hashlib.file_digest(target.open('rb'),'sha256').hexdigest()!=expected:
    raise ValueError('Destination already contains a different model file')
temporary=root/({path.name!r}+'.assembling')
with temporary.open('wb') as output:
    for chunk in {chunks!r}:
        data=(parts/chunk['name']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==chunk['sha256']
        output.write(data)
assert hashlib.file_digest(temporary.open('rb'),'sha256').hexdigest()==expected
assert temporary.stat().st_size=={path.stat().st_size}
os.replace(temporary,target)
print(json.dumps({{'file':{path.name!r},'sha256':expected,'verified':True}}))
"""
        runtime.close()
        result = runtime.execute(code,timeout=120)
        if result['status'] != 'ok':
            raise RuntimeError('Remote assembly verification failed')
        records.append({'name':path.name,'sha256':sha,'bytes':path.stat().st_size})
    receipt = {'connection':connection,'destination':str(destination),'files':records,
               'elapsed_seconds':time.monotonic()-started,'status':'UPLOAD_SHA256_VERIFIED'}
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    runtime.close()
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
