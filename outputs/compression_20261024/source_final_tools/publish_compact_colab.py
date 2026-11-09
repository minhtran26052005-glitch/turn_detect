"""Copy the verified uploaded candidate into its final Colab deployment path."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from colab_runtime import ColabRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-path')
    parser.add_argument('--server-index',type=int,default=0)
    args = parser.parse_args()
    root = Path('outputs/compression_20261024')
    model = Path('models/turn_v2_compact_20261024')
    manifest = json.loads((model/'model_manifest.json').read_text())
    assert manifest['status']=='SIZE_AND_FULL_DEV_TARGET_MET'
    uploaded = json.loads((root/'colab_candidate_upload.json').read_text())
    assert next(f['sha256'] for f in uploaded['files'] if f['name']=='model.onnx')==manifest['files']['model.onnx']['sha256']
    old_contract = next(f['sha256'] for f in uploaded['files'] if f['name']=='turn_detector_config.json')
    payload = {}
    for source, destination in [
        (model/'onnx/turn_detector_config.json','models/turn_v2_compact_20261024/onnx/turn_detector_config.json'),
        (model/'model_manifest.json','models/turn_v2_compact_20261024/model_manifest.json'),
        (root/'weight_only_full_dev_cached/report.json','outputs/compression_20261024/weight_only_full_dev_cached/report.json')]:
        payload[destination] = base64.b64encode(source.read_bytes()).decode('ascii')
    code = f'''import base64, hashlib, json, shutil
from pathlib import Path
project=Path('/content/turn_detect_v2')
source=project/'models/compression_candidate_20261024/onnx'
target=project/'models/turn_v2_compact_20261024/onnx'
expected={manifest!r}
assert hashlib.file_digest((source/'model.onnx').open('rb'),'sha256').hexdigest()==expected['files']['model.onnx']['sha256']
if not target.exists():
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,target)
assert hashlib.file_digest((target/'model.onnx').open('rb'),'sha256').hexdigest()==expected['files']['model.onnx']['sha256']
assert hashlib.file_digest((target/'turn_detector_config.json').open('rb'),'sha256').hexdigest() in ({old_contract!r},expected['files']['turn_detector_config.json']['sha256'])
for relative,encoded in {payload!r}.items():
    path=project/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(base64.b64decode(encoded))
for filename,identity in expected['files'].items():
    path=target/filename
    assert path.stat().st_size==identity['bytes']
    assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==identity['sha256']
assert sum(p.stat().st_size for p in target.rglob('*') if p.is_file())==expected['deployment_bytes']
print(json.dumps({{'status':'FINAL_COLAB_FILES_VERIFIED','model_sha256':expected['files']['model.onnx']['sha256'],'deployment_bytes':expected['deployment_bytes'],'dev':expected['dev_metrics'],'holdout_evaluated':False}}))
'''
    runtime = ColabRuntime(args.server_index,args.session_path)
    connection = runtime.connect()
    result = runtime.execute(code,timeout=60)
    runtime.close()
    (root/'colab_final_verification.json').write_text(json.dumps({'connection':connection,**result},indent=2)+'\n')
    if result['status']!='ok':
        raise RuntimeError('Final deployment verification failed')


if __name__ == '__main__':
    main()
