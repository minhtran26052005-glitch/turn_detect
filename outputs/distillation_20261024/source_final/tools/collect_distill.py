"""Collect owned student artifacts and receipts without copying the large target cache."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from colab_runtime import ColabRuntime


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--session-path');p.add_argument('--tag',required=True);a=p.parse_args()
    if any(Path(n).name!=n for n in a.runs) or Path(a.tag).name!=a.tag:raise ValueError('Simple names required')
    runtime=ColabRuntime(0,a.session_path);connection=runtime.connect()
    archive_name=f'content/distill_artifacts_{a.tag}.zip'
    code=f'''from pathlib import Path
import hashlib,json,zipfile
root=Path('/content/turn_detect_v2')
archive=Path('/{archive_name}')
names=['student.pt','config.json','tokenizer.json','tokenizer_config.json','turn_detector_config.json','history.json','training_report.json','best_epoch.json','dev_predictions.npz','run_identity.json']
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for run in {a.runs!r}:
        folder=root/'runs'/run
        for name in names:
            path=folder/name
            if not path.is_file():raise ValueError('Training artifact missing: '+name)
            z.write(path,'runs/'+run+'/'+name)
        for name in ['status.json','stdout.log']:
            path=root/'jobs'/run/name
            if path.is_file():z.write(path,'jobs/'+run+'/'+name)
    for name in ['kept_teacher_ids.json','vocabulary_report.json']:
        z.write(root/'runs/distill_vocabulary_20261024'/name,'vocabulary/'+name)
    for name in ['cache_report.json','teacher_parity.json']:
        z.write(root/'runs/distill_cache_20261024'/name,'cache/'+name)
print(json.dumps({{'bytes':archive.stat().st_size,'sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()}}))
'''
    result=runtime.execute(code,timeout=90,echo=False)
    if result['status']!='ok':raise ValueError('Remote archive creation failed: '+str(result['outputs']))
    metadata=json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type']=='stream'))
    out=Path('outputs/distillation_20261024');destination=out/f'artifacts_{a.tag}.zip'
    runtime.download(archive_name,destination);runtime.close()
    assert destination.stat().st_size==metadata['bytes']
    assert hashlib.file_digest(destination.open('rb'),'sha256').hexdigest()==metadata['sha256']
    with zipfile.ZipFile(destination) as z:
        for entry in z.infolist():
            target=(out/entry.filename).resolve()
            if not target.is_relative_to(out.resolve()):raise ValueError('Unsafe archive entry')
            if target.exists():
                if target.read_bytes()!=z.read(entry):raise ValueError('Existing artifact differs: '+entry.filename)
            else:
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(entry))
    receipt={'connection':connection,'runs':a.runs,'status':'COLLECTED_SHA256_VERIFIED',**metadata}
    (out/f'collection_{a.tag}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
