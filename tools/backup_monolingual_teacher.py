"""Exact teacher backup with10GiB reserve and no duplicate local chunk set."""
import argparse
import json
from pathlib import Path
from colab_runtime import ColabRuntime
from storage_guard import download_checkpoint,ensure_free_space,reuse_checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('--language',choices=['en','zh'],required=True);a=p.parse_args()
    root=Path('outputs/monolingual_20261025');folder=root/('backup_'+a.language)
    if reuse_checkpoint(root,folder/'transfer.json'):return
    ensure_free_space(root,1_200_000_000)
    r=ColabRuntime(0);r.connect();name='mono_'+a.language+'_teacher_20261025'
    code=f'''from pathlib import Path
import hashlib,json
root=Path('/content/turn_detect_v2');name={name!r};run=root/'runs'/name
state=json.loads((root/'jobs'/name/'status.json').read_text());assert state['status']=='COMPLETED' and state['exit_code']==0
report=json.loads((run/'training_report.json').read_text());assert report['language']=={a.language!r}
source=run/'best/model.safetensors';assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==report['checkpoint_sha256']
print(json.dumps({{'path':source.relative_to(root).as_posix(),'bytes':source.stat().st_size,'sha256':report['checkpoint_sha256']}}))
'''
    result=r.execute(code,timeout=120,echo=False);assert result['status']=='ok'
    metadata=json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type']=='stream'))
    target=(root/metadata['path']).resolve();assert target.is_relative_to(root.resolve())
    folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'transfer.json'
    if manifest.exists():manifest=folder/'direct_transfer.json'
    if manifest.exists():assert json.loads(manifest.read_text())==metadata
    else:manifest.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    progress=download_checkpoint(r,'content/turn_detect_v2/'+metadata['path'],target,metadata['bytes'],metadata['sha256']);r.close()
    receipt={'status':'EXACT_PRIVATE_TEACHER_SAVED','language':a.language,'path':str(target),'bytes':metadata['bytes'],'sha256':metadata['sha256'],**progress}
    (folder/'direct_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
