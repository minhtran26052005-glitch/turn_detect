"""Exact selected V5 teacher backup, resumable by verified independent chunks."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import threading
from urllib.parse import quote

from colab_runtime import ColabRuntime
from qat_colab import SESSION
from storage_guard import ensure_free_space,reuse_checkpoint


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    root=Path('outputs/natural_v5_20261025')
    if reuse_checkpoint(root,root/'checkpoint_transfer.json'):return
    ensure_free_space(root,2_500_000_000)
    runtime=ColabRuntime(0,SESSION);runtime.connect()
    result=runtime.execute('''from pathlib import Path
import hashlib,json
root=Path('/content/turn_detect_v2');run=root/'runs/natural_v5_train_20261025'
status=json.loads((root/'jobs/natural_v5_train_20261025/status.json').read_text());assert status['status']=='COMPLETED' and status['exit_code']==0
report=json.loads((run/'report.json').read_text());source=(root/report['best_path']/'model.safetensors').resolve()
assert source.is_relative_to((root/'runs/natural_v5_train_20261025').resolve()) or source==(root/'runs/stream_v4_campaign_20261025/round_02/best/model.safetensors').resolve()
assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==report['checkpoint_sha256']
parts=root/'outputs/natural_v5_20261025/checkpoint_parts';parts.mkdir(parents=True,exist_ok=True);items=[]
with source.open('rb') as stream:
    index=0
    while True:
        data=stream.read(16*1024*1024)
        if not data:break
        path=parts/('part_%03d.bin'%index)
        if path.exists():assert path.read_bytes()==data
        else:path.write_bytes(data)
        items.append({'path':'content/turn_detect_v2/'+str(path.relative_to(root)),'name':path.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});index+=1
print(json.dumps({'selected_path':str(source.relative_to(root)),'sha256':report['checkpoint_sha256'],'bytes':source.stat().st_size,'parts':items}))
''',timeout=120,echo=False)
    assert result['status']=='ok'
    metadata=json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type']=='stream'))
    (root/'checkpoint_transfer.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');runtime.close()
    assert shutil.disk_usage(root).free>metadata['bytes']*2+100_000_000
    parts=root/'checkpoint_parts';parts.mkdir(exist_ok=True);local=threading.local()
    def fetch(item):
        path=(parts/item['name']).resolve();assert path.is_relative_to(parts.resolve())
        if path.exists():assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'];return
        if not hasattr(local,'runtime'):local.runtime=ColabRuntime(0,SESSION)
        client=local.runtime
        response=client.http.get(client.base_url+'files/'+quote(item['path'],safe='/'),timeout=180,stream=True,allow_redirects=False)
        assert response.status_code==200 and 'html' not in response.headers.get('content-type','')
        partial=path.with_suffix('.partial');assert partial.is_relative_to(parts.resolve())
        with partial.open('wb') as stream:
            for block in response.iter_content(1024*1024):stream.write(block)
        assert partial.stat().st_size==item['bytes'] and sha(partial)==item['sha256'];partial.rename(path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for count,_ in enumerate(pool.map(fetch,metadata['parts']),1):
            (root/'checkpoint_backup_progress.json').write_text(json.dumps({'verified_parts':count,'total':len(metadata['parts'])}),encoding='utf-8')
            if count%10==0:print(json.dumps({'verified_parts':count,'total':len(metadata['parts'])}),flush=True)
    target=(root/metadata['selected_path']).resolve();assert target.is_relative_to(root.resolve());target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        partial=target.with_suffix('.partial')
        with partial.open('wb') as stream:
            for item in metadata['parts']:
                with (parts/item['name']).open('rb') as chunk:shutil.copyfileobj(chunk,stream)
        assert partial.stat().st_size==metadata['bytes'] and sha(partial)==metadata['sha256'];partial.rename(target)
    assert target.stat().st_size==metadata['bytes'] and sha(target)==metadata['sha256']
    receipt={'status':'EXACT_SELECTED_TEACHER_SAVED','path':str(target),'sha256':metadata['sha256'],'bytes':metadata['bytes']}
    (root/'checkpoint_backup.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
