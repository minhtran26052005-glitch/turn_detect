"""Backup the selected completed campaign checkpoint with exact SHA256 checks."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import threading
from urllib.parse import quote

from colab_runtime import ColabRuntime
from qat_colab import SESSION


def main():
    evidence=Path("outputs/stream_v4_20261025");r=ColabRuntime(0,SESSION);r.connect()
    result=r.execute('''from pathlib import Path
import hashlib,json
root=Path('/content/turn_detect_v2');run=root/'runs/stream_v4_campaign_20261025'
status=json.loads((root/'jobs/stream_v4_campaign_20261025/status.json').read_text());assert status['status']=='COMPLETED' and status['exit_code']==0
campaign=json.loads((run/'campaign.json').read_text());assert campaign['best_path']
source=(root/campaign['best_path']/'model.safetensors').resolve();assert source.is_relative_to(run.resolve())
parts=root/'outputs/stream_v4_20261025/checkpoint_parts';parts.mkdir(parents=True,exist_ok=True)
items=[]
with source.open('rb') as stream:
    index=0
    while True:
        data=stream.read(16*1024*1024)
        if not data:break
        path=parts/('part_%03d.bin'%index)
        if path.exists():assert path.read_bytes()==data
        else:path.write_bytes(data)
        items.append({'path':'content/turn_detect_v2/'+str(path.relative_to(root)),'name':path.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});index+=1
print(json.dumps({'selected_path':str(source.relative_to(root)),'bytes':source.stat().st_size,'sha256':hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),'parts':items}))
''',timeout=90,echo=False)
    assert result["status"]=="ok"
    metadata=json.loads("".join(o["text"] for o in result["outputs"] if o["output_type"]=="stream"))
    (evidence/"checkpoint_transfer.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8");r.close()
    assert shutil.disk_usage(Path.cwd()).free>metadata["bytes"]*2+100_000_000
    parts=evidence/"checkpoint_parts";parts.mkdir(exist_ok=True)
    local=threading.local()
    def fetch(item):
        path=parts/item["name"]
        if not path.exists():
            if not hasattr(local,"runtime"):local.runtime=ColabRuntime(0,SESSION)
            runtime=local.runtime
            response=runtime.http.get(runtime.base_url+'files/'+quote(item['path'],safe='/'),timeout=120,stream=True,allow_redirects=False)
            assert response.status_code==200 and 'html' not in response.headers.get('content-type','')
            with path.open('wb') as out:
                for chunk in response.iter_content(1024*1024):out.write(chunk)
        assert path.stat().st_size==item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        return path.name
    with ThreadPoolExecutor(max_workers=4) as pool:
        for count,_ in enumerate(pool.map(fetch,metadata['parts']),1):
            progress={'verified_parts':count,'total':len(metadata['parts'])}
            (evidence/'checkpoint_backup_progress.json').write_text(json.dumps(progress),encoding='utf-8')
            if count%10==0 or count==len(metadata['parts']):print(json.dumps(progress),flush=True)
    target=(evidence/metadata['selected_path']).resolve();assert target.is_relative_to(evidence.resolve())
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        with target.open('wb') as out:
            for item in metadata['parts']:
                with (parts/item['name']).open('rb') as source:shutil.copyfileobj(source,out)
    with target.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    assert target.stat().st_size==metadata['bytes'] and sha==metadata['sha256']
    receipt={'status':'EXACT_SELECTED_CHECKPOINT_SAVED','path':str(target),'bytes':metadata['bytes'],'sha256':sha}
    (evidence/'checkpoint_backup.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
