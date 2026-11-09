"""Resumable, verified backup of one completed private language teacher."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import threading
from urllib.parse import quote
from colab_runtime import ColabRuntime


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--language',choices=['en','zh'],required=True);a=p.parse_args()
    root=Path('outputs/monolingual_20261025');r=ColabRuntime(0);r.connect();name='mono_'+a.language+'_teacher_20261025'
    code=f'''from pathlib import Path
import hashlib,json
root=Path('/content/turn_detect_v2');name={name!r};run=root/'runs'/name
state=json.loads((root/'jobs'/name/'status.json').read_text());assert state['status']=='COMPLETED' and state['exit_code']==0
report=json.loads((run/'training_report.json').read_text());assert report['language']=={a.language!r}
source=run/'best/model.safetensors';assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==report['checkpoint_sha256']
folder=root/'colab'/('backup_'+name);folder.mkdir(exist_ok=True);parts=[]
with source.open('rb') as stream:
 index=0
 while True:
  data=stream.read(16*1024*1024)
  if not data:break
  path=folder/('part_%03d.bin'%index)
  if path.exists():assert path.read_bytes()==data
  else:path.write_bytes(data)
  parts.append({{'path':'content/turn_detect_v2/'+path.relative_to(root).as_posix(),'name':path.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}});index+=1
print(json.dumps({{'path':source.relative_to(root).as_posix(),'bytes':source.stat().st_size,'sha256':report['checkpoint_sha256'],'parts':parts}}))
'''
    result=r.execute(code,timeout=120,echo=False);assert result['status']=='ok'
    metadata=json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type']=='stream'));r.close()
    folder=root/('backup_'+a.language);folder.mkdir(exist_ok=True);(folder/'transfer.json').write_text(json.dumps(metadata,indent=2))
    assert shutil.disk_usage(root).free>metadata['bytes']*2+100_000_000
    clients=threading.local()
    def fetch(item):
        target=folder/item['name']
        if target.exists():assert target.stat().st_size==item['bytes'] and sha(target)==item['sha256'];return
        if not hasattr(clients,'runtime'):clients.runtime=ColabRuntime(0)
        client=clients.runtime;response=client.http.get(client.base_url+'files/'+quote(item['path'],safe='/'),timeout=180,stream=True,allow_redirects=False)
        assert response.status_code==200 and 'html' not in response.headers.get('content-type','')
        partial=target.with_suffix('.partial')
        with partial.open('wb') as stream:
            for block in response.iter_content(1024*1024):stream.write(block)
        assert partial.stat().st_size==item['bytes'] and sha(partial)==item['sha256'];partial.replace(target)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for count,_ in enumerate(pool.map(fetch,metadata['parts']),1):
            (folder/'progress.json').write_text(json.dumps({'verified_parts':count,'total':len(metadata['parts'])}))
            if count%10==0:print(json.dumps({'language':a.language,'verified_parts':count,'total':len(metadata['parts'])}),flush=True)
    target=(root/metadata['path']).resolve();assert target.is_relative_to(root.resolve());target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        partial=target.with_suffix('.partial')
        with partial.open('wb') as out:
            for item in metadata['parts']:
                with (folder/item['name']).open('rb') as source:shutil.copyfileobj(source,out)
        assert partial.stat().st_size==metadata['bytes'] and sha(partial)==metadata['sha256'];partial.replace(target)
    assert target.stat().st_size==metadata['bytes'] and sha(target)==metadata['sha256']
    receipt={'status':'EXACT_PRIVATE_TEACHER_SAVED','language':a.language,'path':str(target),'bytes':metadata['bytes'],'sha256':metadata['sha256']}
    (folder/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
