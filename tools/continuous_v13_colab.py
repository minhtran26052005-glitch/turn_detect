"""Launch/resume an unbounded worker; poll and snapshot without big local data."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import quote
import zipfile
from colab_runtime import ColabRuntime
from storage_guard import ensure_free_space

ROOT=Path('outputs/continuous_v13_20261026');NAME='continuous_v13_20261026'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','status','snapshot','resume','restore']);parser.add_argument('--worker-module',choices=['continuous_v13','continuous_v13_r2','continuous_v13_r3'],default='continuous_v13_r3');a=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    r=ColabRuntime(0);connection=r.connect()
    if a.action=='restore':
        archive=ROOT/'latest_snapshot.zip';metadata=json.loads((ROOT/'snapshot_receipt.json').read_text(encoding='utf-8'))
        assert archive.stat().st_size==metadata['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==metadata['sha256']
        result=r.execute("from pathlib import Path\nPath('/content/turn_detect_v2/colab').mkdir(parents=True,exist_ok=True)",timeout=30,echo=False);assert result['status']=='ok'
        r.request('PUT','api/contents/content/turn_detect_v2/colab/continuous_restore.zip',json={'type':'file','format':'base64','content':base64.b64encode(archive.read_bytes()).decode('ascii')})
        code=f'''from pathlib import Path
import hashlib,json,zipfile,subprocess,sys
root=Path('/content/turn_detect_v2');archive=root/'colab/continuous_restore.zip'
assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()=={metadata['sha256']!r}
state=root/'outputs/{NAME}/state.sqlite';assert not state.exists(),'Remote state exists; resume it, do not restore over it'
with zipfile.ZipFile(archive) as z:
 for member in z.infolist():
  relative=member.filename
  if relative=='outputs/{NAME}/recovery/state_snapshot.sqlite':relative='outputs/{NAME}/state.sqlite'
  target=(root/relative).resolve();assert target.is_relative_to(root.resolve())
  data=z.read(member)
  if target.exists():assert target.read_bytes()==data,'Existing recovery file differs: '+relative
  else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
subprocess.run([sys.executable,'-m','pip','install','onnx==1.19.1','onnxruntime==1.23.2'],check=True,stdout=subprocess.DEVNULL)
print(json.dumps({{'status':'RESTORED_CHECKPOINT_NOT_YET_LAUNCHED','cycle':{metadata['cycle']!r},'resume_command':'python tools/continuous_v13_colab.py resume --worker-module continuous_v13_r3'}}))
'''
    elif a.action in ['launch','resume']:
        files=['turn_v2/'+a.worker_module+'.py','turn_v2/continuous_v13_state.py','turn_v2/continuous_v13_objective.py'];digests={}
        for name in files:
            data=Path(name).read_bytes();digests[name]=hashlib.sha256(data).hexdigest()
            r.request('PUT','api/contents/content/turn_detect_v2/colab/'+Path(name).name,json={'type':'file','format':'base64','content':base64.b64encode(data).decode('ascii')})
        code=f'''from pathlib import Path
import hashlib,json,subprocess,sys,os
root=Path('/content/turn_detect_v2');os.chdir(root);name={NAME!r}
for relative,digest in {digests!r}.items():
 data=(root/'colab'/Path(relative).name).read_bytes();assert hashlib.sha256(data).hexdigest()==digest;target=root/relative
 if target.exists():assert target.read_bytes()==data,'Existing worker source differs; preserve running code'
 else:target.write_bytes(data)
pidfile=root/'outputs'/name/'launcher.json'
if pidfile.exists():
 previous=json.loads(pidfile.read_text());pid=previous['pid'];proc=Path('/proc')/str(pid)/'cmdline'
 if proc.exists():
  command=proc.read_bytes().replace(bytes([0]),b' ').decode();assert 'turn_v2.continuous_v13' not in command,'Continuous worker already active; never launch duplicate'
folder=root/'outputs'/name;folder.mkdir(parents=True,exist_ok=True)
log=folder/('worker_'+str(len(list(folder.glob('worker_*.log')))+1)+'.log')
command=[sys.executable,'-u','-m',{'turn_v2.'+a.worker_module!r},'--output','outputs/'+name,'--max-cycles','0','--steps','200','--cadence','300']
with log.open('w') as stream:
 process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1'}})
receipt={{'pid':process.pid,'command':command,'source_sha256':{digests!r},'global_timeout':None,'max_cycles':0,'log':str(log.relative_to(root))}}
(folder/'launcher.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
    elif a.action=='status':
        code=f'''from pathlib import Path
import json,subprocess,os,time
root=Path('/content/turn_detect_v2');folder=root/'outputs/{NAME}';values={{}}
for relative in ['status.json','launcher.json','latest_error.json','recovery/latest_snapshot.json']:
 path=folder/relative
 if path.exists():values[relative]=json.loads(path.read_text())
launcher=values.get('launcher.json',{{}});pid=launcher.get('pid');proc=Path('/proc')/str(pid)
if pid and (proc/'cmdline').exists():
 command=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode();values['process_verified']='turn_v2.continuous_v13' in command
 values['process_start_ticks']=(proc/'stat').read_text().split()[21]
else:values['process_verified']=False
if not values['process_verified'] and launcher.get('log'):
 log=root/launcher['log']
 if log.exists():values['log_tail']=log.read_text()[-5000:]
database=folder/'state.sqlite'
if database.exists():
 import sqlite3
 c=sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True)
 values['cycle']=dict(c.execute('SELECT key,value FROM meta WHERE key=?',('cycle',))).get('cycle')
 values['pool_counts']=[dict(zip(['language','split','label','samples'],row)) for row in c.execute('SELECT language,split,label,COUNT(*) FROM samples GROUP BY language,split,label')]
 values['champions']={{key.split(':')[1]:json.loads(value) for key,value in c.execute("SELECT key,value FROM meta WHERE key LIKE 'champion:%'")}}
 values['reference_disagreements']=[]
 for lang in ['vi','en','zh']:
  rows=[json.loads(row[0]) for row in c.execute("SELECT record FROM samples WHERE language=? AND split='semantic_dev'",(lang,))]
  paired=[row for row in rows if row.get('observed_floor_reference') in ['EOT','NEOT']]
  values['reference_disagreements'].append({{'language':lang,'paired_natural_dev_cases':len(paired),'semantic_vs_floor_disagree':sum(row['label']!=row['observed_floor_reference'] for row in paired)}})
 c.close()
values['GPU']=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
print(json.dumps(values))
'''
    else:
        code=f'''from pathlib import Path
import hashlib,json
root=Path('/content/turn_detect_v2');folder=root/'outputs/{NAME}';receipt=json.loads((folder/'recovery/latest_snapshot.json').read_text());path=root/receipt['path']
assert path.stat().st_size==receipt['bytes'] and hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==receipt['sha256']
print(json.dumps(receipt))
'''
    result=r.execute(code,timeout=120,echo=False)
    receipt=ROOT/(a.action+'_receipt.json');receipt.write_text(json.dumps({'connection':connection,**result},indent=2),encoding='utf-8')
    assert result['status']=='ok','Inspect saved receipt; execution never silently repeats'
    value=json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type']=='stream'))
    if a.action=='snapshot':
        ensure_free_space(ROOT,value['bytes']*2);target=ROOT/'latest_snapshot.zip';partial=ROOT/'download.partial'
        # Stream a single small archive; credentials never leave this client.
        response=r.http.get(r.base_url+'files/'+quote('content/turn_detect_v2/'+value['path'],safe='/'),timeout=180,stream=True,allow_redirects=False)
        assert response.status_code==200 and 'html' not in response.headers.get('content-type','')
        with partial.open('wb') as stream:
            for block in response.iter_content(1024*1024):stream.write(block)
        assert partial.stat().st_size==value['bytes'] and hashlib.sha256(partial.read_bytes()).hexdigest()==value['sha256'];partial.replace(target)
        (ROOT/'snapshot_receipt.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    if a.action=='status':
        for lang,item in value.get('champions',{}).items():
            value['champions'][lang]={'path':item['path'],'semantic_silver_dev':item['semantic_dev']['metrics'],'natural_floor_dev':item['natural_floor_dev'],'precision':item.get('precision')}
    print(json.dumps(value),flush=True);r.close()


if __name__=='__main__':main()
