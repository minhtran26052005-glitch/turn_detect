"""Control the durable VI/EN generator using a.ipynb's current Colab session."""
import argparse
import base64
import json
from pathlib import Path
import sys
from colab_runtime import assigned_servers,ColabRuntime

REPO=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','start','stop']);a=p.parse_args()
    matches=[]
    for i in range(len(assigned_servers())):
        rt=ColabRuntime(server_index=i)
        try:
            response=rt.http.get(rt.base_url+'api/sessions',timeout=15);response.raise_for_status()
            for session in response.json():
                name=Path(session.get('path','')).name
                if name=='a.ipynb' or name.startswith('a-jvsc-'):matches.append((i,session))
        except Exception as e:print(json.dumps({'runtime':i,'error':type(e).__name__}))
    if len(matches)!=1:print(json.dumps({'stage':'NEED_CURRENT_A_IPYNB_SESSION','matches':len(matches)}));return 2
    i,session=matches[0]
    if session['kernel'].get('execution_state')!='idle':print(json.dumps({'stage':'KERNEL_BUSY_WAIT'}));return 3
    payload={name:base64.b64encode((REPO/'turn_v2'/name).read_bytes()).decode() for name in ['factory_paragraphs_v4.py','factory_dialogue_campaign_v6.py']}
    code=f'''import os,json,pathlib,subprocess,shutil,sys,base64,hashlib,time
root=pathlib.Path('/content/factory_dialogue_campaign_v6')
drive=pathlib.Path('/content/drive/MyDrive/turn_detect_backups/translation_segmenter_v15d_20261026')
out=drive/'factory_dialogue_campaign_v6'
gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free,name','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
value={{'action':{a.action!r},'gpu':gpu.stdout.strip(),'colab_free_gib':round(shutil.disk_usage('/content').free/2**30,2),'processes':[]}}
for proc in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  command=proc.read_bytes().replace(b'\\0',b' ').decode(errors='replace')
  if '/content/' in command and any(marker in command for marker in ['/factory_dialogue_campaign_v6.py','/factory_paragraphs_v','/factory_paragraphs_zh_v','/factory_consult_dialogue_v','/factory_dialogue_pilot_v']):
   value['processes'].append({{'pid':int(proc.parent.name),'ticks':(proc.parent/'stat').read_text().split()[21],'command':command[:250]}})
 except (OSError,IndexError):pass
for name in ('status.json','error.json','launcher.json'):
 path=root/name
 if path.exists():value[name]=json.loads(path.read_text())
if (root/'worker.log').exists():
 with (root/'worker.log').open('rb') as f:
  f.seek(max(0,(root/'worker.log').stat().st_size-1800));value['log_tail']=f.read().decode(errors='replace')
if {a.action!r}=='stop':
 assert out.exists(),'Campaign does not exist'
 (out/'STOP_REQUESTED').write_text('Explicit user stop requested.\\n')
 root.mkdir(exist_ok=True);(root/'STOP_REQUESTED').write_text('Explicit user stop requested.\\n')
 value['stop_requested']=True
if {a.action!r}=='start':
 assert os.path.ismount('/content/drive') and drive.is_dir(),'Mount the existing Google Drive first'
 identity=json.loads((drive/'destination_identity.json').read_text())
 assert identity.get('nonce')=='9e618b0f2db442a6a9ee3ee18f1d7143' and identity.get('intended_path')==str(drive),'Drive destination mismatch'
 assert not value['processes'],'GPU generation worker already present; do not duplicate'
 assert not (out/'STOP_REQUESTED').exists() and not (root/'STOP_REQUESTED').exists(),'Stopped campaign needs an explicit resume'
 memory=gpu.stdout.strip().splitlines()[0].split(',')
 assert int(memory[0].strip())>=22528 and int(memory[1].strip())>=19456,'Insufficient free GPU memory'
 assert value['colab_free_gib']>=43,'Insufficient Colab disk reserve'
 root.mkdir(exist_ok=True);source_dir=out/'source';source_dir.mkdir(parents=True,exist_ok=True)
 bundle=json.loads({json.dumps(json.dumps(payload))})
 for name,encoded in bundle.items():
  data=base64.b64decode(encoded);compile(data,name,'exec');(root/name).write_bytes(data)
  path=source_dir/name
  if path.exists():assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256(data).digest(),'Source version changed; preserve and version it'
  else:
   with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
   assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256(data).digest()
 with (root/'worker.log').open('a') as log:
  proc=subprocess.Popen([sys.executable,'-u',str(root/'factory_dialogue_campaign_v6.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 time.sleep(2)
 assert proc.poll() is None,'Worker exited; inspect worker.log'
 launch={{'pid':proc.pid,'start_ticks':pathlib.Path('/proc/'+str(proc.pid)+'/stat').read_text().split()[21],'module':'factory_dialogue_campaign_v6','drive_output':str(out)}}
 (root/'launcher.json').write_text(json.dumps(launch,indent=2));value['launched']=launch
print(json.dumps(value,ensure_ascii=False))'''
    rt=ColabRuntime(server_index=i,session_path=session['path']);rt.connect()
    try:
        result=rt.execute(code,timeout=90,echo=True)
        return 0 if result['status']=='ok' else 1
    finally:rt.close()

if __name__=='__main__':sys.exit(main())
