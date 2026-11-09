"""Discover a.ipynb's current session and safely launch/read the Qwen pilot."""
import argparse
import base64
import json
from pathlib import Path
import sys

from colab_runtime import assigned_servers, ColabRuntime

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'turn_v2/factory_paragraphs_v4.py'


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','start']);a=p.parse_args()
    servers=assigned_servers();matches=[]
    for index in range(len(servers)):
        runtime=ColabRuntime(server_index=index)
        try:
            response=runtime.http.get(runtime.base_url+'api/sessions',timeout=15)
            response.raise_for_status()
            for session in response.json():
                name=Path(session.get('path','')).name
                if name=='a.ipynb' or name.startswith('a-jvsc-'):
                    matches.append((index,session))
        except Exception as exc:
            print(json.dumps({'runtime':index,'error_type':type(exc).__name__}))
    if len(matches)!=1:
        print(json.dumps({'status':'NEED_A_IPYNB_SESSION','matches':len(matches)}));return 2
    index,session=matches[0]
    if session.get('kernel',{}).get('execution_state')!='idle':
        print(json.dumps({'status':'KERNEL_BUSY_WAIT','path':session.get('path')}));return 3
    rt=ColabRuntime(server_index=index,session_path=session['path']);rt.connect()
    payload=base64.b64encode(SCRIPT.read_bytes()).decode('ascii')
    code=f'''import os, json, pathlib, subprocess, shutil, sys, base64
root=pathlib.Path('/content/factory_dialogue_pilot_20261030_v4_paragraphs')
dest=root/'output'
gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free,name','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
procs=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
status={{'disk_free_gib':round(shutil.disk_usage('/content').free/2**30,2),'gpu':gpu.stdout.strip(),'gpu_processes':procs.stdout.strip(),'output_exists':dest.exists()}}
lock=root/'LAUNCH_LOCK'
status['launch_lock_exists']=lock.exists()
log_path=root/'worker.log'
if log_path.exists():status['worker_log_tail']=log_path.read_text(errors='replace')[-4000:]
status['pilot_processes']=[]
for proc_path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  cmd=proc_path.read_bytes().replace(b'\\0',b' ').decode(errors='replace')
  if '/content/factory_dialogue_pilot_20261030_' in cmd and ('/factory_dialogue_pilot_v' in cmd or '/factory_consult_dialogue_v' in cmd or '/factory_paragraphs_v' in cmd):
   pid=proc_path.parent.name
   status['pilot_processes'].append({{'pid':int(pid),'start_ticks':(proc_path.parent/'stat').read_text().split()[21]}})
 except (OSError,IndexError):pass
if dest.exists():
 for name in ('status.json','manifest.json'):
  path=dest/name
  if path.exists():status[name]=json.loads(path.read_text())
 for name in ('dialogues.jsonl','examples.jsonl','rejected.jsonl'):
  path=dest/name
  if path.exists():
   with path.open(encoding='utf-8') as f:status[name+'_sample']=[json.loads(line) for _,line in zip(range(3),f)]
if {a.action!r}=='start' and not dest.exists() and not lock.exists() and not status['pilot_processes']:
 import importlib.util
 required=['torch','transformers','accelerate','safetensors']
 status['missing_packages']=[x for x in required if importlib.util.find_spec(x) is None]
 parts=gpu.stdout.strip().splitlines()[0].split(',') if gpu.returncode==0 and gpu.stdout.strip() else []
 suitable=len(parts)>=2 and int(parts[0].strip())>=22528 and int(parts[1].strip())>=19456
 if status['disk_free_gib']>=43 and suitable and not status['missing_packages']:
  root.mkdir(parents=True,exist_ok=True)
  fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
  source=root/'factory_paragraphs_v4.py';data=base64.b64decode({payload!r});compile(data,str(source),'exec');source.write_bytes(data)
  with (root/'worker.log').open('a') as log:
   proc=subprocess.Popen([sys.executable,'-u',str(source),'--output',str(dest),'--count','8'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  status['started_pid']=proc.pid
 else:status['status']='PREFLIGHT_BLOCKED_REVIEW_GPU_DISK_OR_PACKAGES'
print(json.dumps(status,ensure_ascii=False))'''
    try:
        result=rt.execute(code,timeout=90,echo=True)
        return 0 if result.get('status')=='ok' else 1
    finally:rt.close()


if __name__=='__main__':sys.exit(main())
