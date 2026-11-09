"""Control the three-language spoken-text campaign in the current a.ipynb."""
import argparse
import base64
import json
from pathlib import Path
import sys
from colab_runtime import ColabRuntime
from factory_spoken_session import select_current_a_session, SessionSelectionError

REPO=Path(__file__).resolve().parents[1]
SOURCES=['factory_spoken_campaign_v9.py','factory_spoken_spec_v1.py','factory_spoken_topics_v1.py','factory_paragraphs_v4.py']


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','start','stop','stop-legacy']);a=p.parse_args()
    state_path=REPO/'outputs/factory_spoken_campaign_v9/task_state.json'
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    if a.action!='status' and state.get('must_not_auto_resume') and state.get('superseded_by')=='factory_spoken_campaign_v10':
        print(json.dumps({'stage':'SUPERSEDED_BY_BILINGUAL_V10','use':'tools/factory_spoken_colab_v10.py','action_refused':a.action}));return 2
    try:i,s,selection=select_current_a_session()
    except SessionSelectionError as error:print(json.dumps(error.details));return 2
    payload={n:base64.b64encode((REPO/'turn_v2'/n).read_bytes()).decode() for n in SOURCES} if a.action=='start' else {}
    code=f'''import pathlib,os,json,subprocess,shutil,sys,time,hashlib,base64,importlib.util,importlib.metadata
root=pathlib.Path('/content/factory_spoken_campaign_v9')
drive=pathlib.Path('/content/drive/MyDrive/turn_detect_backups/translation_segmenter_v15d_20261026')
out=drive/'factory_spoken_campaign_v9';legacy=drive/'factory_dialogue_campaign_v6'
v={{'action':{a.action!r},'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'drive_mounted':os.path.ismount('/content/drive'),'colab_free_gib':round(shutil.disk_usage('/content').free/2**30,3),'processes':[],'session_selection':{selection!r}}}
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  cmd=p.read_bytes().replace(b'\\0',b' ').decode(errors='replace')
  if '/content/' in cmd and any(x in cmd for x in ['/factory_dialogue_campaign_v','/factory_spoken_campaign_v','/factory_paragraphs_v','/factory_paragraphs_zh_v','/factory_consult_dialogue_v','/factory_dialogue_pilot_v']):
   raw=(p.parent/'stat').read_text();v['processes'].append({{'pid':int(p.parent.name),'start_ticks':raw[raw.rfind(')')+2:].split()[19],'cmdline':cmd[:400]}})
 except OSError:pass
v['gpu']=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free,utilization.gpu,name','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20).stdout.strip()
v['dependencies']={{n:importlib.metadata.version(n) if importlib.util.find_spec(n) else None for n in ['torch','transformers','accelerate','safetensors','jieba']}}
v['identity_verified']=False
if v['drive_mounted'] and (drive/'destination_identity.json').exists():
 identity=json.loads((drive/'destination_identity.json').read_text());v['identity_verified']=identity.get('nonce')=='9e618b0f2db442a6a9ee3ee18f1d7143' and identity.get('intended_path')==str(drive)
for prefix,folder in [('spoken',out),('legacy',legacy)]:
 for name in ['owner.json','state.json','status.json']:
  p=folder/name
  if p.exists():v[prefix+'_'+name]=json.loads(p.read_text())
for n in ['status.json','error.json','launcher.json']:
 if (root/n).exists():v[n]=json.loads((root/n).read_text())
if {a.action!r}=='stop-legacy':
 assert v['identity_verified'],'Drive identity must be verified before stopping legacy worker'
 old=[x for x in v['processes'] if '/factory_dialogue_campaign_v8.py' in x['cmdline']]
 if old:
  owner=json.loads((legacy/'owner.json').read_text());assert owner['boot_id']==v['boot_id']
  assert len(old)==1 and old[0]['pid']==owner['pid'] and old[0]['start_ticks']==owner['start_ticks'],'Legacy identity mismatch'
  data=b'User requested regenerating all three languages as spoken transcripts; stop after committing current batch.\\n'
  for path in [legacy/'STOP_REQUESTED',pathlib.Path('/content/factory_dialogue_campaign_v8/STOP_REQUESTED')]:
   with path.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  v['legacy_stop_requested']=old[0]
 else:
  assert legacy.exists(),'No legacy output exists'
  data=b'User requested new spoken VI/EN/ZH campaign. Stop old generation after committing pending batch. Previous VM liveness remains unverified.\\n'
  with (legacy/'STOP_REQUESTED').open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  assert (legacy/'STOP_REQUESTED').read_bytes()==data
  v['legacy_stop_requested_on_drive']=True;v['legacy_worker_liveness']='UNKNOWN_OTHER_VM_NO_LOCAL_PROCESS'
if {a.action!r}=='stop':
 assert v['identity_verified'] and out.exists()
 for path in [out/'STOP_REQUESTED',root/'STOP_REQUESTED']:
  with path.open('w') as f:f.write('Explicit user stop request.\\n');f.flush();os.fsync(f.fileno())
 v['stop_requested']=True
if {a.action!r}=='start':
 assert v['identity_verified'],'Mount existing Drive destination first'
 assert not v['processes'],'Existing generation worker must finish/stop first; do not duplicate'
 assert all(v['dependencies'].values()),'Missing generation dependencies'
 assert v['colab_free_gib']>=43,'Keep Colab disk reserve'
 gpu=v['gpu'].splitlines()[0].split(',');assert int(gpu[0].strip())>=22528 and int(gpu[1].strip())>=19456,'Insufficient free GPU memory'
 assert not (out/'STOP_REQUESTED').exists() and not (root/'STOP_REQUESTED').exists(),'Stopped campaign needs explicit resume'
 owner=json.loads((out/'owner.json').read_text()) if (out/'owner.json').exists() else None
 if owner and not owner.get('stopped'):
  assert owner['boot_id']==v['boot_id'],'Previous spoken campaign owner unknown; do not overwrite'
 root.mkdir(exist_ok=True);source_dir=out/'source_v9';source_dir.mkdir(parents=True,exist_ok=True)
 for n,b64 in json.loads({json.dumps(json.dumps(payload))}).items():
  data=base64.b64decode(b64);compile(data,n,'exec');p=source_dir/n
  if p.exists():assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(data).digest(),'Immutable source mismatch'
  else:
   with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
   assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(data).digest()
  (root/n).write_bytes(data)
 with (root/'worker.log').open('ab') as log:proc=subprocess.Popen([sys.executable,'-u',str(root/'factory_spoken_campaign_v9.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 time.sleep(2);assert proc.poll() is None,'Worker exited; inspect version9 error/log'
 raw=pathlib.Path('/proc/'+str(proc.pid)+'/stat').read_text();launch={{'pid':proc.pid,'start_ticks':raw[raw.rfind(')')+2:].split()[19],'boot_id':v['boot_id'],'module':'factory_spoken_campaign_v9','output':str(out)}}
 (root/'launcher.json').write_text(json.dumps(launch,indent=2));v['launched']=launch
print(json.dumps(v,ensure_ascii=False))'''
    rt=ColabRuntime(server_index=i,session_path=s['path']);rt.kernel_id=s['kernel']['id']
    try:result=rt.execute(code,timeout=90,echo=True);return 0 if result['status']=='ok' else 1
    finally:rt.close()


if __name__=='__main__':sys.exit(main())
