"""Current-session control of bilingual spoken generation, preserving v9 commits."""
import argparse
import base64
import json
from pathlib import Path
import sys
from colab_runtime import ColabRuntime
from factory_spoken_session import select_owned_a_session,SessionSelectionError

REPO=Path(__file__).resolve().parents[1]
SOURCES=['factory_spoken_campaign_v10.py','factory_spoken_spec_v2.py','factory_spoken_spec_v1.py','factory_spoken_topics_v1.py','factory_paragraphs_v4.py']


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','transition-v9','start','stop']);a=p.parse_args()
    authorization=json.loads((REPO/'outputs/factory_spoken_campaign_v10/task_state.json').read_text(encoding='utf-8'))
    if a.action!='status':
        assert authorization['authorized_active_languages']==['vi','en'] and authorization['language_change_user_authorized']
    try:i,session,selection=select_owned_a_session(authorization.get('actual_colab_boot_id',authorization['authorized_transition_boot']))
    except SessionSelectionError as e:print(json.dumps(e.details));return 2
    payload={n:base64.b64encode((REPO/'turn_v2'/n).read_bytes()).decode() for n in SOURCES} if a.action in ('start','transition-v9') else {}
    code=r'''import pathlib,json,os,subprocess,time,hashlib,base64,shutil,sys
action=__ACTION__;payload=__PAYLOAD__;authorization=__AUTH__
root=pathlib.Path('/content/factory_spoken_campaign_v10');previous=pathlib.Path('/content/factory_spoken_campaign_v9')
drive=pathlib.Path('/content/drive/MyDrive/turn_detect_backups/translation_segmenter_v15d_20261026');out=drive/'factory_spoken_campaign_v9'
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
v={'action':action,'boot_id':boot,'drive_mounted':os.path.ismount('/content/drive'),'processes':[],'colab_free_gib':shutil.disk_usage('/content').free/2**30}
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  cmd=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if '/content/' in cmd and any(x in cmd for x in ['/factory_dialogue_campaign_v','/factory_spoken_campaign_v','/factory_paragraphs_v','/factory_consult_dialogue_v','/factory_dialogue_pilot_v']):
   raw=(p.parent/'stat').read_text();v['processes'].append({'pid':int(p.parent.name),'start_ticks':raw[raw.rfind(')')+2:].split()[19],'cmdline':cmd[:400]})
 except OSError:pass
v['gpu']=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free,utilization.gpu,name','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
v['identity_verified']=False
if v['drive_mounted'] and (drive/'destination_identity.json').exists():
 identity=json.loads((drive/'destination_identity.json').read_text());v['identity_verified']=identity.get('nonce')=='9e618b0f2db442a6a9ee3ee18f1d7143' and identity.get('intended_path')==str(drive)
for name in ['owner.json','state.json','status.json','config_v10.json','language_policy_v10.json']:
 if (out/name).exists():v['drive_'+name]=json.loads((out/name).read_text())
for name in ['launcher.json','status.json','error.json']:
 if (root/name).exists():v[name]=json.loads((root/name).read_text())
def digest(b):return hashlib.sha256(b).hexdigest()
def immutable(path,b):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():assert path.read_bytes()==b;return digest(b)
 with path.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 assert path.read_bytes()==b;return digest(b)
def js(v):return (json.dumps(v,ensure_ascii=False,sort_keys=True)+'\n').encode()
if action=='stop':
 assert v['identity_verified'] and v['drive_owner.json']['boot_id']==boot
 owner=v['drive_owner.json'];assert owner['module']=='factory_spoken_campaign_v10'
 assert any(p['pid']==owner['pid'] and p['start_ticks']==owner['start_ticks'] for p in v['processes'])
 for path in [out/'STOP_REQUESTED',root/'STOP_REQUESTED']:
  with path.open('w') as f:f.write('Explicit user stop request for bilingual campaign.\n');f.flush();os.fsync(f.fileno())
 v['stop_requested']=True
if action in ('transition-v9','start'):
 assert v['identity_verified'] and v['drive_mounted']
 assert not v['processes'],'Generation process still active; wait for commit/stop, never duplicate'
 owner=v.get('drive_owner.json');assert owner is not None
 if not owner.get('stopped'):assert owner['boot_id']==boot,'Prior owner unknown on another VM'
 if action=='transition-v9':
  assert boot==authorization['authorized_transition_boot']=='eb59f31a-2c7b-4231-a29a-aa835d106e3d'
  assert owner['module']=='factory_spoken_campaign_v9' and owner['pid']==10402 and owner['start_ticks']=='245097' and owner.get('stopped')
  assert not pathlib.Path('/proc/10402').exists()
  assert json.loads((previous/'status.json').read_text())['stage']=='STOPPED_BY_REQUEST'
  for marker in (previous/'pending').glob('*/commit.json'):
   raw=marker.read_bytes();c=json.loads(raw);p=out/'commits'/('%09d.json'%c['index']);assert p.exists() and digest(p.read_bytes())==digest(raw),'Uncommitted pending batch preserved; resolve it before handover'
  baseline=json.loads((out/'state.json').read_text());cp=out/'commits'/('%09d.json'%baseline['commit_index']);raw=cp.read_bytes();c=json.loads(raw)
  assert digest(raw)==baseline['last_commit_sha256'] and digest((out/c['object']).read_bytes())==c['object_sha256']
  old_hashes={'factory_spoken_campaign_v9.py':'703bb102d54d78440e2b94a359f0f5fe5fdecaaf6b2b307dd8ea37e6a8768101','factory_spoken_spec_v1.py':'1ec84d29e30fff3f259165a1b473a112bcf44f0b09915155c2fa7264ca6a0391','factory_spoken_topics_v1.py':'50af8f498dae74dd608b3b542d3bbb345068cf742c7b5230d3cccb94a61c9ff9','factory_paragraphs_v4.py':'a7f057ce13d9172da12d82f37b2fbad1a27c1bb98c3595faf46fd81dcf7fcaeb'}
  for n,h in old_hashes.items():assert digest((out/'source_v9'/n).read_bytes())==h
 root.mkdir(exist_ok=True);source=out/'source_v10';source.mkdir(exist_ok=True);source_hashes={}
 for name,encoded in payload.items():
  b=base64.b64decode(encoded);compile(b,name,'exec');source_hashes[name]=immutable(source/name,b);(root/name).write_bytes(b)
 if action=='transition-v9':
  policy={'version':10,'active_languages':['vi','en'],'user_authorized':True,'user_instruction':'Drop Chinese everywhere; focus English and Vietnamese','quota':10000,'repeat_until_user_stop':True,'baseline_commit':baseline['commit_index'],'baseline_commit_sha256':baseline['last_commit_sha256'],'baseline_counts':baseline['total_counts'],'destination_nonce':'9e618b0f2db442a6a9ee3ee18f1d7143'}
  policy_sha=immutable(out/'language_policy_v10.json',js(policy))
  handover={'old_owner':owner,'same_boot_verified':boot,'old_process_absent':True,'old_graceful_stop_verified':True,'user_authorized_language_change':True,'active_languages':['vi','en'],'baseline':baseline,'baseline_last_object_sha256':c['object_sha256'],'baseline_last_object_readback_verified':True,'new_source_sha256':source_hashes,'policy_sha256':policy_sha,'counts_not_reset':True}
  receipt=out/'handover'/'engine_v9_to_v10_vi_en_eb59f31a.json';receipt_sha=immutable(receipt,js(handover))
  stop=out/'STOP_REQUESTED';archive=out/'handover'/'STOP_REQUESTED_v9_to_v10_vi_en.txt'
  assert stop.exists() and (previous/'STOP_REQUESTED').exists()
  if archive.exists():assert archive.read_bytes()==stop.read_bytes()
  os.replace(stop,archive);assert archive.exists() and not stop.exists()
  v['transition']={'receipt':str(receipt),'sha256':receipt_sha,'baseline':baseline,'source_sha256':source_hashes,'policy_sha256':policy_sha}
 assert json.loads((out/'language_policy_v10.json').read_text())['active_languages']==['vi','en']
 assert not (out/'STOP_REQUESTED').exists() and not (root/'STOP_REQUESTED').exists(),'Explicit stopped campaign cannot auto-resume'
 assert v['colab_free_gib']>=43
 gpu=v['gpu'].splitlines()[0].split(',');assert int(gpu[0])>=22528 and int(gpu[1])>=19456
 with (root/'worker.log').open('ab') as log:p=subprocess.Popen([sys.executable,'-u',str(root/'factory_spoken_campaign_v10.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 time.sleep(.8);proc=pathlib.Path('/proc')/str(p.pid);raw=(proc/'stat').read_text();launch={'pid':p.pid,'start_ticks':raw[raw.rfind(')')+2:].split()[19],'boot_id':boot,'module':'factory_spoken_campaign_v10','output':str(out),'active_languages':['vi','en']}
 with (root/'launcher.json').open('w') as f:json.dump(launch,f,indent=2);f.flush();os.fsync(f.fileno())
 v['launched']=launch
print(json.dumps(v,ensure_ascii=False))'''
    code=code.replace('__ACTION__',repr(a.action)).replace('__PAYLOAD__',repr(payload)).replace('__AUTH__',repr(authorization))
    rt=ColabRuntime(server_index=i,session_path=session['path']);rt.kernel_id=session['kernel']['id']
    try:result=rt.execute(code,timeout=90,echo=False)
    finally:rt.close()
    if result['status']!='ok':
        print(json.dumps({'stage':'COLAB_ACTION_ERROR','action':a.action,'errors':[x.get('evalue') for x in result['outputs'] if x['output_type']=='error']}));return 1
    raw=''.join(x.get('text','') for x in result['outputs'] if x['output_type']=='stream');value=json.loads(raw);value['session_selection']=selection
    local=REPO/'outputs/factory_spoken_campaign_v10';local.mkdir(exist_ok=True)
    (local/(a.action.replace('-','_')+'_receipt.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value,ensure_ascii=False));return 0


if __name__=='__main__':sys.exit(main())
