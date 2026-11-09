"""Small reports only: pretrained mALBERT training on SSH."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

REMOTE='/home/edabk/hoangbpm/turn_detect/outputs/multilingual_malbert_vi_en_20261103_v2'
LOCAL=Path('outputs/multilingual_malbert_vi_en_20261103_v2')


def main():
    assert shutil.disk_usage(Path.cwd()).free>10*2**30+5*2**20
    code=f'''import pathlib,json,os,subprocess
r=pathlib.Path({REMOTE!r});v={{'files':{{}},'process':{{'present':False}}}}
names=['launcher.json','status.json','error.json','migration_receipt.json','initialization_receipt.json','recipe.json','train_probe/final.json','cache/progress.json','cache/receipt.json','technical_fixture/verification.json','technical_fixture/int4_verification.json','first_checkpoint_verification.json','latest_dev.json','candidate_results.json','final_report.json']
for n in names:
 p=r/n
 if p.exists():
  assert p.stat().st_size<2**20
  v['files'][n]=json.loads(p.read_text())
l=v['files'].get('launcher.json')
if l:
 p=pathlib.Path('/proc')/str(l['pid']);v['process']['present']=p.exists()
 if p.exists():
  s=(p/'stat').read_text();t=s[s.rfind(')')+2:].split()[19]
  v['process'].update(pid=l['pid'],start_ticks=t,expected_ticks_match=t==l['start_ticks'],cmdline=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode())
v['runs']={{}}
for p in (r/'runs').glob('*'):
 v['runs'][p.name]={{}}
 for n in ['resume_receipt.json','best_report.json','training_report.json']:
  if (p/n).exists():v['runs'][p.name][n]=json.loads((p/n).read_text())
v['old_large_trainer_absent']=not pathlib.Path('/proc/2827094').exists()
v['old_compact_trainer_absent']=not pathlib.Path('/proc/427431').exists()
v['old_trilingual_malbert_absent']=not pathlib.Path('/proc/889316').exists()
v['gpu_processes']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
v['disks']={{str(p):os.statvfs(p).f_bavail*os.statvfs(p).f_frsize/2**30 for p in [r,pathlib.Path('/mnt/hdd500')]}}
v['mount']=subprocess.run(['findmnt','-n','-o','SOURCE,TARGET,OPTIONS','/mnt/hdd500'],capture_output=True,text=True).stdout.strip()
print(json.dumps(v))'''
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','edabk@100.91.209.124','python3 -'],input=code,capture_output=True,text=True,encoding='utf-8',timeout=45,check=True)
    d=json.loads(p.stdout);d['checked_utc']=datetime.now(timezone.utc).isoformat()
    d['windows_free_gib']=shutil.disk_usage(Path.cwd()).free/2**30
    data=(json.dumps(d,ensure_ascii=False,indent=2)+'\n').encode('utf-8');assert len(data)<5*2**20
    LOCAL.mkdir(parents=True,exist_ok=True);(LOCAL/'heartbeat_verification.json').write_bytes(data)
    for n,value in d['files'].items():
        path=LOCAL/n;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    path=LOCAL/'task_state.json';state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    state.update(checked_utc=d['checked_utc'],phase=d['files'].get('status.json',{}).get('stage','UNKNOWN'),
          process=d['process'],latest_status=d['files'].get('status.json'),source_spoken_v9_used=False,
          active_large_trainer_must_not_resume=True,model_parameters=11941122,first_checkpoint_verified=d['files'].get('first_checkpoint_verification.json'))
    final=d['files'].get('final_report.json',{})
    if final.get('complete'):state.update(complete=True,accepted=final['accepted'])
    path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:d[k] for k in ['checked_utc','process','disks','gpu_processes','old_large_trainer_absent']},ensure_ascii=False))
    print(json.dumps({n:d['files'].get(n) for n in ['status.json','error.json','first_checkpoint_verification.json','latest_dev.json','final_report.json']},ensure_ascii=False))


if __name__=='__main__':main()
