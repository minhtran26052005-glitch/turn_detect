"""Collect only compact reports from the <20M SSH branch."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

REMOTE = '/home/edabk/hoangbpm/turn_detect/outputs/multilingual_under20m_20261103_v1'
LOCAL = Path('outputs/multilingual_under20m_20261103_v1')


def main():
    assert shutil.disk_usage(Path.cwd()).free > 10*2**30+5*2**20
    code = f'''import json,pathlib,os,subprocess
r=pathlib.Path({REMOTE!r});v={{'files':{{}}}}
names=['launcher.json','status.json','error.json','architecture.json','recipe.json','first_checkpoint_verification.json','latest_dev.json','candidate_results.json','final_report.json','cache/progress.json','cache/receipt.json','technical_fixture/verification.json']
for name in names:
 p=r/name
 if p.exists():
  assert p.stat().st_size<2**20
  v['files'][name]=json.loads(p.read_text())
launch=v['files'].get('launcher.json')
if launch:
 p=pathlib.Path('/proc')/str(launch['pid'])
 v['process']={{'present':p.exists()}}
 if p.exists():
  s=(p/'stat').read_text();ticks=s[s.rfind(')')+2:].split()[19]
  v['process'].update(pid=launch['pid'],start_ticks=ticks,expected_ticks_match=ticks==launch['start_ticks'],cmdline=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode())
v['runs']={{}}
for run in (r/'runs').glob('*'):
 v['runs'][run.name]={{}}
 for name in ['resume_receipt.json','best_report.json','training_report.json']:
  if (run/name).exists():v['runs'][run.name][name]=json.loads((run/name).read_text())
v['main_status']=json.loads((r.parent/'multilingual_stream_v2_20261102/status.json').read_text())
v['gpu_processes']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True,capture_output=True).stdout.strip()
v['disks']={{str(p):os.statvfs(p).f_bavail*os.statvfs(p).f_frsize/2**30 for p in [r,pathlib.Path('/mnt/hdd500')]}}
v['mount']=subprocess.run(['findmnt','-n','-o','SOURCE,TARGET,OPTIONS','/mnt/hdd500'],text=True,capture_output=True).stdout.strip()
print(json.dumps(v))'''
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                        'edabk@100.91.209.124', 'python3 -'], input=code, capture_output=True,
                       text=True, encoding='utf-8', timeout=45, check=True)
    d = json.loads(p.stdout); d['checked_utc'] = datetime.now(timezone.utc).isoformat()
    d['windows_free_gib'] = shutil.disk_usage(Path.cwd()).free/2**30
    b = (json.dumps(d, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
    assert len(b) < 5*2**20
    LOCAL.mkdir(parents=True, exist_ok=True); (LOCAL/'heartbeat_verification.json').write_bytes(b)
    for name, value in d['files'].items():
        target = LOCAL/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    ledger = LOCAL/'task_state.json'
    state = json.loads(ledger.read_text(encoding='utf-8')) if ledger.exists() else {}
    state.update(checked_utc=d['checked_utc'], phase=d['files'].get('status.json', {}).get('stage', 'UNKNOWN'),
                 process=d.get('process'), latest_status=d['files'].get('status.json'),
                 first_checkpoint_verified=d['files'].get('first_checkpoint_verification.json'),
                 source_spoken_v9_used=False, active_main_model_replaced=False,
                 new_checkpoint_drive_backup_configured=False)
    if d['files'].get('final_report.json', {}).get('complete'):
        state['complete'] = True; state['accepted'] = d['files']['final_report.json']['accepted']
    ledger.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: d[k] for k in ['checked_utc','process','disks','main_status'] if k in d}, ensure_ascii=False))
    print(json.dumps({k: d['files'].get(k) for k in ['status.json','architecture.json','error.json','first_checkpoint_verification.json','final_report.json']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
