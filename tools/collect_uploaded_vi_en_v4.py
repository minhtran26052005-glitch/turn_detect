"""Fetch small manifests/reports only and verify the actual uploaded-data worker."""
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import shutil
import subprocess

REMOTE='/home/edabk/hoangbpm/turn_detect/outputs/multilingual_malbert_vi_en_uploaded_20261104_v4'
LOCAL=Path(__file__).resolve().parents[1]/'outputs/multilingual_malbert_vi_en_uploaded_20261104_v4'


def main():
    assert shutil.disk_usage(LOCAL.parent).free>10*2**30+5*2**20
    code=f'''import json,pathlib,shutil,subprocess
r=pathlib.Path({REMOTE!r});v={{'files':{{}},'process':{{'present':False}}}}
names=['launcher.json','status.json','error.json','preparation_status.json','dataset_manifest.json','source_audit.json','independent_source_verification.json','independent_causal_training_verification.json','migration_receipt.json','token_digest_repair_receipt.json','independent_training_verification.json','warm_start_new_dev_baseline.json','latest_dev.json','candidate_results.json','final_report.json','teacher_new_dev.json','recovery_launcher.json','recovery_status.json','recovery_error.json','recovery_training_verification.json']
for n in names:
 p=r/n
 if p.exists():
  assert p.stat().st_size<2**20
  v['files'][n]=json.loads(p.read_text())
l=v['files'].get('launcher.json')
if l:
 p=pathlib.Path('/proc')/str(l['pid']);v['process']['present']=p.exists()
 if p.exists():
  raw=(p/'stat').read_text();ticks=raw[raw.rfind(')')+2:].split()[19];cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
  boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
  v['process'].update(pid=l['pid'],start_ticks=ticks,cmdline=cmd,boot_id=boot,identity_match=ticks==l['start_ticks'] and boot==l['boot_id'] and l['module'] in cmd)
v['runs']={{}}
for folder in (r/'runs').glob('*'):
 v['runs'][folder.name]={{}}
 for name in ['resume_receipt.json','best_report.json','training_report.json']:
  p=folder/name
  if p.exists():v['runs'][folder.name][name]=json.loads(p.read_text())
v['gpu_processes']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
v['host_boot_id']=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
v['hdd_mounted']=pathlib.Path('/mnt/hdd500').is_mount()
v['disks']={{str(r):shutil.disk_usage(r).free/2**30,'/mnt/hdd500':shutil.disk_usage('/mnt/hdd500').free/2**30 if v['hdd_mounted'] else None}}
v['mount']=subprocess.run(['findmnt','-n','-o','SOURCE,TARGET,OPTIONS','/mnt/hdd500'],capture_output=True,text=True).stdout.strip()
v['recovery_process']={{'present':False}}
recovery=v['files'].get('recovery_launcher.json')
if recovery:
 p=pathlib.Path('/proc')/str(recovery['pid'])
 if p.exists():
  raw=(p/'stat').read_text();ticks=raw[raw.rfind(')')+2:].split()[19];cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
  v['recovery_process']=dict(present=True,pid=recovery['pid'],start_ticks=ticks,cmdline=cmd,identity_match=ticks==recovery['start_ticks'] and recovery['boot_id']==v['host_boot_id'] and 'turn_v2.uploaded_vi_en_recover_after_reboot_v1' in cmd)
v['old_parent_process_absent']=not pathlib.Path('/proc/967949').exists()
print(json.dumps(v))'''
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','edabk@100.91.209.124','python3 -'],input=code,text=True,capture_output=True,check=True,encoding='utf-8',timeout=60)
    value=json.loads(p.stdout);value.update(checked_utc=datetime.now(timezone.utc).isoformat(),windows_free_gib=shutil.disk_usage(LOCAL.parent).free/2**30)
    data=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8');assert len(data)<4*2**20
    LOCAL.mkdir(exist_ok=True);(LOCAL/'heartbeat_verification.json').write_bytes(data)
    for name,item in value['files'].items():(LOCAL/name).write_text(json.dumps(item,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    path=LOCAL/'task_state.json';state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    latest=value['files'].get('status.json',value['files'].get('preparation_status.json',{}))
    phase=latest.get('stage','UNKNOWN')
    if not value['process']['present']:
        if value['recovery_process'].get('identity_match'):
            latest=value['files'].get('recovery_status.json',latest);phase=latest.get('stage','RECOVERY_PROCESS_ACTIVE')
        elif value['files'].get('launcher.json',{}).get('boot_id')!=value['host_boot_id']:
            phase='INTERRUPTED_AFTER_HOST_REBOOT'
        else:phase='WORKER_NOT_RUNNING'
    state.pop('dataset_manifest_sha_verified_locally_from_remote_receipt',None)
    state.update(phase=phase,process=value['process'],checked_utc=value['checked_utc'],
        desired_state='ACTIVE',active_languages=['vi','en'],drive_data_excluded=True,parameters=11941122,
        latest_status=latest,host_boot_id=value['host_boot_id'],hdd_mounted=value['hdd_mounted'],recovery_process=value['recovery_process'],dataset_manifest_collected_from_remote_snapshot=True,
        compression_accepted=value['files'].get('final_report.json',{}).get('accepted',False))
    path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest=value['files'].get('dataset_manifest.json',{})
    dev=value['files'].get('latest_dev.json',{})
    cards=[]
    for r in dev.get('dev',{}).get('cases',[]):
        cards.append('<article><b>'+escape(r['language']+' · '+r['domain']+' · '+r['case'])+'</b><p>'+escape(r['context_tail'])+' <mark>'+escape(r['future'])+'</mark></p><small>p(EOT)='+str(round(r['p_eot'],5))+'</small></article>')
    review='<!doctype html><html lang=vi><meta charset=utf-8><title>Train dữ liệu tải lên · DEV mới</title><style>body{font:16px/1.6 system-ui;max-width:1000px;margin:30px auto;background:#f5f7fa;color:#234;padding:0 20px}pre,article{background:white;padding:18px;border:1px solid #dde3ed;border-radius:8px}pre{white-space:pre-wrap}mark{background:#d7f4e5}</style><h1>Việt / Anh · dữ liệu tải lên · DEV mới</h1><p>mALBERT 11.941.122 tham số. Không đưa replay Drive vào các batch mới. Dấu câu là nhãn silver, không phải human/audio gold. TRAIN và DEV theo nhóm nguồn; TEST giữ ngoài chọn model/ngưỡng.</p><pre>'+escape(json.dumps({'train':manifest.get('train'),'dev_monitor':manifest.get('dev_monitor'),'dev_calibration':manifest.get('dev_calibration'),'exclusions':manifest.get('exclusions'),'phase':latest.get('stage')},ensure_ascii=False,indent=2))+'</pre>'+''.join(cards)+'</html>'
    assert len(review.encode())<2**20;(LOCAL/'review.html').write_text(review,encoding='utf-8')
    print(json.dumps({'process':value['process'],'status':latest,'disks':value['disks'],'error':value['files'].get('error.json'),
        'train':manifest.get('train'),'dev_monitor':manifest.get('dev_monitor'),'new_training_verified':value['files'].get('independent_training_verification.json')},ensure_ascii=False))


if __name__=='__main__':main()
