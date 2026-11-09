"""Read compact SSH compression status/reports; never download weights/cache."""
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime,timezone

REMOTE='/home/edabk/hoangbpm/turn_detect/outputs/multilingual_tiny_compression_20261102_v1'
LOCAL=Path('outputs/multilingual_tiny_compression_20261102_v1')


def main():
    code=f'''import json,pathlib,os,subprocess
r=pathlib.Path({REMOTE!r});names=['launcher.json','preparation_launcher.json','status.json','preparation_status.json','error.json','preparation_error.json','teacher_receipt.json','train_receipt.json','vocabulary_report.json','technical_checks.json','teacher_dev.json','candidate_results.json','cpu_benchmark.json','final_report.json']
value={{'files':{{}},'processes':{{}},'disk_free_gib':{{}}}}
for n in names:
 p=r/n
 if p.exists():value['files'][n]=json.loads(p.read_text())
for n in ['launcher.json','preparation_launcher.json']:
 if n not in value['files']:continue
 expected=value['files'][n];p=pathlib.Path('/proc')/str(expected['pid'])
 if not p.exists():value['processes'][n]={{'present':False}};continue
 raw=(p/'stat').read_text();ticks=raw[raw.rfind(')')+2:].split()[19]
 value['processes'][n]={{'present':True,'pid':expected['pid'],'start_ticks':ticks,'expected_ticks_match':ticks==expected['start_ticks'],'cmdline':(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()}}
for path in [str(r),'/mnt/hdd500']:
 s=os.statvfs(path);value['disk_free_gib'][path]=s.f_bavail*s.f_frsize/2**30
value['hdd_mount']=subprocess.run(['findmnt','-n','-o','SOURCE,TARGET,OPTIONS','/mnt/hdd500'],capture_output=True,text=True).stdout.strip()
value['gpu_processes']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
if (r/'cache/progress.json').exists():value['cache_progress']=json.loads((r/'cache/progress.json').read_text())
c=r.parent/'multilingual_transformer_compression_20261102_v1'
if c.exists():
 value['transformer_control']={{}}
 for n in ['launcher.json','status.json','error.json','candidate_results.json','cpu_benchmark.json','final_report.json','package_verification.json']:
  if (c/n).exists():value['transformer_control'][n]=json.loads((c/n).read_text())
 if (c/'launcher.json').exists():
  expected=value['transformer_control']['launcher.json'];p=pathlib.Path('/proc')/str(expected['pid'])
  if p.exists():
   raw=(p/'stat').read_text();ticks=raw[raw.rfind(')')+2:].split()[19]
   value['transformer_control']['process']={{'present':True,'pid':expected['pid'],'start_ticks':ticks,'expected_ticks_match':ticks==expected['start_ticks'],'cmdline':(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()}}
  else:value['transformer_control']['process']={{'present':False}}
value['run_summaries']={{}}
for run in sorted((r/'runs').glob('*')):
 p=run/'training_report.json'
 if p.exists():
  v=json.loads(p.read_text());v=v['selected'];value['run_summaries'][run.name]={{'epoch':v['epoch'],'step':v['step'],'retention_passed':v['retention_passed'],'by_language':v['dev']['by_language']}}
print(json.dumps(value))'''
    proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','edabk@100.91.209.124','python3 -'],input=code,capture_output=True,text=True,encoding='utf-8',timeout=45,check=True)
    d=json.loads(proc.stdout);d['checked_utc']=datetime.now(timezone.utc).isoformat();d['windows_free_gib']=shutil.disk_usage(Path.cwd()).free/2**30
    encoded=json.dumps(d,ensure_ascii=False,indent=2)+'\n';assert len(encoded.encode())<5*2**20
    LOCAL.mkdir(exist_ok=True,parents=True);(LOCAL/'heartbeat_verification.json').write_text(encoded,encoding='utf-8')
    if d['windows_free_gib']>=10+5/1024:
        for n,v in d['files'].items():(LOCAL/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        if 'transformer_control' in d:
            target=LOCAL.parent/'multilingual_transformer_compression_20261102_v1';target.mkdir(exist_ok=True)
            for n,v in d['transformer_control'].items():
                if n.endswith('.json'):(target/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    state_path=LOCAL/'task_state.json'
    if state_path.exists():
        state=json.loads(state_path.read_text(encoding='utf-8'))
        state.update(checked_utc=d['checked_utc'],phase=d['files'].get('status.json',{}).get('stage','UNKNOWN'),
                     preparation_complete=d['files'].get('preparation_status.json',{}).get('stage')=='COMPLETE',
                     latest_pipeline_status=d['files'].get('status.json'),pipeline_process_verified=d['processes'].get('launcher.json'),
                     transformer_control_status=d.get('transformer_control',{}).get('status.json'),
                     transformer_control_process=d.get('transformer_control',{}).get('process'))
        tiny_final=d['files'].get('final_report.json',{})
        control_final=d.get('transformer_control',{}).get('final_report.json',{})
        if tiny_final.get('accepted'):
            state.update(phase='COMPLETE_ACCEPTED_TINY',accepted_family='tiny_GRU_KD_QAT')
        elif control_final.get('accepted'):
            state.update(phase='COMPLETE_ACCEPTED_TRANSFORMER_INT8_CONTROL',accepted_family='original_architecture_PTQ_control',
                         tiny_retention_passed=False,pending_transformer_control=False)
        state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary={k:d['files'].get(k) for k in ['launcher.json','status.json','preparation_status.json','error.json','preparation_error.json']}
    if 'final_report.json' in d['files']:
        f=d['files']['final_report.json'];summary['final']={k:f.get(k) for k in ['complete','accepted','package','archive_sha256','winner','reason']}
    summary.update(processes=d['processes'],cache_progress=d.get('cache_progress'),disk=d['disk_free_gib'])
    if 'transformer_control' in d:summary['transformer_control']={k:d['transformer_control'].get(k) for k in ['status.json','error.json','process']}
    control_final=d.get('transformer_control',{}).get('final_report.json',{})
    if control_final.get('accepted'):summary['accepted_compression']={'family':'original_architecture_PTQ_control','package':control_final['package'],'model_bytes':control_final['winner']['model_bytes'],'archive_sha256':control_final['archive_sha256']}
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
