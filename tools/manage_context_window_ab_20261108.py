"""Launch/collect the user-authorized context A/B; bounded metadata only."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NAME = 'context_window_ab_20261108_v1'
OUT = ROOT/'outputs'/NAME
FILES = ['launcher.json', 'status.json', 'task_state.json', 'error.json', 'locked_protocol.json',
         'dataset_manifest.json', 'migration_receipt.json', 'independent_preparation_verification.json',
         'technical_fixture_verification.json', 'report.json', 'completion_verification.json',
         'paired_family_bootstrap.json', 'README.md', 'review.html', 'user_summary.json',
         'paired_dev_cases.json', 'review_verification.json', 'independent_metric_verification.json',
         'notification_verification.json']
for arm in ('predicted_reset', 'source_window'):
    FILES += [arm+'/'+name for name in ('status.json', 'resume_receipt.json', 'training_report.json',
              'first_training_verification.json', 'independent_final_training_verification.json',
              'epoch_0001_coverage.json', 'monitor.json', 'monitor_verification.json', 'selection_receipt.json')]


def remote(code, timeout=45):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             'edabk@100.91.209.124', 'python3 -'], input=code, text=True,
                            encoding='utf-8', capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('launch', 'status'))
    parser.add_argument('--brief', action='store_true')
    args = parser.parse_args()
    assert shutil.disk_usage('C:/').free > 10*2**30+5*2**20
    OUT.mkdir(exist_ok=True)
    if args.action == 'launch':
        raw = (ROOT/'turn_v2'/f'{NAME}.py').read_bytes()
        compile(raw.decode('utf-8'), NAME, 'exec')
        code = '''import base64,json,pathlib,hashlib,os,subprocess,time,shutil
r=pathlib.Path('/home/edabk/hoangbpm/turn_detect');p=r/'outputs/NAME';p.mkdir(exist_ok=True)
assert not (p/'launcher.json').exists() and not (p/'report.json').exists(),'Never duplicate existing run'
m=r/'outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7'
assert json.loads((m/'task_state.json').read_text())['desired_state']=='PAUSED_BY_USER_FOR_TEST'
assert (m/'STOP_REQUESTED').exists() and json.loads((m/'status.json').read_text())['stage']=='STOPPED_WITH_CHECKPOINT'
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==json.loads((m/'launcher.json').read_text())['boot_id']
for q in pathlib.Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:c=(q/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 except (OSError,UnicodeError):continue
 assert not c.startswith(str(r/'.venv/bin/python')),c
mount=subprocess.run(['findmnt','-no','SOURCE,UUID,OPTIONS','/mnt/hdd500'],capture_output=True,text=True,check=True).stdout
assert '/dev/sda1' in mount and '37622A5A7A5A4B57' in mount and 'rw,' in mount
assert min(shutil.disk_usage(r).free,shutil.disk_usage('/mnt/hdd500').free)>=13*2**30
b=base64.b64decode(__SOURCE_B64__);dst=r/'turn_v2/NAME.py'
if dst.exists():assert dst.read_bytes()==b,'Never edit active/fingerprinted source'
else:dst.write_bytes(b)
assert hashlib.sha256(dst.read_bytes()).hexdigest()==hashlib.sha256(b).hexdigest()
env=os.environ.copy();env.update(LD_LIBRARY_PATH='/home/edabk/miniconda3/envs/.VLLM/lib/python3.10/site-packages/nvidia/cu13/lib',PYTHONPATH=str(r/'outputs/multilingual_tiny_compression_20261102_v1/runtime_deps')+':'+str(r),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
with (p/'launch.log').open('ab',buffering=0) as log:
 worker=subprocess.Popen([str(r/'.venv/bin/python'),'-u','-m','turn_v2.NAME'],cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
time.sleep(.4);q=pathlib.Path('/proc',str(worker.pid));assert q.exists();s=(q/'stat').read_text()
launcher=dict(pid=worker.pid,start_ticks=s[s.rfind(')')+2:].split()[19],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),module='turn_v2.NAME',source_sha256=hashlib.sha256(b).hexdigest())
(p/'launcher.json').write_text(json.dumps(launcher,indent=2)+'\\n')
(p/'task_state.json').write_text(json.dumps(dict(complete=False,reported=False,phase='RUNNING',parent_paused=True,adopted=False,auto_extend_authorized=False))+'\\n')
print(json.dumps(launcher))
'''.replace('NAME', NAME).replace('__SOURCE_B64__', repr(base64.b64encode(raw).decode()))
        result = remote(code)
        (OUT/'launcher.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(result)); return
    code = '''import pathlib,json,base64,hashlib,shutil,subprocess,datetime
r=pathlib.Path('/home/edabk/hoangbpm/turn_detect');p=r/'outputs/NAME';files={};total=0
for name in FILES:
 q=p/name
 if q.exists():
  b=q.read_bytes();assert len(b)<=2*2**20;total+=len(b);assert total<=5*2**20
  files[name]=dict(base64=base64.b64encode(b).decode(),sha256=hashlib.sha256(b).hexdigest())
launch=json.loads((p/'launcher.json').read_text());q=pathlib.Path('/proc',str(launch['pid']));process=dict(present=q.exists())
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
if q.exists():
 raw=(q/'stat').read_text();v=raw[raw.rfind(')')+2:].split();cmd=(q/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 process.update(pid=launch['pid'],start_ticks=v[19],cpu_ticks=int(v[11])+int(v[12]),cmdline=cmd,identity_match=(v[19]==str(launch['start_ticks']) and boot==launch['boot_id'] and launch['module'] in cmd))
workers=[]
for q in pathlib.Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:c=(q/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 except (OSError,UnicodeError):continue
 if c.startswith(str(r/'.venv/bin/python')):workers.append(dict(pid=int(q.name),cmdline=c))
m=r/'outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7'
proof=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),boot_id=boot,process=process,project_workers=workers,main_stop_requested=(m/'STOP_REQUESTED').exists(),main_status=json.loads((m/'status.json').read_text())['stage'],ssd_free_gib=shutil.disk_usage(r).free/2**30,hdd_free_gib=shutil.disk_usage('/mnt/hdd500').free/2**30,mount=subprocess.run(['findmnt','-no','SOURCE,UUID,OPTIONS','/mnt/hdd500'],capture_output=True,text=True).stdout,gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout)
print(json.dumps(dict(files=files,proof=proof)))
'''.replace('NAME', NAME).replace('FILES', repr(FILES))
    result = remote(code)
    for name, item in result['files'].items():
        raw = base64.b64decode(item['base64'])
        assert hashlib.sha256(raw).hexdigest() == item['sha256']
        path = OUT/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    proof = result['proof']; proof['windows_free_gib'] = shutil.disk_usage('C:/').free/2**30
    (OUT/'heartbeat_verification.json').write_text(json.dumps(proof, indent=2)+'\n', encoding='utf-8')
    summary = dict(proof=proof)
    for name in ('status.json', 'error.json', 'task_state.json', 'predicted_reset/status.json', 'source_window/status.json'):
        path = OUT/name
        if path.exists():
            summary[name] = json.loads(path.read_text(encoding='utf-8'))
    if args.brief:
        arm_progress = {}
        for arm in ('predicted_reset', 'source_window'):
            final_path = OUT/arm/'training_report.json'
            if final_path.exists():
                final = json.loads(final_path.read_text(encoding='utf-8'))
                arm_progress[arm] = dict(stage='FULL_EPOCH_VERIFIED' if (OUT/arm/'independent_final_training_verification.json').exists() else 'FULL_EPOCH_CHECKING',
                                         updates=final['updates'], seen=final['seen'], complete=final['complete'])
            else:
                arm_progress[arm] = summary.get(arm+'/status.json')
        summary = dict(at_utc=proof['at_utc'], process=proof['process'],
                       worker_count=len(proof['project_workers']),
                       stage=summary.get('status.json'),
                       arms=arm_progress,
                       error=summary.get('error.json'), complete=summary.get('task_state.json', {}).get('complete'),
                       free_gib=dict(ssd=proof['ssd_free_gib'], hdd=proof['hdd_free_gib'], windows=proof['windows_free_gib']))
        summary['process'].pop('cmdline', None)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
