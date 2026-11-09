"""Launch once, or collect only bounded reports for the DEV calibration audit."""
import argparse,base64,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAME='malbert_calibration_audit_20261107_v1'
FILES=['launcher.json','task_state.json','status.json','error.json','locked_protocol.json','selection_receipt.json',
       'report.json','completion_verification.json','selected_monitor_verification.json','selected_monitor.json']

def remote(code):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','edabk@100.91.209.124','python3 -'],
        input=code,encoding='utf-8',text=True,capture_output=True,timeout=45)
    if r.returncode:raise RuntimeError(r.stderr)
    return json.loads(r.stdout)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['launch','status']);args=ap.parse_args()
    assert shutil.disk_usage('C:/').free>10*2**30+5*2**20
    if args.action=='launch':
        source=(ROOT/'turn_v2'/f'{NAME}.py').read_bytes();compile(source.decode(),NAME,'exec')
        code='''import pathlib,json,base64,hashlib,subprocess,os,time,datetime,shutil
r=pathlib.Path('/home/edabk/hoangbpm/turn_detect');out=r/'outputs/NAME';out.mkdir(exist_ok=True)
assert not (out/'launcher.json').exists() and not (out/'report.json').exists(),'Inspect existing job, never duplicate'
mount=next(v.split() for v in pathlib.Path('/proc/mounts').read_text().splitlines() if v.split()[1]=='/mnt/hdd500')
assert mount[0]=='/dev/sda1' and 'rw' in mount[3].split(',')
assert min(shutil.disk_usage(r).free,shutil.disk_usage('/mnt/hdd500').free)>=13*2**30
for q in pathlib.Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:cmd=(q/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 except (OSError,UnicodeError):continue
 assert not (cmd.startswith(str(r/'.venv/bin/python')) and (' -m turn_v2.NAME' in cmd or ' -m turn_v2.multilingual_malbert_vi_en_sentence_mix_train_v7' in cmd or ' -m turn_v2.asr_only_adaptation_20261107_v1' in cmd)),cmd
b=base64.b64decode('SOURCE');dst=r/'turn_v2/NAME.py';dst.write_bytes(b);assert hashlib.sha256(dst.read_bytes()).hexdigest()==hashlib.sha256(b).hexdigest()
env=os.environ.copy();env.update(LD_LIBRARY_PATH='/home/edabk/miniconda3/envs/.VLLM/lib/python3.10/site-packages/nvidia/cu13/lib',PYTHONPATH=str(r/'outputs/multilingual_tiny_compression_20261102_v1/runtime_deps')+':'+str(r),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
with (out/'launch.log').open('ab',buffering=0) as log:
 p=subprocess.Popen([str(r/'.venv/bin/python'),'-u','-m','turn_v2.NAME'],cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
time.sleep(.4);proc=pathlib.Path('/proc',str(p.pid));assert proc.exists()
stat=(proc/'stat').read_text();launch=dict(pid=p.pid,start_ticks=stat[stat.rfind(')')+2:].split()[19],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),module='turn_v2.NAME',source_sha256=hashlib.sha256(b).hexdigest(),at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(out/'launcher.json').write_text(json.dumps(launch,indent=2)+'\\n');(out/'task_state.json').write_text(json.dumps(dict(phase='RUNNING',complete=False,reported=False,parent_paused=True,auto_resume_authorized=False))+'\\n')
print(json.dumps(launch))
'''.replace('NAME',NAME).replace('SOURCE',base64.b64encode(source).decode())
        launch=remote(code);out=ROOT/'outputs'/NAME;out.mkdir(exist_ok=True);(out/'launcher.json').write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch));return
    code='''import pathlib,json,base64,shutil,datetime,hashlib,subprocess
r=pathlib.Path('/home/edabk/hoangbpm/turn_detect');p=r/'outputs/NAME';files={};size=0
for n in FILES:
 q=p/n
 if q.exists():
  assert q.stat().st_size<=2*2**20;b=q.read_bytes();size+=len(b);assert size<=5*2**20
  files[n]={'base64':base64.b64encode(b).decode(),'sha256':hashlib.sha256(b).hexdigest()}
launch=json.loads((p/'launcher.json').read_text());proc=pathlib.Path('/proc',str(launch['pid']));boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip();process={'present':proc.exists()}
if proc.exists():
 stat=(proc/'stat').read_text();v=stat[stat.rfind(')')+2:].split();cmd=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 process.update(pid=launch['pid'],start_ticks=v[19],cpu_ticks=int(v[11])+int(v[12]),cmdline=cmd,identity_match=(v[19]==str(launch['start_ticks']) and boot==launch['boot_id'] and launch['module'] in cmd))
mount=[v.split() for v in pathlib.Path('/proc/mounts').read_text().splitlines() if v.split()[1]=='/mnt/hdd500']
proof=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),boot_id=boot,process=process,ssd_free_gib=shutil.disk_usage(r).free/2**30,hdd_free_gib=shutil.disk_usage('/mnt/hdd500').free/2**30 if mount else None,hdd_mount=mount,gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout)
print(json.dumps({'files':files,'proof':proof}))
'''.replace('NAME',NAME).replace('FILES',repr(FILES))
    x=remote(code);out=ROOT/'outputs'/NAME;out.mkdir(exist_ok=True)
    for n,v in x['files'].items():
        b=base64.b64decode(v['base64']);assert hashlib.sha256(b).hexdigest()==v['sha256'];(out/n).write_bytes(b)
    x['proof']['windows_free_gib']=shutil.disk_usage('C:/').free/2**30
    (out/'heartbeat_verification.json').write_text(json.dumps(x['proof'],indent=2)+'\n')
    print(json.dumps(dict(proof=x['proof'],status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else None,error=json.loads((out/'error.json').read_text()) if (out/'error.json').exists() else None)))

if __name__=='__main__':main()
