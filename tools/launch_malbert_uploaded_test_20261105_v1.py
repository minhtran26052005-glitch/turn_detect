"""Launch exactly one newly requested uploaded-v4 TEST, preserving the active trainer."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/home/edabk/hoangbpm/turn_detect'
MODULE = 'malbert_uploaded_test_20261105_v1'


def main():
    assert shutil.disk_usage(ROOT).free > 10*2**30+5*2**20
    path = ROOT / 'turn_v2' / (MODULE+'.py')
    compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    subprocess.run(['scp', str(path), f'edabk@100.91.209.124:{REMOTE}/turn_v2/'], check=True)
    code = f'''import hashlib,json,pathlib,subprocess,os,time,shutil
r=pathlib.Path({REMOTE!r});module={MODULE!r};out=r/'outputs'/module
assert hashlib.sha256((r/'turn_v2'/(module+'.py')).read_bytes()).hexdigest()=={expected!r}
main=r/'outputs/multilingual_malbert_vi_en_uploaded_20261104_v4'
l=json.loads((main/'launcher.json').read_text());p=pathlib.Path('/proc')/str(l['pid'])
raw=(p/'stat').read_text();cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
assert raw[raw.rfind(')')+2:].split()[19]==str(l['start_ticks']) and l['module'] in cmd
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip();assert boot==l['boot_id']
found=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  parts=(p/'cmdline').read_bytes().split(b'\\0')
  if ('turn_v2.'+module).encode() in parts:found.append(p.name)
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not found,found
assert not out.exists(),'Inspect the existing test instead of launching it again'
mount=next(s for s in pathlib.Path('/proc/mounts').read_text().splitlines() if s.split()[1]=='/mnt/hdd500')
assert mount.split()[0]=='/dev/sda1' and 'rw' in mount.split()[3].split(',')
assert min(shutil.disk_usage(r).free,shutil.disk_usage('/mnt/hdd500').free)>=13*2**30
out.mkdir();(out/'inputs').mkdir()
old=r/'outputs/malbert_eval_zip_20261104_v2'
for name in ['eval.zip','usage_plan.json']:os.link(old/'inputs'/name,out/'inputs'/name)
env=os.environ.copy();env.update(LD_LIBRARY_PATH='/home/edabk/miniconda3/envs/.VLLM/lib/python3.10/site-packages/nvidia/cu13/lib',PYTHONPATH=str(r/'outputs/multilingual_tiny_compression_20261102_v1/runtime_deps')+':'+str(r),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
status=json.loads((main/'status.json').read_text());minimum=status.get('branch_step')
if minimum is None:minimum=json.loads((main/'runs/finetune/resume_receipt.json').read_text())['branch_step']
# Write the launch intent before execution; the worker reads its temporal lower bound.
value=dict(module='turn_v2.'+module,minimum_branch_step=minimum,source_sha256={expected!r},boot_id=boot,reported=False)
(out/'launcher.json').write_text(json.dumps(value,indent=2)+'\\n')
with (out/'worker.log').open('ab') as log:
 child=subprocess.Popen([str(r/'.venv/bin/python'),'-u','-m','turn_v2.'+module],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(.5);p=pathlib.Path('/proc')/str(child.pid);raw=(p/'stat').read_text();cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
assert 'turn_v2.'+module in cmd
value.update(pid=child.pid,start_ticks=raw[raw.rfind(')')+2:].split()[19],cmdline=cmd)
tmp=out/'launcher.json.tmp'
with tmp.open('w') as f:json.dump(value,f,indent=2);f.write('\\n');f.flush();os.fsync(f.fileno())
tmp.replace(out/'launcher.json');print(json.dumps(value))
'''
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'edabk@100.91.209.124', 'python3 -'],
                       input=code, text=True, capture_output=True, check=True, encoding='utf-8', timeout=60)
    local = ROOT / 'outputs' / MODULE
    local.mkdir(exist_ok=True)
    (local / 'launcher.json').write_bytes(p.stdout.encode('utf-8'))
    (local / 'task_state.json').write_text(json.dumps(dict(phase='ACTIVE',reported=False,request='Test the current uploaded-only v4 model',
          main_training_continues=True),indent=2)+'\n',encoding='utf-8')
    print(p.stdout)


if __name__ == '__main__': main()
