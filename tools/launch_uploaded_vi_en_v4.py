"""One explicit uploaded-data cutover. No Windows model/cache downloads."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/home/edabk/hoangbpm/turn_detect'
FILES=['uploaded_vi_en_data_v4.py','uploaded_vi_en_evaluation_v4.py','multilingual_malbert_vi_en_uploaded_train_v4.py']


def main():
    assert shutil.disk_usage(ROOT).free>10*2**30+5*2**20
    for name in FILES:
        path=ROOT/'turn_v2'/name
        compile(path.read_text(encoding='utf-8'),str(path),'exec')
    subprocess.run(['scp',*[str(ROOT/'turn_v2'/n) for n in FILES],f'edabk@100.91.209.124:{REMOTE}/turn_v2/'],check=True)
    hashes={n:hashlib.sha256((ROOT/'turn_v2'/n).read_bytes()).hexdigest() for n in FILES}
    code=f'''import hashlib,json,pathlib,subprocess,os,time
r=pathlib.Path({REMOTE!r});out=r/'outputs/multilingual_malbert_vi_en_uploaded_20261104_v4';out.mkdir(exist_ok=True)
expected={hashes!r}
for n,d in expected.items():assert hashlib.sha256((r/'turn_v2'/n).read_bytes()).hexdigest()==d
assert not pathlib.Path('/proc/967949').exists()
assert json.loads((r/'outputs/multilingual_malbert_vi_en_checked_20261103_v3/status.json').read_text())['stage']=='STOPPED_BY_REQUEST'
assert not (out/'launcher.json').exists(),'Do not launch twice; inspect the existing launcher/process/full checkpoint'
env=os.environ.copy();env.update(LD_LIBRARY_PATH='/home/edabk/miniconda3/envs/.VLLM/lib/python3.10/site-packages/nvidia/cu13/lib',PYTHONPATH=str(r/'outputs/multilingual_tiny_compression_20261102_v1/runtime_deps')+':'+str(r),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
with (out/'worker.log').open('ab') as f:
 p=subprocess.Popen([str(r/'.venv/bin/python'),'-u','-m','turn_v2.multilingual_malbert_vi_en_uploaded_train_v4'],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(.5);proc=pathlib.Path('/proc')/str(p.pid);s=(proc/'stat').read_text();cmd=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
assert 'turn_v2.multilingual_malbert_vi_en_uploaded_train_v4' in cmd
value=dict(pid=p.pid,start_ticks=s[s.rfind(')')+2:].split()[19],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),module='turn_v2.multilingual_malbert_vi_en_uploaded_train_v4',cmdline=cmd,source_sha256=expected)
with (out/'launcher.json').open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(value))
'''
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','edabk@100.91.209.124','python3 -'],input=code,text=True,capture_output=True,check=True)
    local=ROOT/'outputs/multilingual_malbert_vi_en_uploaded_20261104_v4';local.mkdir(exist_ok=True)
    (local/'launcher.json').write_text(p.stdout,encoding='utf-8')
    print(p.stdout)


if __name__=='__main__':main()
