"""Explicit metadata-only repair before the first uploaded-data training step."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/home/edabk/hoangbpm/turn_detect'


def remote(code):
    result=subprocess.run(['ssh','edabk@100.91.209.124','python3 -'],input=code,text=True,capture_output=True,check=True)
    return result.stdout


def main():
    # Preserve exact failed source/metadata/full inherited checkpoint before replacing the helper.
    code=f'''import pathlib,json,os,shutil
r=pathlib.Path({REMOTE!r});o=r/'outputs/multilingual_malbert_vi_en_uploaded_20261104_v4'
assert not pathlib.Path('/proc/2828870').exists()
h=o/'history/token_digest_uint32_repair01';h.mkdir(parents=True,exist_ok=False)
for n in ['launcher.json','status.json','error.json','dataset_manifest.json','migration_receipt.json']:
 shutil.copyfile(o/n,h/n)
shutil.copyfile(r/'turn_v2/uploaded_vi_en_evaluation_v4.py',h/'uploaded_vi_en_evaluation_v4.py')
os.link(o/'runs/finetune/latest_resume.pt',h/'inherited_full_resume_before_repair.pt')
print(str(h))
'''
    print(remote(code))
    path=ROOT/'turn_v2/uploaded_vi_en_evaluation_v4.py'
    subprocess.run(['scp',str(path),f'edabk@100.91.209.124:{REMOTE}/turn_v2/'],check=True)
    code=f"""import os,pathlib,json,hashlib,subprocess,time
r=pathlib.Path({REMOTE!r});o=r/'outputs/multilingual_malbert_vi_en_uploaded_20261104_v4';h=o/'history/token_digest_uint32_repair01'
env=os.environ.copy();env.update(LD_LIBRARY_PATH='/home/edabk/miniconda3/envs/.VLLM/lib/python3.10/site-packages/nvidia/cu13/lib',PYTHONPATH=str(r/'outputs/multilingual_tiny_compression_20261102_v1/runtime_deps')+':'+str(r),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
script='''import json,torch
from turn_v2.uploaded_vi_en_data_v4 import OUT,ROOT,MODULES,sha,save
from turn_v2.multilingual_malbert_vi_en_uploaded_train_v4 import equal,atomic_torch,fingerprint
p=OUT/'runs/finetune/latest_resume.pt';x=torch.load(p,map_location='cpu',weights_only=False)
assert x['branch_step']==0 and x['uploaded_rows_seen']==0
before=x['fingerprint'];m=json.loads((OUT/'dataset_manifest.json').read_text());files=m['files'].copy()
m['code_sha256']={{n:sha(ROOT/'turn_v2'/n) for n in MODULES}}
save('dataset_manifest.json',m);x['fingerprint']=fingerprint(m);atomic_torch(p,x)
y=torch.load(p,map_location='cpu',weights_only=False)
assert all(equal(x[k],y[k]) for k in ['model','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng','order','stream_state'])
assert files==json.loads((OUT/'dataset_manifest.json').read_text())['files']
save('token_digest_repair_receipt.json',dict(complete=True,reason='Frozen teacher vocabulary119547 requires uint32 digest; no input or model computation changed',before_fingerprint=before,after_fingerprint=y['fingerprint'],source_data_and_dev_file_shas_unchanged=True,model_optimizer_all_rng_order_preserved=True,branch_step=0,uploaded_rows_seen=0,full_resume_sha256=sha(p)))
'''
subprocess.run([str(r/'.venv/bin/python'),'-c',script],cwd=r,env=env,check=True)
# Retain the technical failure in history, without keeping it as the active error.
os.replace(o/'error.json',h/'active_error_archived.json')
with (o/'worker.log').open('ab') as f:p=subprocess.Popen([str(r/'.venv/bin/python'),'-u','-m','turn_v2.multilingual_malbert_vi_en_uploaded_train_v4'],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(.5);proc=pathlib.Path('/proc')/str(p.pid);raw=(proc/'stat').read_text()
v=dict(pid=p.pid,start_ticks=raw[raw.rfind(')')+2:].split()[19],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),module='turn_v2.multilingual_malbert_vi_en_uploaded_train_v4',repair='token_digest_uint32_repair01')
with (o/'launcher.json').open('w') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(v))
"""
    print(remote(code))


if __name__=='__main__':main()
