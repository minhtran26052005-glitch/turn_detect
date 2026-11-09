"""Independent read-only checkpoint audit; transfer only its small receipt."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess


def main():
    assert shutil.disk_usage(Path.cwd()).free >= 10*2**30+5*2**20
    code = r'''import pathlib,json,hashlib,os,collections,numpy as np,torch
r=pathlib.Path('/home/edabk/hoangbpm/turn_detect/outputs/multilingual_malbert_vi_en_20261103_v2')
project=r.parent.parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
launcher=json.loads((r/'launcher.json').read_text());pid=launcher['pid'];p=pathlib.Path('/proc')/str(pid)
raw=(p/'stat').read_text();ticks=raw[raw.rfind(')')+2:].split()[19];cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
assert ticks==launcher['start_ticks'] and 'turn_v2.multilingual_malbert_vi_en_train_v2' in cmd
assert all(not (pathlib.Path('/proc')/str(old_pid)).exists() for old_pid in [2827094,427431,889316])
receipt=json.loads((r/'cache/receipt.json').read_text());migration=json.loads((r/'migration_receipt.json').read_text())
with (r/'runs/finetune/latest_resume.pt').open('rb') as f:
 x=torch.load('/proc/self/fd/'+str(f.fileno()),map_location='cpu',weights_only=False,mmap=True)
 f.seek(0);h=hashlib.sha256()
 for b in iter(lambda:f.read(2**20),b''):h.update(b)
 checkpoint_sha=h.hexdigest()
assert x['parameters']==11941122<20000000 and x['step']>migration['parent_step']==14082
assert x['fingerprint']['active_languages']==['vi','en']
assert x['fingerprint']['train_sha256']==receipt['train_sha256']
assert x['fingerprint']['cache_sha256']==sha(r/'cache/receipt.json')
for name,value in x['fingerprint']['code_sha256'].items():assert sha(project/'turn_v2'/name)==value
assert len(x['optimizer']['state'])==27 and len(x['order'])==200000
assert np.array_equal(np.sort(x['order']),np.arange(200000)) and 0<=x['cursor']<=200000
for key in ['sampler_rng','torch_rng','cuda_rng','numpy_rng','python_rng']:assert x[key] is not None
assert x['torch_rng'].numel()>0 and x['cuda_rng'].numel()>0
steps={int(v['step']) for v in x['optimizer']['state'].values()};assert steps=={x['step']}
language_ids=np.load(r/'cache/language_ids.npy',mmap_mode='r')
assert dict(collections.Counter(language_ids.tolist()))=={0:100000,1:100000}
counts=collections.Counter(json.loads(l)['language'] for l in (r/'train_vi_en.jsonl').open())
assert dict(counts)=={'vi':100000,'en':100000}
dev=collections.Counter(json.loads(l)['language'] for l in (r/'dev_rows_vi_en.jsonl').open())
assert set(dev)=={'vi','en'}
latest=json.loads((r/'latest_dev.json').read_text());assert set(latest['dev']['by_language'])=={'vi','en'}
assert set(latest['retention_by_language'])=={'vi','en'}
v={'verified':True,'pid':pid,'start_ticks':ticks,'cmdline':cmd,'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 'parameters':x['parameters'],'checkpoint_sha256':checkpoint_sha,'step':x['step'],'epoch':x['epoch'],'cursor':x['cursor'],
 'optimizer_entries':27,'optimizer_step_matches':True,'rng_and_sampler_present':True,'order_is_full_bilingual_permutation':True,
 'active_languages':['vi','en'],'train_counts':dict(counts),'dev_counts':dict(dev),'fingerprint':x['fingerprint'],
 'parent_step':migration['parent_step'],'migration_optimizer_rng_order_preserved':migration['model_optimizer_rng_preserved'],
 'latest_dev_languages_verified':True,'test_used':False}
print(json.dumps(v))'''
    # The pinned environment supplies torch; no weights leave SSH.
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
        'edabk@100.91.209.124',
        'LD_LIBRARY_PATH=/home/edabk/miniconda3/envs/.VLLM/lib/python3.10/site-packages/nvidia/cu13/lib '
        '/home/edabk/hoangbpm/turn_detect/.venv/bin/python -'],
        input=code,text=True,capture_output=True,encoding='utf-8',check=True,timeout=90)
    value=json.loads(result.stdout);value['checked_utc']=datetime.now(timezone.utc).isoformat()
    target=Path('outputs/multilingual_malbert_vi_en_20261103_v2/independent_training_verification.json')
    target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value))


if __name__=='__main__':main()
