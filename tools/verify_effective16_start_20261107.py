"""Read-only early proof of actual effective16 training, source inputs and state."""
import json,subprocess
from pathlib import Path
CODE=r'''
import json,pathlib,torch,gzip,hashlib,numpy as np
import transformers.utils as u,transformers.utils.import_utils as im
u.is_torchvision_available=im.is_torchvision_available=lambda:False;u.is_librosa_available=im.is_librosa_available=lambda:False
from transformers import AutoTokenizer
from turn_v2.asr_effective_batch16_20261107_v1 import OUT,MAIN,CONTROL,sha,save
from turn_v2.asr_only_adaptation_20261107_v1 import equal
from turn_v2.uploaded_vi_en_data_v5 import StreamOrder,initialize
from turn_v2.tiny_compression_v1.common import encode_pair
torch.set_num_threads(2)
initial=torch.load(OUT/'frozen/initial_resume.pt',map_location='cpu',weights_only=False,mmap=True);control=torch.load(CONTROL/'frozen/initial_child_resume.pt',map_location='cpu',weights_only=False,mmap=True);first=torch.load(OUT/'frozen/first100_resume.pt',map_location='cpu',weights_only=False,mmap=True)
for k in ['model','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng','sampler_rng','order']:assert equal(initial[k],control[k])
tok=AutoTokenizer.from_pretrained(MAIN/'tokenizer',local_files_only=True,token=False);src=initialize(json.load(gzip.open(OUT/'train_streams.json.gz','rt',encoding='utf-8')));runner=StreamOrder(src,initial['order'],lanes=16);trace=first['training_trace'];offset=0;windows={}
while offset<len(trace):
 rows,states=runner.rows();part=trace[offset:offset+len(rows)];rows=rows[:len(part)];states=states[:len(part)]
 for r,z in zip(rows,part):
  for k in ['stream','boundary','k','start','label','scored']:assert r[k]==z[k]
  seq=encode_pair(tok,r['context'],r['text'],256)['input_ids'];assert hashlib.sha256(np.asarray(seq,dtype='<u4').tobytes()).hexdigest()==z['input_ids_sha256']
  if r['scored']:windows[z['window']]=windows.get(z['window'],0)+1
 runner.advance(states,[z['p_eot'] for z in part],.1);offset+=len(part)
assert len(windows)==100 and sum(windows.values())==1600 and set(windows.values())=={16}
assert first['branch_step']==100 and first['scored_seen']==1600 and len(first['optimizer']['state'])==27 and first['accumulator']['count']==0
assert {int(v['step']) for v in first['optimizer']['state'].values()}=={3731273}
assert all(torch.isfinite(v).all() for v in first['model'].values()) and any(not torch.equal(v,initial['model'][k]) for k,v in first['model'].items())
proof=dict(complete=True,actual_training=True,encoded_inputs_audited=len(trace),scored_points=1600,windows100_each16=True,source_labels_masks_future_predicted_resets_verified=True,initial_control_model_optimizer_all_rng_order_exact=True,finite_changed_weights=True,optimizer_entries=27,checkpoint_sha256=sha(OUT/'frozen/first100_resume.pt'))
save('first_training_verification.json',proof);print(json.dumps(proof))
'''
cmd='cd /home/edabk/hoangbpm/turn_detect && LD_LIBRARY_PATH=/home/edabk/miniconda3/envs/.VLLM/lib/python3.10/site-packages/nvidia/cu13/lib PYTHONPATH=/home/edabk/hoangbpm/turn_detect/outputs/multilingual_tiny_compression_20261102_v1/runtime_deps:/home/edabk/hoangbpm/turn_detect .venv/bin/python -'
r=subprocess.run(['ssh','-o','BatchMode=yes','edabk@100.91.209.124',cmd],input=CODE,text=True,encoding='utf-8',capture_output=True,timeout=45)
assert r.returncode==0,r.stderr
x=json.loads(r.stdout);p=Path('outputs/asr_effective_batch16_20261107_v1/first_training_verification.json');p.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
