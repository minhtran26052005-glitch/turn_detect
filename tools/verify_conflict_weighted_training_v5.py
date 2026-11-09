"""Audit actual v5 inputs against masks and inherited causal lane starts."""
import subprocess,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 code=r'''
import pathlib,json,gzip,hashlib,torch,numpy as np
from transformers import AutoTokenizer
from turn_v2.uploaded_vi_en_data_v5 import OUT,initialize,sha
from turn_v2.tiny_compression_v1.common import encode_pair
from turn_v2.multilingual_malbert_vi_en_conflict_weighted_train_v5 import fingerprint,equal
proof=json.loads((OUT/'independent_weighted_training_verification.json').read_text());assert proof['complete']
parent=torch.load(OUT/'frozen/parent_full_resume.pt',map_location='cpu',weights_only=False,mmap=True)
initial=torch.load(OUT/'frozen/initial_child_resume.pt',map_location='cpu',weights_only=False,mmap=True)
for k in ['model','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng','sampler_rng','order','cursor','stream_state','best_threshold']:assert equal(parent[k],initial[k]),k
streams=initialize(json.load(gzip.open(OUT/'train_streams.json.gz','rt',encoding='utf-8')))
trace=json.loads((OUT/'first_training_causal_trace.json').read_text());assert trace
tok=AutoTokenizer.from_pretrained(OUT/'tokenizer',local_files_only=True,token=False)
starts={(v['stream'],v['k']):v['start'] for v in parent['stream_state']['slots']};last={(v['stream'],v['k']):v['b']-1 for v in parent['stream_state']['slots']};masked=0
for r in trace:
 i,k,b=r['stream'],r['k'],r['boundary'];s=streams[i];key=i,k
 assert r['start']==starts.get(key,0) and b>last.get(key,0)
 seg=next(x for x in s['segments'] if x['start']<=b-1<x['end']);assert b+k<=seg['end']
 expected=b not in s['_unknown'] and b not in s['conflict_mask'][str(k)]
 assert r['scored']==expected and r['label']==int(b in s['_eots'])
 if not expected:masked+=1
 seq=encode_pair(tok,' '.join(s['words'][r['start']:b]),' '.join(s['words'][b:b+k]),256)['input_ids']
 assert hashlib.sha256(np.asarray(seq,dtype='<u2').tobytes()).hexdigest()==r['input_ids_sha256']
 if r['p_eot']>=r['threshold']:starts[key]=b
 last[key]=b
manifest=json.loads((OUT/'dataset_manifest.json').read_text())
with (OUT/'runs/finetune/latest_resume.pt').open('rb') as f:x=torch.load('/proc/self/fd/'+str(f.fileno()),map_location='cpu',weights_only=False,mmap=True)
assert x['fingerprint']==fingerprint(manifest) and x['loss_policy']=={'NEOT':1.,'EOT':2.} and x['branch_step']>=500
assert len(x['optimizer']['state'])==27 and {int(v['step']) for v in x['optimizer']['state'].values()}=={x['step']}
l=json.loads((OUT/'launcher.json').read_text());p=pathlib.Path('/proc',str(l['pid']));raw=(p/'stat').read_text();assert raw[raw.rfind(')')+2:].split()[19]==l['start_ticks'] and l['module'].encode() in (p/'cmdline').read_bytes().split(bytes([0]))
value=dict(complete=True,actual_training_trace_rows=len(trace),masked_or_ambiguous_trace_rows=masked,every_actual_label_mask_encoded_input_and_predicted_only_reset_verified=True,initial_model_optimizer_all_rng_order_cursor_lanes_exactly_equal_to_parent=True,step=x['step'],branch_step=x['branch_step'],new_filtered_exposures=x['uploaded_rows_seen']-initial['uploaded_rows_seen'],optimizer_entries=27,loss=x['loss_policy'],parameters=x['parameters'],process=l,parent_full_sha256=sha(OUT/'frozen/parent_full_resume.pt'),initial_child_full_sha256=sha(OUT/'frozen/initial_child_resume.pt'),manifest_sha256=sha(OUT/'dataset_manifest.json'))
(OUT/'independent_actual_weighted_training_audit.json').write_text(json.dumps(value,indent=2)+chr(10));print(json.dumps(value))
'''
 cmd='cd /home/edabk/hoangbpm/turn_detect && LD_LIBRARY_PATH=/home/edabk/miniconda3/envs/.VLLM/lib/python3.10/site-packages/nvidia/cu13/lib PYTHONPATH=outputs/multilingual_tiny_compression_20261102_v1/runtime_deps .venv/bin/python -'
 p=subprocess.run(['ssh','-o','BatchMode=yes','edabk@100.91.209.124',cmd],input=code,text=True,capture_output=True,encoding='utf-8',check=True,timeout=60)
 out=ROOT/'outputs/multilingual_malbert_vi_en_conflict_weighted_20261106_v5';out.mkdir(exist_ok=True)
 (out/'independent_actual_weighted_training_audit.json').write_text(p.stdout,encoding='utf-8');print(p.stdout)
if __name__=='__main__':main()
