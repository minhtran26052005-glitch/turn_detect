"""Verify the already-scored 135M control on new DEV; do not rescore/select on TEST."""
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert shutil.disk_usage(ROOT).free > 10*2**30+5*2**20
    code = r'''
import os
os.environ.update(LD_LIBRARY_PATH='/home/edabk/miniconda3/envs/.VLLM/lib/python3.10/site-packages/nvidia/cu13/lib',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import sys
from pathlib import Path
root=Path('/home/edabk/hoangbpm/turn_detect')
sys.path[:0]=[str(root/'outputs/multilingual_tiny_compression_20261102_v1/runtime_deps'),str(root)]
import hashlib,json,gzip,math,datetime
from collections import Counter,defaultdict
import numpy as np
out=root/'outputs/multilingual_malbert_vi_en_uploaded_20261104_v4'
teacher=root/'outputs/multilingual_tiny_compression_20261102_v1/teacher'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
report=json.loads((out/'teacher_new_dev.json').read_text())
receipt=json.loads((teacher.parent/'teacher_receipt.json').read_text())
assert sha(teacher/'model.safetensors')==receipt['weights_sha256']==report['frozen_teacher_sha256']=='15d4ebfc7fcba7c3c842b7b7815ac309cf349f0436dca69522d9d529295954cc'
manifest=json.loads((out/'dataset_manifest.json').read_text())
for n in ['dev_calibration.json.gz','dev_monitor.json.gz']:assert sha(out/n)==manifest['files'][n]
streams=json.load(gzip.open(out/'dev_monitor.json.gz','rt',encoding='utf-8'))
assert len(streams)==394
import torch
import transformers.utils as u
import transformers.utils.import_utils as im
u.is_torchvision_available=im.is_torchvision_available=lambda:False
u.is_librosa_available=im.is_librosa_available=lambda:False
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from turn_v2.tiny_compression_v1.common import encode_pair
from turn_v2.malbert_eval_zip_v2 import metric
tok=AutoTokenizer.from_pretrained(teacher,local_files_only=True,token=False)
starts=defaultdict(int);last={};counts=defaultdict(Counter);visits=Counter();selected={};digest=hashlib.sha256();n=0
for s in streams:s['_eots']=set(s['eots']);s['_unknown']=set(s['unknown'])
trace=out/'teacher_new_dev_dev_trace.jsonl.gz'
with gzip.open(trace,'rt',encoding='utf-8') as f:
 for line in f:
  digest.update(line[:-1].encode('utf-8'));r=json.loads(line);i=r['stream'];b=r['boundary'];s=streams[i]
  assert s['split']=='dev' and r['context_start']==starts[i] and b>last.get(i,0)
  seg=next(p for p in s['segments'] if p['start']<=b-1<p['end'])
  assert b+2<=seg['end'] and seg.get('punctuation_present',True) and r['language']==seg['language']
  assert r['label']==int(b in s['_eots']) and r['scored']==(b not in s['_unknown'])
  context=' '.join(s['words'][starts[i]:b]);future=' '.join(s['words'][b:b+2])
  seq=encode_pair(tok,context,future,256)['input_ids']
  assert hashlib.sha256(np.asarray(seq,dtype='<u4').tobytes()).hexdigest()==r['encoded_sha256']
  p=r['p_eot'];assert math.isfinite(p) and 0<=p<=1 and r['predicted_eot']==(p>=report['threshold'])
  if r['scored']:counts[r['language']]['tp' if r['label'] and r['predicted_eot'] else 'fn' if r['label'] else 'fp' if r['predicted_eot'] else 'tn']+=1
  bucket='short' if len(seq)<=32 else 'medium' if len(seq)<=128 else 'long'
  selected.setdefault((s['domain'],r['language'],bucket),dict(row=r,context=context,text=future))
  if r['predicted_eot']:starts[i]=b
  visits[i]+=1;last[i]=b;n+=1
for i,s in enumerate(streams):assert visits[i]==sum(max(0,p['end']-p['start']-2) for p in s['segments'] if p.get('punctuation_present',True))
assert n==report['inferences']==35270 and digest.hexdigest()==report['causal_input_trace_digest']
assert {l:metric(c) for l,c in counts.items()}==report['by_language']
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25)
model=AutoModelForSequenceClassification.from_pretrained(teacher,local_files_only=True,token=False).cuda().eval()
from turn_v2.uploaded_vi_en_evaluation_v4 import predictor
predict=predictor(model,tok,teacher=True);cases=list(selected.values());delta=0.
for start in range(0,len(cases),8):
 part=cases[start:start+8];probs,_=predict([dict(context=c['context'],text=c['text']) for c in part])
 for c,p in zip(part,probs):
  delta=max(delta,abs(p-c['row']['p_eot']));assert (p>=report['threshold'])==c['row']['predicted_eot']
assert delta<1e-4
launch=json.loads((out/'launcher.json').read_text());proc=Path('/proc')/str(launch['pid']);raw=(proc/'stat').read_text();fields=raw[raw.rfind(')')+2:].split();assert fields[19]==launch['start_ticks'] and launch['module'].encode() in (proc/'cmdline').read_bytes().split(b'\0')
current=json.loads((out/'latest_dev.json').read_text());assert all(current['dev']['by_language'][l]['samples']==report['by_language'][l]['samples'] for l in ['vi','en'])
proof=dict(complete=True,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),teacher_checkpoint=receipt['step'],teacher_branch_step=receipt['branch_step'],teacher_parameters=sum(p.numel() for p in model.parameters()),teacher_weights_sha256=receipt['weights_sha256'],teacher_report_sha256=sha(out/'teacher_new_dev.json'),teacher_trace_sha256=sha(trace),same_new_dev_monitor_and_calibration_sha_verified=True,all_trace_inputs_labels_future_and_predicted_resets_verified=n,all_confusion_counts_verified=True,canonical_predictor_parity_cases=len(cases),max_probability_delta=delta,all_parity_decisions_identical=True,threshold=report['threshold'],teacher_metrics=report['by_language'],student_checkpoint=current['step'],student_branch_step=current['branch_step'],student_threshold=current['dev']['threshold'],student_metrics=current['dev']['by_language'],main_process=launch['pid'],main_status=json.loads((out/'status.json').read_text()),test_scored=False,previous_135m_dev_run_reused_verified=True,no_training_or_threshold_policy_changes=True)
target=out/'teacher_new_dev_user_recheck_20261105.json';target.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(proof))
'''
    # Bootstrap with the same shared CUDA/runtime libraries as the active campaign.
    command = "cd /home/edabk/hoangbpm/turn_detect && LD_LIBRARY_PATH=/home/edabk/miniconda3/envs/.VLLM/lib/python3.10/site-packages/nvidia/cu13/lib .venv/bin/python -"
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','edabk@100.91.209.124',command],input=code,text=True,capture_output=True,encoding='utf-8',timeout=240,check=True)
    proof=json.loads(p.stdout);out=ROOT/'outputs/multilingual_malbert_vi_en_uploaded_20261104_v4'
    (out/'teacher_new_dev_user_recheck_20261105.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(proof,ensure_ascii=False))


if __name__=='__main__':main()
