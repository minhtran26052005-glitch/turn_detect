"""Final read-only checks complementing all-trace worker verification."""
import subprocess,json
from pathlib import Path
CODE=r'''
import torch,json
from turn_v2.asr_effective_batch16_20261107_v1 import OUT,MAIN,CONTROL,sha,save,guard
torch.set_num_threads(2);guard()
proof=json.loads((OUT/'completion_verification.json').read_text());report=json.loads((OUT/'report.json').read_text());assert proof['complete'] and proof['report_sha256']==sha(OUT/'report.json')
manifest=json.loads((CONTROL/'dataset_manifest.json').read_text())
for n,h in manifest['dev_files'].items():assert sha(OUT/n)==sha(CONTROL/n)==h
old=json.loads((CONTROL/'training_report.json').read_text());assert report['arms']['control']['weights_sha256']==old['weights_sha256']==sha(CONTROL/'runs/finetune/weights.pt')
oldcal=json.loads((CONTROL/'dev_report.json').read_text())['calibration'];anchor=next(v for v in oldcal if v['threshold']==.1);newanchor=json.loads((OUT/'control_calibration_11.json').read_text());assert anchor['by_language']==newanchor['by_language']
x=torch.load(OUT/'runs/finetune/latest_resume.pt',map_location='cpu',weights_only=False,mmap=True);w=torch.load(OUT/'runs/finetune/weights.pt',map_location='cpu',weights_only=True)
assert all(torch.equal(v,w[k]) for k,v in x['model'].items());assert report['arms']['effective16']['weights_sha256']==sha(OUT/'runs/finetune/weights.pt')
for state in x['optimizer']['state'].values():
 for value in state.values():
  if torch.is_tensor(value):assert torch.isfinite(value).all()
assert not x['accumulator']['gradients'] and x['accumulator']['count']==0 and x['accumulator']['weight']==0
for arm in ['control','effective16']:
 receipt=json.loads((OUT/(arm+'_selection_receipt.json')).read_text());best=max(receipt['calibration'],key=lambda z:(min((v['eot_f1'] or 0) for v in z['by_language'].values()),sum((v['eot_f1'] or 0) for v in z['by_language'].values())))
 assert best['threshold']==receipt['threshold']==report['arms'][arm]['selected']['threshold']
out=dict(complete=True,report_sha256=sha(OUT/'report.json'),source_dev_sha_unchanged=True,control_weights_match_original_verified_trial=True,control_point1_calibration_anchor_exact=True,treatment_weights_equal_final_fullresume=True,all_optimizer_states_finite=True,accumulator_empty_at_full_epoch_end=True,selection_recomputed_calibration_only=True,training_audit_scope='Full source/label counts, task completion and window histogram;1622actualencodedinputs/1600scored first100updates audited, not every training forward',parent_paused=True,no_test=True,no_default_change=True)
save('independent_final_identity_verification.json',out);print(json.dumps(out))
'''
cmd='cd /home/edabk/hoangbpm/turn_detect && LD_LIBRARY_PATH=/home/edabk/miniconda3/envs/.VLLM/lib/python3.10/site-packages/nvidia/cu13/lib PYTHONPATH=/home/edabk/hoangbpm/turn_detect .venv/bin/python -'
r=subprocess.run(['ssh','-o','BatchMode=yes','edabk@100.91.209.124',cmd],input=CODE,text=True,encoding='utf-8',capture_output=True,timeout=45);assert r.returncode==0,r.stderr
x=json.loads(r.stdout);Path('outputs/asr_effective_batch16_20261107_v1/independent_final_identity_verification.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
