"""One guarded handover after the authorized v4 checkpointed stop."""
import pathlib,json,os,time,subprocess,hashlib,fcntl,traceback
ROOT=pathlib.Path('/home/edabk/hoangbpm/turn_detect')
OUT=ROOT/'outputs/multilingual_malbert_vi_en_conflict_weighted_20261106_v5'
PARENT=ROOT/'outputs/multilingual_malbert_vi_en_uploaded_20261104_v4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
 p=OUT/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def run():
 assert not (OUT/'cutover_complete.json').exists()
 launch=json.loads((PARENT/'launcher.json').read_text());assert launch['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 assert (PARENT/'STOP_REQUESTED').exists() and json.loads((OUT/'independent_preparation_verification.json').read_text())['complete']
 while pathlib.Path('/proc',str(launch['pid'])).exists():
  proc=pathlib.Path('/proc',str(launch['pid']));s=(proc/'stat').read_text();assert s[s.rfind(')')+2:].split()[19]==launch['start_ticks']
  save('cutover_status.json',dict(stage='WAITING_FOR_V4_CHECKPOINTED_EXIT',parent_pid=launch['pid'],parent_status=json.loads((PARENT/'status.json').read_text())))
  time.sleep(5)
 assert json.loads((PARENT/'status.json').read_text())['stage'] in ('STOPPED_WITH_CHECKPOINT','STOPPED_BY_REQUEST')
 assert not (OUT/'launcher.json').exists()
 manifest=json.loads((OUT/'dataset_manifest.json').read_text());old=sha(OUT/'dataset_manifest.json');core='multilingual_malbert_vi_en_conflict_weighted_train_v5.py';old_source=manifest['code_sha256'][core];manifest['code_sha256'][core]=sha(ROOT/'turn_v2'/core)
 save('dataset_manifest.json',manifest)
 proof=json.loads((OUT/'independent_preparation_verification.json').read_text());proof['manifest_sha256']=sha(OUT/'dataset_manifest.json');proof['prelaunch_stop_guard_clarification']=dict(old_manifest_sha256=old,old_source_sha256=old_source,new_source_sha256=manifest['code_sha256'][core],reason='Accept actual v4 STOPPED_WITH_CHECKPOINT only with explicit STOP_REQUESTED and observed PID absence',data_mask_unchanged=True);save('independent_preparation_verification.json',proof)
 env=os.environ.copy();env.update(LD_LIBRARY_PATH='/home/edabk/miniconda3/envs/.VLLM/lib/python3.10/site-packages/nvidia/cu13/lib',PYTHONPATH=str(ROOT/'outputs/multilingual_tiny_compression_20261102_v1/runtime_deps')+':'+str(ROOT),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
 with (OUT/'worker.log').open('ab') as f:p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','turn_v2.multilingual_malbert_vi_en_conflict_weighted_train_v5'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 time.sleep(.5);proc=pathlib.Path('/proc',str(p.pid));raw=(proc/'stat').read_text();l=dict(pid=p.pid,start_ticks=raw[raw.rfind(')')+2:].split()[19],boot_id=launch['boot_id'],module='turn_v2.multilingual_malbert_vi_en_conflict_weighted_train_v5',source_sha256=manifest['code_sha256']);save('launcher.json',l)
 save('cutover_status.json',dict(stage='VERIFYING_NEW_WEIGHTED_TRAINING',launcher=l))
 while True:
  assert proc.exists(),'New worker exited; inspect error.json, never blindly restart'
  assert not (OUT/'error.json').exists(),(OUT/'error.json').read_text() if (OUT/'error.json').exists() else ''
  f=OUT/'runs/finetune/resume_receipt.json'
  if f.exists() and json.loads(f.read_text())['branch_step']>=500:break
  time.sleep(3)
 import torch
 from .multilingual_malbert_vi_en_conflict_weighted_train_v5 import fingerprint
 path=OUT/'runs/finetune/latest_resume.pt'
 with path.open('rb') as f:x=torch.load('/proc/self/fd/'+str(f.fileno()),map_location='cpu',weights_only=False,mmap=True)
 initial=torch.load(OUT/'frozen/parent_full_resume.pt',map_location='cpu',weights_only=False,mmap=True)
 assert x['fingerprint']==fingerprint(manifest) and x['loss_policy']=={'NEOT':1.,'EOT':2.}
 assert x['step']>initial['step'] and x['uploaded_rows_seen']>initial['uploaded_rows_seen']
 assert len(x['optimizer']['state'])==27 and {int(v['step']) for v in x['optimizer']['state'].values()}=={x['step']}
 assert all(k in x for k in ['torch_rng','cuda_rng','numpy_rng','python_rng','sampler_rng','order','cursor','stream_state'])
 assert any(not torch.equal(v,initial['model'][k]) for k,v in x['model'].items())
 assert all(bool(torch.isfinite(v).all()) for v in x['model'].values() if v.is_floating_point())
 pin=OUT/'frozen'/('verified_new_training_step_%d.pt'%x['step'])
 # Hardlink the open inode, not a subsequently replaced latest checkpoint.
 with pin.open('xb') as dst:
  import shutil
  with path.open('rb') as src:shutil.copyfileobj(src,dst)
 check=torch.load(pin,map_location='cpu',weights_only=False,mmap=True)
    assert check['step']==x['step'] and check['fingerprint']==x['fingerprint']
    proof=dict(complete=True,phase='FILTERED_WEIGHTED_TRAINING_VERIFIED',launcher=l,parent_step=initial['step'],step=x['step'],branch_step=x['branch_step'],new_filtered_exposures=x['uploaded_rows_seen']-initial['uploaded_rows_seen'],optimizer_entries=27,full_rng_order_cursor_lanes_present=True,all_weights_finite=True,weights_changed_after_weighted_training=True,loss_policy=x['loss_policy'],parameters=x['parameters'],manifest_sha256=sha(OUT/'dataset_manifest.json'),parent_proc_absent=True,base_reset=False)
 save('independent_weighted_training_verification.json',proof);save('cutover_complete.json',proof);save('cutover_status.json',dict(stage='COMPLETE',proof=proof))
if __name__=='__main__':
 with (OUT/'CUTOVER_LOCK').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:run()
  except Exception as e:save('cutover_error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
