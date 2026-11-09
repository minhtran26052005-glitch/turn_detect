"""One full ASR epoch with exactly16 scored samples per optimizer window."""
import copy,fcntl,gc,gzip,hashlib,json,math,os,signal,traceback
from collections import Counter
from pathlib import Path
import numpy as np
from . import asr_only_adaptation_20261107_v1 as old
from .uploaded_vi_en_data_v5 import StreamOrder,initialize
from .tiny_compression_v1.common import encode_pair

ROOT=old.ROOT;MAIN=old.MAIN;CONTROL=old.OUT
OUT=ROOT/'outputs/asr_effective_batch16_20261107_v1'
STOP=False
sha=old.sha
def save(n,v):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 with t.open('w',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def status(stage,**v):
 from datetime import datetime,timezone
 raw=Path('/proc/self/stat').read_text();v=dict(stage=stage,pid=os.getpid(),start_ticks=raw[raw.rfind(')')+2:].split()[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),at_utc=datetime.now(timezone.utc).isoformat(),**v);save('status.json',v);print(json.dumps(v),flush=True)
def guard():old.parent_paused();old.reserve()

class Accumulator:
 def __init__(self,model,opt,runner,tok,device='cuda',state=None):
  self.m=model;self.o=opt;self.r=runner;self.t=tok;self.device=device
  self.pending=[];self.count=0;self.weight=0.;self.updates=0;self.windows=Counter()
  if state:
   self.pending=list(state['pending']);self.count=state['count'];self.weight=state['weight'];self.updates=state['updates'];self.windows=Counter(state['windows'])
   for n,p in model.named_parameters():p.grad=state['gradients'][n].to(device).clone() if n in state['gradients'] else None
  else:opt.zero_grad(set_to_none=True)
 def state(self):
  return dict(pending=self.pending,count=self.count,weight=self.weight,updates=self.updates,windows=dict(self.windows),gradients={n:p.grad.detach().cpu().clone() for n,p in self.m.named_parameters() if p.grad is not None})
 def flush(self):
  import torch
  if not self.count:return False
  for p in self.m.parameters():
   if p.grad is not None:p.grad.mul_(16/self.weight)
  torch.nn.utils.clip_grad_norm_(self.m.parameters(),1.);self.o.step();self.o.zero_grad(set_to_none=True)
  self.windows[str(self.count)]+=1;self.updates+=1;self.count=0;self.weight=0.;return True
 def next(self):
  import torch
  from torch.nn import functional as F
  if not self.pending:
   rows,states=self.r.rows()
   if not rows:return None
   assert len(states)==len(self.r.slots);self.pending=list(range(len(states)))
  rows=[];indices=[];valid=0
  for idx in self.pending:
   state=self.r.slots[idx];s=self.r.streams[state['stream']];b=state['b'];k=state['k'];seg=s['segments'][state['segment']]
   scored=b not in s['_unknown'] and b not in s.get('conflict_mask',{}).get(str(k),[])
   rows.append(dict(context=' '.join(s['words'][state['start']:b]),text=' '.join(s['words'][b:b+k]),label=int(b in s['_eots']),scored=scored,language=seg['language'],origin=s['domain'],stream=state['stream'],boundary=b,k=k,start=state['start']))
   indices.append(idx);valid+=int(scored)
   if valid==16-self.count:break
  seqs=[encode_pair(self.t,r['context'],r['text'],256)['input_ids'] for r in rows];width=max(map(len,seqs))
  ids=torch.zeros((len(rows),width),dtype=torch.long,device=self.device);mask=torch.zeros_like(ids)
  for i,s in enumerate(seqs):ids[i,:len(s)]=torch.tensor(s,device=self.device);mask[i,:len(s)]=1
  with torch.autocast(device_type=self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):logits=self.m(ids,mask)
  logits=logits.float();probs=logits.detach().softmax(-1)[:,1].cpu().tolist();scored=torch.tensor([r['scored'] for r in rows],dtype=torch.bool,device=self.device)
  window=self.updates+1
  if valid:
   y=torch.tensor([r['label'] for r in rows],device=self.device);loss=F.cross_entropy(logits[scored],y[scored],weight=logits.new_tensor([1.,2.]),reduction='sum')/16
   assert torch.isfinite(loss);loss.backward();self.count+=valid;self.weight+=sum(1+r['label'] for r in rows if r['scored'])
  self.r.advance([self.r.slots[i] for i in indices],probs,.1);self.pending=self.pending[len(indices):]
  updated=self.flush() if self.count==16 else False
  return rows,seqs,probs,window,updated

def fixture():
 import torch
 class Tok:
  cls_token_id=1;sep_token_id=3
  def encode(self,t,add_special_tokens=False):return [4+int(w[1:]) for w in t.split()]
 class Tiny(torch.nn.Module):
  def __init__(self):
   super().__init__();self.emb=torch.nn.Embedding(32,8);self.drop=torch.nn.Dropout(.2);self.head=torch.nn.Linear(8,2)
  def forward(self,ids,mask):return self.head(self.drop((self.emb(ids)*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]))
 src=initialize([dict(words=['w'+str(j) for j in range(1,12)],eots=[4,8,11],unknown=[2],conflict_mask={'1':[6],'2':[]},domain='fixture',segments=[dict(start=0,end=11,language='vi')]) for _ in range(3)])
 torch.manual_seed(917);m=Tiny();initial=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone();order=np.array([1,4,0,3,5,2])
 def run(cut):
  m=Tiny();m.load_state_dict(initial);o=torch.optim.AdamW(m.parameters(),lr=4e-6,weight_decay=.01);torch.set_rng_state(rng)
  r=StreamOrder(src,order,lanes=2);a=Accumulator(m,o,r,Tok(),'cpu');seen=0;interrupted=False
  while True:
   z=a.next()
   if z is None:break
   seen+=sum(x['scored'] for x in z[0])
   if cut and a.count>0 and a.updates>=1 and not interrupted:
    snap=copy.deepcopy(dict(model=m.state_dict(),opt=o.state_dict(),runner=r.state(),acc=a.state(),rng=torch.get_rng_state()))
    m=Tiny();m.load_state_dict(snap['model']);o=torch.optim.AdamW(m.parameters(),lr=4e-6,weight_decay=.01);o.load_state_dict(snap['opt']);r=StreamOrder(src,order,snap['runner'],lanes=2);a=Accumulator(m,o,r,Tok(),'cpu',snap['acc']);torch.set_rng_state(snap['rng']);interrupted=True
  a.flush();return dict(model=m.state_dict(),opt=o.state_dict(),runner=r.state(),acc=a.state(),rng=torch.get_rng_state(),seen=seen),interrupted
 a,_=run(False);b,interrupted=run(True);assert interrupted and old.equal(a,b)
 # Gradient normalization matches concatenated weighted mean before clipping.
 p=torch.tensor([[.2,.5],[.7,.1],[-.1,.4]],requires_grad=True);y=torch.tensor([1,0,1]);w=torch.tensor([1.,2.])
 from torch.nn import functional as F
 exact=torch.autograd.grad(F.cross_entropy(p,y,weight=w),p,retain_graph=True)[0]
 split=torch.autograd.grad((F.cross_entropy(p[:1],y[:1],weight=w,reduction='sum')+F.cross_entropy(p[1:],y[1:],weight=w,reduction='sum'))/16,p)[0]*16/5
 assert torch.allclose(exact,split,atol=1e-7,rtol=1e-6)
 save('technical_fixture_verification.json',dict(complete=True,technical_only=True,actual_accumulator_mid_window_gradient_rng_resume_bitwise_equal=True,weighted_mean_gradient_parity=True,scored=a['seen'],windows=a['acc']['windows']))

def prepare():
 guard();manifest=json.loads((CONTROL/'dataset_manifest.json').read_text());assert sha(CONTROL/'train_streams.json.gz')==manifest['train_sha256']
 for n in ['train_streams.json.gz','dev_calibration.json.gz','dev_monitor.json.gz']:
  if not (OUT/n).exists():os.link(CONTROL/n,OUT/n)
 source_hashes={n:sha(ROOT/'turn_v2'/n) for n in ['asr_effective_batch16_20261107_v1.py','asr_only_adaptation_20261107_v1.py','uploaded_vi_en_data_v4.py','uploaded_vi_en_data_v5.py','malbert_turn_student_v1.py','tiny_compression_v1/common.py']}
 protocol=dict(domain='asr/run2b',points=333315,effective_scored_samples=16,final_remainder=3,expected_updates=20833,control_updates=22147,control_output=str(CONTROL),same_initial_model_optimizer_all_rng_order=True,
  loss='weighted CE sum divided by sum weights across effective window, NEOT1 EOT2',lr=4e-6,wd=.01,training_reset_threshold=.1,
  changed='Grouping scored samples into effective16 windows; update count and dropout grouping/dynamic reset histories can differ. Not pure loss-normalizer attribution.',
  calibration_grid=[.00001,.00003,.0001,.0003,.001,.003,.01,.02,.03,.05,.075,.1,.2,.3,.4,.5,.6,.7,.8,.9],
  selection='maximum worst-language F1 then sum F1; ascending threshold tie break; calibration only; one monitor per arm',no_test=True,no_default_change=True,budget_epochs=1,source_hashes=source_hashes,train_sha256=manifest['train_sha256'])
 if (OUT/'locked_protocol.json').exists():assert json.loads((OUT/'locked_protocol.json').read_text())==protocol
 else:save('locked_protocol.json',protocol)
 save('dataset_manifest.json',manifest)
 path=OUT/'runs/finetune/latest_resume.pt'
 if not path.exists():
  import torch
  x=torch.load(CONTROL/'frozen/initial_child_resume.pt',map_location='cpu',weights_only=False,mmap=True)
  assert x['step']==3731173 and x['scored_seen']==0 and x['stream_state'] is None
  x=dict(x);x.update(fingerprint=protocol,accumulator=None,branch_step=0,training_trace=[])
  old.atomic_torch(path,x);(OUT/'frozen').mkdir(exist_ok=True);os.link(path,OUT/'frozen/initial_resume.pt')
  save('migration_receipt.json',dict(complete=True,initial_sha256=sha(path),control_initial_sha256=sha(CONTROL/'frozen/initial_child_resume.pt'),same_model_optimizer_rng_order=True,parent_step=3731173))
 return manifest,protocol

def train(manifest,protocol):
 import torch
 from transformers import AutoTokenizer
 if (OUT/'training_report.json').exists():return
 x=torch.load(OUT/'runs/finetune/latest_resume.pt',map_location='cpu',weights_only=False);assert x['fingerprint']==protocol
 m=old.make_model();m.load_state_dict(x['model']);m.train();o=torch.optim.AdamW(m.parameters(),lr=4e-6,weight_decay=.01);o.load_state_dict(x['optimizer'])
 assert all(abs(g['lr']-4e-6)<1e-15 and g['weight_decay']==.01 for g in o.param_groups)
 tok=AutoTokenizer.from_pretrained(MAIN/'tokenizer',local_files_only=True,token=False)
 src=initialize(json.load(gzip.open(OUT/'train_streams.json.gz','rt',encoding='utf-8')));r=StreamOrder(src,x['order'],x['stream_state'],lanes=16)
 a=Accumulator(m,o,r,tok,state=x.get('accumulator'));seen=x['scored_seen'];by=Counter(x['seen_by_key']);trace=x.get('training_trace',[]);order=x['order'];sampler=x['sampler_rng'];old.restore_rng(x);del x
 def checkpoint(epoch=0):
  guard();x=dict(model={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},optimizer=o.state_dict(),step=3731173+a.updates,branch_step=a.updates,parent_step=3731173,parameters=11941122,epoch=epoch,order=order,sampler_rng=sampler,cursor=r.next,stream_state=r.state(),accumulator=a.state(),scored_seen=seen,seen_by_key=dict(by),training_trace=trace,quant_scales={},fingerprint=protocol,**old.rng_state())
  old.atomic_torch(OUT/'runs/finetune/latest_resume.pt',x)
  save('resume_receipt.json',dict(step=x['step'],branch_step=a.updates,seen=seen,expected=333315,pending_gradient_samples=a.count,pending_rows=len(a.pending),optimizer_entries=len(o.state),windows=dict(a.windows),checkpoint_sha256=sha(OUT/'runs/finetune/latest_resume.pt')))
 last_saved=a.updates
 while True:
  if STOP or (OUT/'STOP_REQUESTED').exists():checkpoint();raise InterruptedError('User stop with accumulator gradients and pending lane queue saved')
  z=a.next()
  if z is None:break
  rows,seqs,probs,window,updated=z
  for row,seq,p in zip(rows,seqs,probs):
   if row['scored']:seen+=1;by[row['language']+':k'+str(row['k'])+':'+str(row['label'])]+=1
   if window<=100:trace.append(dict(stream=row['stream'],boundary=row['boundary'],k=row['k'],start=row['start'],label=row['label'],scored=row['scored'],p_eot=p,window=window,input_ids_sha256=hashlib.sha256(np.asarray(seq,dtype='<u4').tobytes()).hexdigest()))
  if updated and a.updates%50==0:status('TRAINING_EFFECTIVE16',updates=a.updates,seen=seen,expected=333315)
  if updated and (a.updates==100 or a.updates%500==0) and a.updates!=last_saved:
   checkpoint();last_saved=a.updates
   if a.updates==100 and not (OUT/'frozen/first100_resume.pt').exists():os.link(OUT/'runs/finetune/latest_resume.pt',OUT/'frozen/first100_resume.pt')
 a.flush();assert seen==333315 and a.updates==20833 and a.windows==Counter({'16':20832,'3':1})
 assert dict(by)=={k:v for k,v in manifest['counts'].items() if ':k' in k}
 assert r.next==420 and not r.slots and not a.pending and a.count==0
 assert len(o.state)==27 and {int(v['step']) for v in o.state.values()}=={3731173+a.updates}
 assert all(torch.isfinite(v).all() for v in m.state_dict().values());checkpoint(1)
 old.atomic_torch(OUT/'runs/finetune/weights.pt',{k:v.detach().cpu().clone() for k,v in m.state_dict().items()})
 save('epoch_0001_coverage.json',dict(complete=True,expected=333315,seen=seen,by_key=dict(by),windows=dict(a.windows),updates=a.updates,all420tasks=True))
 save('training_report.json',dict(complete=True,step=3731173+a.updates,updates=a.updates,seen=seen,weights_sha256=sha(OUT/'runs/finetune/weights.pt'),resume_sha256=sha(OUT/'runs/finetune/latest_resume.pt')))
 del m,o;gc.collect();torch.cuda.empty_cache()

def evaluate(protocol):
 import torch
 from transformers import AutoTokenizer
 from . import uploaded_vi_en_sentence_mix_evaluation_v7 as ev
 from .malbert_calibration_audit_20261107_v1 import verify_trace as verifier,scorekey
 import turn_v2.malbert_calibration_audit_20261107_v1 as cal
 cal.OUT=OUT;ev.OUT=OUT
 tok=AutoTokenizer.from_pretrained(MAIN/'tokenizer',local_files_only=True,token=False)
 calibration=initialize(json.load(gzip.open(OUT/'dev_calibration.json.gz','rt',encoding='utf-8')));monitor=initialize(json.load(gzip.open(OUT/'dev_monitor.json.gz','rt',encoding='utf-8')))
 arms={};proofs=[]
 for arm,weightpath in [('control',CONTROL/'runs/finetune/weights.pt'),('effective16',OUT/'runs/finetune/weights.pt')]:
  m=old.make_model();m.load_state_dict(torch.load(weightpath,map_location='cpu',weights_only=True));m.eval();predict=ev.predictor(m,tok);curves=[]
  for j,t in enumerate(protocol['calibration_grid']):
   guard();name=f'{arm}_calibration_{j:02d}';trace=name+'.jsonl.gz'
   if (OUT/(name+'_proof.json')).exists():
    result=json.loads((OUT/(name+'.json')).read_text());proof=json.loads((OUT/(name+'_proof.json')).read_text());assert sha(OUT/trace)==proof['trace_sha256']
   else:
    status('CALIBRATING',arm=arm,index=j+1,total=len(protocol['calibration_grid']),threshold=t)
    result=ev.rollout(predict,calibration,t,trace);save(name+'.json',result);proof=verifier(trace,result,calibration,tok,predict);save(name+'_proof.json',proof)
   curves.append(result);proofs.append(proof)
  chosen=max(curves,key=scorekey);save(arm+'_selection_receipt.json',dict(threshold=chosen['threshold'],calibration=curves,selected_before_monitor=True))
  name=arm+'_monitor';status('SCORING_DEV_MONITOR',arm=arm,threshold=chosen['threshold']);result=ev.rollout(predict,monitor,chosen['threshold'],name+'.jsonl.gz');save(name+'.json',result)
  proof=verifier(name+'.jsonl.gz',result,monitor,tok,predict);save(name+'_proof.json',proof);proofs.append(proof)
  arms[arm]=dict(selected=result,weights_sha256=sha(weightpath));del m;gc.collect();torch.cuda.empty_cache()
 save('dev_report.json',dict(complete=True,arms=arms,no_test=True,calibration_policy=protocol['selection']))
 return arms,proofs

def verify_training():
 import torch
 from transformers import AutoTokenizer
 guard();init=torch.load(OUT/'frozen/initial_resume.pt',map_location='cpu',weights_only=False,mmap=True);control=torch.load(CONTROL/'frozen/initial_child_resume.pt',map_location='cpu',weights_only=False,mmap=True)
 for k in ['model','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng','sampler_rng','order']:assert old.equal(init[k],control[k])
 first=torch.load(OUT/'frozen/first100_resume.pt',map_location='cpu',weights_only=False,mmap=True);tok=AutoTokenizer.from_pretrained(MAIN/'tokenizer',local_files_only=True,token=False)
 src=initialize(json.load(gzip.open(OUT/'train_streams.json.gz','rt',encoding='utf-8')));runner=StreamOrder(src,init['order'],lanes=16);trace=first['training_trace'];offset=0;windowcounts=Counter()
 while offset<len(trace):
  rows,states=runner.rows();part=trace[offset:offset+len(rows)];rows=rows[:len(part)];states=states[:len(part)]
  for r,z in zip(rows,part):
   for k in ['stream','boundary','k','start','label','scored']:assert r[k]==z[k]
   seq=encode_pair(tok,r['context'],r['text'],256)['input_ids'];assert hashlib.sha256(np.asarray(seq,dtype='<u4').tobytes()).hexdigest()==z['input_ids_sha256']
   if r['scored']:windowcounts[z['window']]+=1
  runner.advance(states,[z['p_eot'] for z in part],.1);offset+=len(part)
 assert sum(windowcounts.values())==1600 and all(n==16 for n in windowcounts.values())
 latest=torch.load(OUT/'runs/finetune/latest_resume.pt',map_location='cpu',weights_only=False,mmap=True)
 assert latest['epoch']==1 and latest['scored_seen']==333315 and latest['accumulator']['count']==0 and not latest['accumulator']['pending']
 assert latest['branch_step']==20833 and latest['accumulator']['windows']=={'16':20832,'3':1}
 assert sorted(map(int,latest['order']))==list(range(420)) and latest['cursor']==420 and not latest['stream_state']['slots']
 assert len(latest['optimizer']['state'])==27 and {int(v['step']) for v in latest['optimizer']['state'].values()}=={3752006}
 assert all(k in latest for k in ['torch_rng','cuda_rng','numpy_rng','python_rng','sampler_rng'])
 for state in latest['optimizer']['state'].values():
  for value in state.values():
   if torch.is_tensor(value):assert torch.isfinite(value).all()
 assert all(torch.isfinite(v).all() for v in latest['model'].values()) and any(not torch.equal(v,init['model'][k]) for k,v in latest['model'].items())
 save('independent_training_verification.json',dict(complete=True,initial_control_model_optimizer_all_rng_order_exact=True,first100_actual_training_inputs=len(trace),first100_scored=1600,all100windows16=True,causal_inputs_labels_masks_predicted_resets_verified=True,full_coverage333315=True,full_resume27optimizer_rng_order_lanes_accumulator_verified=True,updates=20833))

def run():
 import torch
 import transformers.utils as u
 import transformers.utils.import_utils as im
 u.is_torchvision_available=im.is_torchvision_available=lambda:False;u.is_librosa_available=im.is_librosa_available=lambda:False
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.5);assert not (OUT/'report.json').exists()
 status('PREPARING');manifest,protocol=prepare()
 if not (OUT/'technical_fixture_verification.json').exists():fixture()
 train(manifest,protocol);status('VERIFYING_TRAINING');verify_training();arms,proofs=evaluate(protocol)
 guard();cp=json.loads((old.BASE/'checkpoint_receipt.json').read_text());assert sha(MAIN/'runs/finetune/latest_resume.pt')==cp['full_checkpoint_sha256']
 for n,h in protocol['source_hashes'].items():assert sha(ROOT/'turn_v2'/n)==h
 save('report.json',dict(complete=True,arms=arms,protocol=protocol,training=json.loads((OUT/'training_report.json').read_text()),no_test=True,no_default_replacement=True,parent_paused=True))
 save('completion_verification.json',dict(complete=True,report_sha256=sha(OUT/'report.json'),calibration_and_monitor_inferences_audited=sum(p['inferences'] for p in proofs),canonical_cases=sum(p['canonical_single_case_count'] for p in proofs),max_probability_delta=max(p['max_probability_delta'] for p in proofs),all_new_traces_audited=True,full_training_verified=True,parent_checkpoint_unchanged=True,no_test=True))
 save('task_state.json',dict(complete=True,reported=False,phase='COMPLETE',parent_paused=True,adopted=False,auto_extend_authorized=False));status('COMPLETE',report_sha256=sha(OUT/'report.json'))

if __name__=='__main__':
 OUT.mkdir(exist_ok=True);os.nice(10)
 def stop(*args):
  global STOP
  STOP=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 with (OUT/'CAMPAIGN_LOCK').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:run()
  except InterruptedError as exc:status('STOPPED_WITH_CHECKPOINT',reason=str(exc))
  except Exception as exc:save('error.json',dict(error=repr(exc),traceback=traceback.format_exc()));status('ERROR',error=repr(exc));raise
