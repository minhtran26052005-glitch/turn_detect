"""Isolated CPU fixture for partial-gradient and pending-lane resume."""
import subprocess,json
from pathlib import Path
CODE=r'''
import torch,copy,json,numpy as np
from turn_v2.asr_effective_batch16_20261107_v1 import Accumulator,save
from turn_v2.asr_only_adaptation_20261107_v1 import equal
from turn_v2.uploaded_vi_en_data_v5 import initialize,StreamOrder
torch.set_num_threads(2)
class Tok:
 cls_token_id=1;sep_token_id=3
 def encode(self,t,add_special_tokens=False):return [4+int(w[1:]) for w in t.split()]
class Tiny(torch.nn.Module):
 def __init__(self):
  super().__init__();self.e=torch.nn.Embedding(32,8);self.d=torch.nn.Dropout(.2);self.h=torch.nn.Linear(8,2)
 def forward(self,ids,mask):return self.h(self.d((self.e(ids)*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]))
src=initialize([dict(words=['w'+str(j) for j in range(1,22)],eots=[4,8,21],unknown=[2,5],conflict_mask={'1':[6],'2':[]},domain='fixture',segments=[dict(start=0,end=21,language='vi')]) for _ in range(5)])
order=np.array([1,4,0,3,5,2,8,9,6,7]);torch.manual_seed(919);m=Tiny();initial=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
def run(mode):
 m=Tiny();m.load_state_dict(initial);o=torch.optim.AdamW(m.parameters(),lr=4e-6,weight_decay=.01);torch.set_rng_state(rng);r=StreamOrder(src,order,lanes=16);a=Accumulator(m,o,r,Tok(),'cpu');cut=False;seen=0
 while True:
  z=a.next()
  if z is None:break
  seen+=sum(v['scored'] for v in z[0])
  condition=(mode=='pending' and bool(a.pending) and a.updates>0) or (mode=='gradient' and a.count>0 and a.updates>0)
  if condition and not cut:
   x=copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),runner=r.state(),acc=a.state(),rng=torch.get_rng_state()))
   m=Tiny();m.load_state_dict(x['model']);o=torch.optim.AdamW(m.parameters(),lr=4e-6,weight_decay=.01);o.load_state_dict(x['optimizer']);r=StreamOrder(src,order,x['runner'],lanes=16);a=Accumulator(m,o,r,Tok(),'cpu',x['acc']);torch.set_rng_state(x['rng']);cut=True
 a.flush();return dict(model=m.state_dict(),optimizer=o.state_dict(),runner=r.state(),acc=a.state(),rng=torch.get_rng_state(),seen=seen),cut
ref,_=run('none');a,cut1=run('pending');b,cut2=run('gradient');assert cut1 and cut2 and equal(ref,a) and equal(ref,b)
proof=dict(complete=True,technical_only=True,actual_accumulator_used=True,pending_lane_queue_resume_bitwise_equal=True,partial_gradients_resume_bitwise_equal=True,model_optimizer_rng_lanes_counts_equal=True,scored_points=ref['seen'],windows=ref['acc']['windows'])
save('pending_and_gradient_resume_verification.json',proof);print(json.dumps(proof))
'''
cmd='cd /home/edabk/hoangbpm/turn_detect && LD_LIBRARY_PATH=/home/edabk/miniconda3/envs/.VLLM/lib/python3.10/site-packages/nvidia/cu13/lib PYTHONPATH=/home/edabk/hoangbpm/turn_detect .venv/bin/python -'
r=subprocess.run(['ssh','-o','BatchMode=yes','edabk@100.91.209.124',cmd],input=CODE,text=True,encoding='utf-8',capture_output=True,timeout=45);assert r.returncode==0,r.stderr
x=json.loads(r.stdout);Path('outputs/asr_effective_batch16_20261107_v1/pending_and_gradient_resume_verification.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
