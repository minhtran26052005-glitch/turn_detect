"""Independent source/mask accounting and causal-resume/loss fixtures."""
import json,gzip,hashlib
from collections import Counter
from .uploaded_vi_en_data_v5 import OUT,PARENT,sha,initialize,StreamOrder,save
def run():
 import torch,numpy as np
 torch.set_num_threads(2)
 from torch.nn import functional as F
 a=initialize(json.load(gzip.open(PARENT/'train_streams.json.gz','rt',encoding='utf-8')))
 b=initialize(json.load(gzip.open(OUT/'train_streams.json.gz','rt',encoding='utf-8')))
 assert len(a)==len(b)==17301
 labels={};original=Counter();filtered=Counter();removed=Counter()
 def rows(s):
  for p in s['segments']:
   for k in (1,2):
    for n in range(p['start']+1,p['end']-k+1):
     if n in s['_unknown']:continue
     text=' '.join(s['words'][max(0,n-8):n])+'\0'+' '.join(s['words'][n:n+k])
     yield n,k,int(n in s['_eots']),p['language'],hashlib.sha256(text.encode()).digest()
 for x,y in zip(a,b):
  for k in ['key','family','domain','words','segments','eots','unknown','source_rows']:assert x[k]==y[k]
  for n,k,v,lang,h in rows(x):labels[h]=labels.get(h,0)|(1<<v);original[lang+':'+str(v)]+=1
 nconflict=sum(v==3 for v in labels.values())
 for s in b:
  masks={k:set(v) for k,v in s['conflict_mask'].items()}
  for n,k,v,lang,h in rows(s):
   excluded=n in masks[str(k)];assert excluded==(labels[h]==3)
   if excluded:removed[lang+':'+str(v)]+=1
   else:filtered[lang+':k'+str(k)+':'+('EOT' if v else 'NEOT')]+=1
 manifest=json.loads((OUT/'dataset_manifest.json').read_text())
 assert sum(original.values())==6169873
 assert all(manifest['train'][k]==v for k,v in filtered.items())
 audit=json.loads((OUT/'conflict_filter_audit.json').read_text());assert nconflict==audit['conflicting_keys'] and dict(removed)==audit['removed_by_language_label']
 for n in ['dev_full.json.gz','dev_calibration.json.gz','dev_monitor.json.gz','dev_diagnostic.json.gz','teacher_new_dev.json']:assert sha(PARENT/n)==sha(OUT/n)
 fixture=initialize([dict(words=['hi','how','can','i','help','you'],segments=[dict(start=0,end=6,language='en')],eots=[1],unknown=[],domain='fixture',conflict_mask={'1':[1],'2':[1]})])
 order=np.array([1]);runner=StreamOrder(fixture,order,lanes=1);rr,ss=runner.rows();assert not rr[0]['scored'] and rr[0]['label']==1
 runner.advance(ss,[0.],.1);rr,ss=runner.rows();assert rr[0]['scored'] and rr[0]['context']=='hi how'
 restored=StreamOrder(fixture,order,runner.state(),lanes=1);assert restored.rows()[0]==rr
 logits=torch.zeros((2,2),requires_grad=True);y=torch.tensor([0,1]);weights=torch.tensor([1.,2.]);loss=F.cross_entropy(logits,y,weight=weights)
 manual=(F.cross_entropy(logits,y,reduction='none')*weights[y]).sum()/weights[y].sum();assert torch.allclose(loss,manual)
 loss.backward();assert abs(float(logits.grad[1,1]/logits.grad[0,1]))==2.
 save('independent_preparation_verification.json',dict(complete=True,all_17301_source_streams_words_labels_unchanged=True,all_6169873_original_gaps_checked=True,conflicting_keys=nconflict,excluded_gaps=sum(removed.values()),remaining_gaps=sum(filtered.values()),removed_by_language_label=dict(removed),all_masked_keys_verified_both_labels=True,all_surviving_rule_keys_consistent=True,all_dev_and_teacher_files_byte_identical=True,predicted_only_reset_and_restored_lane_fixtures_passed=True,weighted_CE_value_and_twofold_relative_EOT_gradient_verified=True,dev_test_used_to_build_conflicts=False,manifest_sha256=sha(OUT/'dataset_manifest.json')))
if __name__=='__main__':run()
