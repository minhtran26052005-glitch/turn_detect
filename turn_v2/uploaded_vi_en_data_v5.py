"""TRAIN-only conservative conflict masks; DEV and source words stay frozen."""
from .uploaded_vi_en_data_v4 import ROOT,LANGS,initialize,clean,sha,StreamOrder as ParentOrder
from collections import Counter
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,gzip,os,shutil
OUT=ROOT/'outputs/multilingual_malbert_vi_en_conflict_weighted_20261106_v5'
PARENT=ROOT/'outputs/multilingual_malbert_vi_en_uploaded_20261104_v4'
MODULES=['uploaded_vi_en_data_v4.py','uploaded_vi_en_data_v5.py','uploaded_vi_en_evaluation_v5.py','multilingual_malbert_vi_en_conflict_weighted_train_v5.py','malbert_turn_student_v1.py']
def save(name,value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def status(stage,file='status.json',**kw):
 s=Path('/proc/self/stat').read_text();v=dict(stage=stage,pid=os.getpid(),start_ticks=s[s.rfind(')')+2:].split()[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),at_utc=datetime.now(timezone.utc).isoformat(),active_languages=list(LANGS),**kw);save(file,v);print(json.dumps(v),flush=True)
def reserve():
 m=next(x.split() for x in Path('/proc/mounts').read_text().splitlines() if x.split()[1]=='/mnt/hdd500');assert m[0]=='/dev/sda1' and 'rw' in m[3].split(',')
 assert min(shutil.disk_usage(OUT).free,shutil.disk_usage('/mnt/hdd500').free)>=13*2**30
def write_gzip(name,value):
 p=OUT/name;tmp=p.with_name(p.name+'.tmp')
 with gzip.open(tmp,'wt',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,separators=(',',':'))
 with tmp.open('rb') as f:os.fsync(f.fileno())
 tmp.replace(p)
def key(s,b,k):
 # Fixed TRAIN-only rule: identical max-eight-word left suffix + same future.
 # These are feasible normalized causal inputs. Exclude all occurrences conservatively,
 # even where longer actual histories could disambiguate. No gold resets are fed to training.
 return hashlib.sha256((' '.join(s['words'][max(0,b-8):b])+'\0'+' '.join(s['words'][b:b+k])).encode()).digest()
def gaps(s):
 for seg in s['segments']:
  if not seg.get('punctuation_present',True):continue
  for k in (1,2):
   for b in range(seg['start']+1,seg['end']-k+1):
    if b not in s['_unknown']:yield b,k,int(b in s['_eots']),seg['language']
def counts(streams):
 c=Counter()
 for s in streams:
  c['streams']+=1;c['source_rows']+=s['source_rows'];c['words']+=len(s['words']);c['origin:'+s['domain']+':streams']+=1
  masks={k:set(v) for k,v in s.get('conflict_mask',{}).items()}
  for b,k,y,lang in gaps(s):
   if b not in masks.get(str(k),set()):c[lang+':k'+str(k)+':'+('EOT' if y else 'NEOT')]+=1
 return dict(c)
def prepare():
 OUT.mkdir(parents=True,exist_ok=True);reserve()
 if (OUT/'dataset_manifest.json').exists():return json.loads((OUT/'dataset_manifest.json').read_text())
 parent=json.loads((PARENT/'dataset_manifest.json').read_text());assert sha(PARENT/'train_streams.json.gz')==parent['files']['train_streams.json.gz']
 streams=initialize(json.load(gzip.open(PARENT/'train_streams.json.gz','rt',encoding='utf-8')))
 positive=set();conflicts=set();before=Counter();removed=Counter();examples={}
 status('SCANNING_TRAIN_EOT_KEYS',file='preparation_status.json')
 for s in streams:
  for b,k,y,lang in gaps(s):
   if y:positive.add(key(s,b,k))
 status('SCANNING_TRAIN_OPPOSING_LABELS',file='preparation_status.json',positive_keys=len(positive))
 for s in streams:
  for b,k,y,lang in gaps(s):
   if not y:
    h=key(s,b,k)
    if h in positive:conflicts.add(h)
 del positive
 for i,s in enumerate(streams):
  mask={'1':[],'2':[]}
  for b,k,y,lang in gaps(s):
   before[lang+':'+str(y)]+=1;h=key(s,b,k)
   if h in conflicts:
    mask[str(k)].append(b);removed[lang+':'+str(y)]+=1
    if len(examples)<12 or h.hex() in examples:
     group=examples.setdefault(h.hex(),{})
     group.setdefault(str(y),dict(stream=i,boundary=b,k=k,language=lang,context=' '.join(s['words'][max(0,b-8):b]),future=' '.join(s['words'][b:b+k])))
  s['conflict_mask']=mask
 # Independent second traversal: every surviving key must have one label only.
 bits={};masked=0
 for s in streams:
  masks={k:set(v) for k,v in s['conflict_mask'].items()}
  for b,k,y,lang in gaps(s):
   h=key(s,b,k);excluded=b in masks[str(k)];assert excluded==(h in conflicts)
   if excluded:masked+=1;continue
   bits[h]=bits.get(h,0)|(1<<y);assert bits[h]!=3
 assert masked==sum(removed.values())
 surviving_keys_checked=len(bits);write_gzip('train_streams.json.gz',clean(streams));del bits
 for n in ['dev_full.json.gz','dev_calibration.json.gz','dev_monitor.json.gz','dev_diagnostic.json.gz','teacher_new_dev.json']:
  if not (OUT/n).exists():shutil.copyfile(PARENT/n,OUT/n)
  assert sha(OUT/n)==sha(PARENT/n)
 audit=dict(complete=True,rule='Exact normalized eight-word left suffix (shorter at stream start) plus one/two same-turn future words; opposing TRAIN labels exclude BOTH classes at every matching source gap',train_only=True,test_or_dev_labels_used=False,source_words_and_punctuation_unchanged=True,conflicting_keys=len(conflicts),excluded_scored_gaps=masked,before_by_language_label=dict(before),removed_by_language_label=dict(removed),surviving_keys_checked=surviving_keys_checked,examples=list(examples.values()),all_surviving_rule_keys_label_consistent=True,limitations='Conservative local-context exclusion can remove positions resolvable with longer histories. Does not exhaust all dynamic reset contexts, WordPiece-only collisions or paraphrases; historical exposure is preserved, not unlearned.')
 save('conflict_filter_audit.json',audit)
 manifest=dict(parent);manifest.update(version=5,parent_manifest_sha256=sha(PARENT/'dataset_manifest.json'),train=counts(streams),loss=dict(kind='weighted_cross_entropy',class_weights={'NEOT':1.,'EOT':2.},reduction='weighted_mean'),conflict_filter=audit,code_sha256={n:sha(ROOT/'turn_v2'/n) for n in MODULES})
 manifest['files']={n:sha(OUT/n) for n in parent['files']};save('dataset_manifest.json',manifest)
 status('FILTERED_TRAIN_READY',file='preparation_status.json',conflicting_keys=len(conflicts),excluded_scored_gaps=masked,remaining_scored_gaps=sum(v for k,v in manifest['train'].items() if ':k' in k));return manifest
class StreamOrder(ParentOrder):
 def rows(self):
  rows,states=super().rows()
  for r in rows:
   s=self.streams[r['stream']]
   if '_conflict_sets' not in s:s['_conflict_sets']={k:set(v) for k,v in s.get('conflict_mask',{}).items()}
   r['scored']=r['scored'] and r['boundary'] not in s['_conflict_sets'].get(str(r['k']),set())
  return rows,states
if __name__=='__main__':prepare()
