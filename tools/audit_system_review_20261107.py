"""Read-only source/schedule audit. No model inference, training, or config edits."""
import json, subprocess, shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/system_review_20261107'
REMOTE=r'''
import json,gzip,pathlib,hashlib,collections,shutil,datetime,subprocess,torch
torch.set_num_threads(2)
r=pathlib.Path('/home/edabk/hoangbpm/turn_detect'); a=r/'outputs/asr_only_adaptation_20261107_v1'; p=r/'outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7'
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
m=json.loads((a/'dataset_manifest.json').read_text());assert sha(a/'train_streams.json.gz')==m['train_sha256']
streams=json.load(gzip.open(a/'train_streams.json.gz','rt')); x=torch.load(a/'frozen/initial_child_resume.pt',map_location='cpu',weights_only=False,mmap=True)
# Independent schedule from raw segment positions, not production StreamOrder.
tasks=[]
for s in streams:
 for k in (1,2):
  e=set(s['eots']);u=set(s['unknown']);mask=set(s.get('conflict_mask',{}).get(str(k),[]));rows=[]
  for seg in s['segments']:
   if not seg.get('punctuation_present',True):continue
   for b in range(seg['start']+1,seg['end']-k+1):rows.append((b not in u and b not in mask,int(b in e),seg['language']))
  tasks.append(rows)
order=[int(i) for i in x['order']];lanes=[];cursor=0;hist=collections.Counter();points=collections.Counter();coefs=collections.Counter();tail=[]
while True:
 while len(lanes)<16 and cursor<len(order):lanes.append([order[cursor],0]);cursor+=1
 active=[v for v in lanes if v[1]<len(tasks[v[0]])]
 if not active:
  lanes=[]
  if cursor==len(order):break
  continue
 rows=[tasks[t][j] for t,j in active];scored=[row for row in rows if row[0]]
 if scored:
  n=len(scored);pos=sum(row[1] for row in scored);den=n+pos
  hist['updates']+=1;hist['valid_points_'+str(n)]+=1
  hist['all_NEOT_updates' if pos==0 else 'all_EOT_updates' if pos==n else 'mixed_label_updates']+=1
  for _,y,lang in scored:points[lang+':'+str(y)]+=1;coefs[lang+':'+str(y)]+=(1+y)/den
  tail.append({'points':n,'positive':pos})
 lanes=[[t,j+1] for t,j in active]
assert hist['updates']==22147 and sum(points.values())==333315
proc=[]
for q in pathlib.Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:cmd=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 except (OSError,UnicodeError):continue
 if cmd.startswith(str(r/'.venv/bin/python')) and (' -m turn_v2.multilingual_malbert_vi_en_sentence_mix_train_v7' in cmd or ' -m turn_v2.asr_only_adaptation_20261107_v1' in cmd):proc.append({'pid':q.name,'cmd':cmd})
mount=[v.split() for v in pathlib.Path('/proc/mounts').read_text().splitlines() if v.split()[1]=='/mnt/hdd500']
files=['asr_only_adaptation_20261107_v1.py','uploaded_vi_en_data_v4.py','uploaded_vi_en_data_v5.py','uploaded_vi_en_sentence_mix_evaluation_v7.py','malbert_turn_student_v1.py','tiny_compression_v1/common.py']
hashes={n:sha(r/'turn_v2'/n) for n in files};assert hashes==m['code_sha256']
reports={str(q.relative_to(r)):sha(q) for q in [a/'report.json',a/'completion_verification.json',p/'migration_receipt.json',r/'outputs/sentence_mix_end_only_test_20261107_v1/report.json']}
print(json.dumps(dict(complete=True,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Independent source/order-only schedule; no inference or optimizer updates',boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),active_target_processes=proc,parent_stop_marker=(p/'STOP_REQUESTED').exists(),parent_status=json.loads((p/'status.json').read_text()),trial_status=json.loads((a/'status.json').read_text()),update_histogram=dict(hist),points=dict(points),sum_per_batch_loss_coefficients=dict(coefs),last1000_points=sum(z['points'] for z in tail[-1000:]),source_hashes=hashes,report_hashes=reports,mount=mount,ssd_free_gib=shutil.disk_usage(r).free/2**30,hdd_free_gib=shutil.disk_usage('/mnt/hdd500').free/2**30 if mount else None)))
'''

def main():
    assert shutil.disk_usage('C:/').free>10*2**30+5*2**20
    remote_cmd='cd /home/edabk/hoangbpm/turn_detect && LD_LIBRARY_PATH=/home/edabk/miniconda3/envs/.VLLM/lib/python3.10/site-packages/nvidia/cu13/lib .venv/bin/python -'
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','edabk@100.91.209.124',remote_cmd],input=REMOTE,encoding='utf-8',capture_output=True,timeout=60)
    if result.returncode:raise RuntimeError(result.stderr)
    value=json.loads(result.stdout);value['windows_free_gib']=shutil.disk_usage('C:/').free/2**30
    OUT.mkdir(exist_ok=True);(OUT/'independent_schedule_and_state.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value,ensure_ascii=False))

if __name__=='__main__':main()
