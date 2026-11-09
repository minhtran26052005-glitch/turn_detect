"""Run bounded automatic crawl/reward/update loop on the current Colab GPU."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from colab_runtime import ColabRuntime
from storage_guard import ensure_free_space
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.common import input_id,read_jsonl

ROOT=Path('outputs/auto_rl_v11_20261025');NAME='auto_rl_v11_20261025'


def main():
    global ROOT,NAME
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','status','collect']);p.add_argument('--revision',type=int,choices=[1,2],default=1);a=p.parse_args()
    module='auto_rl_v11' if a.revision==1 else 'auto_rl_v12'
    if a.revision==2:ROOT=Path('outputs/auto_rl_v12_20261025');NAME='auto_rl_v12_20261025'
    ROOT.mkdir(parents=True,exist_ok=True)
    r=ColabRuntime(0);connection=r.connect()
    if a.action=='launch':
        sources=set();inputs=set()
        for name in ['data_v4/fresh_crawl_20261025/review_cases.jsonl','data_v6/pho_text_20261025/sealed.jsonl']:
            for row in read_jsonl(Path(name)):sources.add(row['source_id']);inputs.add(row['language']+input_id(row))
        source=Path('turn_v2')/(module+'.py')
        payload={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'protected_sources':sorted(sources),'protected_inputs':sorted(inputs),'targets_included':False}
        for remote,data in [('turn_v2/'+source.name,source.read_bytes()),('data_v10/reward_20261025/auto_rl_guard.json',json.dumps(payload,indent=2).encode())]:
            r.request('PUT','api/contents/content/turn_detect_v2/'+remote,json={'type':'file','format':'base64','content':base64.b64encode(data).decode('ascii')})
        code=f'''from pathlib import Path
import hashlib,json,subprocess,sys
root=Path('/content/turn_detect_v2');name={NAME!r}
state=json.loads((root/'jobs/reward_v10_recovery_20261025/status.json').read_text());assert state['status']=='COMPLETED' and state['exit_code']==0
assert not (root/'jobs'/name).exists() and not (root/'outputs'/name).exists(),'Existing loop; inspect instead of restarting'
assert hashlib.file_digest((root/{source.as_posix()!r}).open('rb'),'sha256').hexdigest()=={payload['source_sha256']!r}
command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','1800','--',sys.executable,'-u','-m',{'turn_v2.'+module!r}]
with (root/'jobs'/(name+'_launcher.log')).open('w') as log:
 process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'job':name,'launcher_pid':process.pid,'source_sha256':{payload['source_sha256']!r}}}))
'''
    elif a.action=='status':
        code=f'''from pathlib import Path
import json,subprocess
root=Path('/content/turn_detect_v2');value={{}}
for relative in ['jobs/{NAME}/status.json','outputs/{NAME}/progress.json','outputs/{NAME}/report.json']:
 p=root/relative
 if p.exists():value[relative]=json.loads(p.read_text())
state=value.get('jobs/{NAME}/status.json',{{}})
if state.get('status') in ['FAILED','TIMEOUT']:value['error']=(root/'jobs/{NAME}/stdout.log').read_text()[-5000:]
probe=root/'outputs/{NAME}/judge_probe.jsonl'
if probe.exists():value['judge_probe']=[{{k:row.get(k) for k in ['language','expected','judge_label','judge_accepted']}} for row in [json.loads(line) for line in probe.read_text().splitlines()]]
value['GPU']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
print(json.dumps(value))
'''
    else:
        code=f'''from pathlib import Path
import hashlib,json,zipfile
root=Path('/content/turn_detect_v2');state=json.loads((root/'jobs/{NAME}/status.json').read_text());assert state['status']=='COMPLETED' and state['exit_code']==0
target=Path('/content/auto_rl_v11_results.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
 for directory in ['jobs/{NAME}','outputs/{NAME}']:
  for p in (root/directory).rglob('*'):
   if p.is_file():z.write(p,p.relative_to(root))
print(json.dumps({{'path':'content/'+target.name,'bytes':target.stat().st_size,'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest()}}))
'''
    result=r.execute(code,timeout=120,echo=False);destination=ROOT/(a.action+'_receipt.json');destination.write_text(json.dumps({'connection':connection,**result},indent=2),encoding='utf-8')
    assert result['status']=='ok','Inspect saved receipt; no silent repeat'
    value=json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type']=='stream'))
    if a.action=='collect':
        ensure_free_space(ROOT,value['bytes']*4);archive=ROOT/'results.zip';r.download(value['path'],archive);assert hashlib.sha256(archive.read_bytes()).hexdigest()==value['sha256']
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                name=member.filename
                if name.startswith('outputs/'+NAME+'/'):name=name[len('outputs/'+NAME+'/'):]
                target=(ROOT/name).resolve();assert target.is_relative_to(ROOT.resolve());data=z.read(member)
                if target.exists():assert target.read_bytes()==data
                else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    print(json.dumps(value),flush=True);r.close()


if __name__=='__main__':main()
