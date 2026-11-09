"""Disk-bounded launch/status/collection; no full teachers downloaded locally."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from colab_runtime import ColabRuntime
from storage_guard import ensure_free_space

ROOT=Path('outputs/reward_v10_20261025');NAME='reward_v10_campaign_20261025'
INITIAL={'vi':'outputs/qat_20261024/runs/qat_gru96_g64_20261024_r2','en':'outputs/monolingual_20261025/runs/mono_en_student_20261025/qat_best','zh':'outputs/monolingual_20261025/runs/mono_zh_student_20261025/qat_best'}


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','recover','status','collect']);a=p.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    r=ColabRuntime(0);connection=r.connect()
    if a.action=='launch':
        ensure_free_space(ROOT,300_000_000)
        paths=[*Path('turn_v2').glob('*.py'),Path('tools/distill_job.py'),Path('tools/reward_v10_campaign.py'),*Path('data_v10/reward_20261025').rglob('*.json*')]
        for initial in INITIAL.values():
            paths.extend(Path(initial)/n for n in ['config.json','tokenizer.json','tokenizer_config.json','turn_detector_config.json','student.pt','quant_params.pt'])
        archive=ROOT/'upload.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for path in paths:z.write(path,path.as_posix())
        data=archive.read_bytes();checksum=hashlib.sha256(data).hexdigest()
        result=r.execute("from pathlib import Path\nPath('/content/turn_detect_v2/colab').mkdir(parents=True,exist_ok=True)",timeout=30,echo=False);assert result['status']=='ok'
        r.request('PUT','api/contents/content/turn_detect_v2/colab/reward_v10_upload.zip',json={'type':'file','format':'base64','content':base64.b64encode(data).decode('ascii')})
        code=f'''from pathlib import Path
import hashlib,json,subprocess,sys,zipfile,shutil
root=Path('/content/turn_detect_v2');name={NAME!r};assert not (root/'jobs'/name).exists(),'Existing job; inspect instead of relaunch'
assert shutil.disk_usage(root).free>15_000_000_000
archive=root/'colab/reward_v10_upload.zip';assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()=={checksum!r}
with zipfile.ZipFile(archive) as z:
 for member in z.infolist():
  target=(root/member.filename).resolve();assert target.is_relative_to(root.resolve());data=z.read(member)
  if target.exists():assert target.read_bytes()==data,'Existing source differs: '+member.filename
  else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
(root/'jobs').mkdir(exist_ok=True)
command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','9300','--',sys.executable,'-u','tools/reward_v10_campaign.py']
with (root/'jobs'/(name+'_launcher.log')).open('w') as log:
 process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'job':name,'launcher_pid':process.pid,'upload_bytes':archive.stat().st_size,'upload_sha256':{checksum!r}}}))
'''
    elif a.action=='recover':
        source=Path('tools/reward_v10_recover.py');checksum=hashlib.sha256(source.read_bytes()).hexdigest()
        r.request('PUT','api/contents/content/turn_detect_v2/colab/reward_v10_recover.py',json={'type':'file','format':'base64','content':base64.b64encode(source.read_bytes()).decode('ascii')})
        code=f'''from pathlib import Path
import hashlib,json,subprocess,sys
root=Path('/content/turn_detect_v2');name='reward_v10_recovery_20261025';assert not (root/'jobs'/name).exists()
data=(root/'colab/reward_v10_recover.py').read_bytes();assert hashlib.sha256(data).hexdigest()=={checksum!r}
target=root/'tools/reward_v10_recover.py'
if target.exists():assert target.read_bytes()==data
else:target.write_bytes(data)
command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','8100','--',sys.executable,'-u','tools/reward_v10_recover.py']
with (root/'jobs'/(name+'_launcher.log')).open('w') as log:
 process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'job':name,'launcher_pid':process.pid,'source_sha256':{checksum!r}}}))
'''
    elif a.action=='status':
        code=f'''from pathlib import Path
import json,subprocess
root=Path('/content/turn_detect_v2');value={{}}
names=['{NAME}','reward_v10_qwen_20261025','reward_v10_recovery_20261025']+['reward_v10_'+lang+'_20261025'+suffix for lang in ['vi','en','zh'] for suffix in ['','_r2']]
for name in names:
 p=root/'jobs'/name/'status.json'
 if p.exists():
  state=json.loads(p.read_text());value[name]={{k:state.get(k) for k in ['status','exit_code','elapsed_seconds']}}
  if state['status'] in ['FAILED','TIMEOUT']:value[name]['log_tail']=(p.parent/'stdout.log').read_text()[-5000:]
for relative in ['progress.json','recovery_progress.json','qwen/progress.json','qwen/report.json']+[lang+'/'+file for lang in ['vi','en','zh'] for file in ['progress.json','report.json']]:
 p=root/'outputs/reward_v10_20261025'/relative
 if p.exists():value[relative]=json.loads(p.read_text())
value['GPU']=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
print(json.dumps(value))
'''
    else:
        code=f'''from pathlib import Path
import hashlib,json,zipfile
root=Path('/content/turn_detect_v2');proof=root/'jobs/reward_v10_recovery_20261025/status.json'
if not proof.exists():proof=root/'jobs/{NAME}/status.json'
state=json.loads(proof.read_text());assert state['status']=='COMPLETED' and state['exit_code']==0
target=Path('/content/reward_v10_results.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
 for folder in ['outputs/reward_v10_20261025']+['jobs/'+name for name in ['{NAME}','reward_v10_qwen_20261025','reward_v10_recovery_20261025']+['reward_v10_'+lang+'_20261025'+suffix for lang in ['vi','en','zh'] for suffix in ['','_r2']]]:
  for p in (root/folder).rglob('*'):
   if p.is_file():z.write(p,p.relative_to(root))
print(json.dumps({{'path':'content/'+target.name,'bytes':target.stat().st_size,'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest()}}))
'''
    result=r.execute(code,timeout=120,echo=False);receipt=ROOT/(a.action+'_receipt.json');receipt.write_text(json.dumps({'connection':connection,**result},indent=2),encoding='utf-8')
    if result['status']!='ok':raise RuntimeError('Inspect saved receipt; no automatic execution resubmit')
    value=json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type']=='stream'))
    if a.action=='collect':
        ensure_free_space(ROOT,value['bytes']*4);archive=ROOT/'results.zip';r.download(value['path'],archive)
        assert archive.stat().st_size==value['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==value['sha256']
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                name=member.filename
                if name.startswith('outputs/reward_v10_20261025/'):name=name[len('outputs/reward_v10_20261025/'):]
                target=(ROOT/name).resolve();assert target.is_relative_to(ROOT.resolve());data=z.read(member)
                if target.exists():assert target.read_bytes()==data
                else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    if a.action=='status':
        compact={}
        for key,item in value.items():
            if key.endswith('/report.json') and 'phases' in item:
                compact[key]={'selected_phase':item['selected_phase'],'RL_beats_CE':item['RL_beats_matched_CE_on_selection_score'],'RL_beats_cost_CE':item['RL_beats_cost_CE_on_selection_score'],
                    'natural_dev':{v['phase']:{k:v['dev']['slices']['natural'][k] for k in ['accuracy','eot_f1','false_eot_rate']} for v in item['phases']}}
            else:compact[key]=item
        value=compact
    print(json.dumps(value),flush=True);r.close()


if __name__=='__main__':main()
