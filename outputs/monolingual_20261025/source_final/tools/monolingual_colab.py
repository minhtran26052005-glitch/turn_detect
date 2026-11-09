"""Prepare/launch/inspect independent language models in the selected Colab VM."""
import argparse
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import zipfile
from colab_runtime import ColabRuntime

ROOT=Path('outputs/monolingual_20261025');NAME='monolingual_campaign_20261025'


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','status','collect']);p.add_argument('--session');a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);r=ColabRuntime(0,a.session);connection=r.connect()
    if a.action=='launch':
        # Only exports missing from a stock Colab VM. Torch/Transformers stay as
        # installed; their actual versions are captured by the training receipt.
        dependencies={n:importlib.metadata.version(n) for n in ['onnx','onnxruntime']}
        (ROOT/'dependency_versions.json').write_text(json.dumps(dependencies,indent=2))
        paths=[*Path('turn_v2').glob('*.py'),Path('tools/distill_job.py'),Path('tools/monolingual_campaign.py'),ROOT/'dependency_versions.json',
            *Path('data_v9/monolingual_20261025').glob('*/*.json*'),Path('data_v9/monolingual_20261025/recipe.json')]
        archive=ROOT/'upload.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for path in paths:z.write(path,path.as_posix())
        data=archive.read_bytes();digest=hashlib.sha256(data).hexdigest()
        result=r.execute("from pathlib import Path\nPath('/content/turn_detect_v2/colab').mkdir(parents=True,exist_ok=True)",timeout=30,echo=False)
        assert result['status']=='ok'
        r.request('PUT','api/contents/content/turn_detect_v2/colab/monolingual_upload.zip',json={'type':'file','format':'base64','content':base64.b64encode(data).decode('ascii')})
        code=f'''from pathlib import Path
import hashlib,json,subprocess,sys,zipfile,shutil
root=Path('/content/turn_detect_v2');name={NAME!r}
assert not (root/'jobs'/name).exists(),'Inspect previous campaign instead of launching twice'
assert shutil.disk_usage(root).free>10_000_000_000,'Need10GB for two independent teachers/caches; do not delete existing files automatically'
archive=root/'colab/monolingual_upload.zip';assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()=={digest!r}
with zipfile.ZipFile(archive) as z:
 for member in z.infolist():
  path=(root/member.filename).resolve();assert path.is_relative_to(root.resolve());data=z.read(member)
  if path.exists():assert path.read_bytes()==data,'Remote file version differs: '+member.filename
  else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
(root/'jobs').mkdir(exist_ok=True)
command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','10200','--',sys.executable,'-u','tools/monolingual_campaign.py']
with (root/'jobs'/(name+'_launcher.log')).open('w') as log:
 process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'job':name,'launcher_pid':process.pid,'upload_sha256':{digest!r}}}))
'''
    elif a.action=='status':
        code=f'''from pathlib import Path
import json
root=Path('/content/turn_detect_v2');values={{}}
for rel in ['jobs/{NAME}/status.json','outputs/monolingual_20261025/progress.json']+[p for lang in ['en','zh'] for phase in ['teacher','student'] for p in ['jobs/mono_'+lang+'_'+phase+'_20261025/status.json','runs/mono_'+lang+'_'+phase+'_20261025/progress.json','runs/mono_'+lang+'_'+phase+'_20261025/training_report.json','runs/mono_'+lang+'_'+phase+'_20261025/monolingual_report.json']]:
 path=root/rel
 if path.exists():values[rel]=json.loads(path.read_text())
for name in ['{NAME}']+['mono_'+lang+'_'+phase+'_20261025' for lang in ['en','zh'] for phase in ['teacher','student']]:
 path=root/'jobs'/name/'status.json'
 if path.exists() and json.loads(path.read_text())['status'] in ['FAILED','TIMEOUT']:
  log=path.parent/'stdout.log';values[name+'_error']=log.read_text()[-4500:] if log.exists() else ''
print(json.dumps(values))
'''
    else:
        code=f'''from pathlib import Path
import hashlib,json,zipfile
root=Path('/content/turn_detect_v2');state=json.loads((root/'jobs/{NAME}/status.json').read_text());assert state['status']=='COMPLETED' and state['exit_code']==0
target=Path('/content/monolingual_results.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
 for folder in ['outputs/monolingual_20261025','jobs/{NAME}']+['runs/mono_'+lang+'_student_20261025' for lang in ['en','zh']]+['jobs/mono_'+lang+'_'+phase+'_20261025' for lang in ['en','zh'] for phase in ['teacher','student']]:
  for p in (root/folder).rglob('*'):
   if p.is_file():z.write(p,p.relative_to(root))
 for lang in ['en','zh']:
  for p in (root/('runs/mono_'+lang+'_teacher_20261025')).glob('*'):
   if p.is_file():z.write(p,p.relative_to(root))
  for p in (root/('runs/mono_'+lang+'_teacher_20261025/best')).glob('*'):
   if p.is_file() and p.name!='model.safetensors':z.write(p,p.relative_to(root))
print(json.dumps({{'path':'content/'+target.name,'bytes':target.stat().st_size,'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest()}}))
'''
    result=r.execute(code,timeout=120,echo=False)
    receipt=ROOT/(a.action+'_receipt.json');receipt.write_text(json.dumps({'connection':connection,**result},indent=2))
    if result['status']!='ok':raise RuntimeError('Inspect '+str(receipt)+'; execution is not silently repeated')
    value=json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type']=='stream'))
    if a.action=='collect':
        archive=ROOT/'results.zip';r.download(value['path'],archive);assert hashlib.sha256(archive.read_bytes()).hexdigest()==value['sha256']
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                name=member.filename
                if name.startswith('outputs/monolingual_20261025/'):name=name[len('outputs/monolingual_20261025/'):]
                target=(ROOT/name).resolve();assert target.is_relative_to(ROOT.resolve());data=z.read(member)
                if target.exists():assert target.read_bytes()==data
                else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    if a.action=='status':
        compact={}
        for key,v in value.items():
            if isinstance(v,dict):
                compact[key]={k:x for k,x in v.items() if k in ['status','stage','language','phase','step','steps','steps_max','epoch','loss','elapsed_seconds','exit_code','overall_dev','natural_target_met','natural_dev_target_met','model_bytes']}
                if 'dev_slices' in v:compact[key]['natural_dev']=v['dev_slices'].get('natural')
            else:compact[key]=v
        value=compact
    print(json.dumps(value),flush=True);r.close()


if __name__=='__main__':main()
