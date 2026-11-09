"""Prepare a corrected causal text packet while preserving V5 and blind cases."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from colab_runtime import ColabRuntime
from qat_colab import SESSION

NAME='natural_v6_prepare_20261025'
ROOT=Path('outputs/natural_v6_20261025')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','status','collect'])
    args=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    runtime=ColabRuntime(0,SESSION);connection=runtime.connect()
    if args.action=='prepare':
        source=Path('turn_v2/natural_v6_data.py');digest=hashlib.sha256(source.read_bytes()).hexdigest()
        runtime.upload(source,'content/turn_detect_v2/colab/natural_v6_data.py')
        code=f'''from pathlib import Path
import hashlib,json,os,subprocess,sys
root=Path('/content/turn_detect_v2');name={NAME!r}
for job in ['natural_v5_train_20261025','natural_v5_qwen_dev_20261025']:
    status=json.loads((root/'jobs'/job/'status.json').read_text());assert status['status']=='COMPLETED' and status['exit_code']==0
assert not (root/'jobs'/name).exists(), 'Inspect existing preparation; no duplicate launch'
data=(root/'colab/natural_v6_data.py').read_bytes();assert hashlib.sha256(data).hexdigest()=={digest!r}
target=root/'turn_v2/natural_v6_data.py'
if target.exists():assert target.read_bytes()==data
else:target.write_bytes(data)
env=os.environ.copy();env['PYTHONPATH']=str(root/'.deps-v5-vad')+os.pathsep+str(root)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','3600','--',sys.executable,'-u','-m','turn_v2.natural_v6_data']
with (root/'jobs'/(name+'_launcher.log')).open('w') as stream:
    process=subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'job':name,'launcher_pid':process.pid,'source_sha256':{digest!r}}}))
'''
    elif args.action=='status':
        code=f'''from pathlib import Path
import json
root=Path('/content/turn_detect_v2');value={{}}
for rel in ['jobs/{NAME}/status.json','data_v6/turn_context_20261025/progress.json','data_v6/turn_context_20261025/manifest.json']:
    path=root/rel
    if path.exists():value[rel]=json.loads(path.read_text())
log=root/'jobs/{NAME}/stdout.log'
if log.exists():value['log_tail']=log.read_text()[-2500:]
print(json.dumps(value))
'''
    else:
        code=f'''from pathlib import Path
import hashlib,json,zipfile
root=Path('/content/turn_detect_v2')
status=json.loads((root/'jobs/{NAME}/status.json').read_text());assert status['status']=='COMPLETED' and status['exit_code']==0
target=Path('/content/natural_v6_data_receipts.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for folder in ['data_v6/turn_context_20261025','jobs/{NAME}']:
        for path in sorted((root/folder).rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(root))
print(json.dumps({{'path':'content/'+target.name,'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),'bytes':target.stat().st_size}}))
'''
    result=runtime.execute(code,timeout=120,echo=False)
    (ROOT/(args.action+'.json')).write_text(json.dumps({'connection':connection,**result},indent=2),encoding='utf-8')
    assert result['status']=='ok','Inspect saved response; never blindly resubmit execution'
    value=json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type']=='stream'))
    if args.action=='collect':
        archive=ROOT/'data_receipts.zip';runtime.download(value['path'],archive)
        assert archive.stat().st_size==value['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==value['sha256']
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                if item.filename.startswith('jobs/'):target=(ROOT/item.filename).resolve();assert target.is_relative_to(ROOT.resolve())
                else:target=Path(item.filename).resolve();assert target.is_relative_to((Path.cwd()/'data_v6').resolve())
                data=source.read(item)
                if target.exists():assert target.read_bytes()==data
                else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    elif args.action=='status':
        value={key:(entry.get('files',entry) if not key.endswith('/status.json') else {k:entry.get(k) for k in ['status','exit_code','elapsed_seconds']}) if isinstance(entry,dict) else entry for key,entry in value.items()}
        if value.get('jobs/'+NAME+'/status.json',{}).get('status')!='FAILED':value.pop('log_tail',None)
    print(json.dumps(value),flush=True);runtime.close()


if __name__=='__main__':main()
