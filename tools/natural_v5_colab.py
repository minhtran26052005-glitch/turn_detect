"""Manage new text-only data preparation without disturbing completed controls."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from colab_runtime import ColabRuntime
from qat_colab import SESSION

NAME='natural_v5_prepare_20261025'
EVIDENCE=Path('outputs/natural_v5_20261025')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','expand-vi','status','collect-data'])
    args=parser.parse_args();EVIDENCE.mkdir(parents=True,exist_ok=True)
    runtime=ColabRuntime(0,SESSION);connection=runtime.connect()
    if args.action=='prepare':
        probe=runtime.execute("""from pathlib import Path
import json,shutil
root=Path('/content/turn_detect_v2')
for name in ['stream_v4_campaign_20261025','stream_v4_balanced_20261025','stream_v4_final_20261025']:
    value=json.loads((root/'jobs'/name/'status.json').read_text());assert value['status']=='COMPLETED' and value['exit_code']==0
assert not (root/'jobs/natural_v5_prepare_20261025').exists(), 'Inspect existing job; no duplicate submission'
assert shutil.disk_usage(root).free>10_000_000_000
print('ready')
""",timeout=60,echo=False)
        assert probe['status']=='ok','Inspect existing state; no restart or duplicate launch'
        source_files=[Path('turn_v2/natural_v5_data.py'),Path('tools/natural_v5_prepare_job.py')]
        archive=EVIDENCE/'prepare_upload.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as target:
            for path in source_files:target.write(path,path.as_posix())
        digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        runtime.upload(archive,'content/turn_detect_v2/colab/natural_v5_prepare.zip')
        code=f'''from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
root=Path('/content/turn_detect_v2');name={NAME!r}
assert not (root/'jobs'/name).exists()
archive=root/'colab/natural_v5_prepare.zip'
assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()=={digest!r}
with zipfile.ZipFile(archive) as source:
    for item in source.infolist():
        target=(root/item.filename).resolve();assert target.is_relative_to(root.resolve())
        data=source.read(item)
        if target.exists():assert target.read_bytes()==data
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','3600','--',sys.executable,'-u','tools/natural_v5_prepare_job.py']
with (root/'jobs'/(name+'_launcher.log')).open('w') as stream:
    process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'launcher_pid':process.pid,'job':name,'bundle_sha256':{digest!r},'command':command}}))
'''
    elif args.action=='expand-vi':
        source=Path('turn_v2/natural_v5_vi_expand.py');digest=hashlib.sha256(source.read_bytes()).hexdigest()
        runtime.upload(source,'content/turn_detect_v2/colab/natural_v5_vi_expand.py')
        code=f'''from pathlib import Path
import hashlib,json,os,subprocess,sys
root=Path('/content/turn_detect_v2');name='natural_v5_vi_expand_20261025'
status=json.loads((root/'jobs/natural_v5_prepare_20261025/status.json').read_text())
assert status['status']=='COMPLETED' and status['exit_code']==0
assert not (root/'jobs'/name).exists(), 'Inspect existing expansion; no duplicate'
data=(root/'colab/natural_v5_vi_expand.py').read_bytes();assert hashlib.sha256(data).hexdigest()=={digest!r}
target=root/'turn_v2/natural_v5_vi_expand.py'
if target.exists():assert target.read_bytes()==data
else:target.write_bytes(data)
env=os.environ.copy();env['PYTHONPATH']=str(root/'.deps-v5-vad')+os.pathsep+str(root)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','1800','--',sys.executable,'-u','-m','turn_v2.natural_v5_vi_expand']
with (root/'jobs'/(name+'_launcher.log')).open('w') as stream:
    process=subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'job':name,'launcher_pid':process.pid,'source_sha256':{digest!r}}}))
'''
    elif args.action=='status':
        code=f'''from pathlib import Path
import json
root=Path('/content/turn_detect_v2');value={{}}
for rel in ['jobs/{NAME}/status.json','data_v5/natural_20261025/progress.json','data_v5/natural_20261025/manifest.json',
    'jobs/natural_v5_vi_expand_20261025/status.json','data_v5/natural_vi_expanded_20261025/progress.json','data_v5/natural_vi_expanded_20261025/manifest.json']:
    path=root/rel
    if path.exists():value[rel]=json.loads(path.read_text())
log=root/'jobs/{NAME}/stdout.log'
if log.exists() and value.get('jobs/{NAME}/status.json',{{}}).get('status')=='FAILED':value['log_tail']=log.read_text()[-3500:]
expanded_log=root/'jobs/natural_v5_vi_expand_20261025/stdout.log'
if expanded_log.exists() and value.get('jobs/natural_v5_vi_expand_20261025/status.json',{{}}).get('status')=='FAILED':value['expanded_log_tail']=expanded_log.read_text()[-3500:]
print(json.dumps(value))
'''
    else:
        code=f'''from pathlib import Path
import hashlib,json,zipfile
root=Path('/content/turn_detect_v2')
status=json.loads((root/'jobs/{NAME}/status.json').read_text());assert status['status']=='COMPLETED' and status['exit_code']==0
target=Path('/content/natural_v5_data_receipts.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for folder in ['jobs/{NAME}','data_v5/natural_20261025']:
        for path in sorted((root/folder).rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(root))
print(json.dumps({{'path':'content/'+target.name,'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),'bytes':target.stat().st_size}}))
'''
    result=runtime.execute(code,timeout=120,echo=False)
    (EVIDENCE/(args.action+'.json')).write_text(json.dumps({'connection':connection,**result},indent=2),encoding='utf-8')
    assert result['status']=='ok','Inspect saved receipt; never blindly retry execution'
    value=json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type']=='stream'))
    if args.action=='collect-data':
        archive=EVIDENCE/'prepared_data.zip';runtime.download(value['path'],archive)
        assert archive.stat().st_size==value['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==value['sha256']
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                target=Path(item.filename).resolve();assert target.is_relative_to(Path.cwd().resolve())
                data=source.read(item)
                if target.exists():assert target.read_bytes()==data
                else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    if args.action=='status':
        value={key:({'status':entry.get('status'),'exit_code':entry.get('exit_code'),'elapsed_seconds':entry.get('elapsed_seconds')}
            if key.endswith('/status.json') else entry.get('files',entry)) if isinstance(entry,dict) else entry for key,entry in value.items()}
    print(json.dumps(value),flush=True);runtime.close()


if __name__=='__main__':main()
