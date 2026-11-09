"""Canonicalize the ASR pool, then train only on the corrected call split."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from colab_runtime import ColabRuntime
from qat_colab import SESSION

EVIDENCE=Path('outputs/natural_v5_20261025')
NAME='natural_v5_train_20261025'


def upload_source(runtime,relative):
    path=Path(relative)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    temporary='content/turn_detect_v2/colab/'+path.name
    runtime.upload(path,temporary)
    return f"""source=root/'colab/{path.name}'
data=source.read_bytes();assert hashlib.sha256(data).hexdigest()=={digest!r}
target=root/{relative!r}
if target.exists():assert target.read_bytes()==data
else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
"""


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['canonicalize','train','status','collect','collect-training','evaluate','eval-status','collect-final'])
    args=parser.parse_args();EVIDENCE.mkdir(parents=True,exist_ok=True)
    runtime=ColabRuntime(0,SESSION);connection=runtime.connect()
    if args.action=='canonicalize':
        prefix=upload_source(runtime,'turn_v2/natural_v5_canonical.py')
        code="""from pathlib import Path
import hashlib,json,subprocess,sys
root=Path('/content/turn_detect_v2')
status=json.loads((root/'jobs/natural_v5_prepare_20261025/status.json').read_text())
assert status['status']=='COMPLETED' and status['exit_code']==0
status=json.loads((root/'jobs/natural_v5_vi_expand_20261025/status.json').read_text())
assert status['status']=='COMPLETED' and status['exit_code']==0
assert not (root/'data_v5/canonical_20261025').exists(), 'Inspect existing canonical data; no overwrite'
"""+prefix+"""subprocess.run([sys.executable,'-u','-m','turn_v2.natural_v5_canonical'],cwd=root,check=True)
"""
    elif args.action=='train':
        prefix=upload_source(runtime,'turn_v2/natural_v5_train.py')
        code=f'''from pathlib import Path
import hashlib,json,subprocess,sys
root=Path('/content/turn_detect_v2');name={NAME!r}
status=json.loads((root/'jobs/natural_v5_prepare_20261025/status.json').read_text())
assert status['status']=='COMPLETED' and status['exit_code']==0
manifest=json.loads((root/'data_v5/canonical_20261025/manifest.json').read_text())
assert manifest['status']=='CANONICAL_PUBLIC_NATURAL_TEXT_READY_NOT_HUMAN_GOLD'
assert not (root/'jobs'/name).exists(), 'Inspect existing campaign; no duplicate/restart'
assert hashlib.file_digest((root/'runs/stream_v4_campaign_20261025/round_02/best/model.safetensors').open('rb'),'sha256').hexdigest()=='89ebf4e8c8b99e4ea1ad2eac19f9132b0ce8506ee0836943dba343fc849d6509'
'''+prefix+f'''command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','3600','--',sys.executable,'-u','-m','turn_v2.natural_v5_train','--output','runs/'+name,'--rounds','4','--steps','600']
with (root/'jobs'/(name+'_launcher.log')).open('w') as stream:
    process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'job':name,'launcher_pid':process.pid,'command':command}}))
'''
    elif args.action=='evaluate':
        prefix=upload_source(runtime,'turn_v2/natural_v5_evaluate.py')
        code=f'''from pathlib import Path
import hashlib,json,subprocess,sys
root=Path('/content/turn_detect_v2');name='natural_v5_eval_20261025'
status=json.loads((root/'jobs/{NAME}/status.json').read_text())
assert status['status']=='COMPLETED' and status['exit_code']==0
assert not (root/'jobs'/name).exists(), 'Inspect existing evaluation; no repeat/retuning'
'''+prefix+'''command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','1800','--',sys.executable,'-u','-m','turn_v2.natural_v5_evaluate']
with (root/'jobs'/(name+'_launcher.log')).open('w') as stream:
    process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'job':name,'launcher_pid':process.pid,'command':command}))
'''
    elif args.action=='eval-status':
        code='''from pathlib import Path
import json
root=Path('/content/turn_detect_v2');value={}
status=root/'jobs/natural_v5_eval_20261025/status.json'
if status.exists():value['status']=json.loads(status.read_text())
report=root/'outputs/natural_v5_20261025/final_eval/report.json'
if report.exists():value['report']=json.loads(report.read_text())
log=root/'jobs/natural_v5_eval_20261025/stdout.log'
if log.exists():value['log_tail']=log.read_text()[-2200:]
print(json.dumps(value))
'''
    elif args.action=='collect-final':
        code='''from pathlib import Path
import hashlib,json,zipfile
root=Path('/content/turn_detect_v2')
status=json.loads((root/'jobs/natural_v5_eval_20261025/status.json').read_text());assert status['status']=='COMPLETED' and status['exit_code']==0
target=Path('/content/natural_v5_final_receipts.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for folder in ['jobs/natural_v5_eval_20261025','outputs/natural_v5_20261025/final_eval']:
        for path in sorted((root/folder).rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(root))
    marker=root/'data_v5/canonical_20261025/sealed_consumed.json';archive.write(marker,marker.relative_to(root))
print(json.dumps({'path':'content/'+target.name,'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),'bytes':target.stat().st_size}))
'''
    elif args.action=='status':
        code=f'''from pathlib import Path
import json
root=Path('/content/turn_detect_v2');name={NAME!r};value={{}}
for rel in ['jobs/'+name+'/status.json','runs/'+name+'/progress.json','runs/'+name+'/report.json','runs/'+name+'/initial_dev.json','runs/'+name+'/selection.json']:
    path=root/rel
    if path.exists():value[rel]=json.loads(path.read_text())
history=root/'runs'/name/'history.json'
if history.exists():value['latest_dev']=json.loads(history.read_text())[-1]
log=root/'jobs'/name/'stdout.log'
if log.exists() and value.get('jobs/'+name+'/status.json',{{}}).get('status')=='FAILED':value['log_tail']=log.read_text()[-5000:]
print(json.dumps(value))
'''
    else:
        code=f'''from pathlib import Path
import hashlib,json,zipfile
root=Path('/content/turn_detect_v2');name={NAME!r}
target=Path('/content/natural_v5_local_receipts.zip')
folders=['data_v5/natural_20261025','data_v5/natural_vi_expanded_20261025','data_v5/canonical_20261025','jobs/natural_v5_prepare_20261025','jobs/natural_v5_vi_expand_20261025']
if (root/'jobs'/name/'status.json').exists():
    status=json.loads((root/'jobs'/name/'status.json').read_text())
    assert status['status']=='COMPLETED' and status['exit_code']==0
    folders+=['jobs/'+name,'runs/'+name]
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for folder in folders:
        for path in sorted((root/folder).rglob('*')):
            if path.is_file() and path.suffix!='.safetensors':archive.write(path,path.relative_to(root))
print(json.dumps({{'path':'content/'+target.name,'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),'bytes':target.stat().st_size}}))
'''
    result=runtime.execute(code,timeout=180,echo=False)
    (EVIDENCE/(args.action+'.json')).write_text(json.dumps({'connection':connection,**result},indent=2),encoding='utf-8')
    assert result['status']=='ok','Inspect the saved response; never silently resubmit execution'
    output=''.join(o['text'] for o in result['outputs'] if o['output_type']=='stream')
    if args.action in ['collect','collect-training','collect-final']:
        value=json.loads(output);filename={'collect':'local_receipts.zip','collect-training':'training_receipts.zip','collect-final':'final_receipts.zip'}[args.action]
        archive=EVIDENCE/filename;runtime.download(value['path'],archive)
        assert archive.stat().st_size==value['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==value['sha256']
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                name=item.filename
                if name.startswith('outputs/natural_v5_20261025/'):
                    target=(EVIDENCE/name[len('outputs/natural_v5_20261025/'):]).resolve();assert target.is_relative_to(EVIDENCE.resolve())
                elif name.startswith(('jobs/','runs/')):target=(EVIDENCE/name).resolve();assert target.is_relative_to(EVIDENCE.resolve())
                else:target=Path(name).resolve();assert target.is_relative_to((Path.cwd()/'data_v5').resolve())
                data=source.read(item)
                if target.exists():assert target.read_bytes()==data
                else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    elif args.action=='status':
        value=json.loads(output);status=value.get('jobs/'+NAME+'/status.json',{});selection=value.get('runs/'+NAME+'/selection.json',{})
        latest=value.get('latest_dev',value.get('runs/'+NAME+'/initial_dev.json',{}))
        output=json.dumps({'status':status.get('status'),'exit_code':status.get('exit_code'),
            'progress':value.get('runs/'+NAME+'/progress.json'),'best_dev_minimum':selection.get('best_dev',{}).get('minimum_dev_slice_joint_score'),
            'latest_dev_slices':latest.get('slices'),'stale_evaluations':selection.get('stale_evaluations'),'log_tail':value.get('log_tail')})
    elif args.action=='eval-status':
        value=json.loads(output);report=value.get('report',{});status=value.get('status',{})
        output=json.dumps({'status':status.get('status'),'exit_code':status.get('exit_code'),'held':report.get('metrics',{}).get('new_held_natural'),
            'automatic_target_met':report.get('automatic_public_held_three_language_target_met'),
            'log_tail':value.get('log_tail') if status.get('status')=='FAILED' else None})
    print(output,flush=True);runtime.close()


if __name__=='__main__':main()
