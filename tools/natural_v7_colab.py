"""Train/inspect/collect text-only teacher after identical-window VI ASR repair."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zipfile
from colab_runtime import ColabRuntime
from qat_colab import SESSION

NAME='natural_v7_text_20261025';ROOT=Path('outputs/natural_v7_20261025')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','compare-asr','pipeline','status','collect']);args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);runtime=ColabRuntime(0,SESSION);connection=runtime.connect()
    if args.action=='pipeline':
        files=[Path('turn_v2/natural_v7_train.py'),Path('turn_v2/natural_v7_compare_asr.py'),Path('turn_v2/natural_v8_lora.py'),Path('tools/natural_campaign_pipeline.py')]
        archive=ROOT/'pipeline_sources.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as target:
            for path in files:target.write(path,path.as_posix())
        digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        # This exact parent already exists in the verified connected project;
        # skip expensive directory inventories at /content during an active job.
        runtime.request('PUT','api/contents/content/turn_detect_v2/colab/natural_campaign_pipeline.zip',
            json={'type':'file','format':'base64','content':base64.b64encode(archive.read_bytes()).decode('ascii')})
        code=f'''from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
root=Path('/content/turn_detect_v2');name='natural_campaign_pipeline_20261025'
for job in [name,'natural_v7_compare_asr_20261025','natural_v7_text_20261025','natural_v8_lora_20261025']:
    assert not (root/'jobs'/job).exists(), 'Inspect existing step before starting pipeline; no duplicates'
archive=root/'colab/natural_campaign_pipeline.zip';assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()=={digest!r}
with zipfile.ZipFile(archive) as source:
    for item in source.infolist():
        target=(root/item.filename).resolve();assert target.is_relative_to(root.resolve());data=source.read(item)
        if target.exists():assert target.read_bytes()==data
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','12000','--',sys.executable,'-u','tools/natural_campaign_pipeline.py']
with (root/'jobs'/(name+'_launcher.log')).open('w') as stream:
    process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'job':name,'launcher_pid':process.pid,'bundle_sha256':{digest!r}}}))
'''
    elif args.action in ['launch','compare-asr']:
        source=Path('turn_v2/natural_v7_train.py' if args.action=='launch' else 'turn_v2/natural_v7_compare_asr.py');digest=hashlib.sha256(source.read_bytes()).hexdigest()
        runtime.upload(source,'content/turn_detect_v2/colab/'+source.name)
        job=NAME if args.action=='launch' else 'natural_v7_compare_asr_20261025'
        module='turn_v2.natural_v7_train' if args.action=='launch' else 'turn_v2.natural_v7_compare_asr'
        code=f'''from pathlib import Path
import hashlib,json,subprocess,sys
root=Path('/content/turn_detect_v2');name={job!r}
for dep in ['natural_v6_pho_repair_20261025','natural_v6_finetune_20261025']:
    status=json.loads((root/'jobs'/dep/'status.json').read_text());assert status['status']=='COMPLETED' and status['exit_code']==0
other=root/'jobs'/('natural_v7_compare_asr_20261025' if name=='natural_v7_text_20261025' else 'natural_v7_text_20261025')/'status.json'
if other.exists():assert json.loads(other.read_text())['status']!='RUNNING','Wait for other GPU work'
assert not (root/'jobs'/name).exists(), 'Inspect existing campaign; no duplicate submission'
data=(root/'colab/{source.name}').read_bytes();assert hashlib.sha256(data).hexdigest()=={digest!r}
target=root/{source.as_posix()!r}
if target.exists():assert target.read_bytes()==data
else:target.write_bytes(data)
command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','3600','--',sys.executable,'-u','-m',{module!r}]
with (root/'jobs'/(name+'_launcher.log')).open('w') as stream:
    process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'job':name,'launcher_pid':process.pid,'source_sha256':{digest!r}}}))
'''
    elif args.action=='status':
        code=f'''from pathlib import Path
import json
root=Path('/content/turn_detect_v2');name={NAME!r};value={{}}
for rel in ['jobs/'+name+'/status.json','runs/'+name+'/progress.json','runs/'+name+'/selection.json','runs/'+name+'/report.json','runs/'+name+'/initial_dev.json']:
    path=root/rel
    if path.exists():value[rel]=json.loads(path.read_text())
for rel in ['jobs/natural_v7_compare_asr_20261025/status.json','outputs/natural_v7_20261025/matched_ASR_dev/report.json']:
    p=root/rel
    if p.exists():value[rel]=json.loads(p.read_text())
for rel in ['jobs/natural_campaign_pipeline_20261025/status.json','outputs/natural_v7_20261025/pipeline_progress.json','outputs/natural_v7_20261025/selected_before_held.json']:
    p=root/rel
    if p.exists():value[rel]=json.loads(p.read_text())
history=root/'runs'/name/'history.json'
if history.exists():value['latest_dev']=json.loads(history.read_text())[-1]
status=value.get('jobs/'+name+'/status.json',{{}});log=root/'jobs'/name/'stdout.log'
if status.get('status')=='FAILED' and log.exists():value['log_tail']=log.read_text()[-5000:]
pipeline_state=value.get('jobs/natural_campaign_pipeline_20261025/status.json',{{}})
if pipeline_state.get('status')=='FAILED':value['pipeline_log_tail']=(root/'jobs/natural_campaign_pipeline_20261025/stdout.log').read_text()[-5000:]
print(json.dumps(value))
'''
    else:
        code=f'''from pathlib import Path
import hashlib,json,zipfile
root=Path('/content/turn_detect_v2');name={NAME!r}
status=json.loads((root/'jobs'/name/'status.json').read_text());assert status['status']=='COMPLETED' and status['exit_code']==0
target=Path('/content/natural_v7_receipts.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for folder in ['jobs/'+name,'runs/'+name]:
        for path in sorted((root/folder).rglob('*')):
            if path.is_file() and path.suffix!='.safetensors':archive.write(path,path.relative_to(root))
    folder=root/'outputs/natural_v7_20261025/matched_ASR_dev'
    if folder.exists():
        for path in sorted(folder.rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(root))
print(json.dumps({{'path':'content/'+target.name,'bytes':target.stat().st_size,'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest()}}))
'''
    result=runtime.execute(code,timeout=120,echo=False)
    (ROOT/(args.action+'_receipt.json')).write_text(json.dumps({'connection':connection,**result},indent=2),encoding='utf-8')
    assert result['status']=='ok','Inspect saved result; never repeat submitted execution silently'
    value=json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type']=='stream'))
    if args.action=='status':
        status=value.get('jobs/'+NAME+'/status.json',{});progress=value.get('runs/'+NAME+'/progress.json');selection=value.get('runs/'+NAME+'/selection.json',{});
        initial=value.get('runs/'+NAME+'/initial_dev.json');report=value.get('runs/'+NAME+'/report.json');latest=value.get('latest_dev',{})
        value={'status':status.get('status'),'exit_code':status.get('exit_code'),'progress':progress,
            'best_path':selection.get('best_path'),'best_dev_slices':selection.get('best_dev',{}).get('slices'),
            'latest_dev_slices':latest.get('slices'),'stale_evaluations':selection.get('stale_evaluations'),
            'initial_candidates':initial['candidates'] if initial and not selection else None,
            'stop_reason':report.get('stop_reason') if report else None,'log_tail':value.get('log_tail'),
            'ASR_comparison_status':value.get('jobs/natural_v7_compare_asr_20261025/status.json',{}).get('status'),
            'ASR_comparison':value.get('outputs/natural_v7_20261025/matched_ASR_dev/report.json'),
            'pipeline_status':value.get('jobs/natural_campaign_pipeline_20261025/status.json',{}).get('status'),
            'pipeline_progress':value.get('outputs/natural_v7_20261025/pipeline_progress.json'),'pipeline_log_tail':value.get('pipeline_log_tail')}
    elif args.action=='collect':
        archive=ROOT/'receipts.zip';runtime.download(value['path'],archive)
        assert archive.stat().st_size==value['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==value['sha256']
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                name=item.filename
                if name.startswith('outputs/natural_v7_20261025/'):name=name[len('outputs/natural_v7_20261025/'):]
                target=(ROOT/name).resolve();assert target.is_relative_to(ROOT.resolve());data=source.read(item)
                if target.exists():assert target.read_bytes()==data
                else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    print(json.dumps(value),flush=True);runtime.close()


if __name__=='__main__':main()
