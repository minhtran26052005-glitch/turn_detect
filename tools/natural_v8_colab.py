"""Run/inspect/collect an isolated supervised Qwen LoRA campaign."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zipfile
from colab_runtime import ColabRuntime
from qat_colab import SESSION

NAME='natural_v8_lora_20261025';ROOT=Path('outputs/natural_v8_20261025')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','status','collect']);parser.add_argument('--revision',type=int,choices=[1,2,3],default=1);args=parser.parse_args()
    suffix='' if args.revision==1 else '_r'+str(args.revision)
    name=NAME+suffix
    ROOT.mkdir(parents=True,exist_ok=True);runtime=ColabRuntime(0,SESSION);connection=runtime.connect()
    if args.action=='launch':
        source=Path('turn_v2/natural_v8_lora'+suffix+'.py');digest=hashlib.sha256(source.read_bytes()).hexdigest()
        runtime.request('PUT','api/contents/content/turn_detect_v2/colab/'+source.name,
            json={'type':'file','format':'base64','content':base64.b64encode(source.read_bytes()).decode('ascii')})
        module='turn_v2.natural_v8_lora'+suffix
        code=f'''from pathlib import Path
import hashlib,json,subprocess,sys
root=Path('/content/turn_detect_v2');name={name!r}
for dep in ['natural_v6_pho_repair_20261025','natural_v7_text_20261025']:
    status=json.loads((root/'jobs'/dep/'status.json').read_text());assert status['status']=='COMPLETED' and status['exit_code']==0
assert not (root/'jobs'/name).exists(), 'Inspect existing adapter job; no duplicate launch'
data=(root/'colab/{source.name}').read_bytes();assert hashlib.sha256(data).hexdigest()=={digest!r}
target=root/{source.as_posix()!r}
if target.exists():assert target.read_bytes()==data
else:target.write_bytes(data)
command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','5400','--',sys.executable,'-u','-m',{module!r}]
with (root/'jobs'/(name+'_launcher.log')).open('w') as stream:
    process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'job':name,'launcher_pid':process.pid,'source_sha256':{digest!r}}}))
'''
    elif args.action=='status':
        code=f'''from pathlib import Path
import json
root=Path('/content/turn_detect_v2');name={name!r};value={{}}
for rel in ['jobs/'+name+'/status.json','runs/'+name+'/progress.json','runs/'+name+'/selection.json','runs/'+name+'/report.json','runs/'+name+'/backward_probe.json','runs/'+name+'/initial_dev.json']:
    path=root/rel
    if path.exists():value[rel]=json.loads(path.read_text())
status=value.get('jobs/'+name+'/status.json',{{}});log=root/'jobs'/name/'stdout.log'
if status.get('status')=='FAILED' and log.exists():value['log_tail']=log.read_text()[-6000:]
print(json.dumps(value))
'''
    else:
        code=f'''from pathlib import Path
import hashlib,json,zipfile
root=Path('/content/turn_detect_v2');name={name!r}
status=json.loads((root/'jobs'/name/'status.json').read_text());assert status['status']=='COMPLETED' and status['exit_code']==0
target=Path('/content/natural_v8_adapter_receipts.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for folder in ['jobs/'+name,'runs/'+name]:
        for path in sorted((root/folder).rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(root))
    compatibility=root/'outputs/natural_v8_20261025/r2_compatibility.json'
    if compatibility.exists():archive.write(compatibility,'r2_compatibility.json')
print(json.dumps({{'path':'content/'+target.name,'bytes':target.stat().st_size,'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest()}}))
'''
    result=runtime.execute(code,timeout=120,echo=False)
    (ROOT/(args.action+suffix+'_receipt.json')).write_text(json.dumps({'connection':connection,**result},indent=2),encoding='utf-8')
    assert result['status']=='ok','Inspect saved adapter result; never silently resubmit execution'
    value=json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type']=='stream'))
    if args.action=='status':
        status=value.get('jobs/'+name+'/status.json',{});selected=value.get('runs/'+name+'/selection.json',{});report=value.get('runs/'+name+'/report.json',{})
        initial=value.get('runs/'+name+'/initial_dev.json')
        value={'status':status.get('status'),'exit_code':status.get('exit_code'),'elapsed_seconds':status.get('elapsed_seconds'),'progress':value.get('runs/'+name+'/progress.json'),
            'backward_probe':value.get('runs/'+name+'/backward_probe.json'),'best_path':selected.get('best_path'),'best_dev_slices':selected.get('best_dev',{}).get('slices'),
            'stale_evaluations':selected.get('stale_evaluations'),'stop_reason':report.get('stop_reason'),'initial_dev_slices':initial['slices'] if initial and not selected else None,'log_tail':value.get('log_tail')}
    elif args.action=='collect':
        archive=ROOT/('adapter_receipts'+suffix+'.zip');runtime.download(value['path'],archive)
        assert archive.stat().st_size==value['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==value['sha256']
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                target=(ROOT/item.filename).resolve();assert target.is_relative_to(ROOT.resolve());data=source.read(item)
                if target.exists():assert target.read_bytes()==data
                else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    print(json.dumps(value),flush=True);runtime.close()


if __name__=='__main__':main()
