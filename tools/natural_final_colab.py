"""Launch and retrieve one frozen final held-data diagnostic."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zipfile
from colab_runtime import ColabRuntime
from qat_colab import SESSION

ROOT=Path('outputs/natural_v7_20261025');NAME='natural_final_eval_20261025'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','launch','status','collect']);args=parser.parse_args()
    runtime=ColabRuntime(0,SESSION);connection=runtime.connect()
    if args.action=='freeze':
        source=Path('tools/natural_freeze_selection.py');digest=hashlib.sha256(source.read_bytes()).hexdigest()
        runtime.request('PUT','api/contents/content/turn_detect_v2/colab/natural_freeze_selection.py',
            json={'type':'file','format':'base64','content':base64.b64encode(source.read_bytes()).decode('ascii')})
        calibrator=Path('turn_v2/natural_calibration.py');calibrator_sha=hashlib.sha256(calibrator.read_bytes()).hexdigest()
        runtime.request('PUT','api/contents/content/turn_detect_v2/colab/natural_calibration.py',
            json={'type':'file','format':'base64','content':base64.b64encode(calibrator.read_bytes()).decode('ascii')})
        proof=ROOT/'exact_dev_calibration.json';proof_sha=hashlib.sha256(proof.read_bytes()).hexdigest()
        runtime.request('PUT','api/contents/content/turn_detect_v2/outputs/natural_v7_20261025/exact_dev_calibration.json',
            json={'type':'file','format':'base64','content':base64.b64encode(proof.read_bytes()).decode('ascii')})
        code=f'''from pathlib import Path
import hashlib,json,subprocess,sys
root=Path('/content/turn_detect_v2');name='natural_freeze_selection_20261025'
for phase in ['natural_v7_text_20261025','natural_v8_lora_20261025_r3']:
    state=json.loads((root/'jobs'/phase/'status.json').read_text());assert state['status']=='COMPLETED' and state['exit_code']==0
assert not (root/'jobs'/name).exists(), 'Inspect existing freeze receipt; do not overwrite selection'
assert hashlib.file_digest((root/'outputs/natural_v7_20261025/exact_dev_calibration.json').open('rb'),'sha256').hexdigest()=={proof_sha!r}
calibrator=(root/'colab/natural_calibration.py').read_bytes();assert hashlib.sha256(calibrator).hexdigest()=={calibrator_sha!r}
target=root/'turn_v2/natural_calibration.py'
if target.exists():assert target.read_bytes()==calibrator
else:target.write_bytes(calibrator)
data=(root/'colab/natural_freeze_selection.py').read_bytes();assert hashlib.sha256(data).hexdigest()=={digest!r}
target=root/'tools/natural_freeze_selection.py'
if target.exists():assert target.read_bytes()==data
else:target.write_bytes(data)
command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','300','--',sys.executable,'-u','tools/natural_freeze_selection.py']
with (root/'jobs'/(name+'_launcher.log')).open('w') as stream:
    process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'job':name,'launcher_pid':process.pid,'source_sha256':{digest!r}}}))
'''
    elif args.action=='launch':
        source=Path('turn_v2/natural_final_evaluate.py');digest=hashlib.sha256(source.read_bytes()).hexdigest()
        runtime.request('PUT','api/contents/content/turn_detect_v2/colab/natural_final_evaluate.py',
            json={'type':'file','format':'base64','content':base64.b64encode(source.read_bytes()).decode('ascii')})
        code=f'''from pathlib import Path
import hashlib,json,subprocess,sys
root=Path('/content/turn_detect_v2');name={NAME!r}
selection=json.loads((root/'outputs/natural_v7_20261025/selected_before_held.json').read_text())
control=(root/selection.get('completed_control_flow_receipt','jobs/natural_campaign_pipeline_20261025/status.json')).resolve()
assert control.is_relative_to((root/'jobs').resolve())
state=json.loads(control.read_text());assert state['status']=='COMPLETED' and state['exit_code']==0
assert not (root/'jobs'/name).exists(), 'Inspect previous final-evaluation state; do not consume test twice'
assert (root/'outputs/natural_v7_20261025/selected_before_held.json').exists()
assert not (root/'data_v6/pho_text_20261025/sealed_consumed.json').exists()
data=(root/'colab/natural_final_evaluate.py').read_bytes();assert hashlib.sha256(data).hexdigest()=={digest!r}
target=root/'turn_v2/natural_final_evaluate.py'
if target.exists():assert target.read_bytes()==data
else:target.write_bytes(data)
command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','1800','--',sys.executable,'-u','-m','turn_v2.natural_final_evaluate']
with (root/'jobs'/(name+'_launcher.log')).open('w') as stream:
    process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'job':name,'launcher_pid':process.pid,'source_sha256':{digest!r}}}))
'''
    elif args.action=='status':
        code=f'''from pathlib import Path
import json
root=Path('/content/turn_detect_v2');value={{}}
for rel in ['jobs/{NAME}/status.json','outputs/natural_v7_20261025/final_eval/report.json']:
    path=root/rel
    if path.exists():value[rel]=json.loads(path.read_text())
status=value.get('jobs/{NAME}/status.json',{{}})
log=root/'jobs/{NAME}/stdout.log'
if status.get('status')=='FAILED' and log.exists():value['log_tail']=log.read_text()[-6000:]
print(json.dumps(value))
'''
    else:
        code=f'''from pathlib import Path
import hashlib,json,zipfile
root=Path('/content/turn_detect_v2');name={NAME!r}
status=json.loads((root/'jobs'/name/'status.json').read_text());assert status['status']=='COMPLETED' and status['exit_code']==0
target=Path('/content/natural_final_receipts.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for folder in ['jobs/'+name,'jobs/natural_campaign_pipeline_20261025','jobs/natural_v8_lora_20261025','jobs/natural_v8_lora_20261025_r2','jobs/natural_freeze_selection_20261025','outputs/natural_v7_20261025/final_eval']:
        for path in sorted((root/folder).rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(root))
    folder=root/'jobs/natural_campaign_recovery_20261025'
    if folder.exists():
        for path in sorted(folder.rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(root))
    path=root/'outputs/natural_v7_20261025/recovery_progress.json'
    if path.exists():archive.write(path,path.relative_to(root))
    for rel in ['outputs/natural_v7_20261025/selected_before_held.json','outputs/natural_v7_20261025/pipeline_progress.json','data_v6/pho_text_20261025/sealed_consumed.json']:
        path=root/rel;archive.write(path,rel)
print(json.dumps({{'path':'content/'+target.name,'bytes':target.stat().st_size,'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest()}}))
'''
    result=runtime.execute(code,timeout=120,echo=False)
    (ROOT/('final_'+args.action+'_receipt.json')).write_text(json.dumps({'connection':connection,**result},indent=2),encoding='utf-8')
    assert result['status']=='ok','Inspect final receipt; execution is never silently repeated'
    value=json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type']=='stream'))
    if args.action=='collect':
        archive=ROOT/'final_receipts.zip';runtime.download(value['path'],archive)
        assert archive.stat().st_size==value['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==value['sha256']
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                name=item.filename
                if name.startswith('data_v6/'):
                    target=Path(name).resolve();assert target.is_relative_to((Path.cwd()/'data_v6').resolve())
                else:
                    if name.startswith('outputs/natural_v7_20261025/'):name=name[len('outputs/natural_v7_20261025/'):]
                    target=(ROOT/name).resolve();assert target.is_relative_to(ROOT.resolve())
                data=source.read(item)
                if target.exists():assert target.read_bytes()==data
                else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    elif args.action=='status':
        status=value.get('jobs/'+NAME+'/status.json',{});report=value.get('outputs/natural_v7_20261025/final_eval/report.json',{})
        value={'status':status.get('status'),'exit_code':status.get('exit_code'),'selected_kind':report.get('selected_kind'),
            'held_metrics':report.get('metrics'),'automatic_numeric_target_met':report.get('automatic_held_numeric_target_met'),'log_tail':value.get('log_tail')}
    print(json.dumps(value),flush=True);runtime.close()


if __name__=='__main__':main()
