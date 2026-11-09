"""Managed text feature baselines after corrected data preparation completes."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from colab_runtime import ColabRuntime
from qat_colab import SESSION

ROOT=Path('outputs/natural_v6_20261025')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['download-encoder','features-qwen','features-minilm','finetune','status','collect'])
    args=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    runtime=ColabRuntime(0,SESSION);connection=runtime.connect()
    if args.action in ['download-encoder','features-qwen','features-minilm','finetune']:
        downloading=args.action=='download-encoder';encoder=args.action.removeprefix('features-')
        relative='tools/natural_v6_minilm_download.py' if downloading else ('turn_v2/natural_v6_finetune.py' if args.action=='finetune' else 'turn_v2/natural_v6_features.py')
        local=Path(relative);digest=hashlib.sha256(local.read_bytes()).hexdigest()
        runtime.upload(local,'content/turn_detect_v2/colab/'+local.name)
        name='natural_v6_'+args.action.replace('-','_')+'_20261025'
        guard='' if downloading else '''for dependency in ['natural_v6_prepare_20261025','natural_v6_download_encoder_20261025']:
    status=json.loads((root/'jobs'/dependency/'status.json').read_text());assert status['status']=='COMPLETED' and status['exit_code']==0
for candidate in ['natural_v6_features_qwen_20261025','natural_v6_features_minilm_20261025','natural_v6_finetune_20261025']:
    p=root/'jobs'/candidate/'status.json'
    if p.exists():assert json.loads(p.read_text())['status']!='RUNNING','Another feature GPU job is still running'
'''
        tail="[sys.executable,'-u','tools/natural_v6_minilm_download.py']" if downloading else ("[sys.executable,'-u','-m','turn_v2.natural_v6_finetune']" if args.action=='finetune' else f"[sys.executable,'-u','-m','turn_v2.natural_v6_features','--encoder',{encoder!r}]")
        code=f'''from pathlib import Path
import hashlib,json,subprocess,sys
root=Path('/content/turn_detect_v2');name={name!r}
assert not (root/'jobs'/name).exists(), 'Inspect existing job; no duplicate'
{guard}data=(root/'colab/{local.name}').read_bytes();assert hashlib.sha256(data).hexdigest()=={digest!r}
target=root/{relative!r}
if target.exists():assert target.read_bytes()==data
else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','3600','--']+{tail}
with (root/'jobs'/(name+'_launcher.log')).open('w') as stream:
    process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'job':name,'launcher_pid':process.pid,'source_sha256':{digest!r}}}))
'''
    elif args.action=='status':
        code='''from pathlib import Path
import json
root=Path('/content/turn_detect_v2');value={}
for stage in ['download_encoder','features_qwen','features_minilm','finetune']:
    p=root/('jobs/natural_v6_'+stage+'_20261025/status.json')
    if p.exists():
        value[stage]=json.loads(p.read_text())
        log=p.parent/'stdout.log'
        if value[stage]['status']=='FAILED' and log.exists():value[stage+'_log_tail']=log.read_text()[-4500:]
for part in ['qwen','minilm']:
    for name in ['progress.json','report.json']:
        p=root/'outputs/natural_v6_20261025'/(part+'_features')/name
        if p.exists():value[part+'_'+name]=json.loads(p.read_text())
for name in ['progress.json','selection.json','report.json']:
    p=root/'runs/natural_v6_minilm_finetune_20261025'/name
    if p.exists():value['finetune_'+name]=json.loads(p.read_text())
print(json.dumps(value))
'''
    else:
        code='''from pathlib import Path
import hashlib,json,zipfile
root=Path('/content/turn_detect_v2');output=root/'outputs/natural_v6_20261025';target=Path('/content/natural_v6_features.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for stage in ['qwen','minilm']:
        folder=output/(stage+'_features')
        if not folder.exists():continue
        status=json.loads((root/('jobs/natural_v6_features_'+stage+'_20261025/status.json')).read_text());assert status['status']=='COMPLETED' and status['exit_code']==0
        for path in sorted(folder.rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(output))
    p=output/'encoder_download.json'
    if p.exists():archive.write(p,p.relative_to(output))
    run=root/'runs/natural_v6_minilm_finetune_20261025'
    if (run/'report.json').exists():
        for path in sorted(run.rglob('*')):
            if path.is_file() and path.suffix!='.safetensors':archive.write(path,Path('runs/natural_v6_minilm_finetune_20261025')/path.relative_to(run))
print(json.dumps({'path':'content/'+target.name,'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),'bytes':target.stat().st_size}))
'''
    result=runtime.execute(code,timeout=120,echo=False)
    (ROOT/(args.action+'_receipt.json')).write_text(json.dumps({'connection':connection,**result},indent=2),encoding='utf-8')
    assert result['status']=='ok','Inspect saved result; never blindly repeat execution'
    value=json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type']=='stream'))
    if args.action=='collect':
        archive=ROOT/'feature_receipts.zip';runtime.download(value['path'],archive)
        assert archive.stat().st_size==value['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==value['sha256']
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                target=(ROOT/item.filename).resolve();assert target.is_relative_to(ROOT.resolve());data=source.read(item)
                if target.exists():assert target.read_bytes()==data
                else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    elif args.action=='status':
        value={key:({k:entry.get(k) for k in ['status','exit_code','elapsed_seconds']} if key in ['download_encoder','features_qwen','features_minilm','finetune'] else entry) for key,entry in value.items()}
        if 'finetune_selection.json' in value:
            v=value['finetune_selection.json'];value['finetune_selection.json']={'best_path':v['best_path'],'best_dev_slices':v['best_dev']['slices'],'stale_evaluations':v['stale_evaluations']}
        if 'finetune_report.json' in value:
            v=value['finetune_report.json'];value['finetune_report.json']={k:v[k] for k in ['status','stop_reason','best_path','checkpoint_sha256','elapsed_seconds','sealed_loaded']}
    print(json.dumps(value),flush=True);runtime.close()


if __name__=='__main__':main()
