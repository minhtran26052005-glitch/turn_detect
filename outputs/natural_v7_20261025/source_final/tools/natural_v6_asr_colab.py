"""Download/compare a pinned VI ASR model; no overwrite or duplicate jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from colab_runtime import ColabRuntime
from qat_colab import SESSION


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['download','repair','status','collect']);args=parser.parse_args()
    root=Path('outputs/natural_v6_20261025');runtime=ColabRuntime(0,SESSION);connection=runtime.connect()
    if args.action in ['download','repair']:
        source=Path('tools/natural_v6_pho_download.py') if args.action=='download' else Path('turn_v2/natural_v6_pho.py')
        digest=hashlib.sha256(source.read_bytes()).hexdigest();runtime.upload(source,'content/turn_detect_v2/colab/'+source.name)
        name='natural_v6_pho_'+args.action+'_20261025'
        guard='' if args.action=='download' else '''for dep in ['natural_v6_finetune_20261025','natural_v6_pho_download_20261025']:
    status=json.loads((root/'jobs'/dep/'status.json').read_text());assert status['status']=='COMPLETED' and status['exit_code']==0
'''
        command="[sys.executable,'-u','tools/natural_v6_pho_download.py']" if args.action=='download' else "[sys.executable,'-u','-m','turn_v2.natural_v6_pho']"
        code=f'''from pathlib import Path
import hashlib,json,subprocess,sys
root=Path('/content/turn_detect_v2');name={name!r}
assert not (root/'jobs'/name).exists(), 'Inspect existing ASR job; no duplicate launch'
{guard}data=(root/'colab/{source.name}').read_bytes();assert hashlib.sha256(data).hexdigest()=={digest!r}
target=root/{source.as_posix()!r}
if target.exists():assert target.read_bytes()==data
else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','5400','--']+{command}
with (root/'jobs'/(name+'_launcher.log')).open('w') as stream:
    process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'job':name,'launcher_pid':process.pid,'source_sha256':{digest!r}}}))
'''
    elif args.action=='status':
        code='''from pathlib import Path
import json
root=Path('/content/turn_detect_v2');value={}
for action in ['download','repair']:
    path=root/('jobs/natural_v6_pho_'+action+'_20261025/status.json')
    if path.exists():
        value[action]=json.loads(path.read_text())
        log=path.parent/'stdout.log'
        if value[action]['status']=='FAILED' and log.exists():value[action+'_log_tail']=log.read_text()[-5000:]
for rel in ['outputs/natural_v6_20261025/pho_download.json','data_v6/pho_text_20261025/progress.json','data_v6/pho_text_20261025/manifest.json']:
    path=root/rel
    if path.exists():value[rel]=json.loads(path.read_text())
print(json.dumps(value))
'''
    else:
        code='''from pathlib import Path
import hashlib,json,zipfile
root=Path('/content/turn_detect_v2')
status=json.loads((root/'jobs/natural_v6_pho_repair_20261025/status.json').read_text());assert status['status']=='COMPLETED' and status['exit_code']==0
target=Path('/content/natural_v6_pho_receipts.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for folder in ['data_v6/pho_text_20261025','jobs/natural_v6_pho_download_20261025','jobs/natural_v6_pho_repair_20261025']:
        for path in sorted((root/folder).rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(root))
    p=root/'outputs/natural_v6_20261025/pho_download.json';archive.write(p,p.relative_to(root))
print(json.dumps({'path':'content/'+target.name,'bytes':target.stat().st_size,'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest()}))
'''
    result=runtime.execute(code,timeout=120,echo=False)
    (root/('pho_'+args.action+'_receipt.json')).write_text(json.dumps({'connection':connection,**result},indent=2),encoding='utf-8')
    assert result['status']=='ok','Inspect saved ASR response; do not silently submit again'
    value=json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type']=='stream'))
    if args.action=='status':
        for key in ['download','repair']:
            if key in value:value[key]={k:value[key].get(k) for k in ['status','exit_code','elapsed_seconds']}
        receipt=value.get('outputs/natural_v6_20261025/pho_download.json')
        if receipt:value['outputs/natural_v6_20261025/pho_download.json']={'revision':receipt['revision'],'bytes':sum(f['bytes'] for f in receipt['files'])}
        receipt=value.get('data_v6/pho_text_20261025/manifest.json')
        if receipt:value['data_v6/pho_text_20261025/manifest.json']={k:receipt[k] for k in ['status','files','diagnostics','elapsed_seconds']}
    elif args.action=='collect':
        archive=root/'pho_receipts.zip';runtime.download(value['path'],archive)
        assert archive.stat().st_size==value['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==value['sha256']
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                name=item.filename
                if name.startswith('data_v6/'):
                    target=Path(name).resolve();assert target.is_relative_to((Path.cwd()/'data_v6').resolve())
                else:
                    if name.startswith('outputs/natural_v6_20261025/'):name=name[len('outputs/natural_v6_20261025/'):]
                    target=(root/name).resolve();assert target.is_relative_to(root.resolve())
                data=source.read(item)
                if target.exists():assert target.read_bytes()==data
                else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    print(json.dumps(value),flush=True);runtime.close()


if __name__=='__main__':main()
