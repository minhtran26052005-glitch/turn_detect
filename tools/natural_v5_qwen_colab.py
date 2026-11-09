"""Separate pinned teacher download and dev-only diagnosis."""
import argparse
import hashlib
import json
from pathlib import Path

from colab_runtime import ColabRuntime
from qat_colab import SESSION


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['download','dev','status','collect'])
    args=parser.parse_args();evidence=Path('outputs/natural_v5_20261025')
    runtime=ColabRuntime(0,SESSION);connection=runtime.connect()
    if args.action in ['download','dev']:
        relative='tools/natural_v5_qwen_download.py' if args.action=='download' else 'turn_v2/natural_v5_qwen.py'
        local=Path(relative);digest=hashlib.sha256(local.read_bytes()).hexdigest()
        runtime.upload(local,'content/turn_detect_v2/colab/'+local.name)
        name='natural_v5_qwen_'+args.action+'_20261025'
        prefix="" if args.action=='download' else """status=json.loads((root/'jobs/natural_v5_train_20261025/status.json').read_text())
assert status['status']=='COMPLETED' and status['exit_code']==0
status=json.loads((root/'jobs/natural_v5_qwen_download_20261025/status.json').read_text())
assert status['status']=='COMPLETED' and status['exit_code']==0
"""
        command_tail="[sys.executable,'-u','tools/natural_v5_qwen_download.py']" if args.action=='download' else "[sys.executable,'-u','-m','turn_v2.natural_v5_qwen']"
        code=f'''from pathlib import Path
import hashlib,json,subprocess,sys
root=Path('/content/turn_detect_v2');name={name!r}
assert not (root/'jobs'/name).exists(), 'Inspect existing job; no duplicate launch'
{prefix}data=(root/'colab/{local.name}').read_bytes();assert hashlib.sha256(data).hexdigest()=={digest!r}
target=root/{relative!r}
if target.exists():assert target.read_bytes()==data
else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','3600','--']+{command_tail}
with (root/'jobs'/(name+'_launcher.log')).open('w') as stream:
    process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'job':name,'launcher_pid':process.pid,'source_sha256':{digest!r}}}))
'''
    elif args.action=='status':
        code='''from pathlib import Path
import json
root=Path('/content/turn_detect_v2');value={}
for part in ['download','dev']:
    status=root/('jobs/natural_v5_qwen_'+part+'_20261025/status.json')
    if status.exists():
        value[part]=json.loads(status.read_text())
        log=status.parent/'stdout.log'
        if value[part]['status']=='FAILED' and log.exists():value[part+'_log_tail']=log.read_text()[-4500:]
for name in ['qwen_download.json','qwen_dev/report.json','qwen_dev/progress.json']:
    path=root/'outputs/natural_v5_20261025'/name
    if path.exists():value[name]=json.loads(path.read_text())
print(json.dumps(value))
'''
    else:
        code='''from pathlib import Path
import base64,json
root=Path('/content/turn_detect_v2/outputs/natural_v5_20261025');files={}
for name in ['qwen_download.json','qwen_dev/report.json','qwen_dev/predictions.npz','qwen_dev/identity.json']:
    path=root/name
    if path.exists():files[name]=base64.b64encode(path.read_bytes()).decode()
print(json.dumps(files))
'''
    result=runtime.execute(code,timeout=120,echo=False)
    (evidence/('qwen_'+args.action+'_receipt.json')).write_text(json.dumps({'connection':connection,**result},indent=2),encoding='utf-8')
    assert result['status']=='ok','Inspect saved result, do not blindly resubmit'
    value=json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type']=='stream'))
    if args.action=='collect':
        import base64
        for name,encoded in value.items():
            target=(evidence/name).resolve();assert target.is_relative_to(evidence.resolve());target.parent.mkdir(parents=True,exist_ok=True)
            data=base64.b64decode(encoded)
            if target.exists():assert target.read_bytes()==data
            else:target.write_bytes(data)
        value={'files_collected':list(value)}
    elif args.action=='status':
        if 'qwen_download.json' in value:
            receipt=value['qwen_download.json'];value['qwen_download.json']={'revision':receipt['revision'],'bytes':sum(f['bytes'] for f in receipt['files'])}
    print(json.dumps(value),flush=True);runtime.close()


if __name__=='__main__':main()
