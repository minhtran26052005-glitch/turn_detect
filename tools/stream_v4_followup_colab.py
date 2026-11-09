"""Manage a distinct follow-up only after the original Colab campaign exits."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from colab_runtime import ColabRuntime
from qat_colab import SESSION

NAME = 'stream_v4_balanced_20261025'
ROOT = Path('outputs/stream_v4_20261025')


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['launch', 'status', 'collect'])
    args = parser.parse_args()
    runtime = ColabRuntime(0, SESSION)
    connection = runtime.connect()
    if args.action == 'launch':
        # Inspect before uploading or launching; never overlap/restart a job.
        probe = runtime.execute("""from pathlib import Path
import json
root=Path('/content/turn_detect_v2')
status=json.loads((root/'jobs/stream_v4_campaign_20261025/status.json').read_text())
assert status['status']=='COMPLETED' and status['exit_code']==0
assert not (root/'jobs/stream_v4_balanced_20261025').exists(), 'Inspect existing follow-up; no duplicate launch'
print(json.dumps({'previous_completed':True}))
""", timeout=60, echo=False)
        assert probe['status'] == 'ok', 'Original job is still active or follow-up already exists; inspect, do not relaunch'
        source = Path('turn_v2/stream_v4_balance.py')
        digest = file_hash(source)
        runtime.upload(source, 'content/turn_detect_v2/colab/stream_v4_balance_upload.py')
        code = f'''from pathlib import Path
import hashlib,json,subprocess,sys
root=Path('/content/turn_detect_v2');name={NAME!r}
assert not (root/'jobs'/name).exists()
data=(root/'colab/stream_v4_balance_upload.py').read_bytes()
assert hashlib.sha256(data).hexdigest()=={digest!r}
target=root/'turn_v2/stream_v4_balance.py'
if target.exists():assert target.read_bytes()==data
else:target.write_bytes(data)
command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','3600','--',sys.executable,'-u','-m','turn_v2.stream_v4_balance','--output','runs/'+name,'--rounds','2','--steps','400']
with (root/'jobs'/(name+'_launcher.log')).open('w') as out:
    process=subprocess.Popen(command,cwd=root,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'launcher_pid':process.pid,'job':name,'command':command,'source_sha256':{digest!r}}}))
'''
    elif args.action == 'status':
        code = f'''from pathlib import Path
import json,time
root=Path('/content/turn_detect_v2');name={NAME!r};value={{'current_unix':time.time()}}
for rel in ['jobs/'+name+'/status.json','runs/'+name+'/progress.json','runs/'+name+'/report.json']:
    path=root/rel
    if path.exists():value[rel]=json.loads(path.read_text())
history=root/'runs'/name/'history.json'
if history.exists():value['latest_dev']=json.loads(history.read_text())[-1]
status=value.get('jobs/'+name+'/status.json',{{}})
log=root/'jobs'/name/'stdout.log'
if status.get('status')=='FAILED' and log.exists():value['log_tail']=log.read_text()[-4000:]
print(json.dumps(value))
'''
    else:
        code = f'''from pathlib import Path
import hashlib,json,zipfile
root=Path('/content/turn_detect_v2');name={NAME!r}
status=json.loads((root/'jobs'/name/'status.json').read_text())
assert status['status']=='COMPLETED' and status['exit_code']==0
target=Path('/content/'+name+'_receipts.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for folder in ['jobs/'+name,'runs/'+name]:
        for path in sorted((root/folder).rglob('*')):
            if path.is_file() and path.suffix!='.safetensors':archive.write(path,path.relative_to(root))
print(json.dumps({{'path':'content/'+target.name,'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest()}}))
'''
    result = runtime.execute(code, timeout=90, echo=False)
    (ROOT / ('balanced_' + args.action + '.json')).write_text(json.dumps({'connection': connection, **result}, indent=2), encoding='utf-8')
    assert result['status'] == 'ok', 'Inspect saved response; never silently resubmit'
    value = json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type'] == 'stream'))
    if args.action == 'collect':
        archive = ROOT / 'balanced_receipts.zip'
        runtime.download(value['path'], archive)
        assert file_hash(archive) == value['sha256']
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                target = (ROOT / item.filename).resolve()
                assert target.is_relative_to(ROOT.resolve())
                data = source.read(item)
                if target.exists():
                    assert target.read_bytes() == data
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
    if args.action == 'status':
        status = value.get('jobs/' + NAME + '/status.json', {})
        report = value.get('runs/' + NAME + '/report.json', {})
        latest = value.get('latest_dev', {})
        value = {'status': status.get('status'), 'exit_code': status.get('exit_code'),
            'progress': value.get('runs/' + NAME + '/progress.json'),
            'best_minimum_dev_score': report.get('best_minimum_dev_score'),
            'stop_reason': report.get('stop_reason'), 'latest_dev_slices': latest.get('slices'),
            'log_tail': value.get('log_tail')}
    print(json.dumps(value), flush=True)
    runtime.close()


if __name__ == '__main__':
    main()
