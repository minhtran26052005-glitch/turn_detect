"""One final comparative evaluation after development-only candidate selection."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from colab_runtime import ColabRuntime
from qat_colab import SESSION

NAME = 'stream_v4_final_20261025'
EVIDENCE = Path('outputs/stream_v4_20261025')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['launch', 'status', 'collect'])
    args = parser.parse_args()
    runtime = ColabRuntime(0, SESSION)
    connection = runtime.connect()
    if args.action == 'launch':
        probe = runtime.execute("""from pathlib import Path
import json
root=Path('/content/turn_detect_v2')
for name in ['stream_v4_campaign_20261025','stream_v4_balanced_20261025']:
    status=json.loads((root/'jobs'/name/'status.json').read_text())
    assert status['status']=='COMPLETED' and status['exit_code']==0, 'Wait for training; do not overlap evaluation'
assert not (root/'jobs/stream_v4_final_20261025').exists(), 'Existing evaluation must be inspected; do not resubmit'
print('ready')
""", timeout=60, echo=False)
        assert probe['status'] == 'ok', 'Inspect training/evaluation state first'
        files = [Path('turn_v2/stream_v4_evaluate.py'), Path('data_v4/causal_20261025/sealed.jsonl'),
            Path('data_v4/fresh_crawl_20261025/review_cases.jsonl'), Path('data_v4/vi_natural_pause_20261025/cases.jsonl'),
            Path('data_v4/en_zh_natural_pause_20261025/cases.jsonl')]
        archive = EVIDENCE / 'final_evaluation_upload.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as target:
            for path in files:
                target.write(path, path.as_posix())
        digest = sha(archive)
        runtime.upload(archive, 'content/turn_detect_v2/colab/stream_v4_final_upload.zip')
        code = f'''from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
root=Path('/content/turn_detect_v2');name={NAME!r}
assert not (root/'jobs'/name).exists()
archive=root/'colab/stream_v4_final_upload.zip'
assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()=={digest!r}
with zipfile.ZipFile(archive) as source:
    for item in source.infolist():
        target=(root/item.filename).resolve();assert target.is_relative_to(root.resolve())
        data=source.read(item)
        if target.exists():assert target.read_bytes()==data
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','1800','--',sys.executable,'-u','-m','turn_v2.stream_v4_evaluate','--followup','runs/stream_v4_balanced_20261025','--vi-pilot','data_v4/vi_natural_pause_20261025/cases.jsonl','--en-zh-pilot','data_v4/en_zh_natural_pause_20261025/cases.jsonl','--output','outputs/stream_v4_20261025/colab_final']
with (root/'jobs'/(name+'_launcher.log')).open('w') as stream:
    process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'launcher_pid':process.pid,'job':name,'bundle_sha256':{digest!r}}}))
'''
    elif args.action == 'status':
        code = f'''from pathlib import Path
import json
root=Path('/content/turn_detect_v2');name={NAME!r};value={{}}
for rel in ['jobs/'+name+'/status.json','outputs/stream_v4_20261025/colab_final/report.json']:
    path=root/rel
    if path.exists():value[rel]=json.loads(path.read_text())
log=root/'jobs'/name/'stdout.log'
if log.exists():value['log_tail']=log.read_text()[-2500:]
print(json.dumps(value))
'''
    else:
        code = f'''from pathlib import Path
import hashlib,json,zipfile
root=Path('/content/turn_detect_v2');name={NAME!r}
status=json.loads((root/'jobs'/name/'status.json').read_text());assert status['status']=='COMPLETED' and status['exit_code']==0
target=Path('/content/'+name+'_receipts.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for folder in ['jobs/'+name,'outputs/stream_v4_20261025/colab_final']:
        for path in sorted((root/folder).rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(root))
print(json.dumps({{'path':'content/'+target.name,'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest()}}))
'''
    result = runtime.execute(code, timeout=120, echo=False)
    (EVIDENCE / ('final_' + args.action + '.json')).write_text(json.dumps({'connection': connection, **result}, indent=2), encoding='utf-8')
    assert result['status'] == 'ok', 'Inspect response; execution is never silently retried'
    value = json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type'] == 'stream'))
    if args.action == 'collect':
        archive = EVIDENCE / 'final_receipts.zip'
        runtime.download(value['path'], archive)
        assert sha(archive) == value['sha256']
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                # Keep remote output directly under the local evidence folder.
                name = item.filename
                if name.startswith('outputs/stream_v4_20261025/'):
                    name = name[len('outputs/stream_v4_20261025/'):]
                target = (EVIDENCE / name).resolve()
                assert target.is_relative_to(EVIDENCE.resolve())
                data = source.read(item)
                if target.exists():
                    assert target.read_bytes() == data
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
    print(json.dumps(value), flush=True)
    runtime.close()


if __name__ == '__main__':
    main()
