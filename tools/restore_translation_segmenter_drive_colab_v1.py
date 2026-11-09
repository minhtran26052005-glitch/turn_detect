"""Launch/inspect a full Drive-to-Colab restore; transfer only small controls."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from colab_runtime import ColabRuntime

LOCAL = Path('outputs/translation_segmenter_v15d_20261026')
SOURCE = Path('turn_v2/restore_drive_campaign_v1.py')
CORE = ('__init__.py', 'translation_segmenter_v15c.py', 'translation_segmenter_v15d2.py',
        'translation_segmenter_v15d3.py', 'common.py', 'monolingual_data.py', 'metrics.py')


def cell(runtime, code):
    result = runtime.execute(code, timeout=60, echo=False)
    if result.get('status') != 'ok':
        raise RuntimeError(json.dumps([x for x in result.get('outputs', []) if x.get('output_type') == 'error'], ensure_ascii=False)[:6000])
    return json.loads(''.join(x.get('text', '') for x in result['outputs'] if x.get('output_type') == 'stream'))


def launch(runtime):
    config = json.loads((LOCAL / 'drive_backup_config.json').read_text(encoding='utf-8'))
    source = SOURCE.read_bytes(); checksum = hashlib.sha256(source).hexdigest()
    init = Path('turn_v2/__init__.py').read_bytes()
    inspected = cell(runtime, f'''import hashlib,json,os,shutil
from pathlib import Path
root=Path('/content/turn_detect_v2');run=root/'outputs/translation_segmenter_v15d_20261026';job=root/'restore_control_v1';config={config!r}
assert os.path.ismount('/content/drive'),'Mount the user Drive first'
drive=Path(config['drive_path']);assert json.loads((drive/'destination_identity.json').read_text())==config['identity'],'Drive identity mismatch'
assert not run.exists(),'Existing campaign state must be inspected; refusing to overwrite'
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:argv=(p/'cmdline').read_bytes().split(bytes([0]))
 except OSError:continue
 if b'-m' in argv and any(x.startswith((b'turn_v2.translation_segmenter_',b'turn_v2.restore_drive_campaign_',b'turn_v2.translation_segmenter_drive_backup_')) for x in argv):active.append(p.name)
assert not active,'Campaign/restore process already exists: '+str(active)
assert shutil.disk_usage('/content').free>50*1024**3,'Insufficient Colab reserve for full restore'
m=json.loads((drive/'latest_manifest.json').read_text());assert m.get('pass_complete') and m.get('full_recovery_available')
root.mkdir(exist_ok=True);(root/'turn_v2').mkdir(exist_ok=True);job.mkdir(exist_ok=True)
for name,expected in [('__init__.py',{hashlib.sha256(init).hexdigest()!r}),('restore_drive_campaign_v1.py',{checksum!r})]:
 p=root/'turn_v2'/name
 if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,'Preserve existing source; use a new version'
print(json.dumps({{'manifest_canonical_sha256':hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'manifest_at_utc':m['captured_at_utc'],'data_cycle':m['files']['data/data_snapshot.zip']['source_receipt']['cycle'],'checkpoint_receipts':{{l:m['files']['teachers/'+l+'/latest_resume.pt']['source_receipt'] for l in ['vi','en','zh']}}}}))''')
    config.update({'manifest_canonical_sha256': inspected['manifest_canonical_sha256'],
        'core_source_sha256': {'turn_v2/' + name: hashlib.sha256((Path('turn_v2') / name).read_bytes()).hexdigest() for name in CORE}})
    for path, data in [('turn_v2/__init__.py', init), (str(SOURCE).replace('\\', '/'), source),
                       ('restore_control_v1/config.json', json.dumps(config).encode())]:
        assert len(data) < 5 * 1024**2
        runtime.request('PUT', 'api/contents/content/turn_detect_v2/' + path,
            json={'type': 'file', 'format': 'base64', 'content': base64.b64encode(data).decode()})
    started = cell(runtime, f'''import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path('/content/turn_detect_v2');job=root/'restore_control_v1'
assert not (root/'outputs/translation_segmenter_v15d_20261026').exists(),'Campaign state appeared; preserve it'
assert hashlib.sha256((root/'turn_v2/restore_drive_campaign_v1.py').read_bytes()).hexdigest()=={checksum!r}
cmd=[sys.executable,'-u','-m','turn_v2.restore_drive_campaign_v1'];log=job/'worker.log'
with log.open('a') as stream:p=subprocess.Popen(cmd,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1'}})
record={{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':cmd,'module':'turn_v2.restore_drive_campaign_v1','source_sha256':{checksum!r},'log':str(log.relative_to(root))}}
(job/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))''')
    return {'preflight': inspected, 'launcher': started}


def status(runtime):
    return cell(runtime, '''import json,shutil
from pathlib import Path
root=Path('/content/turn_detect_v2');job=root/'restore_control_v1';run=root/'outputs/translation_segmenter_v15d_20261026';out={}
for base,names in [(job,['launcher.json','status.json','error.json','restore_receipt.json']),(run,['launcher.json','child.json','status.json'])]:
 for name in names:
  p=base/name
  if p.exists():out[('restore/' if base==job else 'run/')+name]=json.loads(p.read_text())
def live(record):
 try:
  argv=record['command'];module=argv[argv.index('-m')+1];p=Path('/proc')/str(record['pid'])
  return module.encode() in (p/'cmdline').read_bytes().split(bytes([0])) and (p/'stat').read_text().split()[21]==record['start_ticks']
 except (OSError,KeyError,ValueError,IndexError):return False
out['restore_process_verified']=live(out.get('restore/launcher.json',{}));out['campaign_parent_verified']=live(out.get('run/launcher.json',{}));out['child_verified']=live(out.get('run/child.json',{}));out['free_bytes']=shutil.disk_usage('/content').free
p=job/'worker.log'
if p.exists():
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-3000));out['log_tail']=f.read().decode(errors='replace')
print(json.dumps(out))''')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('launch', 'status'))
    parser.add_argument('--session-path')
    args = parser.parse_args()
    runtime = ColabRuntime(0, session_path=args.session_path)
    try:
        connection = runtime.connect()
        if connection['execution_state'] != 'idle':
            raise RuntimeError('Kernel busy; restore control cell was not queued')
        result = {'launch': launch, 'status': status}[args.action](runtime)
        receipt = {'connection': connection, 'result': result}
        (LOCAL / f'drive_restore_{args.action}_v1.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(receipt, ensure_ascii=False))
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
