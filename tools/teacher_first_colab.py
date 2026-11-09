"""Handoff to continuous large teachers; local transfers contain metadata only."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import quote

from colab_runtime import ColabRuntime
from storage_guard import ensure_free_space

RUN = Path('outputs/teacher_first_v14_20261026')
REMOTE = '/content/turn_detect_v2'
FILES = ['turn_v2/teacher_first_v14.py', 'turn_v2/teacher_first_protocol.py']


def campaign_runtime(session_path=None):
    selection = RUN / 'runtime_selection.json'
    if session_path is None and selection.exists():
        session_path = json.loads(selection.read_text(encoding='utf-8'))['session_path']
    return ColabRuntime(0, session_path=session_path)


def execute(runtime, action, code, timeout=90):
    result = runtime.execute(code, timeout=timeout, echo=False)
    proof = RUN / (action + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '.json')
    proof.write_text(json.dumps(result, indent=2), encoding='utf-8')
    assert result['status'] == 'ok', 'Inspect ' + str(proof)
    return json.loads(''.join(o['text'] for o in result['outputs'] if o['output_type'] == 'stream'))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['launch', 'status', 'snapshot'])
    parser.add_argument('--session-path', help='Exact existing a.ipynb Jupyter session; never creates a kernel')
    args = parser.parse_args(); RUN.mkdir(parents=True, exist_ok=True)
    runtime = campaign_runtime(args.session_path); runtime.connect()
    if args.action == 'launch':
        hashes = {}
        for relative in FILES:
            data = Path(relative).read_bytes(); hashes[relative] = hashlib.sha256(data).hexdigest()
            runtime.request('PUT', 'api/contents/content/turn_detect_v2/colab/' + Path(relative).name,
                            json={'type': 'file', 'format': 'base64', 'content': base64.b64encode(data).decode()})
        code = f'''import json,os,sys,hashlib,subprocess
from pathlib import Path
root=Path({REMOTE!r});os.chdir(root);folder=root/{str(RUN.as_posix())!r};folder.mkdir(parents=True,exist_ok=True)
for relative,digest in {hashes!r}.items():
 data=(root/'colab'/Path(relative).name).read_bytes();assert hashlib.sha256(data).hexdigest()==digest
 target=root/relative
 if target.exists():assert target.read_bytes()==data,'Preserve executing source; use a new module for fixes'
 else:target.write_bytes(data)
for pid in Path('/proc').iterdir():
 if pid.name.isdigit():
  try:command=(pid/'cmdline').read_bytes()
  except OSError:continue
  assert not (b'turn_v2.teacher_first_v14' in command or b'turn_v2.continuous_v13' in command),'Another campaign worker is active: '+pid.name
command=[sys.executable,'-u','-m','turn_v2.teacher_first_v14','--steps','100','--microbatch','16','--candidates','384']
log=folder/('worker_'+str(len(list(folder.glob('worker_*.log')))+1)+'.log')
with log.open('w') as stream:process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1'}})
receipt={{'pid':process.pid,'start_ticks':(Path('/proc')/str(process.pid)/'stat').read_text().split()[21],'command':command,'log':str(log.relative_to(root)),'source_sha256':{hashes!r},'max_cycles':0,'global_timeout':None}}
(folder/'launcher.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
        value = execute(runtime, 'launch', code)
        (RUN / 'launcher.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    elif args.action == 'status':
        code = f'''import json,sqlite3,subprocess,shutil
from pathlib import Path
root=Path({REMOTE!r});folder=root/{str(RUN.as_posix())!r};values={{}}
for name in ['launcher.json','child.json','status.json','latest_error.json','config.json','distillation_gate.json','recovery/latest_snapshot.json','handoff.json']:
 p=folder/name
 if p.exists():values[name]=json.loads(p.read_text())
for name in ['launcher.json','child.json']:
 receipt=values.get(name,{{}});pid=receipt.get('pid');proc=Path('/proc')/str(pid);verified=False
 if pid and (proc/'cmdline').exists():
  command=(proc/'cmdline').read_bytes();ticks=(proc/'stat').read_text().split()[21]
  verified=b'turn_v2.teacher_first_v14' in command and ticks==receipt.get('start_ticks');values[name+'_current_start_ticks']=ticks
 values[name+'_process_verified']=verified
 if receipt.get('log'):
  p=root/receipt['log']
  if p.exists():values[name+'_log_tail']=p.read_text()[-2200:]
database=folder/'state.sqlite'
if database.exists():
 c=sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True)
 values['teacher_cycle']=dict(c.execute('SELECT key,value FROM meta WHERE key=?',('teacher_cycle',))).get('teacher_cycle','1')
 values['pool_counts']=[dict(zip(['language','split','label','samples'],row)) for row in c.execute('SELECT language,split,label,COUNT(*) FROM samples GROUP BY language,split,label')]
 values['champions']={{key.split(':')[1]:json.loads(value) for key,value in c.execute("SELECT key,value FROM meta WHERE key LIKE 'large_teacher:%'")}}
 c.close()
values['teachers']={{}}
for lang in ['vi','en','zh']:
 values['teachers'][lang]={{}}
 for name in ['identity.json','progress.json','resume_receipt.json','last_training_report.json']:
  p=folder/'teachers'/lang/name
  if p.exists():values['teachers'][lang][name]=json.loads(p.read_text())
values['GPU']=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
values['remote_disk_free_bytes']=shutil.disk_usage('/content').free
print(json.dumps(values))
'''
        value = execute(runtime, 'status', code)
        value['queried_at_utc'] = datetime.now(timezone.utc).isoformat()
        (RUN / 'status_receipt.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    else:
        value = execute(runtime, 'snapshot', f'''import json,hashlib
from pathlib import Path
root=Path({REMOTE!r});meta=json.loads((root/{str(RUN.as_posix())!r}/'recovery/latest_snapshot.json').read_text());p=root/meta['path']
assert p.stat().st_size==meta['bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==meta['sha256']
print(json.dumps(meta))''')
        assert value['scope'] == 'METADATA_ONLY_NOT_A_FULL_TEACHER_RECOVERY'
        assert value['bytes'] < 30 * 1024**2, 'Refuse accidental large teacher download'
        ensure_free_space(RUN, value['bytes'] * 2)
        response = runtime.http.get(runtime.base_url + 'files/' + quote('content/turn_detect_v2/' + value['path'], safe='/'),
                                    stream=True, timeout=120, allow_redirects=False)
        assert response.status_code == 200 and 'html' not in response.headers.get('content-type', '')
        target = RUN / 'metadata_snapshot.zip'; partial = RUN / 'metadata_snapshot.partial'
        with partial.open('wb') as stream:
            for block in response.iter_content(1024 * 1024): stream.write(block)
        assert partial.stat().st_size == value['bytes'] and hashlib.sha256(partial.read_bytes()).hexdigest() == value['sha256']
        partial.replace(target)
        (RUN / 'snapshot_receipt.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    runtime.close()
    if args.action == 'status':
        compact = {k: value[k] for k in ['queried_at_utc', 'launcher.json_process_verified', 'child.json_process_verified', 'status.json', 'GPU', 'remote_disk_free_bytes', 'teacher_cycle', 'teachers'] if k in value}
        compact['champions'] = {lang: {'semantic_dev': v['semantic_dev'], 'natural_floor_dev': v['natural_floor_dev']} for lang, v in value.get('champions', {}).items()}
        if value.get('latest_error.json'): compact['latest_error'] = value['latest_error.json']
        print(json.dumps(compact))
    else: print(json.dumps(value))


if __name__ == '__main__': main()
