"""Relay the full committed Colab snapshot to SSH; no archive is saved on Windows."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from urllib.parse import quote

from colab_runtime import assigned_servers, ColabRuntime

HOST='edabk@100.91.209.124'
REMOTE='/home/edabk/hoangbpm/turn_detect/data/factory_stream_v2_20261102'
EXPORT='/content/factory_stream_full_export_20261102_v1/output'
LOCAL=Path('outputs/multilingual_stream_v2_20261102')


def runtime():
    matches=[]
    for i in range(len(assigned_servers())):
        rt=ColabRuntime(server_index=i)
        for session in rt.request('GET','api/sessions').json():
            name=Path(session.get('path','')).name
            if name=='a.ipynb' or name.startswith('a-jvsc-'):matches.append((i,session))
    if len(matches)!=1 or matches[0][1]['kernel'].get('execution_state')!='idle':raise RuntimeError('Need one idle current a.ipynb session; never reset it')
    i,s=matches[0];rt=ColabRuntime(server_index=i,session_path=s['path']);rt.connect();return rt


def ssh(code):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'python3 -'],input=code,capture_output=True,text=True,encoding='utf-8',timeout=120)
    if r.returncode:raise RuntimeError(r.stderr[-2500:])
    return json.loads(r.stdout)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','relay']);a=p.parse_args()
    assert shutil.disk_usage(Path.cwd()).free>=10*2**30,'Windows reserve below 10 GiB'
    LOCAL.mkdir(parents=True,exist_ok=True)
    rt=runtime()
    try:
        value=rt.execute(f"import pathlib,json;print((pathlib.Path({EXPORT!r})/'status.json').read_text())",timeout=60,echo=False)
        if value['status']!='ok':raise RuntimeError('Cannot read current export status')
        status=json.loads(''.join(x.get('text','') for x in value['outputs'] if x['output_type']=='stream'))
        (LOCAL/'export_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
        if a.action=='status':print(json.dumps(status));return
        assert status['stage']=='COMPLETE','Wait for readback-verified Colab export'
        digest=status['sha256'];size=status['bytes'];assert re.fullmatch('[a-f0-9]{64}',digest)
        assert status['archive']==EXPORT+'/factory_committed_snapshot.tar'
        stem=f"snapshot_{status['frozen_commit_index']:09d}_{digest}";final=REMOTE+'/'+stem+'.tar';partial=final+'.partial'
        before=ssh(f'''from pathlib import Path
import json,shutil,hashlib
root=Path({REMOTE!r});root.mkdir(parents=True,exist_ok=True);p=Path({partial!r});done=Path({final!r})
assert root.resolve().is_relative_to(Path('/home/edabk/hoangbpm/turn_detect').resolve())
assert shutil.disk_usage(root).free>={size}+13*2**30
r={{'offset':p.stat().st_size if p.exists() else 0,'complete':False}}
if done.exists():
 h=hashlib.sha256()
 with done.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 assert done.stat().st_size=={size} and h.hexdigest()=={digest!r};r['complete']=True
print(json.dumps(r))''')
        offset=before['offset'];assert offset<=size
        if not before['complete']:
            writer_code=f'''import sys,os,fcntl
from pathlib import Path
p=Path({partial!r})
with p.with_suffix(p.suffix+'.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with p.open('ab') as f:
  assert f.tell()=={offset}
  while True:
   data=sys.stdin.buffer.read(1024**2)
   if not data:break
   f.write(data)
  f.flush();os.fsync(f.fileno())
'''
            proc=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'python3 -c '+shlex.quote(writer_code)],stdin=subprocess.PIPE)
            try:
                while offset<size:
                    end=min(size,offset+16*1024**2)-1
                    response=rt.request('GET','files/'+quote(status['archive'].lstrip('/'),safe='/'),headers={'Range':f'bytes={offset}-{end}'},stream=True)
                    try:
                        assert response.status_code==206 and response.headers.get('Content-Range')==f'bytes {offset}-{end}/{size}'
                        for chunk in response.iter_content(1024**2):
                            if chunk:proc.stdin.write(chunk);offset+=len(chunk)
                        assert offset==end+1
                        print(json.dumps({'stage':'RELAYING_RAM_ONLY','bytes_sent':offset,'total_bytes':size}),flush=True)
                    finally:response.close()
            finally:
                proc.stdin.close();rc=proc.wait(timeout=45)
                if rc:raise RuntimeError('SSH writer failed; partial retained')
        extracted=ssh(f'''import hashlib,json,tarfile,os,shutil
from pathlib import Path,PurePosixPath
root=Path({REMOTE!r});p=Path({partial!r});done=Path({final!r});expected={digest!r}
if not done.exists():
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 assert p.stat().st_size=={size} and h.hexdigest()==expected
 p.replace(done)
target=root/('source_snapshot_'+expected[:12]);target.mkdir(exist_ok=True)
with tarfile.open(done,'r:') as archive:
 members=archive.getmembers();names=[m.name for m in members];assert len(names)==len(set(names))
 raw=archive.extractfile('export_manifest.json').read();assert hashlib.sha256(raw).hexdigest()=={status['manifest_sha256']!r};manifest=json.loads(raw)
 assert len(members)==len(manifest['files'])+1
 assert shutil.disk_usage(root).free>=sum(m.size for m in members)+13*2**30
 for member in members:
  rel=PurePosixPath(member.name);assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
  path=target.joinpath(*rel.parts);assert path.resolve().is_relative_to(target.resolve())
  data=archive.extractfile(member).read()
  if member.name!='export_manifest.json':
   meta=manifest['files'][member.name];assert len(data)==meta['bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256']
  path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists():assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256(data).digest()
  else:
   with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256(data).digest()
r={{'complete':True,'archive':str(done),'archive_bytes':{size},'archive_sha256':expected,'source_snapshot':str(target),'frozen_commit_index':manifest['frozen_commit_index'],'accepted_dialogues':manifest['source_state']['total_counts'],'files_readback_verified':len(members),'source_export_manifest_sha256':hashlib.sha256(raw).hexdigest(),'windows_archive_saved':False}}
(root/'transfer_receipt.json').write_text(json.dumps(r,indent=2)+'\\n');print(json.dumps(r))''')
        (LOCAL/'snapshot_transfer_receipt.json').write_text(json.dumps(extracted,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(extracted),flush=True)
    finally:rt.close()


if __name__=='__main__':main()
