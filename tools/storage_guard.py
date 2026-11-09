"""Keep local checkpoint transfers bounded and reuse verified assembled files."""
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import quote

RESERVE_BYTES=10*1024**3


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def ensure_free_space(destination,additional_bytes,reserve_bytes=RESERVE_BYTES):
    if additional_bytes<0 or reserve_bytes<0:raise ValueError('Space requirements must be nonnegative')
    ancestor=Path(destination).resolve()
    while not ancestor.exists():ancestor=ancestor.parent
    free=shutil.disk_usage(ancestor).free
    if free<additional_bytes+reserve_bytes:
        raise OSError(f'Local disk guard: {free/2**30:.2f}GiB free; need {additional_bytes/2**30:.2f}GiB for transfer plus {reserve_bytes/2**30:.0f}GiB reserve. No large download started.')
    return free


def reuse_checkpoint(root,metadata_path,relative_path=None):
    root=Path(root).resolve();metadata_path=Path(metadata_path)
    if not metadata_path.exists():return False
    info=json.loads(metadata_path.read_text(encoding='utf-8-sig'))
    relative=relative_path or info.get('selected_path',info.get('path'))
    if not relative:return False
    target=(root/relative).resolve()
    if not target.is_relative_to(root):raise ValueError('Checkpoint path outside intended evidence folder')
    if not target.exists():return False
    if target.stat().st_size!=info['bytes'] or sha(target)!=info['sha256']:raise ValueError('Existing checkpoint differs; keep it for inspection')
    print(json.dumps({'status':'ALREADY_SAVED_NO_TRANSFER','path':str(target),'sha256':info['sha256'],'new_local_bytes':0}),flush=True)
    return True


def download_checkpoint(runtime,remote_path,target,size,expected_sha,reserve_bytes=RESERVE_BYTES):
    """One contiguous partial file; resume only a matching HTTP byte range."""
    target=Path(target).resolve()
    if target.exists():
        if target.stat().st_size!=size or sha(target)!=expected_sha:raise ValueError('Existing destination differs')
        return {'reused':True,'bytes_downloaded':0}
    partial=target.with_suffix(target.suffix+'.partial');identity=partial.with_suffix(partial.suffix+'.json')
    expected={'remote_path':remote_path,'bytes':size,'sha256':expected_sha}
    if partial.exists():
        if not identity.exists() or json.loads(identity.read_text())!=expected:raise ValueError('Partial transfer identity mismatch; keep for inspection')
    offset=partial.stat().st_size if partial.exists() else 0
    if offset>size:raise ValueError('Partial file exceeds checkpoint size')
    ensure_free_space(target,size-offset,reserve_bytes)
    target.parent.mkdir(parents=True,exist_ok=True)
    if not identity.exists():identity.write_text(json.dumps(expected,indent=2),encoding='utf-8')
    if offset<size:
        headers={'Range':f'bytes={offset}-'} if offset else {}
        response=runtime.http.get(runtime.base_url+'files/'+quote(remote_path,safe='/'),headers=headers,timeout=180,stream=True,allow_redirects=False)
        if offset:
            if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('Server did not honor resume range; partial file preserved')
        elif response.status_code!=200:raise ValueError('Checkpoint download failed: HTTP'+str(response.status_code))
        if 'html' in response.headers.get('content-type',''):raise ValueError('Expected model bytes, received HTML')
        completed=offset
        with partial.open('ab' if offset else 'wb') as out:
            for block in response.iter_content(1024*1024):
                if not block:continue
                if completed+len(block)>size:raise ValueError('Response exceeds expected checkpoint size')
                ensure_free_space(target,size-completed,reserve_bytes)
                out.write(block);completed+=len(block)
    if partial.stat().st_size!=size or sha(partial)!=expected_sha:raise ValueError('Downloaded checksum mismatch; partial file preserved')
    if target.exists():raise ValueError('Destination appeared during transfer; keep both files for inspection')
    partial.rename(target)
    return {'reused':False,'bytes_downloaded':size-offset,'temporary_full_copies':0}
