"""Remove only verified SSH duplicates of the user's local copies."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

root = Path('/home/edabk/hoangbpm/turn_detect').resolve()
plan_path = root / 'staging/vi_only_cleanup_plan.json'
receipt_path = root / 'staging/vi_only_cleanup_receipt.json'
if receipt_path.exists():
    raise RuntimeError('Cleanup already has a receipt; do not repeat')
plan = json.loads(plan_path.read_text())['files']
verified = []
for item in plan:
    path = root / item['remote_relative']
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise RuntimeError(f'Unsafe or missing candidate: {path}')
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Candidate changed while hashing: {path}')
    if after.st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
        raise RuntimeError(f'Candidate does not match the Windows source: {path}')
    verified.append((path, after.st_ino, after.st_size, after.st_mtime_ns, item))
    print('verified', item['remote_relative'], flush=True)

free_before = shutil.disk_usage(root).free
removed = []
for path, ino, size, modified, item in verified:
    current = path.stat()
    if (current.st_ino, current.st_size, current.st_mtime_ns) != (ino, size, modified):
        raise RuntimeError(f'Candidate changed after verification: {path}')
    path.unlink()
    if path.exists():
        raise RuntimeError(f'Candidate still exists after deletion: {path}')
    removed.append(item)
    tmp = receipt_path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'stage': 'IN_PROGRESS', 'removed': removed,
                               'free_before_bytes': free_before}, indent=2) + '\n')
    os.replace(tmp, receipt_path)
    print('removed', item['remote_relative'], flush=True)
receipt = {'stage': 'COMPLETE', 'at_unix': time.time(), 'removed': removed,
           'free_before_bytes': free_before, 'free_after_bytes': shutil.disk_usage(root).free}
tmp = receipt_path.with_suffix('.tmp')
tmp.write_text(json.dumps(receipt, indent=2) + '\n')
os.replace(tmp, receipt_path)
print('COMPLETE freed_bytes', receipt['free_after_bytes'] - free_before, flush=True)
