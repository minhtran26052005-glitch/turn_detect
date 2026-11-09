"""Extract a verified data snapshot on the SSH host without touching existing files."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import zipfile

project = Path('/home/edabk/hoangbpm/turn_detect')
staging = project / 'staging'
hdd = Path('/mnt/hdd500/turn_detect_training')
manifest = json.loads((staging / 'drive_manifest.json').read_text(encoding='utf-8'))
entry = manifest['files']['data/data_snapshot.zip']
archive = staging / 'source_snapshot.zip.partial'
target = hdd / 'corpus.sqlite'
partial = hdd / 'corpus.sqlite.partial'
receipt = staging / 'corpus_extraction_receipt.json'
mount = subprocess.check_output(['findmnt', '-rn', '-T', '/mnt/hdd500', '-o', 'SOURCE,OPTIONS'], text=True).split()
assert mount[0] == '/dev/sda1' and 'rw' in mount[1].split(',') and os.path.ismount('/mnt/hdd500')
assert archive.stat().st_size == entry['bytes'] and not target.exists() and not partial.exists() and not receipt.exists()
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()
assert sha(archive) == entry['sha256'], 'Snapshot archive SHA mismatch'
with zipfile.ZipFile(archive) as z:
    info = z.getinfo('state_snapshot.sqlite')
    assert shutil.disk_usage(hdd).free - info.file_size >= 10 * 1024**3
    digest = hashlib.sha256()
    with z.open(info) as src, partial.open('xb') as dst:
        while True:
            block = src.read(8 * 1024**2)
            if not block:
                break
            if shutil.disk_usage(hdd).free - len(block) < 10 * 1024**3:
                raise RuntimeError('HDD reserve reached; partial preserved')
            dst.write(block)
            digest.update(block)
        dst.flush()
        os.fsync(dst.fileno())
    assert partial.stat().st_size == info.file_size
db = sqlite3.connect('file:' + str(partial) + '?mode=ro', uri=True)
try:
    assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
finally:
    db.close()
os.replace(partial, target)
tmp = receipt.with_name(receipt.name + '.tmp')
with tmp.open('x', encoding='utf-8') as f:
    json.dump({'source_archive_sha256': entry['sha256'], 'corpus_sha256': digest.hexdigest(),
               'bytes': info.file_size, 'sqlite_integrity': 'ok'}, f, indent=2)
    f.write('\n')
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp, receipt)
print('Source corpus extracted, checked and stored on HDD', flush=True)
