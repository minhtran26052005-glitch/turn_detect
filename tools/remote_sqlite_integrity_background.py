"""Independently check staged SQLite databases without modifying them."""
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback

root = Path('/home/edabk/hoangbpm/turn_detect')
staging = root / 'staging'
paths = [Path('/mnt/hdd500/turn_detect_training/datasets/cycle000084_full_epoch_contiguous_v1/dataset.sqlite'),
         Path('/mnt/hdd500/turn_detect_training/corpus.sqlite')]
target = staging / 'independent_sqlite_integrity.json'


def write(value):
    tmp = target.with_name(target.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, target)


results = []
try:
    for path in paths:
        write({'stage': 'CHECKING', 'path': str(path), 'completed': results, 'at_unix': time.time()})
        conn = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
        try:
            outcome = [row[0] for row in conn.execute('PRAGMA integrity_check')]
        finally:
            conn.close()
        results.append({'path': str(path), 'result': outcome})
        if outcome != ['ok']:
            write({'stage': 'FAILED', 'completed': results, 'at_unix': time.time()})
            raise RuntimeError('SQLite integrity check failed')
    write({'stage': 'COMPLETE', 'completed': results, 'at_unix': time.time()})
except Exception as exc:
    write({'stage': 'FAILED', 'completed': results, 'error': repr(exc), 'traceback': traceback.format_exc()})
    raise
