"""Run the prepared SSH migration once, leaving integrity checking independent."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

root = Path('/home/edabk/hoangbpm/turn_detect')
staging = root / 'staging'
status_file = staging / 'preparation_status.json'


def status(stage, **extra):
    temp = status_file.with_name(status_file.name + '.tmp')
    temp.write_text(json.dumps({'stage': stage, 'at_unix': time.time(), **extra}, indent=2) + '\n')
    os.replace(temp, status_file)


try:
    os.chdir(root)
    with (staging / 'PREPARATION_LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status('VERIFYING_AND_PREPARING', no_training_started=True, sqlite_integrity_check_pending=True)
        env = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
        result = subprocess.run([str(root / '.venv/bin/python'), '-u', str(root / 'prepare_remote_turn_detect_v3.py')], env=env)
        if result.returncode:
            status('PREPARATION_FAILED_PRESERVED', returncode=result.returncode)
            raise SystemExit(result.returncode)
        status('PREPARED_WAITING_FOR_SAFE_CUTOVER', no_training_started=True, sqlite_integrity_check_pending=True)
except Exception as exc:
    status('PREPARATION_ERROR_PRESERVED', error=repr(exc), traceback=traceback.format_exc())
    raise
