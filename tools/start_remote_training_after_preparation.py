"""Start the staged SSH campaign once its checkpoint preparation has completed."""
import fcntl
import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

PROJECT = Path('/home/edabk/hoangbpm/turn_detect')
STAGING = PROJECT / 'staging'
ROOT = PROJECT / 'outputs/translation_segmenter_v15d_20261026'
HDD = Path('/mnt/hdd500')
STATUS = STAGING / 'ssh_training_launch_status.json'
MODULE = 'turn_v2.translation_segmenter_full_campaign_v8'


def status(stage, **fields):
    tmp = STATUS.with_name(STATUS.name + '.tmp')
    tmp.write_text(json.dumps({'stage': stage, 'at_unix': time.time(), **fields}, indent=2) + '\n')
    os.replace(tmp, STATUS)


def existing_campaign():
    matches = []
    for path in glob.glob('/proc/[0-9]*/cmdline'):
        try:
            command = Path(path).read_bytes().split(b'\0')
            if MODULE.encode() in command:
                pid = int(path.split('/')[2])
                ticks = (Path('/proc') / str(pid) / 'stat').read_text().split()[21]
                matches.append({'pid': pid, 'start_ticks': ticks})
        except (OSError, ValueError, IndexError):
            pass
    return matches


def main():
    os.chdir(PROJECT)
    with (STAGING / 'SSH_TRAINING_LAUNCH_LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            prep = json.loads((STAGING / 'preparation_status.json').read_text())
            stage = prep.get('stage', '')
            if stage == 'PREPARED_WAITING_FOR_SAFE_CUTOVER':
                break
            if 'FAILED' in stage or 'ERROR' in stage:
                status('PREPARATION_FAILED', preparation=prep)
                return
            if stage != 'VERIFYING_AND_PREPARING':
                status('WAITING_FOR_PREPARATION', preparation_stage=stage)
            time.sleep(20)
        receipt = json.loads((STAGING / 'prepared_receipt.json').read_text())
        if not receipt.get('complete') or set(receipt.get('languages', {})) != {'vi', 'en', 'zh'}:
            raise RuntimeError('Incomplete preparation receipt')
        if (ROOT / 'STOP_REQUESTED').exists():
            raise RuntimeError('STOP_REQUESTED exists; preserving it')
        mount = subprocess.check_output(['findmnt', '-rn', '-T', str(HDD), '-o', 'SOURCE,OPTIONS'], text=True).split()
        if not os.path.ismount(HDD) or mount[0] != '/dev/sda1' or 'rw' not in mount[1].split(','):
            raise RuntimeError('Expected writable HDD is not mounted')
        if min(shutil.disk_usage(PROJECT).free, shutil.disk_usage(HDD).free) < 10 * 1024**3:
            raise RuntimeError('SSD or HDD is below the 10 GiB reserve')
        current = existing_campaign()
        if current:
            status('ALREADY_RUNNING', processes=current)
            return
        env = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
        command = [str(PROJECT / '.venv/bin/python'), '-u', '-m', MODULE]
        with (STAGING / 'ssh_campaign_launch.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ticks = (Path('/proc') / str(child.pid) / 'stat').read_text().split()[21]
        status('CAMPAIGN_LAUNCHED', pid=child.pid, start_ticks=ticks, command=command,
               backup_note='New SSH checkpoints are not yet backed up to Google Drive')
        time.sleep(5)
        if child.poll() is not None:
            status('CAMPAIGN_EXITED_EARLY', pid=child.pid, returncode=child.returncode)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('LAUNCH_FAILED', error=repr(exc))
        raise
