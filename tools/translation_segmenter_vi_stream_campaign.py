"""Continue the Vietnamese full-coverage campaign from its saved optimizer."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from turn_v2 import translation_segmenter_full_data_v5 as data
from turn_v2 import translation_segmenter_full_train_v5 as train
from turn_v2.translation_segmenter_v15c import atomic_json

ROOT = data.ROOT
JOB = data.JOB
STOP = False
RESERVE = 10 * 1024**3


def stop(*_):
    global STOP
    STOP = True


def status(stage, **extra):
    value = {'campaign': 'translation_segmenter_vi_stream_v1', 'status': 'RUNNING',
             'stage': stage, 'heartbeat_utc': data.now(), 'pid': os.getpid(),
             'language': 'vi', 'quality_target_certified': False,
             'qwen_judge_enabled': False, **extra}
    atomic_json(JOB / 'vi_only_campaign_status.json', value)
    return value


def main():
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB / 'CAMPAIGN_LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((JOB / 'current_dataset.json').read_text())
        folder = JOB / 'datasets' / manifest['generation']
        proof = json.loads((folder / 'verification_receipt.json').read_text())
        assert manifest['complete'] and proof['complete']
        assert proof['database_sha256'] == manifest['database_sha256']
        checkpoint = ROOT / 'teachers/vi/latest_resume.pt'
        assert checkpoint.is_file() and (ROOT / 'teachers/vi/resume_receipt.json').is_file()
        while not STOP and not (ROOT / 'STOP_REQUESTED').exists():
            # Training checkpoints are written atomically: allow one full temporary copy.
            needed = RESERVE + checkpoint.stat().st_size + 512 * 1024**2
            free_ssd, free_hdd = shutil.disk_usage(ROOT).free, shutil.disk_usage(folder).free
            if free_ssd < needed or free_hdd < RESERVE:
                status('WAITING_FOR_DISK_SPACE', free_ssd_bytes=free_ssd,
                       required_ssd_bytes=needed, free_hdd_bytes=free_hdd)
                time.sleep(30)
                continue
            receipt = json.loads((ROOT / 'teachers/vi/resume_receipt.json').read_text())
            epoch = len(train.history_for(manifest, 'vi')) + 1
            if receipt.get('full_epoch_generation') == manifest['generation']:
                saved_epoch = receipt.get('full_epoch_number')
                if saved_epoch and not receipt.get('full_epoch_complete'):
                    epoch = int(saved_epoch)
            command = [sys.executable, '-u', '-m', 'turn_v2.translation_segmenter_vi_stream_v1',
                       '--generation', manifest['generation'], '--epoch', str(epoch),
                       '--budget', str(train.CHUNK_STEPS)]
            folder_log = JOB / 'runs' / manifest['generation'] / 'vi'
            folder_log.mkdir(parents=True, exist_ok=True)
            with (folder_log / f'epoch_{epoch:04d}.log').open('a') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                ticks = (Path('/proc') / str(child.pid) / 'stat').read_text().split()[21]
                record = {'pid': child.pid, 'start_ticks': ticks, 'command': command,
                          'module': 'turn_v2.translation_segmenter_vi_stream_v1'}
                atomic_json(ROOT / 'child.json', record)
                status('PREPARING_OR_TRAINING', generation=manifest['generation'], epoch=epoch, child=record)
                result = child.wait()
            if result:
                raise RuntimeError(f'Vietnamese trainer exited with code {result}; checkpoint preserved')
        status('STOPPED_BY_REQUEST')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        main()
    except Exception as exc:
        atomic_json(JOB / 'vi_only_campaign_error.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        status('CAMPAIGN_ERROR_PRESERVED', error=repr(exc))
        raise
