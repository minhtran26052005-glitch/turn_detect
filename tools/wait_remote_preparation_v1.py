"""Wait for staged immutable assets, then prepare them; does not start training."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path('/home/edabk/hoangbpm/turn_detect')
STAGE = ROOT / 'staging'
STATUS = STAGE / 'preparation_status.json'


def state(stage, **fields):
    temp = STATUS.with_name(STATUS.name + '.tmp')
    temp.write_text(json.dumps({'stage': stage, 'at_unix': time.time(), **fields}, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, STATUS)


def ready():
    manifest = json.loads((STAGE / 'drive_manifest.json').read_text(encoding='utf-8'))
    files = manifest['files']
    generation = 'cycle000084_full_epoch_contiguous_v1'
    requirements = {
        STAGE / 'cycle84_dataset.zip.partial': files[f'full_epoch_v1/datasets/{generation}/dataset.zip']['bytes'],
        STAGE / 'source_snapshot.zip.partial': files['data/data_snapshot.zip']['bytes'],
        STAGE / 'reports_and_logs.zip': files['reports/reports_and_logs.zip']['bytes'],
    }
    for lang in ('vi', 'en', 'zh'):
        ck = files[f'teachers/{lang}/latest_resume.pt']
        requirements[STAGE / 'checkpoints' / f"{lang}_{ck['sha256']}.pt"] = ck['bytes']
        for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json'):
            requirements[STAGE / 'models' / lang / name] = files[f'teachers/{lang}/best/{name}']['bytes']
    missing = []
    for path, size in requirements.items():
        observed = path.stat().st_size if path.is_file() else 0
        if observed != size:
            missing.append({'file': str(path.relative_to(ROOT)), 'bytes': observed, 'expected': size})
    if not (STAGE / 'corpus_extraction_receipt.json').is_file() or not Path('/mnt/hdd500/turn_detect_training/corpus.sqlite').is_file():
        missing.append({'file': 'corpus.sqlite', 'stage': 'WAITING_FOR_SOURCE_EXTRACTION_AND_INTEGRITY'})
    return missing


def main():
    os.chdir(ROOT)
    with (STAGE / 'PREPARATION_LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            if not os.path.ismount('/mnt/hdd500'):
                state('WAITING_FOR_HDD_MOUNT')
                time.sleep(30)
                continue
            missing = ready()
            if missing:
                state('WAITING_FOR_TRANSFER', files=missing)
                time.sleep(30)
                continue
            state('VERIFYING_AND_PREPARING', no_training_started=True)
            env = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
            result = subprocess.run([str(ROOT / '.venv/bin/python'), '-u', str(ROOT / 'prepare_remote_turn_detect_v1.py')],
                                    cwd=ROOT, env=env)
            if result.returncode:
                state('PREPARATION_FAILED_PRESERVED', returncode=result.returncode, no_training_started=True)
                raise SystemExit(result.returncode)
            state('PREPARED_WAITING_FOR_SAFE_CUTOVER', no_training_started=True)
            return


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        state('PREPARATION_ERROR_PRESERVED', error=repr(exc), traceback=traceback.format_exc(), no_training_started=True)
        raise
