"""Restore a committed mounted-Drive backup on Colab, then resume its optimizer.

This module uses only stdlib until the dependency installation phase. Large
objects never pass through Windows. Existing unrelated state is never replaced.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import traceback
import zipfile

PROJECT = Path('/content/turn_detect_v2')
RUN = PROJECT / 'outputs/translation_segmenter_v15d_20261026'
CONTROL = PROJECT / 'restore_control_v1'
RESERVE = 25 * 1024**3
LANGUAGES = ('vi', 'en', 'zh')


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.restore_tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def state(stage, **values):
    value = {'at_utc': now(), 'stage': stage, **values}
    write_json(CONTROL / 'status.json', value)
    if RUN.exists():
        write_json(RUN / 'status.json', {'campaign': 'translation_segmenter_v15d',
            'status': 'RECOVERING', 'heartbeat_utc': value['at_utc'], 'pid': os.getpid(),
            'quality_target_certified': False, 'qwen_judge_enabled': False, **value})
    print(json.dumps(value), flush=True)


def child_record(proc, command, log):
    return {'pid': proc.pid, 'start_ticks': (Path('/proc') / str(proc.pid) / 'stat').read_text().split()[21],
            'command': command, 'module': command[command.index('-m') + 1] if '-m' in command else command[-1],
            'log': str(Path(log).relative_to(PROJECT))}


def target_under(root, relative):
    value = PurePosixPath(relative)
    if value.is_absolute() or '..' in value.parts or '\\' in relative:
        raise ValueError('Unsafe archive/logical path: ' + relative)
    result = (Path(root) / str(value)).resolve()
    result.relative_to(Path(root).resolve())
    return result


def extract_archive(archive, root, *, protect=()):
    """Never follow ZIP symlinks or replace existing different user state."""
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            path = target_under(root, item.filename)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('ZIP symlink refused: ' + item.filename)
            if item.filename in protect:
                path = target_under(CONTROL / 'historical_metadata', item.filename)
            if item.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + '.restore_extract')
            with source.open(item) as src, temp.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024**2)
            if path.exists() and sha(path) != sha(temp):
                raise RuntimeError('Refusing to overwrite different existing state: ' + str(path))
            temp.replace(path)


def historical_fingerprint(db, language, cycle):
    allowed = {language + ':' + str(row[0]) for row in db.execute(
        'SELECT page_id FROM sources WHERE language=? AND added_cycle<=?', (language, cycle))}
    digest = hashlib.sha256()
    selected = 0
    for source_id, raw in db.execute("SELECT source_id,record_json FROM examples WHERE language=? AND split='train' ORDER BY example_id", (language,)):
        if source_id in allowed:
            digest.update(raw.encode('utf-8')); digest.update(b'\n'); selected += 1
    return digest.hexdigest(), selected


def copy_object(drive, entry, target):
    relative = entry['object']
    if not re.fullmatch(r'objects/[0-9a-f]{64}\.bin', relative):
        raise ValueError('Unexpected content object path')
    source = target_under(drive, relative)
    if source.stem != entry['sha256'] or source.stat().st_size != entry['bytes']:
        raise RuntimeError('Drive object metadata mismatch: ' + relative)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == entry['bytes'] and sha(target) == entry['sha256']:
            return
        raise RuntimeError('Existing target differs; preserving it: ' + str(target))
    if shutil.disk_usage(PROJECT).free < entry['bytes'] + RESERVE:
        raise RuntimeError('Colab disk reserve would be violated')
    temp = target.with_name(target.name + '.restore_partial')
    if temp.exists():
        raise RuntimeError('Preserve prior partial restore; inspect before retry: ' + str(temp))
    digest = hashlib.sha256(); count = 0
    state('COPYING_VERIFIED_DRIVE_OBJECT', logical_target=str(target.relative_to(PROJECT)), bytes=entry['bytes'])
    with source.open('rb') as src, temp.open('xb') as dst:
        for block in iter(lambda: src.read(8 * 1024**2), b''):
            dst.write(block); digest.update(block); count += len(block)
        dst.flush(); os.fsync(dst.fileno())
    if count != entry['bytes'] or digest.hexdigest() != entry['sha256'] or sha(temp) != entry['sha256']:
        raise RuntimeError('Object transfer/readback hash mismatch: ' + relative)
    temp.replace(target)


def subprocess_checked(command, log_name, stage):
    log = CONTROL / log_name
    state(stage)
    with log.open('a', encoding='utf-8') as stream:
        proc = subprocess.Popen(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, 'OMP_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false',
                 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1'})
        write_json(RUN / 'child.json', child_record(proc, command, log))
        code = proc.wait()
    if code:
        raise RuntimeError(f'Child exited {code}; preserve log {log}')


def main():
    config = json.loads((CONTROL / 'config.json').read_text())
    drive = Path(config['drive_path'])
    if not os.path.ismount('/content/drive'):
        raise RuntimeError('User Drive mount is required')
    if json.loads((drive / 'destination_identity.json').read_text()) != config['identity']:
        raise RuntimeError('Mounted Drive destination identity mismatch')
    manifest = json.loads((drive / 'latest_manifest.json').read_text())
    canonical_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if canonical_hash != config['manifest_canonical_sha256']:
        raise RuntimeError('Drive generation changed after preflight; inspect the new committed manifest')
    if not manifest.get('pass_complete') or not manifest.get('full_recovery_available'):
        raise RuntimeError('Only a complete committed backup may restore the campaign')
    files = manifest['files']
    required = ['data/data_snapshot.zip', 'source/source_bundle.zip', 'reports/reports_and_logs.zip']
    required += [f'teachers/{lang}/{name}' for lang in LANGUAGES for name in ('latest_resume.pt', 'best/model.safetensors')]
    if any(name not in files for name in required):
        raise RuntimeError('Backup is missing required data/source/checkpoint objects')
    if RUN.exists():
        raise RuntimeError('Existing campaign state must be inspected; refusing overwrite')
    # Includes raw corpus/frozen DB, temporary copy and the minimum reserve.
    if shutil.disk_usage(PROJECT).free < sum(x['bytes'] for x in files.values()) + 16 * 1024**3 + RESERVE:
        raise RuntimeError('Insufficient Colab disk above the reserve')
    RUN.mkdir(parents=True)
    write_json(CONTROL / 'manifest.json', manifest)
    write_json(RUN / 'launcher.json', json.loads((CONTROL / 'launcher.json').read_text()))
    state('RESTORING_COMMITTED_DRIVE_GENERATION', captured_at_utc=manifest['captured_at_utc'])

    source_zip = CONTROL / 'source_bundle.zip'
    copy_object(drive, files['source/source_bundle.zip'], source_zip)
    extract_archive(source_zip, PROJECT)
    for relative, expected in config['core_source_sha256'].items():
        if sha(target_under(PROJECT, relative)) != expected:
            raise RuntimeError('Pinned local/backup source differs: ' + relative)

    reports_zip = CONTROL / 'reports_and_logs.zip'
    copy_object(drive, files['reports/reports_and_logs.zip'], reports_zip)
    extract_archive(reports_zip, RUN, protect=('launcher.json', 'child.json', 'status.json'))

    data = files['data/data_snapshot.zip']
    data_zip = RUN / 'recovery/data_snapshot.zip'
    copy_object(drive, data, data_zip)
    with zipfile.ZipFile(data_zip) as source:
        if set(source.namelist()) != {'state_snapshot.sqlite', 'config.json', 'status.json', 'data_summary.json'}:
            raise RuntimeError('Unexpected data snapshot contents')
        with source.open('state_snapshot.sqlite') as src, (RUN / 'corpus.sqlite').open('xb') as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024**2)
        recipe = json.loads(source.read('config.json'))
        summary = json.loads(source.read('data_summary.json'))
    if recipe['source_sha256'] != sha(PROJECT / 'turn_v2/translation_segmenter_v15d3.py'):
        raise RuntimeError('Corpus recipe/source hash mismatch')
    write_json(RUN / 'config.json', recipe)
    write_json(RUN / 'data_summary.json', summary)
    write_json(RUN / 'recovery/latest_snapshot.json', data['source_receipt'])
    data_cycle = int(data['source_receipt']['cycle'])
    if summary['cycle'] != data_cycle:
        raise RuntimeError('Data summary cycle mismatch')

    for logical, entry in files.items():
        if logical in required[:3]:
            continue
        if logical == 'heldout_test_v2/frozen_test_database.zip':
            target = RUN / 'drive_backup_v1/frozen_test_database.zip'
        else:
            target = target_under(RUN, logical)
        # Mutable report files were also captured in the historical log archive.
        if target.exists() and sha(target) != entry['sha256']:
            if target.suffix not in ('.json', '.jsonl'):
                raise RuntimeError('Conflicting restored artifact: ' + logical)
            history = target_under(CONTROL / 'historical_metadata', logical)
            history.parent.mkdir(parents=True, exist_ok=True); target.replace(history)
        copy_object(drive, entry, target)
    frozen_zip = RUN / 'drive_backup_v1/frozen_test_database.zip'
    if frozen_zip.exists():
        with zipfile.ZipFile(frozen_zip) as source:
            if source.namelist() != ['test.sqlite']:
                raise RuntimeError('Unexpected frozen-test database archive')
            target = RUN / 'heldout_test_v2/frozen/test.sqlite'
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('test.sqlite') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024**2)
        expected = files['heldout_test_v2/frozen_test_database.zip']['source_receipt']['uncompressed_database_sha256']
        if sha(target) != expected:
            raise RuntimeError('Frozen test database hash mismatch')

    versions = json.loads((PROJECT / 'runtime_versions.json').read_text())
    packages = [f'{name}=={versions[name]}' for name in ('transformers', 'datasets', 'numpy', 'sentencepiece', 'jieba', 'huggingface_hub') if name in versions]
    packages += ['safetensors', 'requests', 'beautifulsoup4', 'scikit-learn']
    subprocess_checked([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--quiet', *packages],
                       'dependency_install.log', 'INSTALLING_RECORDED_DEPENDENCIES')

    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('The restored campaign requires a BF16 Colab GPU')
    state('VERIFYING_SQLITE_AND_OPTIMIZER_PROVENANCE', cycle=data_cycle)
    db = sqlite3.connect('file:' + str(RUN / 'corpus.sqlite') + '?mode=ro', uri=True)
    integrity = db.execute('PRAGMA integrity_check').fetchone()[0]
    next_cycle = int(db.execute("SELECT value FROM meta WHERE key='next_cycle'").fetchone()[0])
    if integrity != 'ok' or next_cycle != data_cycle + 1:
        raise RuntimeError('Restored database integrity/cycle marker mismatch')
    checks = {}
    pending = []
    for lang in LANGUAGES:
        entry = files[f'teachers/{lang}/latest_resume.pt']; receipt = entry['source_receipt']
        folder = RUN / 'teachers' / lang
        saved = torch.load(folder / 'latest_resume.pt', map_location='cpu', weights_only=False)
        for key in ('cycle', 'cycle_step', 'total_steps', 'training_fingerprint'):
            if saved[key] != receipt[key]:
                raise RuntimeError(f'{lang} checkpoint/receipt mismatch: {key}')
        if not saved['optimizer']['state'] or not saved['optimizer']['param_groups'] or not saved['model']:
            raise RuntimeError('Missing model/AdamW state: ' + lang)
        if not all(key in saved for key in ('numpy_rng', 'torch_rng', 'cuda_rng')):
            raise RuntimeError('Missing RNG state: ' + lang)
        fingerprint, rows = historical_fingerprint(db, lang, saved['cycle'])
        if fingerprint != saved['training_fingerprint']:
            raise RuntimeError('Historical training fingerprint mismatch: ' + lang)
        if saved['cycle'] < data_cycle and saved['cycle_step'] < recipe['steps']:
            raise RuntimeError('Historical partial cycle needs a versioned filtered-data recovery: ' + lang)
        report_path = folder / 'last_training_report.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if report.get('cycle', 0) > saved['cycle']:
            history = CONTROL / 'historical_metadata' / f'{lang}_later_report.json'
            report_path.replace(history); report = {}
        if data_cycle % 3 == 0 and (saved['cycle'] < data_cycle or saved['cycle_step'] < recipe['steps'] or report.get('cycle', 0) < data_cycle):
            pending.append(lang)
        checks[lang] = {'cycle': saved['cycle'], 'cycle_step': saved['cycle_step'],
            'total_steps': saved['total_steps'], 'training_fingerprint': fingerprint,
            'training_rows': rows, 'optimizer_states': len(saved['optimizer']['state']),
            'rng_present': True, 'weights_sha256': entry['sha256']}
        write_json(folder / 'resume_receipt.json', receipt)
        write_json(folder / 'best_report.json', files[f'teachers/{lang}/best/model.safetensors']['source_receipt'])
        if not (folder / 'pretrained_baseline.json').exists():
            write_json(folder / 'pretrained_baseline.json', {'skipped': True,
                'reason': 'Restored trained checkpoint; no new random-head baseline is claimed'})
        progress = folder / 'progress.json'
        if progress.exists():
            history = CONTROL / 'historical_metadata' / f'{lang}_prior_progress.json'
            progress.replace(history)
        del saved; gc.collect()
    db.close()
    restore_receipt = {'restored_at_utc': now(), 'manifest_at_utc': manifest['captured_at_utc'],
        'data_cycle': data_cycle, 'next_crawl_cycle': next_cycle, 'database_integrity': integrity,
        'checkpoints': checks, 'pending_training_languages': pending,
        'weights_optimizer_rng_restored': True, 'training_fingerprints_match': True,
        'local_Windows_large_downloads': False, 'recorded_torch_version': versions.get('torch'),
        'restored_torch_version': torch.__version__}
    write_json(RUN / 'recovery/drive_restore_receipt_v1.json', restore_receipt)
    write_json(CONTROL / 'restore_receipt.json', restore_receipt)
    state('RESTORED_VERIFIED_CHECKPOINTS', **restore_receipt)

    backup_job = RUN / 'drive_backup_v1'; backup_job.mkdir(exist_ok=True)
    write_json(backup_job / 'config.json', {k: config[k] for k in ('drive_path', 'identity', 'interval_seconds')})
    command = [sys.executable, '-u', '-m', 'turn_v2.translation_segmenter_drive_backup_v1']
    log = backup_job / 'worker.log'
    with log.open('a') as stream:
        proc = subprocess.Popen(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, env={**os.environ, 'OMP_NUM_THREADS': '1'})
    record = child_record(proc, command, log)
    record.update({'source_sha256': sha(PROJECT / 'turn_v2/translation_segmenter_drive_backup_v1.py'), 'drive_path': str(drive)})
    write_json(backup_job / 'launcher.json', record)

    # The data snapshot already committed its next-cycle cursor. Finish the
    # interrupted training batch before the crawler advances that cursor.
    for lang in pending:
        state('RESUMING_INTERRUPTED_TRAINING_BATCH', cycle=data_cycle, language=lang,
              restored_cycle_step=checks[lang]['cycle_step'], restored_total_steps=checks[lang]['total_steps'])
        command = [sys.executable, '-u', '-m', 'turn_v2.translation_segmenter_v15d2', '--phase', 'train',
                   '--cycle', str(data_cycle), '--steps', str(recipe['steps']),
                   '--microbatch', str(recipe['microbatch']), '--language', lang]
        subprocess_checked(command, f'resume_{lang}.log', 'RESUMING_' + lang.upper())

    command = [sys.executable, '-u', '-m', 'turn_v2.translation_segmenter_v15d3', '--phase', 'run',
               '--docs-per-source', str(recipe['docs_per_hf_source_cycle']), '--steps', str(recipe['steps']),
               '--microbatch', str(recipe['microbatch'])]
    launcher = json.loads((RUN / 'launcher.json').read_text())
    launcher.update({'command': command, 'module': 'turn_v2.translation_segmenter_v15d3',
        'source_sha256': sha(PROJECT / 'turn_v2/translation_segmenter_v15d3.py'),
        'resumed_after_drive_restore': str(RUN / 'recovery/drive_restore_receipt_v1.json')})
    write_json(RUN / 'launcher.json', launcher)
    state('HANDOFF_TO_CONTINUOUS_CRAWLER', next_cycle=next_cycle)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write_json(CONTROL / 'error.json', {'at_utc': now(), 'error': repr(exc), 'traceback': traceback.format_exc()})
        state('RESTORE_FAILED_PRESERVED', error=repr(exc))
        raise
