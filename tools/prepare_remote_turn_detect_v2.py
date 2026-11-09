"""Prepare verified full-epoch training on the SSH host; never starts training."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
import zipfile

import numpy as np
import torch

PROJECT = Path('/home/edabk/hoangbpm/turn_detect')
STAGING = PROJECT / 'staging'
ROOT = PROJECT / 'outputs/translation_segmenter_v15d_20261026'
HDD = Path('/mnt/hdd500/turn_detect_training')
JOB = ROOT / 'full_epoch_v1'
GENERATION = 'cycle000084_full_epoch_contiguous_v1'
MANIFEST_SHA = 'a1f45cb8d1f29bb83f16fb9914397410f3386ea9de6f3b37d3d2e1674f675e05'
RESERVE = 10 * 1024**3


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    if temp.exists():
        raise RuntimeError(f'Existing temporary file: {temp}')
    with temp.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
        raise RuntimeError(f'Existing metadata differs: {path}')
    os.replace(temp, path)


def checked_file(path, expected):
    if not path.is_file() or path.stat().st_size != expected['bytes']:
        raise RuntimeError(f'File missing or incomplete: {path}')
    if sha(path) != expected['sha256']:
        raise RuntimeError(f'SHA mismatch: {path}')


def extract_large(archive, member, destination, expected_sha=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_sha and sha(destination) != expected_sha:
            raise RuntimeError(f'Existing extracted file differs: {destination}')
        if not expected_sha:
            raise RuntimeError(f'Existing corpus requires manual verification: {destination}')
        return
    temporary = destination.with_name(destination.name + '.partial')
    if temporary.exists():
        raise RuntimeError(f'Existing extraction partial preserved: {temporary}')
    with zipfile.ZipFile(archive) as z:
        info = z.getinfo(member)
        if shutil.disk_usage(HDD).free - info.file_size < RESERVE:
            raise RuntimeError('Insufficient HDD free space for extraction and reserve')
        digest = hashlib.sha256()
        with z.open(info) as src, temporary.open('xb') as dst:
            while True:
                block = src.read(8 * 1024**2)
                if not block:
                    break
                if shutil.disk_usage(HDD).free - len(block) < RESERVE:
                    raise RuntimeError('HDD reserve reached; partial preserved')
                dst.write(block)
                digest.update(block)
            dst.flush()
            os.fsync(dst.fileno())
        if temporary.stat().st_size != info.file_size or (expected_sha and digest.hexdigest() != expected_sha):
            raise RuntimeError(f'Extracted file verification failed: {destination}')
    os.replace(temporary, destination)


def main():
    os.chdir(PROJECT)
    out = subprocess.check_output(['findmnt', '-rn', '-T', '/mnt/hdd500', '-o', 'SOURCE,OPTIONS'], text=True).strip().split()
    assert out and out[0] == '/dev/sda1' and 'rw' in out[1].split(','), 'HDD must be mounted read/write'
    assert os.path.ismount('/mnt/hdd500') and HDD.is_dir()
    assert (JOB / 'datasets').resolve() == (HDD / 'datasets').resolve()
    assert (ROOT / 'corpus.sqlite').resolve() == (HDD / 'corpus.sqlite').resolve()
    mpath = STAGING / 'drive_manifest.json'
    assert sha(mpath) == MANIFEST_SHA
    m = json.loads(mpath.read_text(encoding='utf-8'))
    assert m['pass_complete'] and m['full_recovery_available'] and m['full_epoch_recipe_supported']
    assert m['verified_active_campaign_source']['module'] == 'turn_v2.translation_segmenter_full_campaign_v8'
    assert m['verified_active_campaign_source']['source_sha256'] == sha(PROJECT / 'turn_v2/translation_segmenter_full_campaign_v8.py')
    files = m['files']
    dkey = f'full_epoch_v1/datasets/{GENERATION}/dataset.zip'
    dmeta = files[dkey]
    dataset_archive = STAGING / 'cycle84_dataset.zip.partial'
    raw_archive = STAGING / 'source_snapshot.zip.partial'
    checked_file(dataset_archive, dmeta)
    checked_file(raw_archive, files['data/data_snapshot.zip'])
    checked_file(STAGING / 'reports_and_logs.zip', files['reports/reports_and_logs.zip'])
    checked_file(PROJECT / 'source_bundle.zip', files['source/source_bundle.zip'])
    f = json.loads((STAGING / 'frozen_checkpoint_receipt.json').read_text(encoding='utf-8'))['files']
    for lang in ('vi', 'en', 'zh'):
        entry = files[f'teachers/{lang}/latest_resume.pt']
        assert f[lang]['sha256'] == entry['sha256'] and f[lang]['source_receipt'] == entry['source_receipt']
        checked_file(STAGING / 'checkpoints' / f"{lang}_{entry['sha256']}.pt", entry)
        for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json'):
            checked_file(STAGING / 'models' / lang / name, files[f'teachers/{lang}/best/{name}'])
    manifest = json.loads((STAGING / 'cycle84_manifest.json').read_text(encoding='utf-8'))
    assert manifest['generation'] == GENERATION and manifest['archive_sha256'] == dmeta['sha256']
    print('All staged archives, models and checkpoints match the committed Drive manifest', flush=True)
    folder = JOB / 'datasets' / GENERATION
    folder.mkdir(parents=True, exist_ok=True)
    extract_large(dataset_archive, 'dataset.sqlite', folder / 'dataset.sqlite', manifest['database_sha256'])
    print('Training SQLite extracted and SHA-256 verified', flush=True)
    raw_snapshot = HDD / 'corpus.sqlite'
    if raw_snapshot.exists():
        corpus_receipt = json.loads((STAGING / 'corpus_extraction_receipt.json').read_text(encoding='utf-8'))
        assert corpus_receipt['source_archive_sha256'] == files['data/data_snapshot.zip']['sha256']
        assert corpus_receipt['sqlite_integrity'] == 'ok' and sha(raw_snapshot) == corpus_receipt['corpus_sha256']
    else:
        extract_large(raw_archive, 'state_snapshot.sqlite', raw_snapshot)
    for path in (folder / 'dataset.sqlite', raw_snapshot):
        db = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
        try:
            assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', str(path)
        finally:
            db.close()
    print('Training and source SQLite integrity checks passed', flush=True)
    atomic_json(folder / 'manifest.json', manifest)
    atomic_json(JOB / 'current_dataset.json', manifest)
    with zipfile.ZipFile(raw_archive) as z:
        for name in ('config.json', 'data_summary.json'):
            if name in z.namelist():
                atomic_json(ROOT / name, json.loads(z.read(name)))
    atomic_json(ROOT / 'recovery/latest_snapshot.json', files['data/data_snapshot.zip']['source_receipt'])
    with zipfile.ZipFile(STAGING / 'reports_and_logs.zip') as z:
        receipt_name = f'full_epoch_v1/datasets/{GENERATION}/verification_receipt.json'
        vr = json.loads(z.read(receipt_name))
        assert vr['complete'] and vr['database_sha256'] == manifest['database_sha256']
        assert vr['verifier_source_sha256'] == sha(PROJECT / 'turn_v2/translation_segmenter_full_verify_v2.py')
        atomic_json(folder / 'verification_receipt.json', vr)
        for lang in ('vi', 'en', 'zh'):
            meta = files[f'teachers/{lang}/latest_resume.pt']
            rec = meta['source_receipt']
            teacher = ROOT / 'teachers' / lang
            best = teacher / 'best'
            best.mkdir(parents=True, exist_ok=True)
            for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json'):
                src, dst = STAGING / 'models' / lang / name, best / name
                expected = files[f'teachers/{lang}/best/{name}']
                if dst.exists():
                    checked_file(dst, expected)
                else:
                    shutil.copy2(src, dst)
                    checked_file(dst, expected)
            best_report = files[f'teachers/{lang}/best/model.safetensors'].get('source_receipt')
            if best_report:
                atomic_json(teacher / 'best_report.json', best_report)
            src = STAGING / 'checkpoints' / f"{lang}_{meta['sha256']}.pt"
            checkpoint = teacher / 'latest_resume.pt'
            if checkpoint.exists():
                checked_file(checkpoint, meta)
            else:
                os.link(src, checkpoint)
            atomic_json(teacher / 'resume_receipt.json', rec)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            for key in ('cycle', 'cycle_step', 'total_steps', 'training_fingerprint'):
                assert saved[key] == rec[key], (lang, key)
            assert saved['optimizer']['state'] and all(k in saved for k in ('torch_rng', 'cuda_rng', 'numpy_rng'))
            sampler = saved['full_epoch']
            fp = manifest['fingerprints'][lang]['train']
            assert sampler['generation'] == GENERATION and sampler['fingerprint'] == fp['sha256']
            n, cursor, order, visits = fp['rows'], sampler['position'], sampler['order'], sampler['visits']
            assert 0 <= cursor <= n and np.array_equal(np.sort(order), np.arange(n, dtype=np.int64))
            expected = np.zeros(n, dtype=np.uint8)
            expected[order[:cursor]] = 1
            assert np.array_equal(visits, expected)
            completed = sampler['epoch'] - (cursor < n)
            for ep in range(1, completed + 1):
                for prefix in ('epoch_', 'coverage_epoch_'):
                    member = f'full_epoch_v1/runs/{GENERATION}/{lang}/{prefix}{ep:04d}.json'
                    if member in z.namelist():
                        atomic_json(JOB / 'runs' / GENERATION / lang / f'{prefix}{ep:04d}.json', json.loads(z.read(member)))
            print(f'{lang}: checkpoint optimizer/RNG and cursor {cursor}/{n} verified', flush=True)
            del saved, sampler, order, visits, expected
            gc.collect()
    atomic_json(ROOT / 'drive_backup_v1/latest_verified_manifest.json', m)
    atomic_json(STAGING / 'prepared_receipt.json', {'complete': True, 'dataset_sha256': manifest['database_sha256'],
        'manifest_sha256': MANIFEST_SHA, 'languages': {l: files[f'teachers/{l}/latest_resume.pt']['source_receipt'] for l in ('vi', 'en', 'zh')},
        'HDD_dataset_and_corpus': str(HDD), 'no_training_started': True})
    print('RESTORE_PREPARED: no training launched; cutover verification still required', flush=True)


if __name__ == '__main__':
    main()
