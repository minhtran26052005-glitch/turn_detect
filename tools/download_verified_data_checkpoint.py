"""Copy verified campaign data and current model checkpoints to the requested folder."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import threading
import time
from urllib.parse import quote

import requests
from colab_runtime import ColabRuntime

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data_checkpoint"
STATUS = ROOT / "outputs/translation_segmenter_v15d_20261026/drive_status_receipt.json"
CONFIG = ROOT / "outputs/translation_segmenter_v15d_20261026/drive_backup_config.json"
RESERVE = 10 * 1024 ** 3
tls = threading.local()


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def connection():
    if not hasattr(tls, 'runtime'):
        tls.runtime = ColabRuntime()
        tls.runtime.connect()
    return tls.runtime


def transfer(item):
    name, meta, drive = item
    path = TARGET.joinpath(*PurePosixPath(name).parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.partial')
    size = meta['bytes']
    if path.exists():
        if path.stat().st_size == size and sha_file(path) == meta['sha256']:
            print(f'Verified existing {name}', flush=True)
            return
        raise RuntimeError(f'Existing file does not match manifest: {name}')
    if part.exists() and part.stat().st_size > size:
        raise RuntimeError(f'Partial file is too large: {name}')
    offset = part.stat().st_size if part.exists() else 0
    if offset < size:
        object_path = PurePosixPath(meta['object'])
        if len(object_path.parts) != 2 or object_path.parts[0] != 'objects' or object_path.name != meta['sha256'] + '.bin':
            raise RuntimeError(f'Unexpected object path: {name}')
        remote = str(PurePosixPath(drive) / object_path)
        written = offset
        report = written // (256 * 1024 ** 2)
        failures = 0
        with part.open('ab') as out:
            while written < size:
                end = min(size, written + 16 * 1024 ** 2) - 1
                response = None
                try:
                    response = connection().request('GET', 'files/' + quote(remote.lstrip('/'), safe='/'),
                                                    headers={'Range': f'bytes={written}-{end}'}, stream=True)
                    is_full = written == 0 and end == size - 1 and response.status_code == 200 and response.headers.get('Content-Length') == str(size)
                    if not is_full and (response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {written}-{end}/{size}'):
                        raise RuntimeError(f'Unexpected HTTP range response for {name}')
                    start = written
                    for chunk in response.iter_content(chunk_size=1024 ** 2):
                        if not chunk:
                            continue
                        if shutil.disk_usage(TARGET).free - len(chunk) < RESERVE:
                            raise RuntimeError('Local disk reserve of 10 GiB reached')
                        out.write(chunk)
                        written += len(chunk)
                    if written != end + 1 or written < start:
                        raise RuntimeError(f'Incomplete range for {name}')
                    failures = 0
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
                    out.flush()
                    failures += 1
                    if failures > 5:
                        raise RuntimeError(f'Connection repeatedly interrupted for {name}; partial preserved') from exc
                    print(f'{name}: connection interrupted, continuing from {written / 1024 ** 2:.0f} MiB', flush=True)
                    time.sleep(min(failures * 2, 10))
                    continue
                finally:
                    if response is not None:
                        response.close()
                out.flush()
                if written // (256 * 1024 ** 2) > report:
                    report = written // (256 * 1024 ** 2)
                    print(f'{name}: {written / 1024 ** 2:.0f}/{size / 1024 ** 2:.0f} MiB', flush=True)
            os.fsync(out.fileno())
    if part.stat().st_size != size or sha_file(part) != meta['sha256']:
        raise RuntimeError(f'Size or SHA-256 mismatch: {name}; partial preserved')
    part.rename(path)
    print(f'SHA-256 verified {name}', flush=True)


def main():
    result = json.loads(STATUS.read_text(encoding='utf-8'))['result']
    manifest = result['latest_verified_manifest.json']
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    if not manifest.get('pass_complete') or not manifest.get('full_recovery_available'):
        raise RuntimeError('No completed, verified recovery manifest')
    if manifest['destination_identity'] != config['identity']:
        raise RuntimeError('Drive destination identity does not match')
    drive = manifest['destination_identity']['intended_path']
    files = manifest['files']
    dataset = 'full_epoch_v1/datasets/cycle000084_full_epoch_contiguous_v1/'
    mapping = {'data/train/cycle84_dataset.zip': dataset + 'dataset.zip',
               'data/train/cycle84_manifest.json': dataset + 'manifest.json',
               'data/train/source_snapshot.zip': 'data/data_snapshot.zip'}
    for lang in ('vi', 'en', 'zh'):
        for filename in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json'):
            mapping[f'models/{lang}/{filename}'] = f'teachers/{lang}/best/{filename}'
    selected = {dest: files[src] for dest, src in mapping.items()}
    diagnostic = json.loads((ROOT / 'outputs/translation_segmenter_v15d_20261026/diagnostic_test_c75_v1/drive_manifest.json').read_text(encoding='utf-8'))
    if not diagnostic.get('complete') or diagnostic['destination_identity'] != config['identity']:
        raise RuntimeError('C75 test manifest is not verified or Drive identity differs')
    test_drive = drive + '/diagnostic_test_c75_v1'
    tests = {f'data/test/{name}': diagnostic['files'][name] for name in (
        'test_predictions_vi.jsonl.gz', 'test_predictions_en.jsonl.gz', 'test_predictions_zh.jsonl.gz', 'report.json')}
    if not all(v.get('readback_verified') for v in tests.values()):
        raise RuntimeError('C75 test traces have not been verified on Drive')
    selected.update(tests)
    TARGET.mkdir(parents=True, exist_ok=True)
    required_remaining = sum(max(0, meta['bytes'] - ((TARGET / (name + '.partial')).stat().st_size
        if (TARGET / (name + '.partial')).exists() else 0)) for name, meta in selected.items()
        if not (TARGET / name).exists())
    if shutil.disk_usage(TARGET).free - required_remaining < RESERVE:
        raise RuntimeError('Insufficient disk space to preserve 10 GiB reserve')
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(transfer, [(name, meta, test_drive if name in tests else drive) for name, meta in selected.items()]))
    for lang in ('vi', 'en', 'zh'):
        receipt = selected[f'models/{lang}/model.safetensors'].get('source_receipt')
        if receipt:
            p = TARGET / 'models' / lang / 'best_report.json'
            if p.exists() and json.loads(p.read_text(encoding='utf-8')) != receipt:
                raise RuntimeError(f'Existing receipt differs: {p}')
            p.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    compact_root = ROOT / 'models/turn_v2_distilled_qat4_20261024'
    compact = json.loads((compact_root / 'model_manifest.json').read_text(encoding='utf-8'))
    compact_out = TARGET / 'models/vi_int4'
    compact_out.mkdir(parents=True, exist_ok=True)
    for name in ('model.onnx', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'turn_detector_config.json'):
        src, dest = compact_root / 'onnx' / name, compact_out / name
        spec = compact['files'][name]
        if src.stat().st_size != spec['bytes'] or sha_file(src) != spec['sha256']:
            raise RuntimeError(f'INT4 source integrity mismatch: {name}')
        if dest.exists() and sha_file(dest) != spec['sha256']:
            raise RuntimeError(f'Existing INT4 file differs: {name}')
        if not dest.exists():
            shutil.copyfile(src, dest)
        if sha_file(dest) != spec['sha256']:
            raise RuntimeError(f'INT4 copy integrity mismatch: {name}')
    (compact_out / 'model_manifest.json').write_text(json.dumps(compact, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (TARGET / 'download_receipt.json').write_text(json.dumps({'verified': True, 'files': list(selected),
        'manifest_captured_at_utc': manifest['captured_at_utc'], 'data_cycle': 84,
        'bytes': sum(v['bytes'] for v in selected.values())}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('COMPLETE: train/test data, three models and Vietnamese INT4 verified', flush=True)


if __name__ == '__main__':
    main()
