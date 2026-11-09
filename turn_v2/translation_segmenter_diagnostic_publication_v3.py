"""Publish reviewed metadata onto an already verified diagnostic manifest.

CPU-only; no model/data download, inference, threshold selection or bulk rehash.
The earlier publication v2 failed before import. Its log remains untouched.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback

ROOT = Path('/content/turn_detect_v2')
JOB = ROOT / 'outputs/translation_segmenter_v15d_20261026/diagnostic_test_c75_v1'
CONTROL = JOB / 'review_publication_v3'
REPORT_SHA = 'fe4555399de148fe34c2e96f7d95d4edb2aa453caa618f0d69a333ebbb238650'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def atomic(path, raw):
    temp = path.with_name(path.name + '.publication_v3.tmp')
    with temp.open('wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if temp.read_bytes() != raw:
        raise RuntimeError('Readback mismatch: ' + str(path))
    temp.replace(path)


def write_json(path, value):
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def state(stage, **details):
    write_json(CONTROL / 'status.json', {'at_utc': datetime.now(timezone.utc).isoformat(), 'stage': stage, **details})


def run():
    import fcntl
    with (CONTROL / 'LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (CONTROL / 'completed_receipt.json').exists():
            raise RuntimeError('Already published; preserve committed evidence')
        assert read(JOB / 'status.json')['stage'] == 'COMPLETE'
        report_raw = (JOB / 'report.json').read_bytes()
        assert digest(report_raw) == REPORT_SHA
        report = json.loads(report_raw)
        assert report['complete'] and not report['test_used_for_selection']
        config = read(JOB.parent / 'drive_backup_v1/config.json')
        drive_root = Path(config['drive_path'])
        assert os.path.ismount('/content/drive')
        assert read(drive_root / 'destination_identity.json') == config['identity']
        destination = drive_root / JOB.name
        committed_raw = (destination / 'latest_manifest.json').read_bytes()
        manifest = json.loads(committed_raw)
        assert manifest == read(JOB / 'drive_manifest.json')
        assert manifest['complete'] and manifest['destination_identity'] == config['identity']
        entries = manifest['files']
        assert entries['report.json']['sha256'] == REPORT_SHA
        for lang, item in report['languages'].items():
            for logical, checksum in ((f'test_predictions_{lang}.jsonl.gz', item['full_trace_sha256']),
                                      (f'frozen/{lang}/model.safetensors', item['weights_sha256'])):
                assert entries[logical]['sha256'] == checksum and entries[logical]['readback_verified']
        request = read(CONTROL / 'request.json')
        assert request['raw_report_sha256'] == REPORT_SHA
        published = {}
        for logical, expected in request['files_sha256'].items():
            path = JOB / logical
            assert path.resolve().is_relative_to(JOB.resolve())
            raw = path.read_bytes()
            assert digest(raw) == expected and len(raw) <= 5 * 1024**2
            state('PUBLISHING_REVIEWED_METADATA', logical_path=logical, bytes=len(raw))
            target = destination / 'objects' / (expected + '.bin')
            if not target.exists():
                atomic(target, raw)
            assert target.read_bytes() == raw
            entry = {'sha256': expected, 'bytes': len(raw), 'object': str(target.relative_to(destination)), 'readback_verified': True}
            entries[logical] = entry
            published[logical] = entry
        assert (JOB / 'report.json').read_bytes() == report_raw
        assert (destination / 'latest_manifest.json').read_bytes() == committed_raw, 'Concurrent manifest change; preserve both versions'
        manifest.update(at_utc=datetime.now(timezone.utc).isoformat(), publisher='reviewed_metadata_overlay_v3',
                        metadata_publication={'inherited_bulk_objects_verified_in_prior_pass': True,
                                              'bulk_objects_rehashed_again': False, 'new_metadata_readback_verified': True,
                                              'previous_manifest_sha256': digest(committed_raw), 'files': list(published)})
        atomic(destination / 'previous_manifest.json', committed_raw)
        write_json(destination / 'latest_manifest.json', manifest)
        assert read(destination / 'latest_manifest.json') == manifest
        write_json(JOB / 'drive_manifest.json', manifest)
        receipt = {'at_utc': datetime.now(timezone.utc).isoformat(), 'stage': 'COMPLETE', 'readback_verified': True,
                   'raw_report_sha256': REPORT_SHA, 'drive_manifest_sha256': digest((destination / 'latest_manifest.json').read_bytes()),
                   'published_metadata': published, 'model_inference_or_test_rescore': False,
                   'campaign_process_changed': False, 'large_artifacts_inherited_from_verified_manifest': True}
        write_json(CONTROL / 'completed_receipt.json', receipt)
        state('COMPLETE', readback_verified=True, drive_manifest_sha256=receipt['drive_manifest_sha256'], files=len(published))


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        write_json(CONTROL / 'error.json', {'at_utc': datetime.now(timezone.utc).isoformat(), 'error': repr(exc), 'traceback': traceback.format_exc()})
        state('FAILED_PRESERVED', error=repr(exc))
        raise
