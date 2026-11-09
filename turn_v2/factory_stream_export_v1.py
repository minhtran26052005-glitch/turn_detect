"""Freeze and package every committed generation shard, without stopping Qwen."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import traceback

DRIVE = Path('/content/drive/MyDrive/turn_detect_backups/translation_segmenter_v15d_20261026')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, obj):
    data = (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode()
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def run(output):
    os.nice(5)
    output.mkdir(parents=True, exist_ok=True)
    status = lambda stage, **kw: atomic(output/'status.json', {'stage': stage, 'pid': os.getpid(), **kw})
    try:
        assert os.path.ismount('/content/drive') and shutil.disk_usage('/content').free >= 25*2**30
        identity = json.loads((DRIVE/'destination_identity.json').read_bytes())
        assert identity['nonce'] == '9e618b0f2db442a6a9ee3ee18f1d7143' and identity['intended_path'] == str(DRIVE)
        campaign = DRIVE/'factory_dialogue_campaign_v6'
        raw_state = (campaign/'state.json').read_bytes(); snapshot = json.loads(raw_state)
        cutoff = snapshot['commit_index']; commits = []; previous = None; totals = Counter()
        status('READING_MANIFESTS', frozen_commit_index=cutoff)
        for index in range(1, cutoff+1):
            raw = (campaign/'commits'/f'{index:09d}.json').read_bytes(); c = json.loads(raw)
            assert c['index'] == index and c['previous_commit_sha256'] == previous and c['object_readback_verified']
            rel = PurePosixPath(c['object']); assert not rel.is_absolute() and '..' not in rel.parts
            previous = digest(raw); totals[c['language']] += c['accepted']; commits.append((c, raw))
            if index % 25 == 0: status('READING_MANIFESTS', current=index, total=cutoff)
        assert previous == snapshot['last_commit_sha256'] and dict(totals) == snapshot['total_counts']
        manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'frozen_commit_index': cutoff,
                    'frozen_commit_sha256': previous, 'source_state': snapshot, 'drive_identity': identity,
                    'files': {}, 'source_read_only': True, 'generation_continues': True,
                    'new_training_authorized_by_user_20261102': True,
                    'training_scope': 'Only accepted dialogues after structural checks and source-group splits; raw UNREVIEWED flags remain unchanged.'}
        target = output/'factory_committed_snapshot.tar'; assert not target.exists()
        partial = target.with_suffix('.tar.partial')

        def fetch(entry):
            c, raw = entry; data = (campaign/c['object']).read_bytes()
            assert len(data) == c['bytes'] and digest(data) == c['object_sha256']
            return c, raw, data

        with partial.open('xb') as stream, tarfile.open(fileobj=stream, mode='w|') as archive:
            def add(name, data):
                info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o600
                archive.addfile(info, io.BytesIO(data)); manifest['files'][name] = {'bytes': len(data), 'sha256': digest(data)}
            add('metadata/destination_identity.json', (DRIVE/'destination_identity.json').read_bytes())
            add('metadata/config.json', (campaign/'config.json').read_bytes())
            add('metadata/state_snapshot.json', raw_state)
            for path in sorted((campaign/'source_v8').glob('*.py')):
                add('source_v8/'+path.name, path.read_bytes())
            with ThreadPoolExecutor(max_workers=4) as pool:
                for ordinal, (c, raw, data) in enumerate(pool.map(fetch, commits), 1):
                    add(f"commits/{c['index']:09d}.json", raw); add(c['object'], data)
                    if ordinal % 10 == 0: status('PACKING_VERIFIED_SHARDS', current=ordinal, total=cutoff)
            encoded = (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode()
            info = tarfile.TarInfo('export_manifest.json'); info.size = len(encoded); info.mode = 0o600
            archive.addfile(info, io.BytesIO(encoded))
        with partial.open('rb') as f: os.fsync(f.fileno())
        h = hashlib.sha256()
        with partial.open('rb') as f:
            for block in iter(lambda:f.read(4*1024**2), b''): h.update(block)
        partial.replace(target)
        receipt = {'stage': 'COMPLETE', 'archive': str(target), 'bytes': target.stat().st_size,
                   'sha256': h.hexdigest(), 'manifest_sha256': digest(encoded),
                   'frozen_commit_index': cutoff, 'accepted_dialogues': dict(totals),
                   'files': len(manifest['files']), 'pid': os.getpid(), 'generation_interrupted': False}
        atomic(output/'export_receipt.json', receipt); atomic(output/'status.json', receipt)
    except Exception as error:
        atomic(output/'error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
        status('ERROR', error=repr(error)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
