"""Freeze the newer verified Drive checkpoint objects on the current Colab VM."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

ROOT = Path('/content/turn_detect_v2/migration_snapshot_v2')
DRIVE = Path('/content/drive/MyDrive/turn_detect_backups/translation_segmenter_v15d_20261026')
RESERVE = 25 * 1024**3


def write(name, value):
    p = ROOT / name
    temp = p.with_name(p.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, p)


def sha(path):
    d = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            d.update(block)
    return d.hexdigest()


def main():
    m = json.loads((ROOT / 'drive_manifest.json').read_text())
    assert m['pass_complete'] and m['full_recovery_available']
    assert os.path.ismount('/content/drive') and json.loads((DRIVE / 'destination_identity.json').read_text()) == m['destination_identity']
    out = {}
    for lang in ('vi', 'zh'):
        e = m['files'][f'teachers/{lang}/latest_resume.pt']
        src = DRIVE / e['object']
        assert src.parent == DRIVE / 'objects' and src.name == e['sha256'] + '.bin'
        target = ROOT / f"{lang}_{e['sha256']}.pt"
        partial = target.with_name(target.name + '.partial')
        if target.exists():
            assert target.stat().st_size == e['bytes'] and sha(target) == e['sha256']
        else:
            assert not partial.exists(), f'Preserve existing partial: {partial}'
            assert src.stat().st_size == e['bytes'] and shutil.disk_usage('/content').free - e['bytes'] >= RESERVE
            write('status.json', {'stage': 'COPYING_CHECKPOINT', 'language': lang, 'bytes': e['bytes']})
            digest = hashlib.sha256()
            with src.open('rb') as inp, partial.open('xb') as dst:
                while True:
                    block = inp.read(8 * 1024**2)
                    if not block:
                        break
                    if shutil.disk_usage('/content').free - len(block) < RESERVE:
                        raise RuntimeError('Colab disk reserve reached')
                    dst.write(block)
                    digest.update(block)
                dst.flush()
                os.fsync(dst.fileno())
            assert partial.stat().st_size == e['bytes'] and digest.hexdigest() == e['sha256']
            os.replace(partial, target)
        out[lang] = {'path': str(target), 'bytes': e['bytes'], 'sha256': e['sha256'], 'source_receipt': e['source_receipt']}
    write('receipt.json', {'complete': True, 'files': out})
    write('status.json', {'stage': 'COMPLETE', 'languages': ['vi', 'zh']})
    print('Frozen updated VI and ZH checkpoints verified', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write('error.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        write('status.json', {'stage': 'FAILED_PRESERVED', 'error': repr(exc)})
        raise
