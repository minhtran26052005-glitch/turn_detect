"""Record Windows hashes for SSH copies that the user asked to remove."""
import hashlib
import json
from pathlib import Path

local = Path('data_checkpoint')
pairs = [
    ('data/train/cycle84_dataset.zip', 'staging/cycle84_dataset.zip.partial'),
    ('data/train/source_snapshot.zip', 'staging/source_snapshot.zip.partial'),
]
for lang in ('vi', 'en', 'zh'):
    for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json'):
        pairs.append((f'models/{lang}/{name}', f'staging/models/{lang}/{name}'))
        if lang in ('en', 'zh'):
            pairs.append((f'models/{lang}/{name}',
                          f'outputs/translation_segmenter_v15d_20261026/teachers/{lang}/best/{name}'))

results = []
cache = {}
for source, destination in pairs:
    path = local / source
    if source not in cache:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Local source changed during hashing: {path}')
        cache[source] = {'bytes': before.st_size, 'sha256': digest.hexdigest()}
    results.append({'windows_source': str(path.resolve()), 'remote_relative': destination, **cache[source]})
    print('prepared', destination, flush=True)
out = Path('outputs/translation_segmenter_v15d_20261026/ssh_migration_v1/vi_only_cleanup_plan.json')
out.write_text(json.dumps({'files': results}, indent=2) + '\n', encoding='utf-8')
