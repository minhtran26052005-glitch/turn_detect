"""Build a preparation variant that reuses the already verified dataset hash."""
from pathlib import Path

source = Path('tools/prepare_remote_turn_detect_v2.py').read_text(encoding='utf-8')
old = """    extract_large(dataset_archive, 'dataset.sqlite', folder / 'dataset.sqlite', manifest['database_sha256'])
    print('Training SQLite extracted and SHA-256 verified', flush=True)
"""
new = """    dataset_path = folder / 'dataset.sqlite'
    previous_log = STAGING / 'preparation_worker_v2.log'
    assert 'Training SQLite extracted and SHA-256 verified' in previous_log.read_text(encoding='utf-8')
    assert dataset_path.is_file() and dataset_path.stat().st_mtime_ns <= previous_log.stat().st_mtime_ns
    with zipfile.ZipFile(dataset_archive) as z:
        assert dataset_path.stat().st_size == z.getinfo('dataset.sqlite').file_size
    print('Reusing dataset SHA-256 verification from prior preparation', flush=True)
"""
assert old in source
source = source.replace(old, new, 1)
old = """    for path in (folder / 'dataset.sqlite', raw_snapshot):
        db = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
        try:
            assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', str(path)
        finally:
            db.close()
    print('Training and source SQLite integrity checks passed', flush=True)
"""
new = """    with zipfile.ZipFile(STAGING / 'reports_and_logs.zip') as z:
        name = f'full_epoch_v1/datasets/{GENERATION}/verification_receipt.json'
        prior = json.loads(z.read(name))
    assert prior['complete'] and prior['database_sha256'] == manifest['database_sha256']
    assert prior['all_rows_checked_for_encoded_slots_token_bounds_hash_and_provenance']
    assert not prior['violations']
    assert prior['verifier_source_sha256'] == sha(PROJECT / 'turn_v2/translation_segmenter_full_verify_v2.py')
    print('Matched prior all-row verification to the verified dataset SHA; SQLite integrity runs separately', flush=True)
"""
assert old in source
source = source.replace(old, new, 1)
source = source.replace("'HDD_dataset_and_corpus': str(HDD), 'no_training_started': True}",
                        "'HDD_dataset_and_corpus': str(HDD), 'sqlite_integrity_check_pending': True, 'no_training_started': True}")
target = Path('tools/prepare_remote_turn_detect_v3.py')
target.write_text(source, encoding='utf-8')
compile(source, str(target), 'exec')
