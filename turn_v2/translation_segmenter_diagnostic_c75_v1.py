"""Frozen cycle75 prose diagnostic. Its errors may inform TRAIN-only crawling.

The previous cycle45 holdout is never rescored. New test pages are assigned
before scoring to diagnostic (80%) or an unscored reserve (20%). All inference
is on Colab, yielding to the campaign. Large artifacts go directly to Drive.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import traceback
import zipfile

from . import translation_segmenter_v15d_test_v2 as ev

ROOT = ev.ROOT
JOB = ROOT / 'diagnostic_test_c75_v1'
CYCLE = 75
OLD_CYCLE = 45
MIN_FREE = 25 * 1024**3


def now():
    return datetime.now(timezone.utc).isoformat()


def reserve_page(source_id):
    return int(hashlib.sha256(('c75-unscored-reserve-v1:' + source_id).encode()).hexdigest()[:16], 16) % 5 == 0


def disk_guard():
    if shutil.disk_usage(ROOT).free < MIN_FREE:
        raise RuntimeError('Colab free space is below the 25 GiB reserve; artifacts preserved')


def state(stage, **fields):
    disk_guard()
    ev.status(stage, **fields)


def await_reports():
    reports = {}
    while len(reports) < 3:
        for lang in ev.LANGUAGES:
            path = ROOT / 'cycles' / f'{CYCLE:06d}' / (lang + '_dev.json')
            if path.exists():
                value = ev.read_json(path)
                if value.get('cycle') == CYCLE and value.get('stage') == 'trained':
                    reports[lang] = value
        if len(reports) == 3:
            break
        if not ev.process_matches(ev.read_json(ROOT / 'launcher.json')):
            raise RuntimeError('Campaign parent identity unavailable while awaiting cycle75 dev reports')
        state('WAITING_FOR_C75_DEV_THRESHOLDS', ready_languages=sorted(reports))
        time.sleep(10)
    return reports


def pin_inputs():
    """Pin atomic checkpoint inodes now; do not wait for the next crawl cycle."""
    if shutil.disk_usage(ROOT).free < MIN_FREE + 25 * 1024**3:
        raise RuntimeError('Insufficient headroom for frozen data and three model weights')
    frozen = JOB / 'frozen'
    frozen.mkdir(exist_ok=False)
    receipts = {}
    for lang in ev.LANGUAGES:
        folder = ROOT / 'teachers' / lang
        record = ev.read_json(folder / 'resume_receipt.json')
        if record.get('cycle') != CYCLE or record.get('cycle_step') != 100:
            raise RuntimeError(f'Requested exact cycle75/100 checkpoint unavailable for {lang}; no substitute selected')
        source = folder / 'latest_resume.pt'
        before = source.stat()
        pinned = frozen / (lang + '_resume_pin.pt')
        os.link(source, pinned)
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError('Checkpoint publication changed during pinning: ' + lang)
        if record != ev.read_json(folder / 'resume_receipt.json'):
            raise RuntimeError('Checkpoint receipt changed during pinning: ' + lang)
        target = frozen / lang
        target.mkdir()
        for p in (folder / 'best').iterdir():
            if p.is_file() and p.name != 'model.safetensors':
                shutil.copy2(p, target / p.name)
        receipts[lang] = record
    state('FREEZING_SQLITE_BACKUP')
    source = sqlite3.connect('file:' + str((ROOT / 'corpus.sqlite').resolve()) + '?mode=ro', uri=True)
    destination = sqlite3.connect(frozen / 'test.sqlite')
    try:
        source.backup(destination, pages=16384, sleep=.05)
    finally:
        destination.close()
        source.close()
    ev.atomic_json(JOB / 'pinned_receipts.json', receipts)
    return receipts


def freeze_models(receipts, reports):
    import torch
    from safetensors.torch import save_file
    checkpoints = {}
    for lang in ev.LANGUAGES:
        state('EXTRACTING_PINNED_C75_WEIGHTS_CPU', language=lang)
        pinned = JOB / 'frozen' / (lang + '_resume_pin.pt')
        checksum = ev.file_sha(pinned)
        saved = torch.load(pinned, map_location='cpu', weights_only=False)
        for key in ('cycle', 'cycle_step', 'total_steps', 'training_fingerprint'):
            if saved.get(key) != receipts[lang].get(key):
                raise RuntimeError('Pinned checkpoint does not match its receipt: ' + lang + '/' + key)
        target = JOB / 'frozen' / lang / 'model.safetensors'
        save_file({k: v.contiguous() for k, v in saved['model'].items()}, str(target), metadata={'format': 'pt'})
        del saved
        gc.collect()
        report = reports[lang]
        ev.atomic_json(JOB / 'frozen' / (lang + '_dev_report.json'), report)
        checkpoints[lang] = {'checkpoint_cycle': CYCLE,
            'cycle_step': 100, 'total_steps': receipts[lang]['total_steps'],
            'source_resume_sha256': checksum, 'training_fingerprint': receipts[lang]['training_fingerprint'],
            'weights_sha256': ev.file_sha(target),
            'threshold_from_dev': float(report['threshold_selected_on_dev']),
            'files_sha256': {p.name: ev.file_sha(p) for p in sorted(target.parent.iterdir()) if p.is_file()}}
    db = sqlite3.connect('file:' + str((JOB / 'frozen/test.sqlite').resolve()) + '?mode=ro', uri=True)
    try:
        integrity = db.execute('PRAGMA integrity_check').fetchone()[0]
        if integrity != 'ok':
            raise RuntimeError('Frozen SQLite integrity check: ' + integrity)
        leakage = db.execute('SELECT COUNT(*) FROM (SELECT source_id FROM examples GROUP BY source_id HAVING COUNT(DISTINCT split)>1)').fetchone()[0]
        if leakage:
            raise RuntimeError('Frozen source groups cross dataset splits')
        counts = [dict(zip(('language', 'split', 'rows'), row)) for row in db.execute('SELECT language,split,COUNT(*) FROM examples GROUP BY 1,2')]
    finally:
        db.close()
    manifest = {'schema': 1, 'frozen_at_utc': now(), 'checkpoints': checkpoints,
        'database_sha256': ev.file_sha(JOB / 'frozen/test.sqlite'), 'database_integrity': integrity,
        'source_split_leakage_groups': leakage, 'split_counts': counts,
        'selection': 'test pages added after45 through75; deterministic source-page80/20 diagnostic/unscored reserve',
        'old_holdout_unchanged': True, 'test_used_for_threshold_or_checkpoint_selection': False,
        'diagnostic_errors_may_inform_future_training_data': True,
        'reserve_scored': False, 'scope': 'PUNCTUATION_DERIVED_PROSE_NOT_HUMAN_AUDIO_GOLD',
        'code_sha256': ev.file_sha(Path(__file__)),
        'shared_evaluator_sha256': ev.file_sha(Path(ev.__file__)),
        'model_encoder_sha256': ev.file_sha(Path('turn_v2/monolingual_data.py'))}
    ev.atomic_json(JOB / 'manifest.json', manifest)
    return manifest


class SelectedTestDB:
    def __init__(self, db, lang, selected):
        self.db, self.lang, self.selected = db, lang, selected

    def execute(self, sql, params=()):
        if "SELECT record_json FROM examples WHERE language=? AND split='test'" in sql:
            assert params == (self.lang,)
            class Rows:
                def fetchall(inner):
                    return [(raw,) for raw in self.selected]
            return Rows()
        return self.db.execute(sql, params)


def select_sources(db, lang):
    sources = {lang + ':' + str(p): {'cycle': c, 'sha': h}
               for p, c, h in db.execute("SELECT page_id,added_cycle,text_sha256 FROM sources WHERE language=? AND split='test'", (lang,))}
    diagnostic, excluded = [], []
    reserve_sources, old_sources = [], []
    for sid, meta in sources.items():
        if OLD_CYCLE < meta['cycle'] <= CYCLE:
            if reserve_page(sid):
                reserve_sources.append(sid)
        else:
            old_sources.append(sid)
    reserve_set, old_set = set(reserve_sources), set(old_sources)
    protected_hashes = {sources[sid]['sha'] for sid in reserve_set | old_set}
    counts = Counter()
    for (raw,) in db.execute("SELECT record_json FROM examples WHERE language=? AND split='test' ORDER BY example_id", (lang,)):
        row = json.loads(raw)
        sid = row['source_id']
        if sid in reserve_set:
            counts['unscored_reserve_rows'] += 1
            excluded.append(raw)
        elif sid in old_set:
            counts['old_test_rows_not_rescored'] += 1
            excluded.append(raw)
        elif sources[sid]['sha'] in protected_hashes:
            counts['diagnostic_rows_source_copy_of_old_or_reserve_removed'] += 1
        else:
            diagnostic.append(raw)
    if not diagnostic or not reserve_sources:
        raise RuntimeError('New diagnostic or unscored source reserve is empty: ' + lang)
    receipt = {**counts, 'diagnostic_rows_before_encoded_audit': len(diagnostic),
        'unscored_reserve_source_groups': len(reserve_sources),
        'unscored_reserve_source_ids_sha256': hashlib.sha256('\n'.join(sorted(reserve_sources)).encode()).hexdigest(),
        'unscored_reserve_source_ids': sorted(reserve_sources), 'reserve_scored': False}
    ev.atomic_json(JOB / (lang + '_partition.json'), receipt)
    return diagnostic, excluded, receipt


def diagnostic_audit(db, lang, rows, encoded, tokenizer, excluded):
    """Match actual truncated 256-token inputs; CPU work does not require GPU."""
    import numpy as np
    from .monolingual_data import encode_tail
    ids, lengths = encoded[:2]
    signatures = [ev.token_hash(ids[i, :int(lengths[i])]) for i in range(len(rows))]
    wanted, leaked, protected = set(signatures), set(), set()
    scanned = 0
    for (raw,) in db.execute("SELECT record_json FROM examples WHERE language=? AND split IN ('train','dev')", (lang,)):
        row = json.loads(raw)
        key = ev.token_hash(encode_tail(tokenizer, row['context'], row['text'], 256)['input_ids'])
        if key in wanted:
            leaked.add(key)
        scanned += 1
        if scanned % 10000 == 0:
            state('AUDITING_TRAIN_DEV_INPUT_OVERLAP_CPU', language=lang, scanned=scanned, overlap_keys=len(leaked))
    for i, raw in enumerate(excluded):
        row = json.loads(raw)
        key = ev.token_hash(encode_tail(tokenizer, row['context'], row['text'], 256)['input_ids'])
        if key in wanted:
            protected.add(key)
        if i % 10000 == 0:
            state('AUDITING_OLD_TEST_AND_UNSCORED_RESERVE_CPU', language=lang, scanned=i, overlap_keys=len(protected))
    source_rows = list(db.execute('SELECT page_id,text_sha256,split FROM sources WHERE language=?', (lang,)))
    train_hashes = {h for _, h, split in source_rows if split != 'test'}
    source_leaks = {lang + ':' + str(p) for p, h, split in source_rows if split == 'test' and h in train_hashes}
    clean = np.asarray([i for i, row in enumerate(rows) if signatures[i] not in leaked | protected and row['source_id'] not in source_leaks], dtype=np.int64)
    if not len(clean):
        raise RuntimeError('No diagnostic rows survive overlap filtering: ' + lang)
    return clean, {'non_test_rows_scanned': scanned, 'test_rows_before_audit': len(rows),
        'overlapping_encoded_train_dev_keys': len(leaked), 'overlapping_old_test_or_reserve_keys': len(protected),
        'test_source_groups_with_exact_non_test_text': len(source_leaks),
        'test_rows_after_audit': int(len(clean)), 'removed_rows': len(rows) - len(clean),
        'policy': 'exclude train/dev, old test and unscored reserve exact encoded inputs and copied source text',
        'near_duplicate_paraphrases_not_exhaustively_detected': True}


def error_summary(db, lang):
    from .common import normalize
    maps = {lang + ':' + str(p): key for p, key in db.execute('SELECT page_id,source_key FROM source_provenance WHERE language=?', (lang,))}
    groups = {name: defaultdict(Counter) for name in ('source', 'punctuation', 'lookahead', 'context_words', 'future_first_word')}
    input_labels = defaultdict(Counter)
    horizon_pairs = defaultdict(dict)
    errors, high_confidence = Counter(), Counter()
    with gzip.open(JOB / f'test_predictions_{lang}.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for raw in stream:
            row = json.loads(raw)
            if not row['included_in_clean_test']:
                continue
            correct = row['correct']
            context, future = normalize(row['context']), normalize(row['text'])
            word_count = len(context.split())
            bucket = '<=5' if word_count <= 5 else '6-15' if word_count <= 15 else '16-40' if word_count <= 40 else '>40'
            features = {'source': maps.get(row['source_id'], 'carryover'), 'punctuation': row.get('target_punctuation', ''),
                'lookahead': str(row['lookahead_words']), 'context_words': bucket,
                'future_first_word': future.split()[0] if future else ''}
            error_type = 'correct' if correct else 'false_EOT' if row['predicted'] == 'EOT' else 'missed_EOT'
            errors[error_type] += 1
            if not correct and abs(row['p_eot'] - row['threshold_from_dev']) >= .30:
                high_confidence[error_type] += 1
            for name, feature in features.items():
                groups[name][feature]['samples'] += 1
                groups[name][feature][error_type] += 1
            input_labels[context + '\0' + future][row['label']] += 1
            pair = (row['source_id'], row.get('paragraph_index'), row.get('sentence_index'), row.get('boundary_type'), row.get('cut_after_word'), row.get('target_punctuation'))
            horizon_pairs[pair][row['lookahead_words']] = correct
    conflicts = [c for c in input_labels.values() if len(c) > 1]
    paired = Counter()
    for value in horizon_pairs.values():
        if 1 in value and 2 in value:
            paired['both_correct' if value[1] and value[2] else 'lookahead2_rescues' if value[2] else 'lookahead2_hurts' if value[1] else 'both_wrong'] += 1
    output = {}
    for name, values in groups.items():
        records = []
        for feature, c in values.items():
            if name == 'future_first_word' and c['samples'] < 50:
                continue
            n = c['samples']; bad = n - c['correct']
            records.append({'feature': feature, **c, 'errors': bad, 'error_rate': bad / n})
        output[name] = sorted(records, key=lambda r: (-r['errors'], -r['samples']))[:100]
    return {'error_counts': dict(errors), 'high_margin_error_counts': dict(high_confidence),
        'slices': output, 'paired_lookahead': dict(paired),
        'normalized_visible_input_label_conflicts': {'keys': len(conflicts), 'rows': sum(sum(c.values()) for c in conflicts),
            'minimum_unavoidable_errors_with_identical_visible_text': sum(min(c.values()) for c in conflicts)},
        'interpretation': 'Slice associations and label collisions are evidence, not causal attribution. Original-source review is required for parser/label error claims.'}


def enhance_review(db):
    path = JOB / 'review_samples.jsonl'
    output = []
    for raw in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(raw)
        lang, page = row['source_id'].split(':', 1)
        value = db.execute('SELECT text FROM sources WHERE language=? AND page_id=?', (lang, page)).fetchone()
        if value:
            import re
            paragraphs = re.split(r'\n+', value[0])
            idx = row.get('paragraph_index', -1)
            row['original_source_paragraph_for_posthoc_label_audit'] = paragraphs[idx][:3000] if 0 <= idx < len(paragraphs) else ''
        output.append(json.dumps(row, ensure_ascii=False))
    path.write_text('\n'.join(output) + '\n', encoding='utf-8')


def drive_publish(complete=False):
    """Separate immutable job destination; never races campaign manifests/pruning."""
    config = ev.read_json(ROOT / 'drive_backup_v1/config.json')
    destination_root = Path(config['drive_path'])
    if not os.path.ismount('/content/drive') or ev.read_json(destination_root / 'destination_identity.json') != config['identity']:
        raise RuntimeError('Mounted Drive destination unavailable or identity mismatch')
    destination = destination_root / JOB.name
    destination.mkdir(exist_ok=True)
    entries = ev.read_json(JOB / 'drive_manifest.json').get('files', {}) if (JOB / 'drive_manifest.json').exists() else {}
    db_zip = JOB / 'frozen_database.zip'
    if not db_zip.exists():
        state('COMPRESSING_FROZEN_DATA_FOR_DIRECT_DRIVE_BACKUP')
        with zipfile.ZipFile(db_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.write(JOB / 'frozen/test.sqlite', 'test.sqlite')
    paths = [p for p in (JOB / 'frozen').rglob('*') if p.is_file() and p.suffix not in ('.sqlite', '.pt')]
    paths += [db_zip, Path(__file__), Path(ev.__file__)]
    paths += [p for p in JOB.iterdir() if p.is_file() and p.name not in ('drive_manifest.json', 'LOCK') and p.suffix in ('.json', '.jsonl', '.gz', '.html', '.md')]
    for source in paths:
        relative = str(source.relative_to(JOB)) if source.is_relative_to(JOB) else 'source/' + source.name
        checksum = ev.file_sha(source)
        if relative in entries and entries[relative]['sha256'] == checksum:
            target = destination / entries[relative]['object']
            if target.exists() and ev.file_sha(target) == checksum:
                continue
        objects = destination / 'objects'
        objects.mkdir(exist_ok=True)
        target = objects / (checksum + '.bin')
        state('BACKING_UP_DIAGNOSTIC_DIRECT_TO_DRIVE', logical_path=relative, bytes=source.stat().st_size)
        if not target.exists():
            temporary = target.with_suffix('.uploading')
            with source.open('rb') as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst, 8 * 1024**2)
                dst.flush(); os.fsync(dst.fileno())
            if ev.file_sha(temporary) != checksum:
                raise RuntimeError('Diagnostic Drive readback SHA mismatch: ' + relative)
            temporary.replace(target)
        if ev.file_sha(target) != checksum:
            raise RuntimeError('Diagnostic Drive object SHA mismatch: ' + relative)
        entries[relative] = {'sha256': checksum, 'bytes': source.stat().st_size, 'object': str(target.relative_to(destination)), 'readback_verified': True}
    manifest = {'at_utc': now(), 'complete': complete, 'destination_identity': config['identity'],
        'scope': 'diagnostic frozen weights/tokenizers/data/protocol, scored traces and reports; excludes redundant optimizer pin intermediates',
        'files': entries, 'reserve_scored': False}
    ev.atomic_json(JOB / 'drive_manifest.json', manifest)
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    temporary = destination / 'latest_manifest.json.tmp'
    with temporary.open('wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    if temporary.read_bytes() != raw:
        raise RuntimeError('Diagnostic Drive manifest readback mismatch')
    temporary.replace(destination / 'latest_manifest.json')


def run():
    import fcntl
    JOB.mkdir(exist_ok=True)
    ev.JOB = JOB
    with (JOB / 'LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (JOB / 'report.json').exists() or (JOB / 'frozen').exists():
            raise RuntimeError('Existing diagnostic state is immutable; inspect it rather than rescore')
        state('PINNING_CYCLE75_INPUTS')
        receipts = pin_inputs()
        reports = await_reports()
        manifest = freeze_models(receipts, reports)
        drive_publish()
        import torch
        import transformers.utils as hf_utils
        import transformers.utils.import_utils as hf_imports
        hf_utils.is_torchvision_available = hf_imports.is_torchvision_available = lambda: False
        hf_utils.is_librosa_available = hf_imports.is_librosa_available = lambda: False
        torch.set_num_threads(1)
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        report = {'created_at_utc': now(), 'manifest': manifest, 'complete': False,
            'scope': 'FROZEN_C75_NEW_SOURCE_PROSE_DIAGNOSTIC_TEST_NOT_HUMAN_AUDIO_GOLD',
            'test_used_for_selection': False, 'future_train_crawl_design_may_use_diagnostic_errors': True,
            'unscored_reserve_preserved': True, 'quality_target_certified': False, 'languages': {}}
        db = sqlite3.connect('file:' + str((JOB / 'frozen/test.sqlite').resolve()) + '?mode=ro', uri=True)
        try:
            for lang in ev.LANGUAGES:
                selected, excluded, partition = select_sources(db, lang)
                ev.audit_test = lambda handle, language, rows, encoded, tokenizer: diagnostic_audit(handle, language, rows, encoded, tokenizer, excluded)
                result = ev.score_language(SelectedTestDB(db, lang, selected), lang, manifest)
                result['partition'] = {k: v for k, v in partition.items() if k != 'unscored_reserve_source_ids'}
                result['error_analysis'] = error_summary(db, lang)
                report['languages'][lang] = result
                ev.atomic_json(JOB / 'partial_report.json', report)
                enhance_review(db)
                drive_publish()
                del selected, excluded
                gc.collect()
        finally:
            db.close()
        report.update(complete=True, completed_at_utc=now())
        ev.atomic_json(JOB / 'report.json', report)
        drive_publish(complete=True)
        state('COMPLETE', report_sha256=ev.file_sha(JOB / 'report.json'), reserve_scored=False,
              languages=list(report['languages']), drive_backup_complete=True)


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        JOB.mkdir(parents=True, exist_ok=True)
        ev.JOB = JOB
        ev.atomic_json(JOB / 'error.json', {'at_utc': now(), 'error': repr(exc), 'traceback': traceback.format_exc()})
        ev.status('FAILED_PRESERVED', error=repr(exc))
        raise
