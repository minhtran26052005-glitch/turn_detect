"""User-requested C75 diagnostic of a fresh streaming-branch checkpoint.

Threshold is frozen from the same checkpoint's streaming DEV.  Earlier C75
cycle84 overlap decisions are reused with input/trace/dataset identity checks;
new factory TRAIN/DEV visible-text substring collisions are audited separately.
No teacher, checkpoint, or operating point is chosen from these test results.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import traceback

from . import factory_stream_test_v1 as stream
from .factory_stream_data_v1 import sha
from .multilingual_student_diagnostic_v1 import metrics
from .common import normalize

PROJECT = Path('/home/edabk/hoangbpm/turn_detect')
CAMPAIGN = PROJECT/'outputs/multilingual_stream_v2_20261102'
JOB = PROJECT/'outputs/multilingual_stream_c75_test_20261102_v1'
PRIOR = Path('/mnt/hdd500/turn_detect_training/multilingual_distill_test_20261101_v1')
PRIOR_REPORT_SHA = '4b3cf677d762d2c0577fcd9523768d5b5453b60763c5e4951c7af608e3bac138'
CYCLE84 = PROJECT/'outputs/translation_segmenter_v15d_20261026/full_epoch_v1/current_dataset.json'
LANGS = ('vi', 'en', 'zh')
stream.JOB = JOB


def save(name, value):
    path = JOB/name
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def status(stage, **values):
    raw = Path(f'/proc/{os.getpid()}/stat').read_text()
    record = {'stage': stage, 'pid': os.getpid(), 'start_ticks': raw[raw.rfind(')')+2:].split()[19],
              'at_utc': datetime.now(timezone.utc).isoformat(), 'training_interrupted': False, **values}
    save('status.json', record)
    print(json.dumps(record), flush=True)


def freeze():
    assert shutil.disk_usage(JOB).free > 15*2**30
    assert not (JOB/'frozen').exists(), 'Checkpoint already pinned; preserve and resume that pin rather than repin'
    (JOB/'frozen').mkdir()
    for src, dst in [('recipe.json', 'source_recipe.json'), ('dataset/manifest.json', 'dataset_manifest.json'),
                     ('dataset/dev_streams.json', 'frozen_dev_streams.json')]:
        os.link(CAMPAIGN/src, JOB/dst)
    os.link(CAMPAIGN/'latest_student.pt', JOB/'frozen/student_resume_pin.pt')
    status('PINNED_LATEST_CHECKPOINT_CALIBRATING_SAME_CHECKPOINT_DEV')
    predict, threshold, receipt = stream.load_and_calibrate()
    # The reused function preserves full AdamW/RNG/sampler in the hardlink and
    # computes the exact current trainer's rolling VI/EN DEV operating point.
    receipt['streaming_branch_checkpoint'] = True
    receipt['source_spoken_v9_used'] = False
    save('checkpoint_receipt.json', receipt)
    return predict, threshold, receipt


def prepare_inputs():
    assert sha(PRIOR/'report.json') == PRIOR_REPORT_SHA, 'Prior audit/report identity changed'
    prior = json.loads((PRIOR/'report.json').read_text())
    old_manifest = json.loads((PRIOR/'input_manifest.json').read_text())
    cycle = json.loads(CYCLE84.read_text())
    assert cycle['complete'] and cycle['database_sha256'] == prior['checkpoint']['dataset_sha256']
    inputs = {}; previous = {}; files = {}
    for lang in LANGS:
        meta = old_manifest['test_files'][lang]
        path = Path(meta['path'])
        assert sha(path) == meta['sha256'] == prior['languages'][lang]['dataset_sha256']
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            inputs[lang] = [json.loads(line) for line in f]
        assert len(inputs[lang]) == meta['expected_rows']
        trace = PRIOR/f'predictions_{lang}.jsonl.gz'
        assert sha(trace) == prior['languages'][lang]['predictions_sha256']
        with gzip.open(trace, 'rt', encoding='utf-8') as f:
            previous[lang] = {r['example_id']: r for r in map(json.loads, f)}
        assert len(previous[lang]) == len(inputs[lang])
        for row in inputs[lang]:
            old = previous[lang][row['example_id']]
            assert all(old[k] == row[k] for k in ['context', 'text', 'label', 'language', 'lookahead_words', 'source_id'])
            assert row['included_in_clean_test'] and row['language'] == lang
            assert row['lookahead_words'] in (1, 2) and len(row['text'].split()) == row['lookahead_words']
            assert normalize(row['context']) == row['context'] and normalize(row['text']) == row['text']
        files[lang] = {**meta, 'path': str(path), 'prior_trace_sha256': sha(trace)}
    save('input_manifest.json', {'test_files': files, 'prior_report_sha256': PRIOR_REPORT_SHA,
        'cycle84_fingerprint': cycle['database_sha256'], 'prior_cycle84_audit_reused_not_rescanned': True,
        'prior_audit_scope': 'Exact teacher-token inputs and source IDs/source hashes, 8339693 accepted cycle84 train/dev rows in the verified prior report',
        'source_sha256': sha(Path(__file__)), 'previous_predictions_are_comparison_only': True,
        'scope': 'Previously exposed C75 fixed-reference-context punctuation diagnostic, not predicted-reset streaming or independent human/audio gold'})
    return inputs, previous, prior


def audit_factory_content(inputs):
    """Conservative exact normalized visible-text audit of the new corpus.

    Detect a C75 context+future span appearing contiguously anywhere inside a
    factory TRAIN/DEV paragraph, regardless of potential reset start.  This does
    not exhaust WordPiece-only collisions, novel synthetic cross-source splices,
    arbitrary truncation-equivalent histories, or semantic near duplicates.
    """
    assert sha(CAMPAIGN/'dataset/manifest.json') == sha(JOB/'dataset_manifest.json')
    manifest = json.loads((JOB/'dataset_manifest.json').read_text())
    database = CAMPAIGN/'dataset/paragraphs.sqlite'
    assert sha(database) == manifest['files']['paragraphs.sqlite']['sha256']
    lookup = defaultdict(list); sizes = set()
    for lang in ('vi', 'en'):
        for row in inputs[lang]:
            words = tuple((row['context']+' '+row['text']).split())
            key = words[-8:]; sizes.add(len(key))
            lookup[(lang, key)].append((row['example_id'], words))
    matches = defaultdict(set); scanned = Counter()
    db = sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True)
    sql = "SELECT language,split,record FROM paragraphs WHERE split IN ('train','dev') ORDER BY id"
    for lang, split, record in db.execute(sql):
        para = json.loads(record); words = tuple(para['words']); scanned[split] += 1
        for end in range(1, len(words)+1):
            for size in sizes:
                if end < size: continue
                for identity, pattern in lookup.get((lang, words[end-size:end]), ()):
                    if len(pattern) <= end and words[end-len(pattern):end] == pattern:
                        matches[identity].add('factory_'+split+'_normalized_context_plus_future_substring')
        if sum(scanned.values()) % 10000 == 0:
            status('AUDITING_NEW_FACTORY_TRAIN_DEV', paragraphs_scanned=dict(scanned), matching_examples=len(matches))
    db.close()
    result = {'complete': True, 'paragraphs_scanned': dict(scanned), 'matching_examples': len(matches),
              'dataset_manifest_sha256': sha(JOB/'dataset_manifest.json'),
              'paragraph_database_sha256': sha(database),
              'scope': 'Exact normalized context+future contiguous spans in source722 factory TRAIN/DEV paragraphs; matches conservatively excluded',
              'limits': 'WordPiece-only collisions, equivalent truncated histories, synthetic cross-source splices and paraphrases not exhaustively audited',
              'matches': {k: sorted(v) for k, v in matches.items()}}
    save('factory_overlap_audit.json', result)
    return matches, result


def predict_trace(predict, threshold, checkpoint, inputs, previous, prior, collisions, new_audit):
    report = {'complete': False, 'checkpoint': checkpoint, 'threshold_from_same_checkpoint_dev': threshold,
              'threshold_selected_on_test': False, 'checkpoint_selected_on_test': False,
              'quality_target_certified': False, 'source_spoken_v9_used': False, 'training_interrupted': False,
              'scope': 'Previously exposed C75 punctuation fixed-context diagnostic; not predicted-reset streaming or independent conversational/audio gold',
              'languages': {}, 'factory_overlap_audit': new_audit,
              'prior_cycle84_audit_reused_not_rescanned': True,
              'comparison_to_previous_c75': {'checkpoint_step': prior['checkpoint']['step'],
                  'old_threshold': prior['threshold_from_same_checkpoint_dev'], 'report_sha256': PRIOR_REPORT_SHA,
                  'scope': 'Identical retained input rows; prior predictions reused only for comparison; thresholds and training domains differ',
                  'by_language': {}}}
    for lang in LANGS:
        rows = inputs[lang]; counter = Counter(); raw = Counter(); before = Counter()
        by_k = {'1': Counter(), '2': Counter()}; audit = Counter(); review = []
        trace = JOB/f'predictions_{lang}.jsonl.gz'
        status('SCORING_C75', language=lang, rows_done=0, checkpoint_step=checkpoint['step'])
        with gzip.open(trace, 'wt', encoding='utf-8') as f:
            for start in range(0, len(rows), 64):
                part = rows[start:start+64]; probs = predict(part)
                for row, prob in zip(part, probs):
                    assert 0 <= prob <= 1
                    old = previous[lang][row['example_id']]
                    reasons = sorted(set(old['exclusion_reasons']) | collisions.get(row['example_id'], set()))
                    keep = old['included_after_cycle84_audit'] and not reasons
                    actual = row['label'] == 'EOT'; predicted = prob >= threshold
                    label = stream.kind(actual, predicted); raw[label] += 1
                    audit['retained_rows' if keep else 'excluded_rows'] += 1
                    for reason in reasons: audit[reason] += 1
                    if keep:
                        counter[label] += 1; by_k[str(row['lookahead_words'])][label] += 1
                        before[stream.kind(actual, old['predicted'] == 'EOT')] += 1
                    record = {k: row[k] for k in ['example_id', 'language', 'context', 'text', 'label', 'lookahead_words', 'source_id']}
                    record.update(p_eot=prob, predicted='EOT' if predicted else 'NEOT', correct=actual == predicted,
                                  included_after_audit=keep, exclusion_reasons=reasons,
                                  input_only_context_and_text=True)
                    f.write(json.dumps(record, ensure_ascii=False)+'\n')
                    if keep and (len(review) < 8 or (not record['correct'] and sum(not r['correct'] for r in review) < 8)):
                        review.append(record)
                end = start+len(part)
                if end % 3200 == 0:
                    status('SCORING_C75', language=lang, rows_done=end, rows_total=len(rows), excluded=audit['excluded_rows'])
        result = {'input_rows': len(rows), 'metrics_after_audit': stream.metric(counter),
                  'metrics_before_audit': stream.metric(raw), 'by_lookahead_words': {k: stream.metric(v) for k, v in by_k.items()},
                  'overlap_audit': dict(audit), 'dataset_sha256': json.loads((JOB/'input_manifest.json').read_text())['test_files'][lang]['sha256'],
                  'predictions_sha256': sha(trace), 'predictions_bytes': trace.stat().st_size}
        report['languages'][lang] = result
        old_metrics, new_metrics = stream.metric(before), result['metrics_after_audit']
        assert old_metrics['samples'] == new_metrics['samples']
        report['comparison_to_previous_c75']['by_language'][lang] = {'old': old_metrics, 'new': new_metrics,
            'accuracy_change_pp': 100*(new_metrics['accuracy']-old_metrics['accuracy']),
            'eot_f1_change_pp': 100*(new_metrics['eot_f1']-old_metrics['eot_f1'])}
        save(f'review_{lang}.json', review); save('partial_report.json', report)
        status('LANGUAGE_COMPLETE', language=lang, metrics=result['metrics_after_audit'])
    report.update(complete=True, completed_utc=datetime.now(timezone.utc).isoformat())
    save('report.json', report)
    status('COMPLETE', checkpoint_step=checkpoint['step'], branch_step=checkpoint['branch_step'], report_sha256=sha(JOB/'report.json'))


def run():
    import torch
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    utils.is_torchvision_available = imports.is_torchvision_available = lambda: False
    utils.is_librosa_available = imports.is_librosa_available = lambda: False
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.30); os.nice(5)
    assert not (JOB/'report.json').exists(), 'Never rerun this completed test'
    predict, threshold, checkpoint = freeze()
    inputs, previous, prior = prepare_inputs()
    status('AUDITING_NEW_FACTORY_TRAIN_DEV')
    collisions, audit = audit_factory_content(inputs)
    predict_trace(predict, threshold, checkpoint, inputs, previous, prior, collisions, audit)


if __name__ == '__main__':
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB/'EVALUATOR_LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: run()
        except Exception as error:
            save('error.json', {'error': repr(error), 'traceback': traceback.format_exc(),
                                'at_utc': datetime.now(timezone.utc).isoformat()})
            status('ERROR', error=repr(error)); raise
