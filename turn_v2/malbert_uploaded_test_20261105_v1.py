"""One newly authorized TEST of the uploaded-only v4 student; training stays active."""
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback
import zipfile

from . import malbert_eval_zip_v2 as engine

ROOT = engine.ROOT
MAIN = ROOT / 'outputs/multilingual_malbert_vi_en_uploaded_20261104_v4'
JOB = ROOT / 'outputs/malbert_uploaded_test_20261105_v1'
PREVIOUS = ROOT / 'outputs/malbert_eval_zip_20261104_v2'
engine.MAIN = MAIN
engine.JOB = JOB
sha, save, status, reserve = engine.sha, engine.save, engine.status, engine.reserve


def main_identity():
    launch = json.loads((MAIN / 'launcher.json').read_text())
    proc = Path('/proc') / str(launch['pid'])
    raw = (proc / 'stat').read_text()
    ticks = raw[raw.rfind(')') + 2:].split()[19]
    cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert ticks == str(launch['start_ticks']) and boot == launch['boot_id']
    assert 'turn_v2.multilingual_malbert_vi_en_uploaded_train_v4' in cmd
    return dict(pid=launch['pid'], start_ticks=ticks, boot_id=boot, cmdline=cmd)


def pin_checkpoint():
    """Pin the next temporal checkpoint at DEV, never choose by TEST score."""
    import torch
    frozen = JOB / 'frozen'
    frozen.mkdir(exist_ok=True)
    assert not (frozen / 'student_resume_pin.pt').exists()
    minimum = json.loads((JOB / 'launcher.json').read_text())['minimum_branch_step']
    status('WAITING_FOR_NEXT_CHECKPOINT_WITH_MATCHING_NEW_DEV', minimum_branch_step=minimum)
    while True:
        reserve(); main_identity()
        receipt = json.loads((MAIN / 'runs/finetune/resume_receipt.json').read_text())
        branch = receipt['branch_step']
        report_path = MAIN / 'runs/finetune' / ('dev_step_%08d.json' % branch)
        if branch < minimum or (branch % 10000 and not report_path.exists()):
            time.sleep(1); continue
        with (MAIN / 'runs/finetune/latest_resume.pt').open('rb') as source:
            x = torch.load('/proc/self/fd/' + str(source.fileno()), map_location='cpu', weights_only=False, mmap=True)
            branch = x['branch_step']
            report_path = MAIN / 'runs/finetune' / ('dev_step_%08d.json' % branch)
            if branch < minimum or (branch % 10000 and not report_path.exists()):
                del x; gc.collect(); continue
            assert x['parameters'] == 11941122 and len(x['optimizer']['state']) == 27
            assert {int(v['step']) for v in x['optimizer']['state'].values()} == {x['step']}
            assert x['fingerprint']['active_languages'] == ['vi', 'en'] and x['drive_replay_rows'] == 0
            assert sha(MAIN / 'dataset_manifest.json') == x['fingerprint']['dataset_manifest_sha256']
            assert sha(MAIN / 'train_streams.json.gz') == x['fingerprint']['train_streams_sha256']
            for name, expected in x['fingerprint']['code_sha256'].items():
                assert sha(ROOT / 'turn_v2' / name) == expected
            assert all(k in x for k in ['torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng', 'sampler_rng', 'order', 'cursor', 'stream_state'])
            assert len(x['order']) == 34602 and set(map(int, x['order'])) == set(range(34602))
            assert all(bool(torch.isfinite(t).all()) for t in x['model'].values())
            source.seek(0)
            with (frozen / 'student_resume_pin.pt').open('xb') as dst:
                shutil.copyfileobj(source, dst, 2**20); dst.flush(); os.fsync(dst.fileno())
            torch.save(x['model'], frozen / 'weights.pt')
            with (frozen / 'weights.pt').open('rb') as dst: os.fsync(dst.fileno())
        result = dict(step=x['step'], branch_step=branch, epoch=x['epoch'] + 1, parameters=x['parameters'],
                      optimizer_entries=27, all_rng_sampler_conversation_lanes_present=True,
                      uploaded_training_exposures=x['uploaded_rows_seen'], origin_exposures=x['uploaded_origin_exposures'],
                      training_fingerprint=x['fingerprint'], full_checkpoint_sha256=sha(frozen / 'student_resume_pin.pt'),
                      weights_sha256=sha(frozen / 'weights.pt'), storage_compute='FP32 mALBERT, not INT4',
                      checkpoint_selection='First temporal checkpoint with matching new DEV after user request; no test selection')
        del x; gc.collect()
        status('WAITING_FOR_FROZEN_CHECKPOINT_NEW_DEV_THRESHOLD', checkpoint_step=result['step'], branch_step=branch)
        while not report_path.exists():
            reserve(); main_identity(); time.sleep(1)
        dev = json.loads(report_path.read_text())
        assert dev['step'] == result['step'] and dev['branch_step'] == branch and dev['new_uploaded_only_dev']
        result.update(threshold=float(dev['dev']['threshold']), threshold_source='Same checkpoint, new original-DEV calibration families; locked before TEST',
                      matching_new_dev_sha256=sha(report_path))
        save('frozen/matching_new_dev.json', dev)
        coverage = {p.name: json.loads(p.read_text()) for p in sorted((MAIN / 'runs/finetune').glob('epoch_*_coverage.json'))}
        assert all(v['complete'] and v['scored_points'] == v['expected_eligible_points'] == 6169873 for v in coverage.values())
        result['completed_uploaded_epoch_receipts'] = coverage
        save('checkpoint_receipt.json', result)
        save('locked_protocol.json', dict(checkpoint=result, split='test only', lookahead_words=[1, 2],
             primary_label_policy=engine.POLICIES[0], secondary_diagnostic_policy=engine.POLICIES[1],
             source_input='source_raw only', reset='Predicted EOT only, no gold/speaker/language reset',
             final_flush_is_model_eot=False, test_used_for_selection=False, training_interrupted=False))
        return result


def prepare_test():
    # Source reconstruction is reused; no old model is run and no DEV is scored.
    all_streams, manifest = engine.prepare_streams()
    streams = [s for s in all_streams if s['split'] == 'test']
    previous_streams = json.load(gzip.open(PREVIOUS / 'prepared_streams.json.gz', 'rt', encoding='utf-8'))
    assert streams == [s for s in previous_streams if s['split'] == 'test']
    with gzip.open(JOB / 'prepared_streams.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(streams, f, ensure_ascii=False)
    manifest.update(streams=len(streams), split_scored='test only',
                    known_train_overlap_streams_included=sum(s['known_train_source_overlap'] for s in streams),
                    prepared_streams_sha256=sha(JOB / 'prepared_streams.json.gz'),
                    previous_verified_source_sha256=sha(PREVIOUS / 'prepared_streams.json.gz'))
    save('input_manifest.json', manifest)
    # Independent raw-word/label accounting for every included original TEST row.
    from .checked_dialogues_prepare_v1 import punctuation
    from .common import normalize
    sources = {}
    with zipfile.ZipFile(JOB / 'inputs/eval.zip') as z:
        for name in z.namelist():
            if name.endswith('/labels_checked.jsonl'):
                domain = '/'.join(name.split('/')[1:-1])
                sources[domain] = {i: json.loads(line) for i, line in enumerate(z.open(name)) if line.strip()}
    seen = Counter(); words = 0; rows = 0
    for stream in streams:
        assert stream['split'] == 'test'
        eots, unknown, joined = [], [], []
        for segment in stream['segments']:
            source = [sources[stream['domain']][i] for i in segment['source_rows']]
            assert all(r['split'] == 'test' and r['src_lang'] == segment['language'] for r in source)
            text = ' '.join(r['source_raw'].strip() for r in source)
            assert text == segment['display']
            w, e, u = punctuation(text); offset = len(joined)
            assert w == stream['words'][segment['start']:segment['end']] and segment['start'] == offset
            joined.extend(w); eots.extend(offset+i for i in e); unknown.extend(offset+i for i in u)
            for i in segment['source_rows']: seen[(stream['domain'], i)] += 1
            words += len(w); rows += len(source)
        assert joined == stream['words'] and eots == stream['eots'] and unknown == stream['unknown']
    for domain, source in sources.items():
        for i, r in source.items():
            eligible = r['split'] == 'test' and r.get('src_lang') in ('vi', 'en') and isinstance(r.get('source_raw'), str) and bool(normalize(r['source_raw']))
            assert seen[(domain, i)] == int(eligible), (domain, i)
    save('input_source_order_audit.json', dict(complete=True, source_rows=rows, source_words=words, streams=len(streams),
         every_nonempty_original_test_source_accounted_once=True, source_structures_match_previous_verified_test=True,
         raw_words_and_punctuation_labels_verified=True, prepared_streams_sha256=manifest['prepared_streams_sha256']))
    return streams, manifest


def verify_numerical(streams, checkpoint, report):
    import torch
    from transformers import AlbertConfig, AutoTokenizer
    from .malbert_turn_student_v1 import MalbertTurnStudent
    from .uploaded_vi_en_evaluation_v4 import predictor
    selected = {}; seen = Counter(); overlaps = set(); low, high = 1., 0.
    with gzip.open(JOB / 'predictions.jsonl.gz', 'rt') as trace:
        for line in trace:
            r = json.loads(line); p = r['p_eot']
            assert math.isfinite(p) and 0 <= p <= 1 and r['predicted_eot'] == (p >= checkpoint['threshold'])
            seen[r['lookahead_words']] += 1; low = min(low, p); high = max(high, p)
            s = streams[r['stream']]; segment = engine.segment_for(s, r['boundary'], r['lookahead_words'])
            bucket = 'short' if r['input_length'] <= 32 else 'medium' if r['input_length'] <= 128 else 'long'
            selected.setdefault((s['domain'], segment['language'], r['lookahead_words'], bucket), r)
            if s['known_train_source_overlap']: overlaps.add(r['stream'])
    assert sum(seen.values()) == report['inferences']
    model = MalbertTurnStudent(AlbertConfig.from_pretrained(MAIN / 'pretrained', local_files_only=True, token=False), sep_token_id=3)
    model.load_state_dict(torch.load(JOB / 'frozen/weights.pt', map_location='cpu', weights_only=True), strict=True)
    model.cuda().eval()
    tok = AutoTokenizer.from_pretrained(MAIN / 'tokenizer', local_files_only=True, token=False)
    predict = predictor(model, tok); delta = 0.; selected = list(selected.values())
    for start in range(0, len(selected), 16):
        part = selected[start:start+16]
        rows = [dict(context=' '.join(streams[r['stream']]['words'][r['context_start']:r['boundary']]),
                     text=' '.join(streams[r['stream']]['words'][r['boundary']:r['future_end']])) for r in part]
        probs, _ = predict(rows)
        for r, p in zip(part, probs):
            delta = max(delta, abs(p-r['p_eot']))
            assert (p >= checkpoint['threshold']) == r['predicted_eot']
    assert delta < 1e-4
    del model; gc.collect(); torch.cuda.empty_cache()
    return dict(complete=True, all_probabilities_finite_and_in_range=True, probability_min_max=[low, high],
                inferences_by_lookahead=dict(seen), known_legacy_overlap_streams_scored=len(overlaps),
                canonical_v4_predictor_parity_cases=len(selected), max_probability_delta=delta,
                all_canonical_decisions_identical=True, verification_only_no_selection=True)


def run():
    import torch
    import transformers.utils as u
    import transformers.utils.import_utils as im
    u.is_torchvision_available = im.is_torchvision_available = lambda: False
    u.is_librosa_available = im.is_librosa_available = lambda: False
    torch.set_num_threads(2)
    assert not (JOB / 'report.json').exists(), 'Never rerun this completed test'
    reserve(); identity_before = main_identity(); before = json.loads((MAIN / 'status.json').read_text())
    manifest_sha = sha(MAIN / 'dataset_manifest.json')
    streams, manifest = prepare_test()
    checkpoint = pin_checkpoint()
    status('EVALUATING_UPLOADED_V4_TEST', checkpoint_step=checkpoint['step'], threshold=checkpoint['threshold'])
    report = engine.evaluate(streams, checkpoint)
    status('AUDITING_EVERY_TEST_INFERENCE', inferences=report['inferences'])
    proof = engine.verify_traces(streams, checkpoint, report)
    numerical = verify_numerical(streams, checkpoint, report)
    assert sha(MAIN / 'dataset_manifest.json') == manifest_sha
    previous = json.loads((PREVIOUS / 'report.json').read_text())
    keys = ['domain', 'split', 'language', 'lookahead_words', 'label_policy', 'cohort']
    old = {tuple(r[k] for k in keys): r for r in previous['metric_rows'] if r['split'] == 'test'}
    for r in report['metric_rows']:
        prior = old[tuple(r[k] for k in keys)]
        assert r['samples'] == prior['samples'] and r['actual_eot'] == prior['actual_eot']
    comparison = dict(old_checkpoint=previous['checkpoint']['step'], new_checkpoint=checkpoint['step'],
         old_threshold=previous['checkpoint']['threshold'], new_threshold=checkpoint['threshold'],
         old_report_sha256=sha(PREVIOUS / 'report.json'), old_predictions_reused_not_rescored=True,
         same_test_point_population=True, comparison_includes_training_and_new_dev_calibration_changes=True,
         rows=[dict(previous=old[tuple(r[k] for k in keys)], current=r) for r in report['metric_rows']
               if r['domain'] == 'ALL' and r['lookahead_words'] == 2 and r['label_policy'] == engine.POLICIES[0]])
    save('comparison_with_previous.json', comparison)
    report.update(complete=True, checkpoint=checkpoint, input_manifest=manifest, causal_audit=proof,
         label_policies=list(engine.POLICIES), primary_label_policy=engine.POLICIES[0], split_scored='test only',
         main_dataset_sha256=manifest_sha, main_training_dataset_unchanged=True, test_used_for_selection=False,
         main_before=before, main_after=json.loads((MAIN / 'status.json').read_text()),
         main_identity_before=identity_before, main_identity_after=main_identity(),
         evaluator_source_sha256=sha(__file__), engine_source_sha256=sha(ROOT / 'turn_v2/malbert_eval_zip_v2.py'),
         model_scope='11.94M FP32, punctuation silver, not human/audio turn gold', backed_up_to_drive=False)
    save('report.json', report)
    numerical.update(report_sha256=sha(JOB / 'report.json'))
    save('independent_completion_verification.json', numerical)
    save('completion_verification.json', dict(complete=True, report_sha256=sha(JOB / 'report.json'), **proof))
    status('COMPLETE', checkpoint_step=checkpoint['step'], inferences=report['inferences'], report_sha256=sha(JOB / 'report.json'))


if __name__ == '__main__':
    JOB.mkdir(parents=True, exist_ok=True); os.nice(10)
    with (JOB / 'EVALUATOR_LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: run()
        except Exception as e:
            save('error.json', dict(error=repr(e), traceback=traceback.format_exc(), at_utc=datetime.now(timezone.utc).isoformat()))
            status('ERROR', error=repr(e)); raise
