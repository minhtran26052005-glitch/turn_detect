"""Fresh user-requested checkpoint on the fixed 200-paragraph diagnostic.

Reuse the established inference protocol without changing the completed v1
job. Independently verify references, confusion counts and frozen artifacts.
"""
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gzip
import json
import os
from pathlib import Path
import traceback

from . import factory_stream_test_v1 as core
from .factory_stream_data_v1 import sha

JOB = core.PROJECT / 'outputs/multilingual_stream_test_20261103_v1'
PRIOR = core.PROJECT / 'outputs/multilingual_stream_test_20261102_v1'
PRIOR_SHA = '61655f640a634115247af5dcb0a520a2a9dea77d2d3382530e8a9320305da424'
core.JOB = JOB


def status(stage, **values):
    raw = Path(f'/proc/{os.getpid()}/stat').read_text()
    record = {'stage': stage, 'pid': os.getpid(),
              'start_ticks': raw[raw.rfind(')')+2:].split()[19],
              'at_utc': datetime.now(timezone.utc).isoformat(),
              'training_interrupted': False, **values}
    core.save('status.json', record)
    print(json.dumps(record), flush=True)


def verify(streams, report):
    lookup = {s['key']: s for s in streams}
    starts, last = {}, {}
    counts = {m: {str(k): {l: Counter() for l in ('vi', 'en')}
                  for k in (1, 2)} for m in ('monolingual', 'mixed')}
    threshold = report['threshold_from_same_checkpoint_dev']
    total = 0
    with gzip.open(JOB/'predictions.jsonl.gz', 'rt', encoding='utf-8') as f:
        for r in map(json.loads, f):
            stream = lookup[r['stream_key']]
            k, b = r['lookahead_words'], r['boundary_word_index']
            key = (r['stream_key'], k)
            start = starts.get(key, 0)
            assert b == last.get(key, 0)+1 and b+k <= len(stream['words'])
            assert r['buffer_start_word_index'] == start
            assert r['context'] == ' '.join(stream['words'][start:b])
            assert r['text'] == ' '.join(stream['words'][b:b+k])
            assert len(r['text'].split()) == k
            predicted = r['p_eot'] >= threshold
            assert r['predicted'] == ('EOT' if predicted else 'NEOT')
            reference = None if b in stream['unknown'] else 'EOT' if b in stream['eots'] else 'NEOT'
            assert (r['reference']['label'] if r['reference'] else None) == reference
            if reference:
                label = ('tp' if predicted else 'fn') if reference == 'EOT' else ('fp' if predicted else 'tn')
                counts[stream['mode']][str(k)][stream['language_by_word'][b-1]][label] += 1
            if predicted:
                starts[key] = b
            last[key] = b
            total += 1
    for stream in streams:
        for k in (1, 2):
            assert last[(stream['key'], k)] == len(stream['words'])-k
    for mode, ks in counts.items():
        for k, languages in ks.items():
            for language, c in languages.items():
                actual = report['metrics_every_eligible_word_gap'][mode][k][language]
                assert all(actual[name] == c[name] for name in ('tp', 'tn', 'fp', 'fn'))
                assert actual['samples'] == sum(c.values())
    cases = json.loads((JOB/'cases.json').read_text())
    assert len(cases) == 100
    for case in cases:
        for result in case['lookahead'].values():
            assert ' '.join(s['text'] for s in result['segments']).split() == case['stream_input'].split()
    assert total == report['inferences'] == report['causal_audit']['all_inferences_audited']
    checkpoint = report['checkpoint']
    assert checkpoint['optimizer_entries'] == 104
    assert sha(JOB/'frozen/student_resume_pin.pt') == checkpoint['source_checkpoint_sha256']
    assert sha(JOB/'frozen/model.safetensors') == checkpoint['weights_sha256']
    assert sha(JOB/'predictions.jsonl.gz') == report['predictions_sha256']
    return {'independent_inferences_audited': total, 'source_references_and_confusion_counts_verified': True,
            'causal_buffers_future_words_and_predicted_resets_verified': True,
            'all_100_pairs_word_conservation_verified': True,
            'frozen_weights_and_full_optimizer_rng_pin_sha_verified': True}


def run():
    import torch
    assert not (JOB/'report.json').exists(), 'Preserve completed diagnostics'
    torch.cuda.set_per_process_memory_fraction(.25)
    original_prepare = core.prepare_inputs
    prepared = {}

    def prepare():
        streams, audit = original_prepare()
        assert not audit['exact_full_paragraph_train_dev_collisions']
        prepared['streams'] = streams
        return streams, audit

    core.prepare_inputs = prepare
    core.status = status
    core.run()
    status('VERIFYING_FROZEN_CONTROLS_AND_ALL_TRACES')
    report = json.loads((JOB/'report.json').read_text())
    proof = verify(prepared['streams'], report)
    assert sha(PRIOR/'report.json') == PRIOR_SHA
    prior = json.loads((PRIOR/'report.json').read_text())
    assert json.loads((PRIOR/'input_manifest.json').read_text())['samples_sha256'] == core.SAMPLES_SHA
    for mode, ks in report['metrics_every_eligible_word_gap'].items():
        for k, languages in ks.items():
            for language, m in languages.items():
                assert m['samples'] == prior['metrics_every_eligible_word_gap'][mode][k][language]['samples']
    report.update(user_requested_new_diagnostic=True, source_spoken_v9_used=False,
                  evaluator_wrapper_sha256=sha(__file__), completion_verification=proof,
                  previous_stream_test_comparison={'checkpoint': prior['checkpoint'],
                      'threshold': prior['threshold_from_same_checkpoint_dev'],
                      'metrics': prior['metrics_every_eligible_word_gap'], 'report_sha256': PRIOR_SHA,
                      'same_200_paragraphs_and_pair_order': True, 'previous_model_rescored': False,
                      'scope': 'Same seen diagnostic; each checkpoint uses its own pre-test DEV threshold. No model or threshold selected from test.'})
    core.save('report.json', report)
    proof.update(report_sha256=sha(JOB/'report.json'), complete=True)
    core.save('completion_verification.json', proof)
    status('COMPLETE', checkpoint_step=report['checkpoint']['step'],
           branch_step=report['checkpoint']['branch_step'], report_sha256=proof['report_sha256'])


if __name__ == '__main__':
    assert JOB.exists() and (JOB/'frozen/student_resume_pin.pt').exists()
    with (JOB/'EVALUATOR_LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run()
        except Exception as error:
            core.save('error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
            status('ERROR', error=repr(error))
            raise
