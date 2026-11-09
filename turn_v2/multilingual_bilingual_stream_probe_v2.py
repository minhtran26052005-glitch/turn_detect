"""Same frozen weights/threshold as v1; causal buffer resets on predicted EOT."""
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gzip
import json
import os
from pathlib import Path
import traceback

from . import multilingual_bilingual_stream_probe_v1 as base
from .monolingual_data import encode_tail

ORIGINAL = Path('/mnt/hdd500/turn_detect_training/multilingual_vi_en_stream_probe_20261102_v1')
JOB = ORIGINAL.with_name('multilingual_vi_en_stream_probe_20261102_v2')


def save(name, value):
    p = JOB / name; tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(p)


def status(stage, **more):
    save('status.json', {'stage': stage, 'pid': os.getpid(), 'training_interrupted': False,
                        'at_utc': datetime.now(timezone.utc).isoformat(), **more})


def run():
    import torch
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    utils.is_torchvision_available = imports.is_torchvision_available = lambda: False
    utils.is_librosa_available = imports.is_librosa_available = lambda: False
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    from safetensors.torch import load_file
    os.nice(5); torch.set_num_threads(2)
    assert not (JOB / 'report.json').exists(), 'Never repeat a completed probe'
    report = json.loads((ORIGINAL / 'report.json').read_text()); assert report['complete']
    frozen = ORIGINAL / 'frozen'; threshold = report['threshold_from_same_checkpoint_dev']
    assert base.sha(frozen / 'model.safetensors') == report['checkpoint']['weights_sha256']
    dev = json.loads((ORIGINAL / 'frozen_dev.json').read_text())
    assert dev['threshold_locked_before_probe'] and dev['shared_threshold'] == threshold
    cases = json.loads((ORIGINAL / 'cases.json').read_text())
    assert base.sha(ORIGINAL / 'cases.json') == report['cases_sha256']
    samples = json.loads((ORIGINAL / 'inputs/samples.json').read_text())['samples']
    by_lang = {l: sorted([s for s in samples if s['language'] == l], key=lambda s: s['number']) for l in ('vi', 'en')}
    words = [case['stream_input'].split() for case in cases]
    counts = {str(k): {l: Counter() for l in ('vi', 'en')} for k in (1, 2)}
    switches = {str(k): {'total': 0, 'predicted_eot': 0, 'full_stop_reference_total': 0, 'full_stop_predicted_eot': 0} for k in (1, 2)}
    comparisons = {}
    manifest = json.loads((ORIGINAL / 'input_manifest.json').read_text())
    manifest.update(source_module_sha256=base.sha(__file__),
                    inputs='Causal buffer since last predicted EOT; only next 1/2 observed words, no gold resets/language selector.',
                    comparison_scope='Same frozen checkpoint and DEV threshold; compare two inference buffer policies. No model/threshold chosen from these results.',
                    original_report_sha256=base.sha(ORIGINAL / 'report.json'))
    save('input_manifest.json', manifest); save('checkpoint_receipt.json', report['checkpoint']); save('frozen_dev.json', dev)
    status('LOADING_EXISTING_FROZEN_STUDENT', checkpoint_step=report['checkpoint']['step'])
    tok = AutoTokenizer.from_pretrained(frozen, local_files_only=True, token=False)
    cfg = AutoConfig.from_pretrained(frozen, local_files_only=True, token=False)
    model = AutoModelForSequenceClassification.from_config(cfg, dtype=torch.float32)
    model.load_state_dict(load_file(str(frozen / 'model.safetensors')), strict=True); model.cuda().eval()

    def predict(rows):
        seqs = [encode_tail(tok, r['context'], r['text'], 256)['input_ids'] for r in rows]
        width = max(map(len, seqs)); ids = torch.full((len(rows), width), tok.pad_token_id, dtype=torch.long, device='cuda'); mask = torch.zeros_like(ids)
        for i, seq in enumerate(seqs):
            ids[i, :len(seq)] = torch.as_tensor(seq, device='cuda'); mask[i, :len(seq)] = 1
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            return model(input_ids=ids, attention_mask=mask).logits.float().softmax(-1)[:, 1].cpu().tolist()

    total = 0
    with gzip.open(JOB / 'predictions.jsonl.gz', 'wt', encoding='utf-8') as trace:
        for k in (1, 2):
            starts = [0] * len(cases); predictions = [[] for _ in cases]
            refs = [{r['boundary_word_index']: r['reference'] for r in c['lookahead'][str(k)]['anchors'] if r['reference']} for c in cases]
            for b in range(1, max(map(len, words)) - k + 1):
                rows = []; indexes = []
                for i, case in enumerate(cases):
                    if b + k > len(words[i]):
                        continue
                    offset = case['switch_word_index']
                    row = {'case': case['number'], 'stream': 'mixed', 'lookahead_words': k, 'boundary_word_index': b,
                           'context': ' '.join(words[i][starts[i]:b]), 'text': ' '.join(words[i][b:b+k]),
                           'reference': refs[i].get(b), 'crosses_language_splice': b <= offset < b+k,
                           'buffer_start_word_index': starts[i], 'buffer_reset_source': 'predicted_EOT_only'}
                    assert row['context'] and len(row['text'].split()) == k
                    rows.append(row); indexes.append(i)
                for i, row, p in zip(indexes, rows, predict(rows)):
                    row['p_eot'] = p; row['predicted'] = 'EOT' if p >= threshold else 'NEOT'
                    predictions[i].append(row); trace.write(json.dumps(row, ensure_ascii=False) + '\n'); total += 1
                    if row['predicted'] == 'EOT':
                        starts[i] = b
                if b % 30 == 0:
                    status('SCORING_PREDICTED_BUFFER_RESET', lookahead_words=k, boundary_index=b)
            old_metrics = Counter(); new_metrics = Counter(); old_other = 0; new_other = 0
            for i, case in enumerate(cases):
                rows = predictions[i]; offset = case['switch_word_index']
                switch = next(r for r in rows if r['boundary_word_index'] == offset)
                summary = switches[str(k)]; summary['total'] += 1; summary['predicted_eot'] += switch['predicted'] == 'EOT'
                if switch['reference'] and switch['reference']['label'] == 'EOT':
                    summary['full_stop_reference_total'] += 1; summary['full_stop_predicted_eot'] += switch['predicted'] == 'EOT'
                vi_words, _, _ = base.paragraph(by_lang['vi'][i]); en_words, _, _ = base.paragraph(by_lang['en'][i])
                assert words[i] == vi_words + en_words
                old = case['lookahead'][str(k)]
                for current, counter in [(old['anchors'], old_metrics), (rows, new_metrics)]:
                    for row in current:
                        if not row['reference']:
                            continue
                        actual = row['reference']['label'] == 'EOT'; pred = row['predicted'] == 'EOT'
                        kind = 'tp' if actual and pred else 'fn' if actual else 'fp' if pred else 'tn'
                        counter[kind] += 1
                        if current is rows:
                            pos = row['boundary_word_index']; lang = 'vi' if pos <= offset else 'en'; rel = pos if lang == 'vi' else pos-offset
                            if rel + k <= len(vi_words if lang == 'vi' else en_words):
                                counts[str(k)][lang][kind] += 1
                endpoints = {r['boundary_word_index'] for r in old['anchors'] if r['reference'] and r['reference']['label'] == 'EOT'}
                old_other += sum(s['reason'] == 'MODEL_EOT' and s['end'] not in endpoints for s in old['segments'])
                segments = base.decode(words[i], rows, threshold)
                new_other += sum(s['reason'] == 'MODEL_EOT' and s['end'] not in endpoints for s in segments)
                case['lookahead'][str(k)] = {'switch': switch, 'segments': segments,
                                             'anchors': [r for r in rows if r['reference'] or r['boundary_word_index'] == offset],
                                             'predicted_cut_count': sum(r['predicted'] == 'EOT' for r in rows)}
            comparisons[str(k)] = {'rolling_prefix_source_anchors': base.metric(old_metrics),
                                   'predicted_reset_source_anchors': base.metric(new_metrics),
                                   'rolling_prefix_cuts_outside_fullstop': old_other, 'predicted_reset_cuts_outside_fullstop': new_other,
                                   'rolling_prefix_splice': report['language_splice_observations'][str(k)],
                                   'predicted_reset_splice': switches[str(k)]}
    save('cases.json', cases)
    report.update(metrics_on_matched_fullstop_comma_anchors={'monolingual': report['metrics_on_matched_fullstop_comma_anchors']['monolingual'],
                  'mixed_matched': {k: {l: base.metric(c) for l, c in langs.items()} for k, langs in counts.items()}},
                  language_splice_observations=switches, inferences=total, predictions_sha256=base.sha(JOB/'predictions.jsonl.gz'),
                  cases_sha256=base.sha(JOB/'cases.json'), predicted_eot_buffer_reset=True, comparison_to_rolling_prefix=comparisons,
                  original_report_sha256=manifest['original_report_sha256'],
                  completed_utc=datetime.now(timezone.utc).isoformat())
    save('report.json', report); status('COMPLETE', checkpoint_step=report['checkpoint']['step'], report_sha256=base.sha(JOB/'report.json'))


if __name__ == '__main__':
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB/'EVALUATOR_LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run()
        except Exception as error:
            save('error.json', {'error': repr(error), 'traceback': traceback.format_exc()}); status('ERROR', error=repr(error)); raise
