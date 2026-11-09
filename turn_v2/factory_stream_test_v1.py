"""Read-only streaming diagnostic on 200 exposed, training-excluded paragraphs."""
from collections import Counter
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

from .common import normalize
from .factory_stream_data_v1 import paragraph, sha
from .monolingual_data import encode_tail
from .multilingual_bilingual_stream_probe_v1 import decode

PROJECT = Path('/home/edabk/hoangbpm/turn_detect')
CAMPAIGN = PROJECT / 'outputs/multilingual_stream_v2_20261102'
JOB = PROJECT / 'outputs/multilingual_stream_test_20261102_v1'
OLD = Path('/mnt/hdd500/turn_detect_training/multilingual_vi_en_stream_probe_20261102_v2')
SAMPLES_SHA = '1a9157938d43395145d5f8785c8b870d33977adda3823f0a10c449608a80eba7'


def save(name, value):
    p = JOB / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(p)


def status(stage, **more):
    save('status.json', {'stage': stage, 'pid': os.getpid(),
                        'at_utc': datetime.now(timezone.utc).isoformat(),
                        'training_interrupted': False, **more})


def metric(c):
    tp, tn, fp, fn = (int(c.get(k, 0)) for k in ('tp', 'tn', 'fp', 'fn'))
    n = tp + tn + fp + fn
    return {'samples': n, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'accuracy': (tp + tn) / n if n else None,
            'eot_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'eot_precision': tp / (tp + fp) if tp + fp else None,
            'eot_recall': tp / (tp + fn) if tp + fn else None,
            'false_share_of_cuts': fp / (tp + fp) if tp + fp else None,
            'false_positive_rate': fp / (tn + fp) if tn + fp else None}


def kind(actual, predicted):
    return 'tp' if actual and predicted else 'fn' if actual else 'fp' if predicted else 'tn'


def prepare_inputs():
    p = JOB / 'inputs/samples.json'
    assert sha(p) == SAMPLES_SHA
    samples = json.loads(p.read_text())['samples']
    assert Counter(s['language'] for s in samples) == {'vi': 100, 'en': 100}
    by = {l: sorted((s for s in samples if s['language'] == l), key=lambda s: s['number']) for l in ('vi', 'en')}
    ids = {s['dialogue']['dialogue_id'] for s in samples}
    assert len(ids) == 200
    content_hashes = {}
    streams = []
    for lang in ('vi', 'en'):
        for s in by[lang]:
            p = paragraph(s['selected_turn'])
            streams.append({**p, 'key': f'{lang}_{s["number"]:03d}', 'mode': 'monolingual',
                            'number': s['number'], 'language_by_word': [lang] * len(p['words']),
                            'source_ids': [s['dialogue']['dialogue_id']]})
            content_hashes[hashlib.sha256(' '.join(p['words']).encode()).hexdigest()] = s['dialogue']['dialogue_id']
    for i in range(100):
        a, b = paragraph(by['vi'][i]['selected_turn']), paragraph(by['en'][i]['selected_turn'])
        offset = len(a['words'])
        streams.append({'key': f'mixed_{i+1:03d}', 'mode': 'mixed', 'number': i+1,
                        'words': a['words'] + b['words'], 'eots': a['eots'] + [x+offset for x in b['eots']],
                        'unknown': a['unknown'] + [x+offset for x in b['unknown']],
                        'language_by_word': ['vi']*offset + ['en']*len(b['words']), 'splice': offset,
                        'vi_display': a['display'], 'en_display': b['display'],
                        'source_ids': [by['vi'][i]['dialogue']['dialogue_id'], by['en'][i]['dialogue']['dialogue_id']]})
    status('AUDITING_INPUT_SEPARATION')
    db = sqlite3.connect((CAMPAIGN/'dataset/paragraphs.sqlite').as_uri()+'?mode=ro', uri=True)
    selected_splits = Counter(); collisions = []; selected_families = set(); train_families = set(); dev_families = set()
    for identity, family, split, record in db.execute('SELECT dialogue_id,family_id,split,record FROM paragraphs'):
        if identity in ids:
            selected_splits[split] += 1; selected_families.add(family)
        if split == 'train': train_families.add(family)
        if split == 'dev': dev_families.add(family)
        if split in ('train', 'dev'):
            words = json.loads(record)['words']; key = hashlib.sha256(' '.join(words).encode()).hexdigest()
            if key in content_hashes: collisions.append({'test_id': content_hashes[key], 'other_id': identity, 'split': split})
    db.close()
    assert not selected_splits['train'] and not selected_splits['dev']
    assert selected_splits['exposed_regression'] == 800
    audit = {'selected_dialogues': 200, 'selected_paragraphs': 200,
             'source_dialogue_paragraph_splits': dict(selected_splits),
             'exact_full_paragraph_train_dev_collisions': collisions,
             'source_family_overlap_with_train': len(selected_families & train_families),
             'source_family_overlap_with_dev': len(selected_families & dev_families),
             'family_overlap_is_not_disjoint_independent_test': True,
             'student_token_window_and_paraphrase_collisions_exhaustively_audited': False}
    save('input_audit.json', audit)
    save('input_manifest.json', {'samples_sha256': SAMPLES_SHA, 'paragraphs': 200, 'mixed_pairs': 100,
         'pair_policy': 'Locked random-review order: VI_i then EN_i; not translations.',
         'review_pairs': 'First20 fixed before scoring, matching old probe; all100 pairs scored on SSH.',
         'every_eligible_word_gap_scored': True, 'question_exclamation_endpoints_unscored': True,
         'final_flush_is_model_prediction': False, 'lookahead_words': [1, 2],
         'inputs': 'Lowercase unpunctuated buffer since last predicted EOT plus only next1/2 observed words. No language/role/source boundary or gold reset.',
         'source_module_sha256': sha(__file__), 'sampler_module_sha256': sha(paragraph.__code__.co_filename),
         'scope': 'Previously exposed Qwen synthetic punctuation diagnostic, excluded from new TRAIN/DEV by dialogue ID; not independent human/audio gold.'})
    return streams, audit


def load_and_calibrate():
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    frozen = JOB/'frozen'; pin = frozen/'student_resume_pin.pt'
    assert shutil.disk_usage(JOB).free > 13*2**30
    assert torch.cuda.mem_get_info()[0] > 4*2**30
    status('LOADING_PINNED_CHECKPOINT')
    state = torch.load(pin, map_location='cpu', weights_only=False)
    recipe = json.loads((JOB/'source_recipe.json').read_text())
    assert state['dataset_manifest_sha256'] == recipe['dataset_manifest_sha256'] == sha(JOB/'dataset_manifest.json')
    assert state['optimizer']['state'] and all(k in state for k in ('torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng', 'sampler', 'zh_cursor'))
    cfg = AutoConfig.from_pretrained(CAMPAIGN/'student_tokenizer', local_files_only=True, token=False)
    tok = AutoTokenizer.from_pretrained(CAMPAIGN/'student_tokenizer', local_files_only=True, token=False)
    cfg.save_pretrained(frozen); tok.save_pretrained(frozen)
    save_file({k: v.contiguous() for k, v in state['model'].items()}, str(frozen/'model.safetensors'))
    receipt = {'step': state['step'], 'branch_step': state['branch_step'], 'base_step': state['base_step'],
               'source_checkpoint_sha256': sha(pin), 'weights_sha256': sha(frozen/'model.safetensors'),
               'dataset_manifest_sha256': state['dataset_manifest_sha256'],
               'full_optimizer_rng_sampler_preserved': True, 'optimizer_entries': len(state['optimizer']['state']),
               'one_shared_model': True, 'student_base': recipe['student_base'], 'revision': recipe['revision']}
    save('checkpoint_receipt.json', receipt)
    model = AutoModelForSequenceClassification.from_config(cfg, dtype=torch.float32)
    model.load_state_dict(state['model'], strict=True); model.cuda().eval(); del state; gc.collect()

    def predict(rows):
        values = []
        for begin in range(0, len(rows), 64):
            part = rows[begin:begin+64]
            seqs = [encode_tail(tok, r['context'], r['text'], 256)['input_ids'] for r in part]
            width = max(map(len, seqs)); ids = torch.full((len(part), width), tok.pad_token_id, dtype=torch.long, device='cuda'); mask = torch.zeros_like(ids)
            for i, seq in enumerate(seqs):
                ids[i, :len(seq)] = torch.as_tensor(seq, device='cuda'); mask[i, :len(seq)] = 1
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                values.extend(model(input_ids=ids, attention_mask=mask).logits.float().softmax(-1)[:, 1].cpu().tolist())
        return values

    status('CALIBRATING_ONLY_ON_SAME_CHECKPOINT_DEV', branch_step=receipt['branch_step'])
    dev = json.loads((JOB/'frozen_dev_streams.json').read_text()); rows = []
    assert sha(JOB/'frozen_dev_streams.json') == json.loads((JOB/'dataset_manifest.json').read_text())['files']['dev_streams.json']['sha256']
    for s in dev:
        for b in range(1, len(s['words'])-1):
            if b in s['unknown']: continue
            rows.append({'context': ' '.join(s['words'][:b]), 'text': ' '.join(s['words'][b:b+2]),
                         'language': s['language_by_word'][b-1], 'label': b in s['eots']})
    probs = predict(rows)
    def scores(t):
        cs = {l: Counter() for l in ('vi', 'en')}
        for r, p in zip(rows, probs): cs[r['language']][kind(r['label'], p >= t)] += 1
        return {l: metric(c) for l, c in cs.items()}
    threshold = float(max(np.linspace(.10, .95, 86), key=lambda t: min(m['eot_f1'] for m in scores(t).values())))
    save('frozen_dev.json', {'checkpoint_step': receipt['step'], 'branch_step': receipt['branch_step'],
         'shared_threshold': threshold, 'selection': 'Exactly new trainer policy: maximize worst VI/EN rolling DEV F1, lookahead2.',
         'threshold_locked_before_test': True, 'test_used_for_selection': False, 'by_language': scores(threshold),
         'dev_streams_sha256': sha(JOB/'frozen_dev_streams.json')})
    return predict, threshold, receipt


def audit_trace(streams, threshold, total):
    lookup = {s['key']: s for s in streams}; starts = {}; boundaries = {}; count = 0
    with gzip.open(JOB/'predictions.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line); s = lookup[r['stream_key']]; key = (r['stream_key'], r['lookahead_words']); b = r['boundary_word_index']; k = r['lookahead_words']
            start = starts.get(key, 0)
            assert b == boundaries.get(key, 0)+1
            assert r['buffer_start_word_index'] == start and r['context'] == ' '.join(s['words'][start:b])
            assert r['text'] == ' '.join(s['words'][b:b+k]) and b+k <= len(s['words'])
            assert normalize(r['context']) == r['context'] and normalize(r['text']) == r['text']
            assert len(r['text'].split()) == k and r['predicted'] == ('EOT' if r['p_eot'] >= threshold else 'NEOT')
            if r['predicted'] == 'EOT': starts[key] = b
            boundaries[key] = b; count += 1
    assert count == total
    for s in streams:
        for k in (1, 2): assert boundaries[(s['key'], k)] == len(s['words'])-k
    return {'all_inferences_audited': count, 'causal_inputs_passed': True,
            'only_predicted_eot_resets_passed': True, 'word_conservation_passed': True,
            'no_language_role_or_reference_supplied_to_model': True}


def run():
    import torch
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    utils.is_torchvision_available = imports.is_torchvision_available = lambda: False
    utils.is_librosa_available = imports.is_librosa_available = lambda: False
    torch.set_num_threads(2); os.nice(5)
    assert not (JOB/'report.json').exists(), 'Never rerun a completed diagnostic'
    streams, input_audit = prepare_inputs()
    predict, threshold, checkpoint = load_and_calibrate()
    counters = {mode: {str(k): {l: Counter() for l in ('vi', 'en')} for k in (1, 2)} for mode in ('monolingual', 'mixed')}
    subset = {str(k): Counter() for k in (1, 2)}; switches = {str(k): Counter() for k in (1, 2)}
    all_cases = {s['number']: {'number': s['number'], 'vi_source': s['source_ids'][0], 'en_source': s['source_ids'][1],
                    'vi_original': s['vi_display'], 'en_original': s['en_display'],
                    'stream_input': ' '.join(s['words']), 'switch_word_index': s['splice'], 'lookahead': {}}
                 for s in streams if s['mode'] == 'mixed'}
    total = 0; unscored_cuts = {str(k): 0 for k in (1, 2)}
    with gzip.open(JOB/'predictions.jsonl.gz', 'wt', encoding='utf-8') as f:
        for k in (1, 2):
            starts = [0]*len(streams); predictions = [[] for _ in streams]
            for b in range(1, max(len(s['words']) for s in streams)-k+1):
                rows = []; indexes = []
                for i, s in enumerate(streams):
                    if b+k > len(s['words']): continue
                    rows.append({'stream_key': s['key'], 'case': s['number'], 'mode': s['mode'], 'lookahead_words': k,
                         'boundary_word_index': b, 'buffer_start_word_index': starts[i],
                         'context': ' '.join(s['words'][starts[i]:b]), 'text': ' '.join(s['words'][b:b+k])})
                    indexes.append(i)
                if not rows: continue
                for i, row, p in zip(indexes, rows, predict(rows)):
                    s = streams[i]; pred = p >= threshold; row.update(p_eot=p, predicted='EOT' if pred else 'NEOT')
                    actual = b in s['eots']; ref = None if b in s['unknown'] else 'EOT' if actual else 'NEOT'
                    row['reference'] = {'label': ref, 'kind': 'full_stop' if actual else 'word_gap'} if ref else None
                    f.write(json.dumps(row, ensure_ascii=False)+'\n'); total += 1
                    if pred: starts[i] = b
                    if ref:
                        label = kind(actual, pred); lang = s['language_by_word'][b-1]
                        counters[s['mode']][str(k)][lang][label] += 1
                        if s['mode'] == 'mixed' and s['number'] <= 20: subset[str(k)][label] += 1
                    elif pred: unscored_cuts[str(k)] += 1
                    if s['mode'] == 'mixed':
                        predictions[i].append(row)
                        if b == s['splice']:
                            switches[str(k)]['total'] += 1; switches[str(k)]['predicted_eot'] += pred
                            switches[str(k)]['full_stop_reference_total'] += actual
                            switches[str(k)]['full_stop_predicted_eot'] += actual and pred
                if b % 30 == 0: status('SCORING_CAUSAL_STREAMS', branch_step=checkpoint['branch_step'], lookahead_words=k, word_index=b, inferences=total)
            for i, s in enumerate(streams):
                rows = predictions[i]
                if s['mode'] != 'mixed': continue
                segs = decode(s['words'], rows, threshold)
                issues = [r for r in rows if r['boundary_word_index'] == s['splice'] or r['reference'] and r['reference']['label'] != r['predicted']]
                all_cases[s['number']]['lookahead'][str(k)] = {'switch': next(r for r in rows if r['boundary_word_index'] == s['splice']),
                    'segments': segs, 'problem_points': issues,
                    'metrics': metric(Counter(kind(r['reference']['label']=='EOT', r['predicted']=='EOT') for r in rows if r['reference']))}
    status('AUDITING_ALL_INFERENCE_INPUTS', inferences=total)
    causal_audit = audit_trace(streams, threshold, total)
    old_comparison = {}
    if (OLD/'report.json').exists():
        old = json.loads((OLD/'report.json').read_text()); old_cs = {str(k): Counter() for k in (1, 2)}
        mixed = {s['number']: s for s in streams if s['mode']=='mixed'}
        original_cases = json.loads((OLD/'cases.json').read_text())
        assert sha(OLD/'cases.json') == old['cases_sha256']
        for case in original_cases: assert case['stream_input'].split() == mixed[case['number']]['words']
        assert sha(OLD/'predictions.jsonl.gz') == old['predictions_sha256']
        with gzip.open(OLD/'predictions.jsonl.gz', 'rt', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line); s = mixed[r['case']]; b = r['boundary_word_index']
                if b not in s['unknown']: old_cs[str(r['lookahead_words'])][kind(b in s['eots'], r['predicted']=='EOT')] += 1
        old_comparison = {str(k): {'old': metric(old_cs[str(k)]), 'new': metric(subset[str(k)])} for k in (1, 2)}
        for c in old_comparison.values(): assert c['old']['samples'] == c['new']['samples']
    save('cases.json', list(all_cases.values()))
    save('review_cases.json', [all_cases[i] for i in range(1, 21)])
    report = {'complete': True, 'checkpoint': checkpoint, 'threshold_from_same_checkpoint_dev': threshold,
         'source_paragraphs': 200, 'by_language_source_paragraphs': {'vi': 100, 'en': 100}, 'mixed_pairs': 100,
         'metrics_every_eligible_word_gap': {mode: {k: {l: metric(c) for l, c in langs.items()} for k, langs in ks.items()} for mode, ks in counters.items()},
         'language_splice_observations': {k: dict(v) for k, v in switches.items()},
         'unscored_question_exclamation_cuts': unscored_cuts, 'same_first20_old_probe_comparison': old_comparison,
         'comparison_scope': 'Frozen old/new weights and their own DEV-calibrated thresholds differ. Old traces reused, never rescored; no selection from this diagnostic.',
         'input_audit': input_audit, 'causal_audit': causal_audit, 'inferences': total,
         'predictions_sha256': sha(JOB/'predictions.jsonl.gz'), 'cases_sha256': sha(JOB/'cases.json'),
         'review_cases_sha256': sha(JOB/'review_cases.json'), 'review_case_count': 20,
         'predicted_eot_buffer_reset': True, 'test_selected_threshold_or_model': False, 'training_interrupted': False,
         'scope': '200 previously exposed Qwen paragraphs held outside new TRAIN/DEV by ID; synthetic punctuation diagnostic, not independent human/audio conversational gold.',
         'completed_utc': datetime.now(timezone.utc).isoformat()}
    save('report.json', report); status('COMPLETE', checkpoint_step=checkpoint['step'], branch_step=checkpoint['branch_step'], report_sha256=sha(JOB/'report.json'))


if __name__ == '__main__':
    JOB.mkdir(exist_ok=True, parents=True)
    with (JOB/'EVALUATOR_LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: run()
        except Exception as error:
            save('error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
            status('ERROR', error=repr(error)); raise
