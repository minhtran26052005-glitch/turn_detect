"""User-requested VI -> EN splice probe; read-only evaluation, never training."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
import traceback

from .common import normalize
from .multilingual_three_teacher_distill_v1 import (
    OUT as CAMPAIGN, DATA, MODEL, REV, HF_CACHE, rows_after, validate_row,
)
from .monolingual_data import encode_tail

JOB = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    p = JOB / name; tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')
        f.flush(); os.fsync(f.fileno())
    tmp.replace(p)


def status(stage, **extra):
    save('status.json', {'stage': stage, 'pid': os.getpid(),
                        'at_utc': datetime.now(timezone.utc).isoformat(),
                        'training_interrupted': False, **extra})


def metric(counts):
    tp, tn, fp, fn = (counts.get(k, 0) for k in ('tp', 'tn', 'fp', 'fn'))
    n = tp + tn + fp + fn
    return {'samples': n, 'accuracy': (tp + tn) / n if n else None,
            'eot_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def paragraph(sample):
    turn = sample['selected_turn']; words = []; anchors = {}
    for sentence in turn['sentences']:
        text = sentence['display']; start = len(words)
        current = normalize(text).split()
        assert ' '.join(current) == sentence['normalized']
        for match in re.finditer('[,，;；]', text):
            left = normalize(text[:match.start()]).split()
            right = normalize(text[match.end():]).split()
            if left and right:
                anchors[start + len(left)] = {'label': 'NEOT', 'kind': 'internal_comma_semicolon'}
        words.extend(current)
        if text.rstrip().endswith(('.', '。')):
            anchors[len(words)] = {'label': 'EOT', 'kind': 'sentence_final_full_stop'}
    return words, anchors, ' '.join(s['display'] for s in turn['sentences'])


def freeze_and_calibrate():
    import gc
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    assert shutil.disk_usage(JOB).free >= 15 * 2**30
    assert torch.cuda.mem_get_info()[0] >= 4 * 2**30
    frozen = JOB / 'frozen'; frozen.mkdir(exist_ok=False)
    status('PINNING_CURRENT_STUDENT')
    pin = frozen / 'student_resume_pin.pt'
    os.link(CAMPAIGN / 'latest_student.pt', pin)
    checkpoint_sha = sha(pin)
    state = torch.load(pin, map_location='cpu', weights_only=False)
    recipe = json.loads((CAMPAIGN / 'recipe.json').read_text())
    assert state['dataset_sha256'] == recipe['dataset_sha256']
    assert state['teacher_hashes'] == {k: v['sha256'] for k, v in recipe['teachers'].items()}
    assert state['optimizer']['state'] and all(k in state for k in ('torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng', 'cursors'))
    cfg = AutoConfig.from_pretrained(MODEL, revision=REV, cache_dir=str(HF_CACHE), local_files_only=True, token=False)
    cfg.num_labels = 2; cfg.id2label = {0: 'NEOT', 1: 'EOT'}; cfg.label2id = {'NEOT': 0, 'EOT': 1}
    cfg.save_pretrained(frozen)
    tok = AutoTokenizer.from_pretrained(CAMPAIGN / 'student_tokenizer', local_files_only=True, token=False)
    tok.save_pretrained(frozen)
    save_file({k: v.contiguous() for k, v in state['model'].items()}, str(frozen / 'model.safetensors'))
    receipt = {'step': state['step'], 'source_checkpoint_sha256': checkpoint_sha,
               'weights_sha256': sha(frozen / 'model.safetensors'), 'dataset_sha256': state['dataset_sha256'],
               'optimizer_rng_cursors_preserved': True, 'student_base': MODEL, 'revision': REV,
               'one_model_no_language_selector': True, 'train_continues': True}
    save('checkpoint_receipt.json', receipt)
    model = AutoModelForSequenceClassification.from_config(cfg, dtype=torch.float32)
    model.load_state_dict(state['model'], strict=True); model.cuda().eval()
    del state; gc.collect()

    def predict(rows):
        result = []
        for begin in range(0, len(rows), 64):
            batch = rows[begin:begin + 64]
            seqs = [encode_tail(tok, row['context'], row['text'], 256)['input_ids'] for row in batch]
            width = max(map(len, seqs))
            ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long, device='cuda')
            mask = torch.zeros_like(ids)
            for i, seq in enumerate(seqs):
                ids[i, :len(seq)] = torch.as_tensor(seq, device='cuda'); mask[i, :len(seq)] = 1
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                result.extend(model(input_ids=ids, attention_mask=mask).logits.float().softmax(-1)[:, 1].cpu().tolist())
        return result

    status('CALIBRATING_THRESHOLD_ON_FROZEN_DEV', checkpoint_step=receipt['step'])
    db = sqlite3.connect(DATA.resolve().as_uri() + '?mode=ro', uri=True, timeout=30)
    dev = {}
    for lang in ('vi', 'en', 'zh'):
        rows = [validate_row(r) for r in rows_after(db, lang, 'dev', '', 1000)]
        dev[lang] = (np.asarray(predict(rows)), np.asarray([r['label'] == 'EOT' for r in rows]))
    db.close()

    def evaluate(p, y, threshold):
        pred = p >= threshold
        return metric({'tp': int((pred & y).sum()), 'tn': int((~pred & ~y).sum()),
                       'fp': int((pred & ~y).sum()), 'fn': int((~pred & y).sum())})

    threshold = float(max(np.linspace(.1, .9, 81), key=lambda t: min(evaluate(*dev[l], t)['eot_f1'] for l in dev)))
    save('frozen_dev.json', {'checkpoint_step': receipt['step'], 'shared_threshold': threshold,
                           'threshold_locked_before_probe': True, 'probe_used_for_selection': False,
                           'by_language': {l: evaluate(*dev[l], threshold) for l in dev}})
    return predict, threshold, receipt


def decode(words, rows, threshold):
    start = 0; segments = []
    for row in rows:
        if row['p_eot'] >= threshold:
            boundary = row['boundary_word_index']
            segments.append({'text': ' '.join(words[start:boundary]), 'start': start, 'end': boundary,
                             'p_eot': row['p_eot'], 'reason': 'MODEL_EOT'})
            start = boundary
    if start < len(words):
        segments.append({'text': ' '.join(words[start:]), 'start': start, 'end': len(words),
                         'p_eot': None, 'reason': 'EXPLICIT_END_OF_STREAM_FLUSH_NOT_MODEL_PREDICTION'})
    assert ' '.join(s['text'] for s in segments) == ' '.join(words)
    return segments


def run(args):
    import torch
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    utils.is_torchvision_available = imports.is_torchvision_available = lambda: False
    utils.is_librosa_available = imports.is_librosa_available = lambda: False
    torch.set_num_threads(2); os.nice(5)
    assert not (JOB / 'report.json').exists(), 'Completed probe must never run again'
    samples = json.loads(args.samples.read_bytes())['samples']
    assert sha(args.samples) == args.samples_sha256
    assert Counter(s['language'] for s in samples) == {'vi': 100, 'en': 100}
    by_lang = {l: sorted([s for s in samples if s['language'] == l], key=lambda s: s['number']) for l in ('vi', 'en')}
    pairs = [(by_lang['vi'][i], by_lang['en'][i]) for i in range(args.pairs)]
    save('input_manifest.json', {'samples_sha256': sha(args.samples), 'selected_pairs': args.pairs,
                                'pair_policy': 'First 20 rows from each already-frozen random review; VI_i followed by EN_i. Not parallel translations.',
                                'pair_ids': [[a['dialogue']['dialogue_id'], b['dialogue']['dialogue_id']] for a, b in pairs],
                                'source_data_used_for_training': False, 'source_module_sha256': sha(__file__),
                                'inputs': 'Causal rolling observed prefix, normalized lower-case without punctuation; only next 1/2 words. No language, role, punctuation, or gold resets supplied.',
                                'reference': 'Only sentence-final full stop EOT and internal comma/semicolon NEOT; question/exclamation endpoints unscored.',
                                'splice': 'Constructed cross-language and cross-source-paragraph transition; outside same-speaker lookahead training protocol.',
                                'end_of_stream': 'Explicit flush; not credited as a model boundary.',
                                'scope': 'Synthetic user examples, punctuation-derived diagnostic; not independent human/audio gold.'})
    predict, threshold, checkpoint = freeze_and_calibrate()
    counters = {mode: {str(k): {l: Counter() for l in ('vi', 'en')} for k in (1, 2)} for mode in ('monolingual', 'mixed_matched')}
    switches = {str(k): {'total': 0, 'predicted_eot': 0, 'full_stop_reference_total': 0,
                         'full_stop_predicted_eot': 0} for k in (1, 2)}
    cases = []; total_inferences = 0
    trace = JOB / 'predictions.jsonl.gz'
    with gzip.open(trace, 'wt', encoding='utf-8') as target:
        for number, (vi, en) in enumerate(pairs, 1):
            vi_words, vi_anchors, vi_display = paragraph(vi)
            en_words, en_anchors, en_display = paragraph(en)
            offset = len(vi_words); mixed_words = vi_words + en_words
            mixed_anchors = {**vi_anchors, **{b + offset: a for b, a in en_anchors.items()}}
            case = {'number': number, 'vi_source': vi['dialogue']['dialogue_id'], 'en_source': en['dialogue']['dialogue_id'],
                    'vi_original': vi_display, 'en_original': en_display, 'stream_input': ' '.join(mixed_words),
                    'switch_word_index': offset, 'lookahead': {}}
            for k in (1, 2):
                selected = {}; matching = {}
                for lang, words, anchors in [('vi', vi_words, vi_anchors), ('en', en_words, en_anchors), ('mixed', mixed_words, mixed_anchors)]:
                    rows = []
                    for boundary in range(1, len(words) - k + 1):
                        row = {'case': number, 'stream': lang, 'lookahead_words': k,
                               'boundary_word_index': boundary, 'context': ' '.join(words[:boundary]),
                               'text': ' '.join(words[boundary:boundary + k]),
                               'reference': anchors.get(boundary), 'crosses_language_splice': lang == 'mixed' and boundary <= offset < boundary + k}
                        assert len(row['text'].split()) == k and normalize(row['context']) == row['context']
                        rows.append(row)
                    probs = predict(rows); total_inferences += len(rows)
                    for row, probability in zip(rows, probs):
                        row['p_eot'] = probability; row['predicted'] = 'EOT' if probability >= threshold else 'NEOT'
                        target.write(json.dumps(row, ensure_ascii=False) + '\n')
                        if row['reference']:
                            b = row['boundary_word_index']; actual = row['reference']['label'] == 'EOT'; pred = probability >= threshold
                            kind = 'tp' if actual and pred else 'fn' if actual else 'fp' if pred else 'tn'
                            if lang in ('vi', 'en'):
                                counters['monolingual'][str(k)][lang][kind] += 1
                                matching[(lang, b)] = kind
                            else:
                                language = 'vi' if b <= offset else 'en'; relative = b if language == 'vi' else b - offset
                                if (language, relative) in matching:
                                    counters['mixed_matched'][str(k)][language][kind] += 1
                    if lang == 'mixed':
                        switch = next(r for r in rows if r['boundary_word_index'] == offset)
                        score = switches[str(k)]; score['total'] += 1
                        score['predicted_eot'] += switch['predicted'] == 'EOT'
                        if switch['reference'] and switch['reference']['label'] == 'EOT':
                            score['full_stop_reference_total'] += 1; score['full_stop_predicted_eot'] += switch['predicted'] == 'EOT'
                        selected = {'switch': switch, 'segments': decode(words, rows, threshold),
                                    'anchors': [r for r in rows if r['reference'] or r['boundary_word_index'] == offset],
                                    'predicted_cut_count': sum(r['predicted'] == 'EOT' for r in rows)}
                case['lookahead'][str(k)] = selected
            cases.append(case)
            save('cases.json', cases)
            status('SCORING_MIXED_AND_MONOLINGUAL', checkpoint_step=checkpoint['step'], pairs_done=number, pairs_total=args.pairs)
    report = {'complete': True, 'checkpoint': checkpoint, 'threshold_from_same_checkpoint_dev': threshold,
              'pairs': args.pairs, 'source_paragraphs_transferred': 200, 'evaluated_paragraphs': 2 * args.pairs,
              'metrics_on_matched_fullstop_comma_anchors': {mode: {k: {l: metric(c) for l, c in langs.items()} for k, langs in by_k.items()} for mode, by_k in counters.items()},
              'language_splice_observations': switches, 'inferences': total_inferences,
              'predictions_sha256': sha(trace), 'cases_sha256': sha(JOB / 'cases.json'),
              'scope': 'Synthetic punctuation diagnostic on artificial VI->EN splices; not a language-ID test, translations, or independent conversational gold.',
              'language_selector_supplied_to_model': False, 'gold_resets_supplied_to_model': False,
              'threshold_selected_on_probe': False, 'probe_data_used_for_training': False,
              'training_interrupted': False, 'final_flush_is_model_prediction': False,
              'completed_utc': datetime.now(timezone.utc).isoformat()}
    save('report.json', report); status('COMPLETE', checkpoint_step=checkpoint['step'], report_sha256=sha(JOB / 'report.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--samples-sha256', required=True)
    parser.add_argument('--pairs', type=int, default=20)
    args = parser.parse_args(); JOB = args.job; JOB.mkdir(parents=True, exist_ok=True)
    with (JOB / 'EVALUATOR_LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args)
        except Exception as error:
            save('error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
            status('ERROR', error=repr(error)); raise
