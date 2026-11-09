"""Fixed TRAIN-fit diagnostic against the saved DEV-selected v5 checkpoint."""
from pathlib import Path
from collections import Counter, defaultdict
import json, gzip, hashlib, os, time, traceback
import numpy as np
from .uploaded_vi_en_data_v5 import OUT as MAIN, initialize, StreamOrder, sha
from .malbert_eval_zip_v2 import metric, kind

OUT = MAIN / 'train_fit_probe_20261106_v1'


def save(name, value):
    path = OUT / name
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def run():
    import torch
    import transformers.utils as u
    import transformers.utils.import_utils as im
    u.is_torchvision_available = im.is_torchvision_available = lambda: False
    u.is_librosa_available = im.is_librosa_available = lambda: False
    from transformers import AutoTokenizer, AlbertConfig
    from .malbert_turn_student_v1 import MalbertTurnStudent
    from .uploaded_vi_en_evaluation_v5 import predictor
    assert not (OUT / 'report.json').exists()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.2)
    # Read the report around the immutable open weight inode to avoid atomic-save races.
    while True:
        first = (MAIN / 'runs/finetune/best_report.json').read_bytes()
        with (MAIN / 'runs/finetune/student.pt').open('rb') as source:
            weights = torch.load('/proc/self/fd/' + str(source.fileno()), map_location='cpu', weights_only=True)
            if first == (MAIN / 'runs/finetune/best_report.json').read_bytes():
                source.seek(0)
                import shutil
                with (OUT / 'frozen_weights.pt').open('xb') as dst:
                    shutil.copyfileobj(source, dst)
                break
        time.sleep(1)
    dev = json.loads(first)
    save('frozen_dev_report.json', dev)
    manifest = json.loads((MAIN / 'dataset_manifest.json').read_text())
    assert sha(MAIN / 'train_streams.json.gz') == manifest['files']['train_streams.json.gz']
    streams = initialize(json.load(gzip.open(MAIN / 'train_streams.json.gz', 'rt', encoding='utf-8')))
    by_origin = defaultdict(list)
    for stream in streams:
        eligible = sum(max(0, p['end']-p['start']-2) for p in stream['segments'])
        if 20 <= eligible <= 4000:
            by_origin[stream['domain']].append((eligible, stream))
    selected = []
    sample = {}
    for origin, candidates in sorted(by_origin.items()):
        candidates.sort(key=lambda pair: hashlib.sha256(('train-fit-v1:' + pair[1]['key']).encode()).digest())
        count = 0
        chosen = []
        for eligible, stream in candidates:
            if len(chosen) >= 12 or (chosen and count + eligible > 7000):
                continue
            chosen.append(stream)
            count += eligible
        selected.extend(chosen)
        sample[origin] = dict(streams=len(chosen), approximate_unmasked_points=count, keys=[s['key'] for s in chosen])
    model = MalbertTurnStudent(AlbertConfig.from_pretrained(MAIN / 'pretrained', local_files_only=True, token=False), sep_token_id=3)
    model.load_state_dict(weights, strict=True)
    del weights
    model.cuda().eval()
    tok = AutoTokenizer.from_pretrained(MAIN / 'tokenizer', local_files_only=True, token=False)
    predict = predictor(model, tok)
    runner = StreamOrder(selected, np.array([2*i+1 for i in range(len(selected))]), lanes=32)
    counts = defaultdict(Counter)
    domains = defaultdict(Counter)
    n = 0
    nll = defaultdict(float)
    true_prob = defaultdict(float)
    while True:
        rows, states = runner.rows()
        if not rows:
            break
        probs, seqs = predict(rows)
        for row, p in zip(rows, probs):
            if not row['scored']:
                continue
            label = bool(row['label'])
            predicted = p >= dev['dev']['threshold']
            counts[row['language']][kind(label, predicted)] += 1
            domains[(row['origin'], row['language'])][kind(label, predicted)] += 1
            key = row['language'] + (':EOT' if label else ':NEOT')
            import math
            nll[key] += -math.log(max(1e-12, p if label else 1-p))
            true_prob[key] += p if label else 1-p
        runner.advance(states, probs, dev['dev']['threshold'])
        n += len(rows)
        if n % 1000 < len(rows):
            save('status.json', dict(stage='MEASURING_TRAIN_FIT', inferences=n, checkpoint_step=dev['step']))
    result = dict(complete=True, checkpoint_step=dev['step'], checkpoint_branch=dev['branch_step'],
                  weights_sha256=sha(OUT / 'frozen_weights.pt'), threshold=dev['dev']['threshold'],
                  train_sample=sample, train_by_language={l:metric(c) for l,c in counts.items()},
                  train_by_domain=[dict(origin=d, language=l, **metric(c)) for (d,l),c in domains.items()],
                  dev_by_language=dev['dev']['by_language'], dev_by_domain=dev['dev']['by_domain'],
                  inferences=n, sum_log_loss_by_class=dict(nll), sum_correct_label_probability=dict(true_prob),
                  fixed_sample_chosen_without_model_scores=True, lookahead_words=2,
                  gold_or_speaker_reset=False, conflict_masks_applied=True,
                  scope='Sampled TRAIN fit, same weights/threshold/predicted-reset encoding as saved DEV. Selected by source hash within length limits, not all TRAIN or unbiased overall error estimate.',
                  training_modified=False, test_scored=False)
    save('report.json', result)
    save('status.json', dict(stage='COMPLETE', inferences=n))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    os.nice(10)
    try:
        run()
    except Exception as error:
        save('error.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
