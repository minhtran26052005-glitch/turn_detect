"""Bounded context-only ASR A/B, warm-started from the protected paused v7 pin.

The existing effective16 training engine is reused without editing its source.
Only the causal encoder history policy changes between the two NEW arms.
"""
from collections import Counter, defaultdict
from pathlib import Path
import copy
import fcntl
import gc
import gzip
import hashlib
import json
import math
import os
import signal
import traceback
import numpy as np
from . import asr_only_adaptation_20261107_v1 as old
from . import asr_effective_batch16_20261107_v1 as engine
from .uploaded_vi_en_data_v5 import StreamOrder, initialize, gaps
from .tiny_compression_v1.common import encode_pair

ROOT = old.ROOT
MAIN = old.MAIN
DATA = old.OUT
OUT = ROOT / 'outputs/context_window_ab_20261108_v1'
ARMS = ('predicted_reset', 'source_window')
PARENT_STEP = 3731173
WINDOW_WORDS = 256
STOP = False
sha = old.sha


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def status(stage, **values):
    from datetime import datetime, timezone
    raw = Path('/proc/self/stat').read_text()
    values.update(stage=stage, pid=os.getpid(), start_ticks=raw[raw.rfind(')')+2:].split()[19],
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  at_utc=datetime.now(timezone.utc).isoformat())
    save('status.json', values)
    print(json.dumps(values), flush=True)


def guard():
    old.parent_paused(); old.reserve()
    if STOP or (OUT / 'STOP_REQUESTED').exists():
        engine.STOP = True


def context_start(mode, boundary, decoder_start):
    return decoder_start if mode == 'predicted_reset' else max(0, boundary-WINDOW_WORDS)


class ContextAccumulator(engine.Accumulator):
    mode = 'predicted_reset'

    def next(self):
        import torch
        from torch.nn import functional as F
        if self.device == 'cuda' and self.updates == 100:
            p = engine.OUT
            if (p/'frozen/first100_resume.pt').exists() and not (p/'first_training_verification.json').exists():
                verify_first(self.mode)
        if not self.pending:
            rows, states = self.r.rows()
            if not rows:
                return None
            assert len(states) == len(self.r.slots)
            self.pending = list(range(len(states)))
        rows, indices, valid = [], [], 0
        for idx in self.pending:
            state = self.r.slots[idx]
            s = self.r.streams[state['stream']]
            b, k = state['b'], state['k']
            seg = s['segments'][state['segment']]
            scored = b not in s['_unknown'] and b not in s.get('conflict_mask', {}).get(str(k), [])
            start = context_start(self.mode, b, state['start'])
            rows.append(dict(context=' '.join(s['words'][start:b]), text=' '.join(s['words'][b:b+k]),
                             label=int(b in s['_eots']), scored=scored, language=seg['language'],
                             origin=s['domain'], stream=state['stream'], boundary=b, k=k, start=start))
            indices.append(idx); valid += int(scored)
            if valid == 16-self.count:
                break
        seqs = [encode_pair(self.t, r['context'], r['text'], 256)['input_ids'] for r in rows]
        width = max(map(len, seqs))
        ids = torch.zeros((len(rows), width), dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(seqs):
            ids[i, :len(seq)] = torch.tensor(seq, device=self.device)
            mask[i, :len(seq)] = 1
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.device == 'cuda'):
            logits = self.m(ids, mask)
        logits = logits.float()
        probs = logits.detach().softmax(-1)[:, 1].cpu().tolist()
        scored = torch.tensor([r['scored'] for r in rows], dtype=torch.bool, device=self.device)
        window = self.updates+1
        if valid:
            labels = torch.tensor([r['label'] for r in rows], device=self.device)
            loss = F.cross_entropy(logits[scored], labels[scored], weight=logits.new_tensor([1., 2.]), reduction='sum')/16
            assert torch.isfinite(loss)
            loss.backward(); self.count += valid
            self.weight += sum(1+r['label'] for r in rows if r['scored'])
        self.r.advance([self.r.slots[i] for i in indices], probs, .1)
        self.pending = self.pending[len(indices):]
        updated = self.flush() if self.count == 16 else False
        return rows, seqs, probs, window, updated


def fixture():
    """New context-policy fixture, including interrupted accumulated gradients."""
    import torch
    class Tok:
        cls_token_id = 1; sep_token_id = 3
        def encode(self, text, add_special_tokens=False):
            return [4+int(w[1:]) for w in text.split()]
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(40, 8)
            self.drop = torch.nn.Dropout(.2)
            self.head = torch.nn.Linear(8, 2)
        def forward(self, ids, mask):
            return self.head(self.drop((self.emb(ids)*mask[:, :, None]).sum(1)/mask.sum(1)[:, None]))
    streams = initialize([dict(words=['w'+str(j) for j in range(1, 20)], eots=[4, 8, 15, 19],
                                unknown=[2], conflict_mask={'1': [6], '2': []}, domain='fixture',
                                segments=[dict(start=0, end=19, language='vi')]) for _ in range(3)])
    torch.manual_seed(20261108)
    initial = copy.deepcopy(Tiny().state_dict()); rng = torch.get_rng_state().clone()
    order = np.array([1, 4, 0, 3, 5, 2])
    proofs = {}
    for mode in ARMS:
        def run(interrupt):
            m = Tiny(); m.load_state_dict(initial)
            o = torch.optim.AdamW(m.parameters(), lr=4e-6, weight_decay=.01)
            torch.set_rng_state(rng)
            r = StreamOrder(streams, order, lanes=2)
            a = ContextAccumulator(m, o, r, Tok(), 'cpu'); a.mode = mode
            resumed = False; different = 0; seen = 0
            while True:
                z = a.next()
                if z is None:
                    break
                for row in z[0]:
                    if mode == 'source_window' and row['boundary'] > 5:
                        different += row['start'] == 0
                seen += sum(row['scored'] for row in z[0])
                if interrupt and a.count and a.updates >= 1 and not resumed:
                    snap = copy.deepcopy(dict(model=m.state_dict(), opt=o.state_dict(), r=r.state(), a=a.state(), rng=torch.get_rng_state()))
                    m = Tiny(); m.load_state_dict(snap['model'])
                    o = torch.optim.AdamW(m.parameters(), lr=4e-6, weight_decay=.01); o.load_state_dict(snap['opt'])
                    r = StreamOrder(streams, order, snap['r'], lanes=2)
                    a = ContextAccumulator(m, o, r, Tok(), 'cpu', snap['a']); a.mode = mode
                    torch.set_rng_state(snap['rng']); resumed = True
            a.flush()
            return dict(model=m.state_dict(), opt=o.state_dict(), r=r.state(), a=a.state(), rng=torch.get_rng_state(), seen=seen), resumed, different
        uninterrupted, _, count = run(False)
        resumed, did_resume, _ = run(True)
        assert did_resume and old.equal(uninterrupted, resumed)
        if mode == 'source_window':
            assert count > 0
        proofs[mode] = dict(mid_gradient_resume_bitwise_equal=True, seen=uninterrupted['seen'])
    # Same source prefix is invariant to decoder reset history for treatment.
    assert context_start('source_window', 300, 299) == context_start('source_window', 300, 1) == 44
    assert context_start('predicted_reset', 300, 299) == 299
    save('technical_fixture_verification.json', dict(complete=True, technical_only=True, arms=proofs,
                                                    source_window_independent_of_decoder_resets=True))


def prepare():
    import torch
    guard()
    cp = json.loads((old.BASE/'checkpoint_receipt.json').read_text())
    assert sha(MAIN/'runs/finetune/latest_resume.pt') == cp['full_checkpoint_sha256']
    manifest = json.loads((DATA/'dataset_manifest.json').read_text())
    assert sha(DATA/'train_streams.json.gz') == manifest['train_sha256']
    assert manifest['eligible_points_per_epoch'] == 333315 and manifest['streams'] == 210
    for name in ('train_streams.json.gz', 'dev_calibration.json.gz', 'dev_monitor.json.gz'):
        target = OUT/name
        if not target.exists():
            os.link(DATA/name, target)
    assert all(sha(OUT/name) == value for name, value in manifest['dev_files'].items())
    modules = ['context_window_ab_20261108_v1.py', 'asr_effective_batch16_20261107_v1.py',
               'asr_only_adaptation_20261107_v1.py', 'uploaded_vi_en_data_v4.py', 'uploaded_vi_en_data_v5.py',
               'uploaded_vi_en_sentence_mix_evaluation_v7.py', 'malbert_turn_student_v1.py', 'tiny_compression_v1/common.py']
    protocol = dict(version=1, parent_step=PARENT_STEP, parent_full_sha256=cp['full_checkpoint_sha256'],
                    parameters=11941122, arms=list(ARMS), only_intervention='Encoder context policy in TRAIN and DEV',
                    control_context='Since most recent predicted EOT; original encode_pair token limit',
                    treatment_context='Last up to 256 source words, independent of predicted cuts; encode_pair newest tokens, max256 total',
                    future='1/2 words within actual source turn, unchanged; DEV k2 only',
                    points=333315, tasks=420, budget_epochs_per_arm=1, effective_scored_samples=16,
                    expected_updates_per_arm=20833, loss='weighted CE NEOT1/EOT2, sum weights normalization',
                    lr=4e-6, wd=.01, training_reset_threshold=.1, cpu_threads=2, gpu_fraction=.5,
                    calibration_grid=[.00001,.00003,.0001,.0003,.001,.003,.01,.02,.03,.05,.075,.1,.2,.3,.4,.5,.6,.7,.8,.9],
                    selection='Maximum worst-language F1, then sum F1, then first ascending threshold; calibration only',
                    primary='Treatment minus control worst-language overall DEV F1',
                    uncertainty='2000 paired video/dialogue-family bootstrap replicates, fixed seed20261108, percentile95CI',
                    exploratory_promising_rule=dict(min_primary_gain=.03, lower_primary_ci_above=0,
                        max_overall_language_f1_drop=.02, max_overall_language_precision_drop=.03,
                        max_fp_increase_per_1000_scored=5, max_domain_language_f1_drop=.03,
                        domain_guard_min_actual_eot=50),
                    no_test=True, no_default_change=True, no_automatic_adoption_or_extension=True,
                    context_window_cache='Treatment calibration predictions may be reused only for identical encoded inputs; per-threshold decisions/metrics audited',
                    scope='Exploratory single-seed mechanism comparison on previously viewed punctuation-silver DEV, not sealed confirmation or human/audio/MT quality',
                    source_hashes={name: sha(ROOT/'turn_v2'/name) for name in modules},
                    train_sha256=manifest['train_sha256'], dev_files=manifest['dev_files'])
    if (OUT/'locked_protocol.json').exists():
        assert json.loads((OUT/'locked_protocol.json').read_text()) == protocol
    else:
        save('locked_protocol.json', protocol)
    save('dataset_manifest.json', manifest)
    parent = torch.load(MAIN/'runs/finetune/latest_resume.pt', map_location='cpu', weights_only=False, mmap=True)
    assert parent['step'] == PARENT_STEP and len(parent['optimizer']['state']) == 27
    for name, value in parent['fingerprint']['code_sha256'].items():
        assert sha(ROOT/'turn_v2'/name) == value
    sampler = np.random.default_rng(); sampler.bit_generator.state = copy.deepcopy(parent['sampler_rng'])
    order = sampler.permutation(420)
    for mode in ARMS:
        directory = OUT/mode
        (directory/'frozen').mkdir(parents=True, exist_ok=True)
        initial = directory/'frozen/initial_resume.pt'
        if not initial.exists():
            child = dict(parent)
            child.update(parent_step=PARENT_STEP, parent_fingerprint=parent['fingerprint'],
                         step=PARENT_STEP, branch_step=0, epoch=0, order=order, cursor=0, stream_state=None,
                         sampler_rng=sampler.bit_generator.state, fingerprint=dict(protocol, context_mode=mode),
                         accumulator=None, scored_seen=0, seen_by_key={}, training_trace=[], threshold=.1)
            old.atomic_torch(directory/'runs/finetune/latest_resume.pt', child)
            os.link(directory/'runs/finetune/latest_resume.pt', initial)
        for name in ('train_streams.json.gz', 'dev_calibration.json.gz', 'dev_monitor.json.gz'):
            if not (directory/name).exists():
                os.link(OUT/name, directory/name)
    a = torch.load(OUT/ARMS[0]/'frozen/initial_resume.pt', map_location='cpu', weights_only=False, mmap=True)
    b = torch.load(OUT/ARMS[1]/'frozen/initial_resume.pt', map_location='cpu', weights_only=False, mmap=True)
    fields = ['model', 'optimizer', 'torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng', 'order', 'sampler_rng']
    assert all(old.equal(a[k], b[k]) for k in fields)
    assert all(old.equal(a[k], parent[k]) for k in fields[:6])
    save('migration_receipt.json', dict(complete=True, same_initial_fields=fields, parent_preserved_exact=fields[:6],
             explicit_new_asr_order_and_lanes=True, parent_full_sha256=cp['full_checkpoint_sha256'],
             initial_sha256={mode: sha(OUT/mode/'frozen/initial_resume.pt') for mode in ARMS},
             no_deleted_checkpoint_restored=True, no_base_reset=True))
    streams = initialize(json.load(gzip.open(OUT/'train_streams.json.gz', 'rt', encoding='utf-8')))
    counts = Counter()
    for s in streams:
        for boundary, k, y, lang in gaps(s):
            if boundary not in s.get('conflict_mask', {}).get(str(k), []):
                counts[f'{lang}:k{k}:{y}'] += 1
    assert dict(counts) == {k: v for k, v in manifest['counts'].items() if ':k' in k}
    save('independent_preparation_verification.json', dict(complete=True, eligible_points=sum(counts.values()),
         by_key=dict(counts), unchanged_train_dev_bytes=True, source_exclusions_inherited_from_verified_manifest=True,
         no_test_scoring=True))
    return manifest, protocol


def verify_first(mode):
    import torch
    from transformers import AutoTokenizer
    directory = OUT/mode
    first = torch.load(directory/'frozen/first100_resume.pt', map_location='cpu', weights_only=False, mmap=True)
    initial = torch.load(directory/'frozen/initial_resume.pt', map_location='cpu', weights_only=False, mmap=True)
    tok = AutoTokenizer.from_pretrained(MAIN/'tokenizer', local_files_only=True, token=False)
    streams = initialize(json.load(gzip.open(OUT/'train_streams.json.gz', 'rt', encoding='utf-8')))
    expected = {(i, k): [(b, seg['language']) for seg in s['segments'] if seg.get('punctuation_present', True)
                         for b in range(seg['start']+1, seg['end']-k+1)]
                for i, s in enumerate(streams) for k in (1, 2)}
    offset, starts, counts = defaultdict(int), defaultdict(int), Counter()
    changed_contexts = 0
    for row in first['training_trace']:
        i, k = row['stream'], row['k']; s = streams[i]; key = (i, k)
        boundary, language = expected[key][offset[key]]; offset[key] += 1
        assert boundary == row['boundary']
        start = starts[key] if mode == 'predicted_reset' else max(0, boundary-256)
        assert row['start'] == start
        changed_contexts += start != starts[key]
        label = int(boundary in s['_eots'])
        scored = boundary not in s['_unknown'] and boundary not in s.get('conflict_mask', {}).get(str(k), [])
        assert row['label'] == label and row['scored'] == scored
        seq = encode_pair(tok, ' '.join(s['words'][start:boundary]), ' '.join(s['words'][boundary:boundary+k]), 256)['input_ids']
        assert hashlib.sha256(np.asarray(seq, dtype='<u4').tobytes()).hexdigest() == row['input_ids_sha256']
        if scored:
            counts[row['window']] += 1
        if row['p_eot'] >= .1:
            starts[key] = boundary
    assert sum(counts.values()) == 1600 and len(counts) == 100 and set(counts.values()) == {16}
    assert len(first['optimizer']['state']) == 27
    assert {int(v['step']) for v in first['optimizer']['state'].values()} == {PARENT_STEP+100}
    assert all(torch.isfinite(v).all() for v in first['model'].values())
    assert any(not torch.equal(v, initial['model'][k]) for k, v in first['model'].items())
    if mode == 'source_window':
        assert changed_contexts > 0
    proof = dict(complete=True, arm=mode, encoded_inputs=len(first['training_trace']), scored=1600,
                 context_inputs_differ_from_decoder_history=changed_contexts,
                 same_turn_future_masks_labels_input_hashes_verified=True, all100windows16=True,
                 finite_changed_weights_and_27_optimizer_steps=True)
    save(mode+'/first_training_verification.json', proof)
    return proof


def verify_final(mode, manifest, protocol):
    import torch
    directory = OUT/mode
    x = torch.load(directory/'runs/finetune/latest_resume.pt', map_location='cpu', weights_only=False, mmap=True)
    assert x['fingerprint'] == dict(protocol, context_mode=mode)
    assert x['step'] == 3752006 and x['branch_step'] == 20833 and x['epoch'] == 1
    assert x['scored_seen'] == 333315 and x['cursor'] == 420 and not x['stream_state']['slots']
    assert sorted(map(int, x['order'])) == list(range(420))
    assert x['accumulator']['windows'] == {'16': 20832, '3': 1}
    assert x['accumulator']['count'] == 0 and not x['accumulator']['pending'] and not x['accumulator']['gradients']
    assert x['seen_by_key'] == {k: v for k, v in manifest['counts'].items() if ':k' in k}
    assert len(x['optimizer']['state']) == 27
    assert {int(s['step']) for s in x['optimizer']['state'].values()} == {3752006}
    for state in x['optimizer']['state'].values():
        assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
    assert all(torch.isfinite(v).all() for v in x['model'].values())
    weights = torch.load(directory/'runs/finetune/weights.pt', map_location='cpu', weights_only=True, mmap=True)
    assert old.equal(weights, x['model'])
    for key in ('torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng', 'sampler_rng'):
        assert key in x
    save(mode+'/independent_final_training_verification.json', dict(complete=True, full333315_coverage=True,
         updates=20833, task_order420_cursor_lanes_rng_optimizer_verified=True,
         empty_gradient_and_pending_queue=True, weights_equal_full_resume=True,
         all_training_forwards_audited=False, first100_actual_forwards_audited=True))


def verify_trace(mode, trace, result, streams, tok, predict):
    from .malbert_eval_zip_v2 import metric, kind
    expected = {i: [(b, seg['language']) for seg in s['segments'] if seg.get('punctuation_present', True)
                    for b in range(seg['start']+1, seg['end']-1)] for i, s in enumerate(streams)}
    offsets, decoder = defaultdict(int), defaultdict(int)
    counts, domains = defaultdict(Counter), defaultdict(Counter)
    digest = hashlib.sha256(); cases = {}; n = 0
    with gzip.open(OUT/trace, 'rt', encoding='utf-8') as f:
        for line in f:
            digest.update(line.rstrip('\n').encode()); row = json.loads(line)
            i = row['stream']; s = streams[i]
            boundary, lang = expected[i][offsets[i]]; offsets[i] += 1
            start = decoder[i] if mode == 'predicted_reset' else max(0, boundary-256)
            assert row['boundary'] == boundary and row['language'] == lang and row['context_start'] == start
            label = int(boundary in s['_eots']); scored = boundary not in s['_unknown']
            assert row['label'] == label and row['scored'] == scored
            p = row['p_eot']; assert math.isfinite(p) and 0 <= p <= 1
            pred = p >= result['threshold']; assert pred == row['predicted_eot']
            inp = dict(context=' '.join(s['words'][start:boundary]), text=' '.join(s['words'][boundary:boundary+2]))
            seq = encode_pair(tok, inp['context'], inp['text'], 256)['input_ids']
            assert hashlib.sha256(np.asarray(seq, dtype='<u4').tobytes()).hexdigest() == row['encoded_sha256']
            if scored:
                c = kind(bool(label), pred); counts[lang][c] += 1; domains[s['domain'], lang][c] += 1
                cases.setdefault((lang, c, len(seq) > 32), (inp, p, pred))
            if pred:
                decoder[i] = boundary
            n += 1
    assert all(offsets[i] == len(values) for i, values in expected.items())
    assert n == result['inferences'] and digest.hexdigest() == result['causal_input_trace_digest']
    assert {lang: metric(c) for lang, c in counts.items()} == result['by_language']
    assert [dict(domain=d, language=lang, **metric(c)) for (d, lang), c in sorted(domains.items())] == result['by_domain']
    delta = 0.
    for inp, p, pred in cases.values():
        values, _ = predict([inp]); q = values[0]; delta = max(delta, abs(q-p))
        assert (q >= result['threshold']) == pred
    assert delta < 1e-4
    return dict(complete=True, inferences=n, trace_sha256=sha(OUT/trace),
                source_population_context_future_labels_confusions_input_hashes_verified=True,
                canonical_cases=len(cases), max_probability_delta=delta)


def evaluate(mode, protocol):
    import torch
    from transformers import AutoTokenizer
    from . import uploaded_vi_en_sentence_mix_evaluation_v7 as ev
    from .malbert_calibration_audit_20261107_v1 import scorekey
    tok = AutoTokenizer.from_pretrained(MAIN/'tokenizer', local_files_only=True, token=False)
    m = old.make_model()
    m.load_state_dict(torch.load(OUT/mode/'runs/finetune/weights.pt', map_location='cpu', weights_only=True)); m.eval()
    base_predict = ev.predictor(m, tok)
    calibration = initialize(json.load(gzip.open(OUT/'dev_calibration.json.gz', 'rt', encoding='utf-8')))
    monitor = initialize(json.load(gzip.open(OUT/'dev_monitor.json.gz', 'rt', encoding='utf-8')))
    assert len(calibration) == 119 and len(monitor) == 394
    assert {s['family'] for s in calibration}.isdisjoint({s['family'] for s in monitor})
    cache = {}; unique_calibration_forwards = 0
    def predict(rows):
        nonlocal unique_calibration_forwards
        if mode != 'source_window':
            unique_calibration_forwards += len(rows)
            return base_predict(rows)
        missing = []
        for row in rows:
            s = calibration[row['stream']]; b = row['boundary']
            row['start'] = max(0, b-256); row['context'] = ' '.join(s['words'][row['start']:b])
            key = (row['stream'], b)
            if key not in cache:
                missing.append(row)
        if missing:
            probs, seqs = base_predict(missing); unique_calibration_forwards += len(missing)
            for row, p, seq in zip(missing, probs, seqs):
                cache[row['stream'], row['boundary']] = (p, seq, row['context'], row['text'])
        values, seqs = [], []
        for row in rows:
            p, seq, context, text = cache[row['stream'], row['boundary']]
            assert context == row['context'] and text == row['text']
            values.append(p); seqs.append(seq)
        return values, seqs
    ev.OUT = OUT
    curves, proofs = [], []
    for j, threshold in enumerate(protocol['calibration_grid']):
        guard(); assert not STOP
        name = f'{mode}/calibration_{j:02d}'; trace = name+'.jsonl.gz'
        status('CALIBRATING', arm=mode, index=j+1, total=20, threshold=threshold)
        result = ev.rollout(predict, calibration, threshold, trace)
        save(name+'.json', result)
        proof = verify_trace(mode, trace, result, calibration, tok, base_predict)
        save(name+'_verification.json', proof); curves.append(result); proofs.append(proof)
    selected = max(curves, key=scorekey)
    save(mode+'/selection_receipt.json', dict(threshold=selected['threshold'], calibration=curves,
         selected_before_monitor=True, unique_calibration_model_forwards=unique_calibration_forwards))
    del cache
    def monitor_predict(rows):
        if mode == 'source_window':
            for row in rows:
                b = row['boundary']; row['start'] = max(0, b-256)
                row['context'] = ' '.join(monitor[row['stream']]['words'][row['start']:b])
        return base_predict(rows)
    status('SCORING_DEV_MONITOR', arm=mode, threshold=selected['threshold'])
    trace = mode+'/monitor.jsonl.gz'
    result = ev.rollout(monitor_predict, monitor, selected['threshold'], trace)
    save(mode+'/monitor.json', result)
    proof = verify_trace(mode, trace, result, monitor, tok, base_predict)
    save(mode+'/monitor_verification.json', proof); proofs.append(proof)
    del m; gc.collect(); torch.cuda.empty_cache()
    return dict(selected=result, weights_sha256=sha(OUT/mode/'runs/finetune/weights.pt'),
                unique_calibration_model_forwards=unique_calibration_forwards), proofs


def bootstrap(arms, protocol):
    """Paired family bootstrap of frozen selected-monitor confusion counts."""
    monitor = json.load(gzip.open(OUT/'dev_monitor.json.gz', 'rt', encoding='utf-8'))
    families = sorted({s['family'] for s in monitor}); index = {f: i for i, f in enumerate(families)}
    arrays = []
    for mode in ARMS:
        a = np.zeros((len(families), 2, 3), dtype=np.int64)
        with gzip.open(OUT/mode/'monitor.jsonl.gz', 'rt', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                if not row['scored']:
                    continue
                y, pred = row['label'], row['predicted_eot']
                if not (y or pred):
                    continue
                slot = 0 if y and pred else (1 if pred else 2)
                a[index[monitor[row['stream']]['family']], 0 if row['language']=='vi' else 1, slot] += 1
        arrays.append(a)
    rng = np.random.default_rng(20261108)
    diffs = []
    for _ in range(2000):
        sample = rng.integers(0, len(families), len(families))
        values = []
        for a in arrays:
            tp, fp, fn = a[sample].sum(0).T
            f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.zeros(2, dtype=float), where=(2*tp+fp+fn)>0)
            values.append(f1)
        diffs.append([float(values[1].min()-values[0].min()), *map(float, values[1]-values[0])])
    intervals = np.quantile(np.asarray(diffs), [.025, .975], axis=0)
    a, b = (arms[mode]['selected'] for mode in ARMS)
    primary = min(x['eot_f1'] for x in b['by_language'].values())-min(x['eot_f1'] for x in a['by_language'].values())
    rule = protocol['exploratory_promising_rule']; checks = {
        'primary_gain_at_least_3pp': primary >= rule['min_primary_gain'],
        'primary_paired_family_ci_above_zero': float(intervals[0, 0]) > 0,
    }
    for lang in ('vi', 'en'):
        x, y = a['by_language'][lang], b['by_language'][lang]
        checks[lang+'_f1_guard'] = (y['eot_f1'] or 0) >= (x['eot_f1'] or 0)-.02
        checks[lang+'_precision_guard'] = (y['eot_precision'] or 0) >= (x['eot_precision'] or 0)-.03
        checks[lang+'_false_cut_guard'] = (y['fp']-x['fp'])/x['samples']*1000 <= 5
    domains_a = {(d['domain'], d['language']): d for d in a['by_domain']}
    for y in b['by_domain']:
        x = domains_a[y['domain'], y['language']]
        if x['actual_eot'] >= 50:
            checks[y['domain']+':'+y['language']+':domain_f1_guard'] = (y['eot_f1'] or 0) >= (x['eot_f1'] or 0)-.03
    result = dict(complete=True, families=len(families), replicates=2000, seed=20261108,
                  primary_difference=primary, primary_percentile95_ci=list(map(float, intervals[:, 0])),
                  language_difference_ci={lang: list(map(float, intervals[:, j+1])) for j, lang in enumerate(('vi', 'en'))},
                  checks=checks, exploratory_promising=all(checks.values()), adopted=False,
                  limitations='Single seed, previously viewed silver DEV; bootstrap conditional on these frozen outputs, not fresh sealed confirmation')
    save('paired_family_bootstrap.json', result)
    return result


def run():
    import torch
    import transformers.utils as u
    import transformers.utils.import_utils as im
    u.is_torchvision_available = im.is_torchvision_available = lambda: False
    u.is_librosa_available = im.is_librosa_available = lambda: False
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.5)
    assert not (OUT/'report.json').exists(), 'Never rerun completed experiment'
    status('PREPARING'); manifest, protocol = prepare()
    if not (OUT/'technical_fixture_verification.json').exists():
        fixture()
    engine.Accumulator = ContextAccumulator
    engine.guard = guard
    for mode in ARMS:
        ContextAccumulator.mode = mode
        engine.OUT = OUT/mode
        engine.STOP = False
        status('TRAINING_ARM', arm=mode, points=333315, expected_updates=20833)
        engine.train(manifest, dict(protocol, context_mode=mode))
        verify_first(mode); verify_final(mode, manifest, protocol)
    arms, proofs = {}, []
    for mode in ARMS:
        arm, checks = evaluate(mode, protocol)
        arms[mode] = arm; proofs.extend(checks)
    uncertainty = bootstrap(arms, protocol)
    guard()
    assert sha(MAIN/'runs/finetune/latest_resume.pt') == protocol['parent_full_sha256']
    assert all(sha(ROOT/'turn_v2'/name) == value for name, value in protocol['source_hashes'].items())
    assert all(sha(OUT/name) == value for name, value in protocol['dev_files'].items())
    save('report.json', dict(complete=True, protocol=protocol, arms=arms, uncertainty=uncertainty,
         parent_paused=True, no_test=True, no_default_replacement=True, adopted=False))
    save('completion_verification.json', dict(complete=True, report_sha256=sha(OUT/'report.json'),
         audited_trace_records=sum(p['inferences'] for p in proofs),
         canonical_cases=sum(p['canonical_cases'] for p in proofs),
         max_probability_delta=max(p['max_probability_delta'] for p in proofs),
         all_calibration_and_monitor_records_audited=True, both_full_epochs_verified=True,
         first100_training_inputs_each_arm_audited=True, all_training_forwards_audited=False,
         treatment_calibration_uses_identical_input_cache=True, parent_checkpoint_unchanged=True))
    save('task_state.json', dict(complete=True, reported=False, phase='COMPLETE', parent_paused=True,
                                adopted=False, auto_extend_authorized=False))
    status('COMPLETE', report_sha256=sha(OUT/'report.json'))


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True); os.nice(10)
    def stop(*args):
        global STOP
        STOP = True; engine.STOP = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    with (OUT/'CAMPAIGN_LOCK').open('a') as lock, (MAIN/'CUTOVER_LOCK').open('a') as cutover:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(cutover, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run()
        except InterruptedError as exc:
            status('STOPPED_WITH_CHECKPOINT', reason=str(exc))
        except Exception as exc:
            save('error.json', dict(error=repr(exc), traceback=traceback.format_exc()))
            status('ERROR', error=repr(exc)); raise
