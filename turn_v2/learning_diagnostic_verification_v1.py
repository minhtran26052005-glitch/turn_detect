"""Audit the completed learning diagnostic without selecting or training a model."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import gc
import gzip
import hashlib
import json
import math

import numpy as np

from .conflict_weighted_learning_diagnostic_v2 import (
    OUT, MAIN, ROOT, batch, fixed_metrics, new_model, parent_identity, reserve, save,
)
from .uploaded_vi_en_data_v5 import StreamOrder, initialize, sha
from .tiny_compression_v1.common import encode_pair


def check_metrics(value):
    tp, tn, fp, fn = (value[key] for key in ('tp', 'tn', 'fp', 'fn'))
    assert value['samples'] == tp + tn + fp + fn
    assert value['actual_eot'] == tp + fn
    expected = dict(eot_precision=tp/(tp+fp) if tp+fp else None,
                    eot_recall=tp/(tp+fn) if tp+fn else None,
                    eot_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    for key, number in expected.items():
        assert value[key] is None if number is None else abs(value[key]-number) < 1e-12


def population(streams):
    counts = defaultdict(Counter)
    for stream in streams:
        for segment in stream['segments']:
            if not segment.get('punctuation_present', True):
                continue
            for boundary in range(segment['start']+1, segment['end']-1):
                if boundary not in stream['_unknown']:
                    counts[segment['language']]['samples'] += 1
                    counts[segment['language']]['actual_eot'] += int(boundary in stream['_eots'])
    return counts


def verify_trace(path, streams, tokenizer, threshold, model):
    import torch
    runner = StreamOrder(streams, np.asarray([2*i+1 for i in range(len(streams))]), lanes=64)
    count = 0
    parity = []
    maximum_delta = 0.
    with gzip.open(path, 'rt', encoding='utf-8') as trace:
        while True:
            rows, states = runner.rows()
            if not rows:
                break
            probs = []
            for row in rows:
                item = json.loads(next(trace))
                source = streams[row['stream']]
                boundary = row['boundary']
                assert 0 <= row['start'] < boundary
                assert row['context'] == ' '.join(source['words'][row['start']:boundary])
                assert row['text'] == ' '.join(source['words'][boundary:boundary+2])
                assert any(segment['start'] < boundary and boundary+2 <= segment['end'] for segment in source['segments'])
                assert row['label'] == int(boundary in source['_eots'])
                assert row['scored'] == (boundary not in source['_unknown'])
                sequence = encode_pair(tokenizer, row['context'], row['text'], 256)['input_ids']
                for name, expected in [('stream', row['stream']), ('boundary', row['boundary']),
                                       ('context_start', row['start']), ('label', row['label']),
                                       ('scored', row['scored']), ('language', row['language'])]:
                    assert item[name] == expected
                assert hashlib.sha256(np.asarray(sequence, dtype='<u4').tobytes()).hexdigest() == item['encoded_sha256']
                probability = item['p_eot']
                assert math.isfinite(probability) and 0 <= probability <= 1
                assert item['predicted_eot'] == (probability >= threshold)
                if len(parity) < 16 and count % 100 == 0:
                    ids, mask, _ = batch([sequence])
                    with torch.inference_mode():
                        canonical = float(model(ids, mask).float().softmax(-1)[0, 1])
                    maximum_delta = max(maximum_delta, abs(canonical-probability))
                    assert (canonical >= threshold) == item['predicted_eot']
                    parity.append(count)
                probs.append(probability)
                count += 1
            runner.advance(states, probs, threshold)
        assert next(trace, None) is None
    assert len(parity) == 16 and maximum_delta < 2e-5
    return dict(inferences=count, all_inputs_labels_encoded_hashes_and_predicted_resets_verified=True,
                canonical_single_input_cases=len(parity), max_probability_delta=maximum_delta,
                trace_sha256=sha(path))


def run():
    import torch
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    utils.is_torchvision_available = imports.is_torchvision_available = lambda: False
    utils.is_librosa_available = imports.is_librosa_available = lambda: False
    from transformers import AutoTokenizer
    from .uploaded_vi_en_evaluation_v5 import predictor, rollout
    assert not (OUT/'completion_verification.json').exists()
    reserve()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.25)
    report = json.loads((OUT/'report.json').read_text())
    assert report['complete'] and report['sanity']['memorization_passed']
    parent = torch.load(OUT/'frozen/full_resume.pt', map_location='cpu', weights_only=False, mmap=True)
    assert sha(OUT/'frozen/full_resume.pt') == report['parent_sha256']
    assert sha(MAIN/'dataset_manifest.json') == parent['fingerprint']['dataset_manifest_sha256']
    assert sha(MAIN/'train_streams.json.gz') == parent['fingerprint']['train_streams_sha256']
    for name, digest in parent['fingerprint']['code_sha256'].items():
        assert sha(ROOT/'turn_v2'/name) == digest
    tokenizer = AutoTokenizer.from_pretrained(MAIN/'tokenizer', local_files_only=True, token=False)
    streams = initialize(json.load(gzip.open(MAIN/'train_streams.json.gz', 'rt', encoding='utf-8')))
    sample = json.loads((OUT/'fixed_train_sample.json').read_text())
    assert len(sample) == len({row['encoded_sha256'] for row in sample}) == 1000
    assert Counter((row['language'], row['label']) for row in sample) == Counter({('vi', 0):375, ('vi', 1):125, ('en', 0):375, ('en', 1):125})
    for row in sample:
        source = streams[row['stream']]
        boundary = row['boundary']
        assert boundary not in source['_unknown'] and boundary not in source['conflict_mask']['2']
        assert row['label'] == int(boundary in source['_eots'])
        assert row['context'] == ' '.join(source['words'][row['context_start']:boundary])
        assert row['future'] == ' '.join(source['words'][boundary:boundary+2])
        assert any(segment['start'] < boundary and boundary+2 <= segment['end'] for segment in source['segments'])
        assert encode_pair(tokenizer, row['context'], row['future'], 256)['input_ids'] == row['input_ids']
        assert hashlib.sha256(np.asarray(row['input_ids'], dtype='<u4').tobytes()).hexdigest() == row['encoded_sha256']
    del streams
    sanity = torch.load(OUT/'sanity/latest_resume.pt', map_location='cpu', weights_only=False, mmap=True)
    assert sanity['step'] == report['sanity']['steps']
    assert len(sanity['optimizer']['state']) == 27
    assert {int(value['step']) for value in sanity['optimizer']['state'].values()} == {sanity['step']}
    model = new_model()
    model.load_state_dict(sanity['model'], strict=True)
    measured = fixed_metrics(model, sample)
    assert measured == report['sanity']['final']
    del model, sanity
    gc.collect()
    torch.cuda.empty_cache()
    calibration = initialize(json.load(gzip.open(MAIN/'dev_calibration.json.gz', 'rt', encoding='utf-8')))
    monitor = initialize(json.load(gzip.open(MAIN/'dev_monitor.json.gz', 'rt', encoding='utf-8')))
    populations = dict(calibration=population(calibration), monitor=population(monitor))
    proofs = {}
    for item in report['reports']:
        name = item['arm']
        save('verification_status.json', dict(stage='VERIFYING_MONITOR', arm=name))
        weights = OUT/name/'weights.pt'
        assert sha(weights) == item['weights_sha256']
        state = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
        assert all(bool(torch.isfinite(value).all()) for value in state.values())
        assert sum(value.numel() for value in state.values()) == 11941122
        if name == 'baseline':
            assert all(torch.equal(state[key], parent['model'][key]) for key in state)
        else:
            resume = torch.load(OUT/name/'latest_resume.pt', map_location='cpu', weights_only=False, mmap=True)
            assert resume['step'] == 3000 and len(resume['optimizer']['state']) == 27
            assert {int(value['step']) for value in resume['optimizer']['state'].values()} == {parent['step']+3000}
            assert resume['parent_fingerprint'] == parent['fingerprint']
            assert np.array_equal(resume['order'], parent['order'])
            assert len(resume['stream_state']['slots']) <= 16
            assert all(torch.equal(state[key], resume['model'][key]) for key in state)
            assert all(key in resume for key in ['torch_rng','cuda_rng','numpy_rng','python_rng','sampler_rng'])
            assert resume['lr'] == {'lr4e6':4e-6, 'lr1e5':1e-5}[name]
            assert all(group['lr'] == resume['lr'] and group['weight_decay'] == .01 for group in resume['optimizer']['param_groups'])
            assert all(bool(torch.isfinite(value).all()) for entry in resume['optimizer']['state'].values()
                       for value in entry.values() if isinstance(value, torch.Tensor))
            del resume
        for curve in item['calibration_curve']:
            for language, metrics in curve['by_language'].items():
                check_metrics(metrics)
                for key in ['samples', 'actual_eot']:
                    assert metrics[key] == populations['calibration'][language][key]
        eligible = [curve for curve in item['calibration_curve'] if all((metrics['eot_precision'] or 0) >= .5 for metrics in curve['by_language'].values())]
        selected = max(eligible, key=lambda curve:min(metrics['eot_recall'] or 0 for metrics in curve['by_language'].values())) if eligible else max(item['calibration_curve'], key=lambda curve:min(metrics['eot_precision'] or 0 for metrics in curve['by_language'].values()))
        assert selected['threshold'] == item['selected_threshold']
        assert selected['by_language'] == item['calibration_selected']
        assert item['test_used'] is False and item['monitor_used_to_select_threshold'] is False
        model = new_model()
        model.load_state_dict(state, strict=True)
        model.eval()
        trace = OUT/name/'verification_monitor.jsonl.gz'
        actual = rollout(predictor(model, tokenizer), monitor, item['selected_threshold'], str(trace))
        assert actual == item['monitor']
        for language, metrics in actual['by_language'].items():
            check_metrics(metrics)
            for key in ['samples', 'actual_eot']:
                assert metrics[key] == populations['monitor'][language][key]
        proofs[name] = verify_trace(trace, monitor, tokenizer, item['selected_threshold'], model)
        del model, state
        gc.collect()
        torch.cuda.empty_cache()
    proof = dict(complete=True, checked_utc=datetime.now(timezone.utc).isoformat(),
        report_sha256=sha(OUT/'report.json'), fixed_sample_all_1000_source_labels_and_encodings_verified=True,
        train_provenance='Pinned TRAIN file and dataset manifest SHA256 verified against the frozen full checkpoint',
        sanity_saved_model_metrics_exactly_reproduced=True, paired_full_checkpoints_27_adamw_rng_order_verified=True,
        monitor_full_rerollouts_exactly_reproduced=proofs, calibration_only_selection_recomputed=True,
        main_identity=parent_identity(), test_used=False, source_and_code_fingerprints_unchanged=True)
    save('completion_verification.json', proof)
    save('verification_status.json', dict(stage='COMPLETE'))
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    run()
