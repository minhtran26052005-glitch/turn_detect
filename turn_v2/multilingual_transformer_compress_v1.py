"""Quality-preserving compression control for the frozen multilingual teacher.

The separate tiny-GRU KD/QAT campaign is allowed to finish unchanged.  This
control keeps the six-layer architecture and quantizes stored matrices only.
It is never described as another distilled tiny model or native INT4 compute.
"""
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np

ROOT = Path('/home/edabk/hoangbpm/turn_detect')
TINY = ROOT/'outputs/multilingual_tiny_compression_20261102_v1'
OUT = ROOT/'outputs/multilingual_transformer_compression_20261102_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def status(stage, **values):
    v = {'stage': stage, 'pid': os.getpid(),
         'at_utc': datetime.now(timezone.utc).isoformat(), **values}
    save('status.json', v)
    print(json.dumps(v), flush=True)


def wait_tiny():
    while not (TINY/'final_report.json').exists():
        if (TINY/'error.json').exists():
            raise RuntimeError('Tiny pipeline error; preserve its state before control')
        status('WAITING_FOR_EXISTING_TINY_CAMPAIGN')
        time.sleep(20)
    result = json.loads((TINY/'final_report.json').read_text())
    if result['accepted']:
        save('final_report.json', {'complete': True, 'skipped': True,
                                  'reason': 'Existing tiny QAT candidate passed; larger control unnecessary'})
        status('SKIPPED_TINY_PASSED')
        return False
    return True


def export_fp32(dev):
    import torch
    import onnx
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    utils.is_torchvision_available = imports.is_torchvision_available = lambda: False
    utils.is_librosa_available = imports.is_librosa_available = lambda: False
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    from safetensors.torch import load_file
    from .tiny_compression_v1.common import encode_pair

    target = OUT/'fp32'
    if (target/'export_receipt.json').exists():
        return target
    assert not target.exists(), 'Preserve incomplete export for inspection; choose a new attempt path'
    target.mkdir()
    config = AutoConfig.from_pretrained(TINY/'teacher', local_files_only=True, token=False)
    config._attn_implementation = 'eager'
    model = AutoModelForSequenceClassification.from_config(config, dtype=torch.float32).eval()
    model.load_state_dict(load_file(str(TINY/'teacher/model.safetensors')), strict=True)
    tok = AutoTokenizer.from_pretrained(TINY/'teacher', local_files_only=True, token=False)

    class Exportable(torch.nn.Module):
        def __init__(self, source):
            super().__init__()
            self.embeddings = source.distilbert.embeddings
            self.layers = source.distilbert.transformer.layer
            self.pre_classifier = source.pre_classifier
            self.classifier = source.classifier

        def forward(self, input_ids, attention_mask):
            hidden = self.embeddings(input_ids)
            batch, length, width = hidden.shape
            for layer in self.layers:
                def heads(value):
                    return value.reshape(batch, length, config.n_heads, config.dim//config.n_heads).transpose(1, 2)
                q = heads(layer.attention.q_lin(hidden)) / ((config.dim//config.n_heads)**.5)
                k = heads(layer.attention.k_lin(hidden))
                v = heads(layer.attention.v_lin(hidden))
                scores = torch.matmul(q, k.transpose(-2, -1))
                scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, torch.finfo(scores.dtype).min)
                a = torch.matmul(scores.softmax(dim=-1), v).transpose(1, 2).reshape(batch, length, width)
                hidden = layer.sa_layer_norm(hidden + layer.attention.out_lin(a))
                hidden = layer.output_layer_norm(hidden + layer.ffn.lin2(torch.nn.functional.gelu(layer.ffn.lin1(hidden))))
            return self.classifier(torch.relu(self.pre_classifier(hidden[:, 0])))

    wrapped = Exportable(model).eval()
    rows = [dev[i] for i in range(0, len(dev), max(1, len(dev)//24))][:24]
    sequences = [encode_pair(tok, r['context'], r['text'], 256)['input_ids'] for r in rows]
    n = max(map(len, sequences))
    ids = torch.zeros((len(rows), n), dtype=torch.long); mask = torch.zeros_like(ids)
    for i, sequence in enumerate(sequences):
        ids[i, :len(sequence)] = torch.tensor(sequence); mask[i, :len(sequence)] = 1
    with torch.inference_mode():
        reference = model(input_ids=ids, attention_mask=mask).logits
        actual = wrapped(ids, mask)
    delta = float((reference-actual).abs().max())
    assert delta < 1e-4, ('Manual-export parity failed', delta)
    with torch.inference_mode():
        torch.onnx.export(wrapped, (ids[:3], mask[:3]), str(target/'model.onnx'),
                          input_names=['input_ids', 'attention_mask'], output_names=['logits'],
                          dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'},
                                        'attention_mask': {0: 'batch', 1: 'sequence'}, 'logits': {0: 'batch'}},
                          opset_version=18, dynamo=False, external_data=False)
    onnx.checker.check_model(str(target/'model.onnx'))
    import onnxruntime as ort
    options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(target/'model.onnx'), options, providers=['CPUExecutionProvider'])
    observed = session.run(['logits'], {'input_ids': ids.numpy(), 'attention_mask': mask.numpy()})[0]
    ort_delta = float(np.max(np.abs(observed-reference.numpy())))
    padded_ids = np.pad(ids.numpy(), ((0, 0), (0, 8))); padded_mask = np.pad(mask.numpy(), ((0, 0), (0, 8)))
    padded = session.run(['logits'], {'input_ids': padded_ids, 'attention_mask': padded_mask})[0]
    padding_delta = float(np.max(np.abs(observed-padded)))
    assert ort_delta < 1e-3 and padding_delta < 1e-3, (ort_delta, padding_delta)
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
        shutil.copyfile(TINY/'teacher'/name, target/name)
    contract = {'max_length': 256, 'threshold': json.loads((TINY/'teacher_dev.json').read_text())['threshold'],
                'padding_to_multiple_of': 8, 'intra_op_threads': 2, 'lookahead_words': [1, 2],
                'normalization': 'NFKC lowercase punctuation-to-space collapse-whitespace',
                'input_format': 'CLS visible buffer SEP next1/2words SEP; left-truncate history only',
                'label2id': {'NEOT': 0, 'EOT': 1}, 'languages': ['vi', 'en', 'zh'],
                'gold_reset_in_inference': False, 'language_id_supplied': False, 'smoke_only': False,
                'architecture': 'Original six-layer multilingual DistilBERT, not tiny GRU',
                'compute_precision': 'FP32', 'test_used': False}
    (target/'turn_detector_config.json').write_text(json.dumps(contract, indent=2)+'\n')
    receipt = {'frozen_teacher_sha256': sha(TINY/'teacher/model.safetensors'),
               'fp32_onnx_sha256': sha(target/'model.onnx'), 'manual_vs_hf_logits_max_delta': delta,
               'onnx_vs_hf_logits_max_delta': ort_delta, 'padding_logits_max_delta': padding_delta,
               'fixtures': len(rows), 'languages': sorted({r['language'] for r in rows})}
    (target/'export_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    del model, wrapped, session
    gc.collect()
    return target


def quantize(source, name, bits, embedding_bits=None, group=32):
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    from .tiny_compression_v1.compress import sort_graph_with_captures
    dest = OUT/name
    if (dest/'quantization_receipt.json').exists():
        return dest
    assert not dest.exists(), 'Preserve incomplete candidate'
    dest.mkdir()
    graph = onnx.version_converter.convert_version(onnx.load(str(source/'model.onnx')), 21)
    tensors = []; nodes = []; converted = []
    for tensor in graph.graph.initializer:
        if tensor.data_type != TensorProto.FLOAT or len(tensor.dims) < 2:
            tensors.append(tensor); continue
        weight = numpy_helper.to_array(tensor)
        current_bits = embedding_bits if embedding_bits and 'embeddings' in tensor.name else bits
        prefix = tensor.name+'_packed'
        if current_bits == 16:
            tensors.append(numpy_helper.from_array(weight.astype(np.float16), prefix))
            nodes.append(helper.make_node('Cast', [prefix], [tensor.name], to=TensorProto.FLOAT))
        else:
            width = weight.shape[-1]; blocks = (width+group-1)//group
            z = np.pad(weight, [(0, 0)]*(weight.ndim-1)+[(0, blocks*group-width)])
            z = z.reshape(*weight.shape[:-1], blocks, group)
            lo, hi = (-8, 7) if current_bits == 4 else (-128, 127)
            scale = np.maximum(np.max(z, axis=-1)/hi, np.min(z, axis=-1)/lo)
            scale = np.maximum(scale, np.finfo(np.float16).tiny).astype(np.float16)
            codes = np.clip(np.rint(z/scale[..., None].astype(np.float32)), lo, hi).astype(np.int8)
            codes = codes.reshape(*weight.shape[:-1], blocks*group)[..., :width]
            if current_bits == 4:
                flat = (codes.reshape(-1).astype(np.uint8) & 15)
                if flat.size % 2: flat = np.pad(flat, (0, 1))
                packed = flat[::2] | (flat[1::2] << 4)
                encoded = TensorProto(); encoded.name = prefix; encoded.data_type = TensorProto.INT4
                encoded.dims.extend(weight.shape); encoded.raw_data = packed.tobytes()
                tensors.append(encoded)
            else:
                tensors.append(numpy_helper.from_array(codes, prefix))
            tensors.append(numpy_helper.from_array(scale, prefix+'_scale'))
            nodes.extend([helper.make_node('DequantizeLinear', [prefix, prefix+'_scale'], [prefix+'_half'],
                                           axis=weight.ndim-1, block_size=group),
                          helper.make_node('Cast', [prefix+'_half'], [tensor.name], to=TensorProto.FLOAT)])
        converted.append({'name': tensor.name, 'shape': list(weight.shape), 'storage_bits': current_bits})
    original = list(graph.graph.node); del graph.graph.node[:]; graph.graph.node.extend(nodes+original)
    del graph.graph.initializer[:]; graph.graph.initializer.extend(tensors)
    sort_graph_with_captures(graph.graph); onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, str(dest/'model.onnx'))
    for filename in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'turn_detector_config.json']:
        shutil.copyfile(source/filename, dest/filename)
    contract = json.loads((dest/'turn_detector_config.json').read_text())
    contract.update(storage_precision=name, compute_precision='FP32',
                    onnx_session_options={'session.disable_quant_qdq': '1'})
    (dest/'turn_detector_config.json').write_text(json.dumps(contract, indent=2)+'\n')
    receipt = {'complete': True, 'method': 'Post-training stored-weight quantization, not QAT or further distillation',
               'group_size': group, 'converted': converted, 'biases_and_normalization': 'FP32',
               'model_bytes': (dest/'model.onnx').stat().st_size, 'model_sha256': sha(dest/'model.onnx'),
               'compute_precision': 'FP32', 'packed_int4_two_values_per_byte': bits == 4}
    (dest/'quantization_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    del graph, tensors, nodes, original
    gc.collect()
    return dest


def run():
    import torch
    torch.set_num_threads(2); os.nice(7)
    if not wait_tiny(): return
    assert not (OUT/'final_report.json').exists(), 'Never rerun a completed control'
    assert shutil.disk_usage(OUT).free > 15*2**30
    from .multilingual_tiny_compress_v3 import make_dev, evaluate, retained, ort_predictor
    from transformers import AutoTokenizer
    dev, streams = make_dev(); teacher = json.loads((TINY/'teacher_dev.json').read_text())
    status('EXPORTING_MATCHED_TRANSFORMER_CONTROL')
    fp = export_fp32(dev)
    tok = AutoTokenizer.from_pretrained(fp, local_files_only=True, token=False)
    candidates = []
    for name, bits, embedding, group in [('int4_g32', 4, None, 32), ('mixed_int4_int8_embedding_g32', 4, 8, 32),
                                          ('int8_g64', 8, None, 64), ('fp16_storage', 16, None, 64)]:
        status('EXPORTING_QUANTIZED_TRANSFORMER', candidate=name)
        path = quantize(fp, name, bits, embedding, group)
        status('EVALUATING_ACTUAL_ONNX_CPU', candidate=name)
        report = evaluate(ort_predictor(path, tok), dev, streams)
        passed, checks = retained(report, teacher)
        contract = json.loads((path/'turn_detector_config.json').read_text())
        contract.update(threshold=report['threshold'], quality_status='DEV_RETENTION_PASS' if passed else 'DEV_RETENTION_FAIL')
        (path/'turn_detector_config.json').write_text(json.dumps(contract, indent=2)+'\n')
        shutil.copyfile(ROOT/'turn_v2/tiny_compression_v1/portable_runner.py', path/'run_model.py')
        (path/'requirements.txt').write_text('numpy>=2.1\nonnxruntime==1.23.2\ntokenizers==0.23.3\n')
        entry = {'candidate': name, 'path': str(path), 'retention_passed': passed, 'retention_by_language': checks,
                 'model_bytes': (path/'model.onnx').stat().st_size, 'model_sha256': sha(path/'model.onnx'),
                 'dev': report, 'storage_bits': bits, 'embedding_bits': embedding or bits}
        candidates.append(entry); save('candidate_results.json', candidates)
        if passed: break
    if not candidates[-1]['retention_passed']:
        save('final_report.json', {'complete': True, 'accepted': False, 'candidates': candidates,
                                   'reason': 'No quantized original-architecture candidate met frozen retention margins'})
        status('COMPLETE_NO_ACCEPTED_MODEL'); return
    winner = candidates[-1]
    status('BENCHMARKING_CPU', candidate=winner['candidate'])
    benchmarks = {}
    for label, path, kind in [('teacher', TINY/'teacher', 'teacher'), ('compressed', Path(winner['path']), 'onnx')]:
        process = subprocess.run([sys.executable, '-m', 'turn_v2.multilingual_tiny_benchmark_v1',
                                  '--kind', kind, '--model', str(path), '--cases', str(TINY/'dev_rows.jsonl')],
                                 cwd=ROOT, capture_output=True, text=True, timeout=300)
        (OUT/('benchmark_'+label+'.log')).write_text(process.stdout+process.stderr)
        assert process.returncode == 0, 'Benchmark failure: '+label
        benchmarks[label] = json.loads(process.stdout.strip().splitlines()[-1])
    save('cpu_benchmark.json', benchmarks)
    package = ROOT/'models/turn_multilingual_transformer_compressed_20261102_v1'
    model = package/'onnx'; model.mkdir(parents=True, exist_ok=False)
    for path in Path(winner['path']).iterdir():
        if path.is_file(): shutil.copyfile(path, model/path.name)
    files = {str(p.relative_to(package)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in model.iterdir() if p.is_file()}
    manifest = {'status': 'DEV_RETENTION_VERIFIED_SILVER_ONLY', 'files': files, 'winner': winner,
                'teacher': json.loads((TINY/'teacher_receipt.json').read_text()),
                'architecture': 'Original six-layer shared DistilBERT preserved; tiny GRU KD/QAT experiment did not retain all languages',
                'method': 'Post-training matrix weight-storage quantization with FP32 computation',
                'source_spoken_v9_used': False, 'test_used': False, 'human_audio_gold': False,
                'deployment_bytes': sum(x['bytes'] for x in files.values()), 'runtime_libraries_included_in_size': False,
                'main_training_interrupted': False, 'cpu_benchmark': benchmarks}
    (package/'model_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (package/'README.md').write_text('Shared VI/EN/ZH retained Transformer compression control. This is not the tiny GRU, not QAT, and not native INT4 compute. See model_manifest.json for actual CPU DEV retention, hashes, input contract, sizes and benchmark. Source722 plus legacyZH, not spokenv9 or independent conversational/audio gold. No active model replaced.\n')
    archive = package.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in package.rglob('*'):
            if p.is_file(): z.write(p, p.relative_to(package.parent))
    with zipfile.ZipFile(archive) as z:
        for name, meta in files.items(): assert hashlib.sha256(z.read(package.name+'/'+name)).hexdigest() == meta['sha256']
    save('final_report.json', {'complete': True, 'accepted': True, 'winner': winner, 'teacher': teacher,
                              'package': str(package), 'archive': str(archive), 'archive_sha256': sha(archive),
                              'manifest_sha256': sha(package/'model_manifest.json'), 'deployment_bytes': manifest['deployment_bytes'],
                              'candidates': candidates, 'cpu_benchmark': benchmarks, 'test_used': False,
                              'tiny_campaign_report': str(TINY/'final_report.json'),
                              'completed_utc': datetime.now(timezone.utc).isoformat()})
    status('COMPLETE', package=str(package), model_bytes=winner['model_bytes'])


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True, parents=True)
    with (OUT/'CAMPAIGN_LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: run()
        except Exception as error:
            save('error.json', {'error': repr(error), 'traceback': traceback.format_exc(),
                                'at_utc': datetime.now(timezone.utc).isoformat()})
            status('ERROR', error=repr(error))
            raise
