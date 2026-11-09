"""A hard <20M shared VI/EN/ZH branch; never fall back to a larger model."""
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np

ROOT = Path('/home/edabk/hoangbpm/turn_detect')
OLD = ROOT/'outputs/multilingual_tiny_compression_20261102_v1'
OUT = ROOT/'outputs/multilingual_under20m_20261103_v1'
PACKAGE = ROOT/'models/turn_multilingual_under20m_20261103_v1'
TEACHER_SHA = '15d4ebfc7fcba7c3c842b7b7815ac309cf349f0436dca69522d9d529295954cc'
TRAIN_SHA = '51cbacc8758a3f3efa5fc2cc2223747bba09cc045ccacca5668deb440b2c3c41'
STOP = False


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def save(name, value):
    p = OUT/name; p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name+'.tmp')
    with t.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')
        f.flush(); os.fsync(f.fileno())
    t.replace(p)


def status(stage, **values):
    v = dict(stage=stage, pid=os.getpid(), at_utc=datetime.now(timezone.utc).isoformat(),
             hard_parameter_limit_exclusive=20_000_000, main_training_interrupted=False, **values)
    save('status.json', v); print(json.dumps(v), flush=True)


def stopped():
    return STOP or (OUT/'STOP_REQUESTED').exists()


def reserve():
    while True:
        hdd = subprocess.run(['findmnt', '-n', '-o', 'SOURCE,OPTIONS', '/mnt/hdd500'],
                             text=True, capture_output=True, check=True).stdout.strip()
        assert hdd.startswith('/dev/sda1 ') and 'rw' in hdd.split()[1].split(','), hdd
        if shutil.disk_usage(OUT).free >= 13*2**30 and shutil.disk_usage('/mnt/hdd500').free >= 13*2**30:
            return
        status('WAITING_FOR_DISK_RESERVE', checkpoint_preserved=True)
        if stopped(): raise InterruptedError('Stop while preserving disk reserve')
        time.sleep(30)


def atomic_torch(p, state):
    import torch
    reserve(); t = p.with_name(p.name+'.tmp'); torch.save(state, t)
    with t.open('rb') as f: os.fsync(f.fileno())
    t.replace(p)


def setup():
    import torch
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    utils.is_torchvision_available = imports.is_torchvision_available = lambda: False
    utils.is_librosa_available = imports.is_librosa_available = lambda: False
    torch.set_num_threads(2); os.nice(6)
    torch.cuda.set_per_process_memory_fraction(.30)
    random.seed(43); np.random.seed(43); torch.manual_seed(43)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def prepare():
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from .compact_multilingual_student_v1 import CompactTurnTransformer
    from .tiny_compression_v1.common import encode_pair
    receipt = json.loads((OLD/'cache/receipt.json').read_text())
    assert receipt['complete'] and receipt['rows'] == 300000
    assert receipt['train_sha256'] == TRAIN_SHA == sha(OLD/'train.jsonl')
    assert receipt['teacher_weights_sha256'] == TEACHER_SHA == sha(OLD/'teacher/model.safetensors')
    for key in ('labels', 'logits', 'hidden'):
        assert sha(OLD/'cache'/(key+'.npy')) == receipt['files'][key]['sha256']
    tok = AutoTokenizer.from_pretrained(OLD/'teacher', local_files_only=True, token=False)
    assert len(tok) == 119547 and tok.pad_token_id == 0
    if not (OUT/'initial_student.pt').exists():
        status('INITIALIZING_UNDER20M_STUDENT')
        model = CompactTurnTransformer(); info = model.initialize_from_teacher(
            load_file(str(OLD/'teacher/model.safetensors')))
        counts = model.parameter_counts(); assert counts['training_total'] < 20_000_000
        atomic_torch(OUT/'initial_student.pt', model.state_dict())
        save('architecture.json', dict(config=model.config, parameters=counts,
             initialization=info, teacher_sha256=TEACHER_SHA,
             original_tokenizer_sha256=sha(OLD/'teacher/tokenizer.json'),
             source_corpus='300000 locked TRAIN rows:100000 per language, source722 plus legacyZH',
             new_spoken_v9_used=False, test_used=False))
        del model; gc.collect()
    model_info = json.loads((OUT/'architecture.json').read_text())
    assert model_info['parameters']['training_total'] < 20_000_000
    cache = OUT/'cache'; cache.mkdir(exist_ok=True)
    labels = np.load(OLD/'cache/labels.npy', mmap_mode='r')
    n = len(labels)
    if not (cache/'receipt.json').exists():
        reserve()
        paths = {'ids': (np.uint32, (n, 256)), 'lengths': (np.uint16, (n,))}
        arrays = {k: np.load(cache/(k+'.npy'), mmap_mode='r+') if (cache/(k+'.npy')).exists()
                  else np.lib.format.open_memmap(cache/(k+'.npy'), mode='w+', dtype=t, shape=s)
                  for k, (t, s) in paths.items()}
        for k, (t, shape) in paths.items():
            assert arrays[k].shape == shape and arrays[k].dtype == t
        progress = json.loads((cache/'progress.json').read_text()) if (cache/'progress.json').exists() else {'rows': 0}
        done = progress['rows']; counts = Counter(progress.get('languages', {}))
        with (OLD/'train.jsonl').open(encoding='utf-8') as f:
            total = 0
            for i, line in enumerate(f):
                total = i+1
                if i < done: continue
                if stopped(): raise InterruptedError('Stop during resumable retokenization')
                row = json.loads(line)
                assert row['split'] == 'train' and 1 <= len(row['text'].split()) <= 2
                assert int(row['label'] == 'EOT') == int(labels[i])
                ids = encode_pair(tok, row['context'], row['text'], 256)['input_ids']
                arrays['ids'][i] = 0; arrays['ids'][i, :len(ids)] = ids
                arrays['lengths'][i] = len(ids); counts[row['language']] += 1
                if (i+1) % 4096 == 0 or i+1 == n:
                    for a in arrays.values(): a.flush()
                    save('cache/progress.json', {'rows': i+1, 'total': n, 'languages': dict(counts)})
                    status('RETOKENIZING_TRAIN_WITH_TEACHER_VOCABULARY', rows=i+1, total=n)
            assert total == n and counts == {'vi': 100000, 'en': 100000, 'zh': 100000}
        save('cache/receipt.json', dict(complete=True, rows=n, train_sha256=TRAIN_SHA,
             teacher_weights_sha256=TEACHER_SHA, tokenizer_sha256=sha(OLD/'teacher/tokenizer.json'),
             source_targets_receipt_sha256=sha(OLD/'cache/receipt.json'),
             ids_sha256=sha(cache/'ids.npy'), lengths_sha256=sha(cache/'lengths.npy'),
             target_alignment_verified_on_all_rows=True, by_language=dict(counts)))
    data = {k: np.load(cache/(k+'.npy'), mmap_mode='r') for k in ('ids', 'lengths')}
    data.update({k: np.load(OLD/'cache'/(k+'.npy'), mmap_mode='r') for k in ('labels', 'logits', 'hidden')})
    assert sha(cache/'ids.npy') == json.loads((cache/'receipt.json').read_text())['ids_sha256']
    return tok, data


def qat(model, group, scales=None):
    import torch
    from torch.nn.utils import parametrize
    from .tiny_compression_v1.qat_student import GroupInt4FakeQuant
    from .tiny_compression_v1.quantize_student4 import quantize_groups
    result = {}
    matrices = [(k, p) for k, p in model.named_parameters() if k.startswith('base.') and p.ndim == 2]
    for key, weight in matrices:
        parent, leaf = key.rsplit('.', 1)
        if scales is None:
            _, value, _ = quantize_groups(weight.detach().cpu().numpy(), group)
            scale = torch.from_numpy(value).to(weight.device)
        else: scale = scales[key].to(weight.device)
        q = GroupInt4FakeQuant(scale, group)
        parametrize.register_parametrization(model.get_submodule(parent), leaf, q)
        result[key] = q
    return result


def latent(model, quantizers):
    if not quantizers: return {k: v.detach().cpu() for k, v in model.state_dict().items()}
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if '.parametrizations.' not in k}
    for key in quantizers:
        parent, leaf = key.rsplit('.', 1)
        state[key] = getattr(model.get_submodule(parent).parametrizations, leaf).original.detach().cpu()
    return state


def fit(name, init, epochs, group, tok, cache, dev, streams, teacher):
    import torch
    from torch.nn import functional as F
    from .compact_multilingual_student_v1 import CompactTurnTransformer
    from .multilingual_tiny_compress_v3 import evaluate, retained, tensor_predictor
    folder = OUT/'runs'/name; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'training_report.json').exists(): return folder
    path = folder/'latest_resume.pt'
    resume = torch.load(path, map_location='cpu', weights_only=False) if path.exists() else None
    model = CompactTurnTransformer().cuda()
    model.load_state_dict(resume['model'] if resume else torch.load(init, map_location='cpu', weights_only=True), strict=True)
    q = qat(model, group, resume['quant_scales'] if resume else None) if group else {}
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4 if name == 'distill' else 2e-5, weight_decay=.01)
    rng = np.random.default_rng(43); n = len(cache['labels']); batch = 32
    epoch = resume['epoch'] if resume else 0; cursor = resume['cursor'] if resume else 0
    step = resume['step'] if resume else 0; order = resume['order'] if resume else rng.permutation(n)
    best = resume['best_score'] if resume else -1.; best_passed = resume['best_passed'] if resume else False
    if resume:
        assert resume['train_sha256'] == TRAIN_SHA and resume['teacher_sha256'] == TEACHER_SHA
        assert resume['architecture'] == model.config
        opt.load_state_dict(resume['optimizer']); rng.bit_generator.state = resume['sampler_rng']
        torch.set_rng_state(resume['torch_rng']); torch.cuda.set_rng_state(resume['cuda_rng'])
        np.random.set_state(resume['numpy_rng']); random.setstate(resume['python_rng'])
    counts = model.parameter_counts(); assert counts['training_total'] < 20_000_000
    cache_sha = sha(OUT/'cache/receipt.json')
    code_sha = {n: sha(ROOT/'turn_v2'/n) for n in
                ('compact_multilingual_student_v1.py', 'multilingual_under20m_distill_v1.py')}
    if resume:
        assert resume['cache_sha256'] == cache_sha and resume['code_sha256'] == code_sha
    history = json.loads((folder/'history.json').read_text()) if (folder/'history.json').exists() else []
    identity = dict(architecture=model.config, parameters=counts, train_sha256=TRAIN_SHA,
                    teacher_sha256=TEACHER_SHA, tokenizer_sha256=sha(OLD/'teacher/tokenizer.json'),
                    source_cache_sha256=cache_sha, code_sha256=code_sha, group_size=group,
                    training_head_deployed=False, epochs=epochs, batch_size=batch, test_used=False)
    save(str(folder.relative_to(OUT)/'identity.json'), identity)
    def checkpoint():
        full = dict(model=latent(model, q), optimizer=opt.state_dict(), epoch=epoch,
                    cursor=cursor, step=step, order=order, sampler_rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                    best_score=best, best_passed=best_passed, train_sha256=TRAIN_SHA,
                    teacher_sha256=TEACHER_SHA, architecture=model.config,
                    cache_sha256=cache_sha, code_sha256=code_sha,
                    quant_scales={k: v.scale.detach().cpu() for k, v in q.items()})
        atomic_torch(path, full)
        save(str(folder.relative_to(OUT)/'resume_receipt.json'), dict(step=step, epoch=epoch,
             cursor=cursor, optimizer_entries=len(opt.state), sampler_length=len(order),
             optimizer_rng_sampler_scales_preserved=True, parameters=counts))
        if not (OUT/'first_checkpoint_verification.json').exists():
            checked = torch.load(path, map_location='cpu', weights_only=False)
            assert checked['step'] > 0 and len(checked['optimizer']['state']) > 0 and len(checked['order']) == n
            save('first_checkpoint_verification.json', dict(run=name, step=step,
                 optimizer_entries=len(checked['optimizer']['state']), parameters=counts,
                 optimizer_rng_sampler_verified=True, checkpoint_sha256=sha(path), torch_load_passed=True))
    while epoch < epochs:
        model.train()
        for start in range(cursor, n, batch):
            if stopped(): checkpoint(); raise InterruptedError('Explicit stop at student checkpoint')
            if step % 100 == 0: reserve()
            indices = order[start:start+batch]; width = int(cache['lengths'][indices].max())
            ids = torch.tensor(np.asarray(cache['ids'][indices, :width], dtype=np.int64), device='cuda')
            lengths = torch.tensor(np.asarray(cache['lengths'][indices], dtype=np.int64), device='cuda')
            mask = (torch.arange(width, device='cuda')[None, :] < lengths[:, None]).long()
            y = torch.tensor(np.asarray(cache['labels'][indices], dtype=np.int64), device='cuda')
            targets = torch.tensor(np.asarray(cache['logits'][indices], dtype=np.float32), device='cuda')
            features = torch.tensor(np.asarray(cache['hidden'][indices], dtype=np.float32), device='cuda')
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=not bool(group)):
                logits, hidden = model(ids, mask, return_features=True)
            logits = logits.float(); hidden = hidden.float()
            ce = F.cross_entropy(logits, y)
            kd = F.kl_div(F.log_softmax(logits/2, -1), F.softmax(targets/2, -1), reduction='batchmean')*4
            mse = F.mse_loss(F.layer_norm(hidden, (768,)), F.layer_norm(features, (768,)))
            loss = .4*ce+.6*kd+.05*mse; assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if group and step == 0:
                gradients = {}
                for key in q:
                    parent, leaf = key.rsplit('.', 1)
                    g = getattr(model.get_submodule(parent).parametrizations, leaf).original.grad
                    gradients[key] = g is not None and bool(torch.isfinite(g).all()) and bool(g.abs().sum() > 0)
                assert all(gradients.values())
                save(str(folder.relative_to(OUT)/'qat_gradient_checks.json'), gradients)
            total = epochs*math.ceil(n/batch)
            lr = (2e-4 if name == 'distill' else 2e-5)*min(1., (step+1)/300)*(.1+.9*.5*(1+math.cos(math.pi*step/total)))
            for pg in opt.param_groups: pg['lr'] = lr
            opt.step(); step += 1; cursor = min(start+batch, n)
            if step % 50 == 0:
                status('QAT_UNDER20M' if group else 'DISTILLING_UNDER20M_TRANSFORMER',
                       run=name, step=step, epoch=epoch+1, epochs=epochs, cursor=cursor,
                       loss=float(loss.detach()), parameters=counts, group_size=group)
            if step == 100 or step % 500 == 0: checkpoint()
        model.eval(); result = evaluate(tensor_predictor(model, tok, 'cuda'), dev, streams)
        passed, checks = retained(result, teacher)
        record = dict(epoch=epoch+1, step=step, dev=result, retention_passed=passed, retention_by_language=checks)
        history.append(record); save(str(folder.relative_to(OUT)/'history.json'), history)
        save('latest_dev.json', dict(run=name, **record))
        if (passed and not best_passed) or (passed == best_passed and result['score'] > best):
            best = result['score']; best_passed = passed
            atomic_torch(folder/'student.pt', latent(model, q))
            if q: atomic_torch(folder/'quant_params.pt', {'group_size': group, 'scales': {k: v.scale.detach().cpu() for k, v in q.items()}})
            save(str(folder.relative_to(OUT)/'best_report.json'), record)
        epoch += 1; cursor = 0; order = rng.permutation(n); checkpoint()
    save(str(folder.relative_to(OUT)/'training_report.json'), dict(complete=True, steps=step,
         selected=json.loads((folder/'best_report.json').read_text()), **identity))
    del model, opt; gc.collect(); torch.cuda.empty_cache()
    return folder


def export_candidate(folder, name, bits, tok, dev, streams, teacher, technical_only=False):
    import torch
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper, TensorProto
    from .compact_multilingual_student_v1 import CompactTurnTransformer, Exportable, export_key_to_train
    from .tiny_compression_v1.qat_student import pack_parameter
    from .tiny_compression_v1.compress import sort_graph_with_captures
    from .tiny_compression_v1.common import encode_pair
    from .multilingual_tiny_compress_v3 import ort_predictor, evaluate, retained
    path = OUT/'candidates'/name; path.mkdir(parents=True, exist_ok=True)
    if (path/'dev_report.json').exists(): return json.loads((path/'dev_report.json').read_text())
    model = CompactTurnTransformer().eval()
    weights = torch.load(folder/'student.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True); counts = model.parameter_counts()
    assert counts['deployment'] < 20_000_000
    quant = torch.load(folder/'quant_params.pt', map_location='cpu', weights_only=True) if bits == 4 else None
    wrapper = Exportable(model).eval()
    rows = [next(r for r in dev if r['language'] == lang) for lang in ('vi', 'en', 'zh')]
    sequences = [encode_pair(tok, r['context'], r['text'], 256)['input_ids'] for r in rows]
    width = max(map(len, sequences)); ids = torch.zeros((3, width), dtype=torch.long); mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences): ids[i, :len(seq)] = torch.tensor(seq); mask[i, :len(seq)] = 1
    with torch.inference_mode():
        reference = model(ids, mask); manual = wrapper(ids, mask)
    assert float((reference-manual).abs().max()) < 1e-4
    float_graph = path/'unpacked_control.onnx'
    with torch.inference_mode():
        torch.onnx.export(wrapper, (ids, mask), str(float_graph), input_names=['input_ids', 'attention_mask'], output_names=['logits'],
             dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'}, 'attention_mask': {0: 'batch', 1: 'sequence'}, 'logits': {0: 'batch'}},
             opset_version=18, dynamo=False, external_data=False, do_constant_folding=False)
    graph = onnx.load(str(float_graph)); converted = []; nodes = []; tensors = []
    if bits != 32:
        graph = onnx.version_converter.convert_version(graph, 21)
        for t in graph.graph.initializer:
            if t.data_type != TensorProto.FLOAT or len(t.dims) < 2:
                tensors.append(t); continue
            a = numpy_helper.to_array(t).copy(); key = export_key_to_train(t.name)
            assert key in weights and tuple(a.shape) == tuple(weights[key].shape), (t.name, key)
            group = quant['group_size'] if quant else 64
            if quant:
                codes, scale, restored = pack_parameter(a, quant['scales'][key].numpy(), group)
            else:
                blocks = (a.shape[-1]+group-1)//group
                z = np.pad(a, [(0,0)]*(a.ndim-1)+[(0,blocks*group-a.shape[-1])]).reshape(*a.shape[:-1], blocks, group)
                scale = np.maximum(np.maximum(z.max(-1)/127, z.min(-1)/-128), np.finfo(np.float16).tiny).astype(np.float16)
                expanded = np.repeat(scale.astype(np.float32), group, -1)[..., :a.shape[-1]]
                codes = np.clip(np.rint(a/expanded), -128, 127).astype(np.int8)
                restored = (codes.astype(np.float32)*expanded).astype(np.float16).astype(np.float32)
            prefix = t.name+'_packed'
            if bits == 4:
                flat = codes.reshape(-1).astype(np.uint8)&15
                if flat.size % 2: flat = np.pad(flat, (0,1))
                packed = flat[::2] | (flat[1::2] << 4)
                encoded = TensorProto(name=prefix, data_type=TensorProto.INT4, dims=a.shape, raw_data=packed.tobytes())
                tensors.append(encoded)
                decoded = np.empty(flat.size, np.int8); decoded[::2] = packed&15; decoded[1::2] = packed>>4
                decoded[decoded>=8] -= 16
                assert np.array_equal(decoded[:codes.size], codes.reshape(-1))
            else: tensors.append(numpy_helper.from_array(codes, prefix))
            tensors.append(numpy_helper.from_array(scale, prefix+'_scale'))
            nodes.extend([helper.make_node('DequantizeLinear', [prefix, prefix+'_scale'], [prefix+'_half'], axis=a.ndim-1, block_size=group),
                          helper.make_node('Cast', [prefix+'_half'], [t.name], to=TensorProto.FLOAT)])
            wrapper.get_parameter(t.name).data.copy_(torch.from_numpy(restored))
            converted.append(dict(name=t.name, shape=list(a.shape), bits=bits))
        original = list(graph.graph.node); del graph.graph.node[:]; graph.graph.node.extend(nodes+original)
        del graph.graph.initializer[:]; graph.graph.initializer.extend(tensors); sort_graph_with_captures(graph.graph)
        if quant: assert {export_key_to_train(x['name']) for x in converted} == set(quant['scales'])
    onnx.checker.check_model(graph, full_check=True); onnx.save(graph, str(path/'model.onnx'))
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 2; opts.inter_op_num_threads = 1
    opts.add_session_config_entry('session.disable_quant_qdq', '1')
    session = ort.InferenceSession(str(path/'model.onnx'), opts, providers=['CPUExecutionProvider'])
    observed = session.run(['logits'], {'input_ids': ids.numpy(), 'attention_mask': mask.numpy()})[0]
    with torch.inference_mode(): reference = wrapper(ids, mask).numpy()
    delta = float(np.max(np.abs(observed-reference)))
    padded = session.run(['logits'], {'input_ids': np.pad(ids.numpy(), ((0,0),(0,8))), 'attention_mask': np.pad(mask.numpy(), ((0,0),(0,8)))})[0]
    padding_delta = float(np.max(np.abs(observed-padded)))
    assert delta < 1e-3 and padding_delta < 1e-3, (delta, padding_delta)
    for filename in ('tokenizer.json', 'tokenizer_config.json'): shutil.copyfile(OLD/'teacher'/filename, path/filename)
    contract = dict(max_length=256, padding_to_multiple_of=8, intra_op_threads=2, lookahead_words=[1,2],
        normalization='NFKC lowercase punctuation-to-space collapse-whitespace',
        input_format='CLS visible buffer SEP next1/2words SEP; left-truncate history only',
        languages=['vi','en','zh'], label2id={'NEOT':0,'EOT':1}, gold_reset_in_inference=False, language_id_supplied=False,
        storage_precision='INT'+str(bits) if bits != 32 else 'FP32', compute_precision='FP32',
        onnx_session_options={'session.disable_quant_qdq':'1'}, parameters=counts['deployment'],
        hard_parameter_limit_exclusive=20_000_000, test_used=False, smoke_only=technical_only)
    save(str(path.relative_to(OUT)/'config.json'), dict(model_type='distilbert', **model.config))
    save(str(path.relative_to(OUT)/'export_receipt.json'), dict(parameters=counts, logits_parity_delta=delta,
         padding_delta=padding_delta, actual_packed_int4=bits == 4, qat_scales_preserved=quant is not None,
         matrices=converted, trained_steps=json.loads((folder/'training_report.json').read_text())['steps']))
    del model, wrapper, session, graph; gc.collect()
    if technical_only:
        result = {'threshold': .5, 'scope': 'Technical fixture only, not quality evaluation'}
        passed, checks = False, {}
    else:
        result = evaluate(ort_predictor(path, tok), dev, streams); passed, checks = retained(result, teacher)
    contract['threshold'] = result['threshold']; contract['quality_status'] = 'SILVER_DEV_RETENTION_PASS' if passed else 'EXPERIMENTAL_RETENTION_FAIL'
    save(str(path.relative_to(OUT)/'turn_detector_config.json'), contract)
    shutil.copyfile(ROOT/'turn_v2/tiny_compression_v1/portable_runner.py', path/'run_model.py')
    (path/'requirements.txt').write_text('numpy>=2.1\nonnxruntime==1.23.2\ntokenizers==0.23.3\n')
    result = dict(candidate=name, path=str(path), parameters=counts['deployment'], parameter_limit_passed=True,
                  storage_bits=bits, model_bytes=(path/'model.onnx').stat().st_size, model_sha256=sha(path/'model.onnx'),
                  dev=result, retention_passed=passed, retention_by_language=checks, test_used=False)
    save(str(path.relative_to(OUT)/'dev_report.json'), result)
    return result


def finalize(candidates, teacher):
    passing = [c for c in candidates if c['retention_passed'] and c['parameters'] < 20_000_000]
    if not passing:
        save('final_report.json', dict(complete=True, accepted=False, candidates=candidates,
             reason='All candidates meet parameter budget, but none retain all three language quality gates',
             parameter_limit_exclusive=20_000_000, test_used=False))
        status('COMPLETE_NO_ACCEPTED_UNDER20M_MODEL'); return
    winner = min(passing, key=lambda c: c['model_bytes'])
    target = PACKAGE/'onnx'; target.mkdir(parents=True, exist_ok=False)
    names = ['model.onnx', 'config.json', 'tokenizer.json', 'tokenizer_config.json',
             'turn_detector_config.json', 'export_receipt.json', 'run_model.py', 'requirements.txt']
    for name in names: shutil.copyfile(Path(winner['path'])/name, target/name)
    files = {n: dict(bytes=(target/n).stat().st_size, sha256=sha(target/n)) for n in names}
    benchmarks = {}
    for name, kind, model in [('teacher','teacher',OLD/'teacher'), ('student','onnx',target)]:
        p = subprocess.run([sys.executable, '-m', 'turn_v2.multilingual_tiny_benchmark_v2', '--kind', kind,
             '--model', str(model), '--cases', str(OLD/'dev_rows.jsonl')], cwd=ROOT, capture_output=True, text=True, timeout=360)
        save('benchmark_'+name+'.json', dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr))
        assert p.returncode == 0
        benchmarks[name] = json.loads(p.stdout.strip().splitlines()[-1])
    manifest = dict(architecture=json.loads((OUT/'architecture.json').read_text()), winner=winner,
        files=files, benchmark=benchmarks, teacher=teacher, parameter_limit_passed=True,
        quality_scope='Synthetic punctuation DEV retention only; not independent human/audio gold',
        int4_compute=False, computation='FP32; packed matrices dequantized', new_spoken_v9_used=False,
        test_used=False, active_model_replaced=False, backed_up_to_drive=False)
    (PACKAGE/'model_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    archive = PACKAGE.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in PACKAGE.rglob('*'):
            if p.is_file(): z.write(p, p.relative_to(PACKAGE.parent))
    with zipfile.ZipFile(archive) as z:
        for n, meta in files.items(): assert hashlib.sha256(z.read(PACKAGE.name+'/onnx/'+n)).hexdigest() == meta['sha256']
    save('final_report.json', dict(complete=True, accepted=True, winner=winner, candidates=candidates,
        package=str(PACKAGE), archive_sha256=sha(archive), manifest_sha256=sha(PACKAGE/'model_manifest.json'),
        benchmark=benchmarks, test_used=False, main_training_interrupted=False))
    status('COMPLETE', parameters=winner['parameters'], storage_bits=winner['storage_bits'], package=str(PACKAGE))


def run():
    setup(); reserve()
    assert not (OUT/'final_report.json').exists(), 'Do not rerun completed branch'
    tok, cache = prepare()
    from .multilingual_tiny_compress_v3 import make_dev
    dev, streams = make_dev(); teacher = json.loads((OLD/'teacher_dev.json').read_text())
    kd = fit('distill', OUT/'initial_student.pt', 8, None, tok, cache, dev, streams, teacher)
    candidates = []
    control = fit('fp32_control', kd/'student.pt', 4, None, tok, cache, dev, streams, teacher)
    for bits in (32, 8):
        status('EXPORTING_EVALUATING_UNDER20M_CPU', run='fp32_control', bits=bits)
        candidates.append(export_candidate(control, 'compact_fp32' if bits == 32 else 'compact_int8', bits, tok, dev, streams, teacher))
        save('candidate_results.json', candidates)
    for group in (64, 32):
        folder = fit('qat_g'+str(group), kd/'student.pt', 4, group, tok, cache, dev, streams, teacher)
        status('EXPORTING_EVALUATING_UNDER20M_CPU', run='qat_g'+str(group), bits=4)
        candidates.append(export_candidate(folder, 'compact_int4_g'+str(group), 4, tok, dev, streams, teacher))
        save('candidate_results.json', candidates)
    finalize(candidates, teacher)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    with (OUT/'CAMPAIGN_LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: run()
        except InterruptedError as e: status('STOPPED_BY_REQUEST', reason=str(e))
        except Exception as e:
            save('error.json', dict(error=repr(e), traceback=traceback.format_exc(), at_utc=datetime.now(timezone.utc).isoformat()))
            status('ERROR', error=repr(e)); raise
