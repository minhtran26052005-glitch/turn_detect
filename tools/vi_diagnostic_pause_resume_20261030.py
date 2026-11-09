"""Pause VI training safely, score the exposed C75 diagnostic, then resume."""
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

ROOT = Path('/home/edabk/hoangbpm/turn_detect')
OUT = ROOT / 'outputs/translation_segmenter_v15d_20261026'
JOB = OUT / 'full_epoch_v1'
FOLDER = OUT / 'vi_diagnostic_current_20261030_v2'
PYTHON = ROOT / '.venv/bin/python'


def save(name, value):
    p = FOLDER / name
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, p)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def same(pid, ticks, needle):
    p = Path('/proc') / str(pid)
    try:
        return p.joinpath('stat').read_text().split()[21] == str(ticks) and needle in p.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except (OSError, IndexError):
        return False


def main():
    FOLDER.mkdir(parents=True, exist_ok=False)
    parent = json.loads((JOB / 'vi_only_campaign_status.json').read_text())
    child = json.loads((OUT / 'child.json').read_text())
    pp = parent['pid']
    ticks = Path(f'/proc/{pp}/stat').read_text().split()[21]
    assert same(pp, ticks, 'translation_segmenter_vi_stream_campaign.py')
    assert same(child['pid'], child['start_ticks'], 'turn_v2.translation_segmenter_vi_stream_v1')
    save('status.json', {'stage': 'PAUSING', 'parent': pp, 'child': child})
    os.kill(pp, signal.SIGTERM)
    os.kill(child['pid'], signal.SIGTERM)
    try:
        for _ in range(180):
            if not same(pp, ticks, 'translation_segmenter_vi_stream_campaign.py') and not same(child['pid'], child['start_ticks'], 'turn_v2.translation_segmenter_vi_stream_v1'):
                break
            time.sleep(1)
        else:
            raise RuntimeError('Training did not stop cleanly; inspect before restart')
        receipt = json.loads((OUT / 'teachers/vi/resume_receipt.json').read_text())
        checkpoint = OUT / 'teachers/vi/latest_resume.pt'
        assert checkpoint.stat().st_size == receipt['bytes'] and receipt['weights_optimizer_rng_and_sampler']
        frozen = FOLDER / 'frozen_resume.pt'
        os.link(checkpoint, frozen)
        digest = sha(frozen)
        dev = json.loads((JOB / 'runs/cycle000084_full_epoch_contiguous_v1/vi/epoch_0003.json').read_text())
        threshold = float(dev['threshold_selected_on_dev'])
        save('frozen_receipt.json', {'source_receipt': receipt, 'checkpoint_sha256': digest, 'threshold': threshold, 'threshold_source': 'cycle84_epoch3_dev', 'dataset': 'exposed_c75_clean_diagnostic_not_independent'})
        save('status.json', {'stage': 'SCORING', 'checkpoint_sha256': digest, 'threshold': threshold})

        import torch
        import transformers.utils as u
        import transformers.utils.import_utils as ui
        u.is_torchvision_available = ui.is_torchvision_available = lambda: False
        u.is_librosa_available = ui.is_librosa_available = lambda: False
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        from turn_v2.monolingual_data import encode_tail
        torch.set_num_threads(2)
        model_dir = OUT / 'teachers/vi/best'
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(model_dir, local_files_only=True), dtype=torch.float32)
        ckpt = torch.load(frozen, map_location='cpu', weights_only=False)
        assert ckpt['optimizer']['state'] and ckpt['total_steps'] == receipt['total_steps']
        assert ckpt['full_epoch']['position'] == receipt['full_epoch_cursor']
        model.load_state_dict(ckpt['model'], strict=True)
        del ckpt
        model.cuda().eval()
        samples = ROOT / 'staging/test/clean_vi.jsonl.gz'
        counts = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        per = {'1': dict(counts), '2': dict(counts)}
        total = 0
        traces = FOLDER / 'predictions.jsonl.gz'
        with gzip.open(samples, 'rt', encoding='utf-8') as source, gzip.open(traces, 'wt', encoding='utf-8') as target:
            batch = []
            def score(rows):
                nonlocal total
                tokens = [encode_tail(tokenizer, row['context'], row['text'], 256)['input_ids'] for row in rows]
                width = max(map(len, tokens))
                ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long, device='cuda')
                mask = torch.zeros_like(ids)
                for i, seq in enumerate(tokens):
                    ids[i, :len(seq)] = torch.tensor(seq, device='cuda')
                    mask[i, :len(seq)] = 1
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    probs = model(input_ids=ids, attention_mask=mask).logits.float().softmax(-1)[:, 1].cpu().tolist()
                for row, prob in zip(rows, probs):
                    expected = row['label'] == 'EOT'; pred = prob >= threshold
                    key = 'tp' if expected and pred else 'fn' if expected else 'fp' if pred else 'tn'
                    counts[key] += 1; per[str(row['lookahead_words'])][key] += 1
                    target.write(json.dumps({'example_id': row['example_id'], 'label': row['label'], 'lookahead_words': row['lookahead_words'], 'p_eot': prob, 'predicted': 'EOT' if pred else 'NEOT'}, ensure_ascii=False) + '\n')
                total += len(rows)
                if total % 3200 == 0: save('status.json', {'stage': 'SCORING', 'rows_done': total})
            for line in source:
                row = json.loads(line)
                assert row['included_in_clean_test'] and row['language'] == 'vi'
                batch.append(row)
                if len(batch) == 32:
                    score(batch); batch = []
            if batch: score(batch)
        def metrics(v):
            tp,tn,fp,fn = (v[k] for k in ('tp','tn','fp','fn'))
            return {**v, 'samples': tp+tn+fp+fn, 'accuracy': (tp+tn)/(tp+tn+fp+fn), 'eot_f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
        report = {'complete': True, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'scope': 'Previously exposed C75 diagnostic, not independent final test or conversational gold; cycle84 overlap audit not rerun', 'dataset_sha256': sha(samples), 'checkpoint_sha256': digest, 'checkpoint_receipt': receipt, 'threshold_from_dev': threshold, 'test_used_for_selection': False, 'overall': metrics(counts), 'by_lookahead': {k:metrics(v) for k,v in per.items()}, 'predictions_sha256': sha(traces)}
        assert total == 39167
        save('report.json', report)
        save('status.json', {'stage': 'TEST_COMPLETE_RESUMING', 'rows': total})
        del model
        torch.cuda.empty_cache()
    except Exception:
        save('error.json', {'error': traceback.format_exc()})
    finally:
        if same(pp, ticks, 'translation_segmenter_vi_stream_campaign.py') or same(child['pid'], child['start_ticks'], 'turn_v2.translation_segmenter_vi_stream_v1'):
            save('resume_status.json', {'status': 'NOT_RESTARTED_PROCESS_STILL_PRESENT'})
            return
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = '/home/edabk/miniconda3/envs/.VLLM/lib/python3.10/site-packages/nvidia/cu13/lib' + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        with (FOLDER / 'resumed_campaign.log').open('a') as log:
            p = subprocess.Popen([str(PYTHON), '-u', str(ROOT / 'translation_segmenter_vi_stream_campaign.py')], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(4)
        save('resume_status.json', {'status': 'STARTED' if p.poll() is None else 'FAILED', 'pid': p.pid, 'start_ticks': Path(f'/proc/{p.pid}/stat').read_text().split()[21] if p.poll() is None else None, 'returncode': p.poll()})


if __name__ == '__main__':
    main()
