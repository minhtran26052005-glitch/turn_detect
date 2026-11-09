"""Score the saved epoch-2 weights on the already-exposed C75 diagnostic."""
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import traceback

ROOT = Path('/home/edabk/hoangbpm/turn_detect')
OUT = ROOT / 'outputs/translation_segmenter_v15d_20261026'
JOB = OUT / 'vi_diagnostic_epoch2_20261030_v1'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def save(name, value):
    temp = JOB / (name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, JOB / name)


def main():
    JOB.mkdir(parents=True, exist_ok=False)
    try:
        best = OUT / 'teachers/vi/best'
        report = json.loads((OUT / 'teachers/vi/best_report.json').read_text())
        assert report['epoch'] == 2 and report['generation'] == 'cycle000084_full_epoch_contiguous_v1'
        frozen = JOB / 'model.safetensors'
        os.link(best / 'model.safetensors', frozen)
        digest = sha(frozen)
        assert digest == report['weights_sha256']
        threshold = float(report['threshold_selected_on_dev'])
        sample = ROOT / 'staging/test/clean_vi.jsonl.gz'
        expected = json.loads((OUT / 'vi_diagnostic_current_20261030_v2/report.json').read_text())
        assert sha(sample) == expected['dataset_sha256']
        save('status.json', {'stage': 'SCORING', 'epoch': 2})
        import torch
        import transformers.utils as u
        import transformers.utils.import_utils as ui
        u.is_torchvision_available = ui.is_torchvision_available = lambda: False
        u.is_librosa_available = ui.is_librosa_available = lambda: False
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        from turn_v2.monolingual_data import encode_tail
        torch.set_num_threads(2)
        tokenizer = AutoTokenizer.from_pretrained(best, local_files_only=True)
        model = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(best, local_files_only=True), dtype=torch.float32)
        model.load_state_dict(load_file(str(frozen)), strict=True)
        model.cuda().eval()
        counts = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        per = {'1': dict(counts), '2': dict(counts)}
        total = 0
        with gzip.open(sample, 'rt', encoding='utf-8') as source, gzip.open(JOB / 'predictions.jsonl.gz', 'wt', encoding='utf-8') as target:
            batch = []
            def score(rows):
                nonlocal total
                seqs = [encode_tail(tokenizer, x['context'], x['text'], 256)['input_ids'] for x in rows]
                width = max(map(len, seqs))
                ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long, device='cuda')
                mask = torch.zeros_like(ids)
                for i, seq in enumerate(seqs):
                    ids[i, :len(seq)] = torch.tensor(seq, device='cuda'); mask[i, :len(seq)] = 1
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    probs = model(input_ids=ids, attention_mask=mask).logits.float().softmax(-1)[:, 1].cpu().tolist()
                for row, prob in zip(rows, probs):
                    actual = row['label'] == 'EOT'; pred = prob >= threshold
                    key = 'tp' if actual and pred else 'fn' if actual else 'fp' if pred else 'tn'
                    counts[key] += 1; per[str(row['lookahead_words'])][key] += 1
                    target.write(json.dumps({'example_id': row['example_id'], 'label': row['label'], 'lookahead_words': row['lookahead_words'], 'p_eot': prob, 'predicted': 'EOT' if pred else 'NEOT'}, ensure_ascii=False) + '\n')
                total += len(rows)
                if total % 3200 == 0: save('status.json', {'stage':'SCORING', 'rows_done':total})
            for line in source:
                row = json.loads(line)
                assert row['language'] == 'vi' and row['included_in_clean_test']
                batch.append(row)
                if len(batch) == 32: score(batch);batch=[]
            if batch:score(batch)
        assert total == 39167
        def metric(v):
            tp,tn,fp,fn = (v[k] for k in ('tp','tn','fp','fn'))
            return {**v, 'samples':tp+tn+fp+fn, 'accuracy':(tp+tn)/(tp+tn+fp+fn), 'eot_f1':2*tp/(2*tp+fp+fn)}
        save('report.json', {'complete':True, 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'scope':'Previously exposed punctuation-based diagnostic, not independent conversational test; cycle84 overlap audit not rerun', 'checkpoint_epoch':2, 'checkpoint_sha256':digest, 'threshold_from_dev':threshold, 'dataset_sha256':sha(sample), 'test_used_for_selection':False, 'overall':metric(counts), 'by_lookahead':{k:metric(v) for k,v in per.items()}, 'predictions_sha256':sha(JOB / 'predictions.jsonl.gz')})
        save('status.json', {'stage':'COMPLETE', 'rows':total})
    except Exception:
        save('error.json', {'error':traceback.format_exc()})
        raise


if __name__ == '__main__':main()
