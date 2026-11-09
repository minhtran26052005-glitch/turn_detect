"""Export natural-floor inputs and per-case best-teacher predictions from Colab.

Inference runs on Colab CPU so it does not contend for the training GPU. The
export is a separate review artifact; it does not change training state.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import sys
from urllib.parse import quote

from teacher_first_colab import campaign_runtime
from storage_guard import ensure_free_space

RUN = Path("outputs/teacher_first_v14_20261026")
REMOTE = "/content/turn_detect_v2"
LANGUAGES = ("vi", "en", "zh")


REMOTE_CODE = r'''import gc, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, "/content/turn_detect_v2")
import torch
import transformers.utils as hf_utils
import transformers.utils.import_utils as hf_imports
hf_utils.is_torchvision_available = hf_imports.is_torchvision_available = lambda: False
hf_utils.is_librosa_available = hf_imports.is_librosa_available = lambda: False
torch.set_num_threads(2)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from turn_v2.teacher_first_v14 import encode
from turn_v2.continuous_v13_state import atomic_json
from turn_v2.common import read_jsonl
from turn_v2.metrics import classification_metrics

root=Path('/content/turn_detect_v2')
run=root/'outputs/teacher_first_v14_20261026'
data=root/'data_v10/reward_20261025'
out=run/'natural_floor_review'
out.mkdir(parents=True,exist_ok=True)
summary={'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
         'inference_device':'CPU_FP32_READ_ONLY_REVIEW',
         'labels_are':'existing observed-floor/VAD labels; is_human_gold=false',
         'threshold_source':'frozen semantic silver dev threshold in each best_report.json',
         'languages':{}}
for lang in ('vi','en','zh'):
    source=data/lang/'dev.jsonl'
    all_rows=list(read_jsonl(source))
    rows=[r for r in all_rows if r.get('evaluation_slice')=='natural']
    assert rows and all(r.get('label') in ('EOT','NEOT') for r in rows)
    report=json.loads((run/'teachers'/lang/'best_report.json').read_text(encoding='utf-8'))
    threshold=float(report['semantic_dev']['threshold'])
    model_dir=run/'teachers'/lang/'best'
    weights=model_dir/'model.safetensors'
    weights_sha=hashlib.file_digest(weights.open('rb'),'sha256').hexdigest()
    assert weights_sha==report['weights_sha256'], f'{lang}: selected checkpoint hash mismatch'
    result_path=out/f'{lang}_natural_floor_predictions.jsonl'
    cached=[]
    if result_path.exists():
        try:
            cached=[json.loads(line) for line in result_path.read_text(encoding='utf-8').splitlines() if line]
            reusable=(len(cached)==len(rows) and all(x['input']==row and
                x['output']['checkpoint_sha256']==weights_sha for x,row in zip(cached,rows)))
        except Exception:
            reusable=False
    else:
        reusable=False
    if reusable:
        probs=[float(item['output']['p_eot']) for item in cached]
    else:
        tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
        model=AutoModelForSequenceClassification.from_pretrained(model_dir,local_files_only=True,
                 num_labels=2,dtype=torch.float32).cpu().eval()
        ids,lengths,labels=encode(rows,tokenizer)
        probs=[]
        with torch.inference_mode():
            for start in range(0,len(ids),8):
                stop=min(start+8,len(ids)); width=int(lengths[start:stop].max())
                x=torch.from_numpy(ids[start:stop,:width].astype('int64'))
                mask=(torch.arange(width)[None,:] < torch.from_numpy(lengths[start:stop])[:,None]).long()
                logits=model(input_ids=x,attention_mask=mask).logits.float()
                probs.extend(torch.softmax(logits,dim=-1)[:,1].tolist())
        with result_path.open('w',encoding='utf-8',newline='\n') as f:
            for row,prob in zip(rows,probs):
                pred='EOT' if prob>=threshold else 'NEOT'
                item={'input':row,'output':{'language':lang,'teacher':'microsoft/mdeberta-v3-base',
                      'checkpoint_sha256':weights_sha,'threshold':threshold,'p_eot':float(prob),
                      'p_neot':float(1-prob),'predicted_label':pred,'observed_floor_label':row['label'],
                      'correct_against_observed_floor':pred==row['label'],
                      'label_source':row.get('label_source'),'is_human_gold':row.get('is_human_gold',False)}}
                f.write(json.dumps(item,ensure_ascii=False)+'\n')
        del model,tokenizer
        gc.collect()
    labels=[0 if r['label']=='NEOT' else 1 for r in rows]
    metrics=classification_metrics(labels,probs,threshold)
    subset_bytes=('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n').encode('utf-8')
    prediction_bytes=result_path.read_bytes()
    summary['languages'][lang]={'source_path':str(source),'total_dev_rows':len(all_rows),
       'natural_floor_rows':len(rows),'source_dev_sha256':hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),
       'natural_rows_sha256':hashlib.sha256(subset_bytes).hexdigest(),'best_report_cycle':report['cycle'],
       'checkpoint_sha256':weights_sha,'threshold':threshold,'metrics_recomputed_cpu_fp32':metrics,
       'metrics_saved_at_checkpoint':report['natural_floor_dev'],
       'prediction_file':str(result_path.relative_to(root)),
       'prediction_sha256':hashlib.sha256(prediction_bytes).hexdigest(),
       'prediction_bytes':len(prediction_bytes),
       'note':'CPU FP32 per-case recomputation; saved training report used Colab GPU BF16 autocast.'}
    if not reusable:
        gc.collect()
atomic_json(out/'summary.json',summary)
print(json.dumps(summary,ensure_ascii=False))
'''


def main() -> None:
    out = RUN / "natural_floor_review"
    out.mkdir(parents=True, exist_ok=True)
    runtime = campaign_runtime()
    runtime.connect()
    try:
        result = runtime.execute(REMOTE_CODE, timeout=900, echo=False)
        if result["status"] != "ok":
            proof = out / "remote_execution_error.json"
            proof.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            raise RuntimeError(f"Colab review export failed; see {proof}")
        streams = [item["text"] for item in result["outputs"]
                   if item.get("output_type") == "stream" and item.get("name") == "stdout"]
        summary = json.loads("".join(streams))
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
        for language, details in summary["languages"].items():
            response = runtime.http.get(
                runtime.base_url + "files/" + quote("content/turn_detect_v2/" + details["prediction_file"], safe="/"),
                stream=True, timeout=120, allow_redirects=False)
            if response.status_code != 200 or "html" in response.headers.get("content-type", ""):
                raise RuntimeError(f"Cannot download {language} natural-floor review: HTTP {response.status_code}")
            target = out / f"{language}_natural_floor_predictions.jsonl"
            partial = target.with_suffix(target.suffix + ".partial")
            with partial.open("wb") as f:
                for block in response.iter_content(1024 * 1024):
                    f.write(block)
            payload = partial.read_bytes()
            if len(payload) != details["prediction_bytes"] or hashlib.sha256(payload).hexdigest() != details["prediction_sha256"]:
                raise RuntimeError(f"Integrity check failed for {language} natural-floor export")
            partial.replace(target)
    finally:
        runtime.close()

    rows = []
    for language in LANGUAGES:
        with (out / f"{language}_natural_floor_predictions.jsonl").open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    summaries = "".join(
        f"<tr><td>{lang}</td><td>{s['natural_floor_rows']}</td>"
        f"<td>{s['metrics_saved_at_checkpoint']['accuracy']:.3f} / {s['metrics_saved_at_checkpoint']['eot_f1']:.3f}</td>"
        f"<td>{s['metrics_recomputed_cpu_fp32']['accuracy']:.3f} / {s['metrics_recomputed_cpu_fp32']['eot_f1']:.3f}</td>"
        f"<td>{s['threshold']:.3f}</td></tr>"
        for lang, s in summary["languages"].items())
    body = []
    for item in rows:
        x, y = item["input"], item["output"]
        body.append("<tr data-lang='" + html.escape(y["language"], quote=True) + "' data-ok='" +
                    str(y["correct_against_observed_floor"]).lower() + "'><td>" + html.escape(y["language"]) +
                    "</td><td>" + html.escape(str(x.get("case_id", ""))) + "</td><td>" +
                    html.escape(str(y["observed_floor_label"])) + "</td><td>" + html.escape(y["predicted_label"]) +
                    "</td><td>" + f"{y['p_eot']:.4f}" + "</td><td>" + f"{y['threshold']:.3f}" +
                    "</td><td>" + ("✓" if y["correct_against_observed_floor"] else "✗") + "</td><td>" +
                    html.escape(str(x.get("context", ""))) + "</td><td>" + html.escape(str(x.get("text", ""))) +
                    "</td><td>" + html.escape(str(y.get("label_source", ""))) + "</td></tr>")
    page = "\n".join([
        "<!doctype html><meta charset='utf-8'><title>Natural-floor case review</title>",
        "<style>body{font:15px system-ui;margin:24px;color:#203040}table{border-collapse:collapse;width:100%}th,td{padding:7px;border:1px solid #ccd4dc;vertical-align:top}th{position:sticky;top:0;background:#edf2f7}.wrap{overflow:auto;max-height:75vh}input,select{margin:8px;padding:7px}</style>",
        "<h1>Natural-floor — dữ liệu và dự đoán từng mẫu</h1>",
        "<p>Nhãn tham chiếu là observed-floor/VAD, không phải human gold. Xác suất tính lại FP32 trên CPU; summary đối chiếu metric GPU BF16 đã lưu.</p>",
        "<table><tr><th>Tiếng</th><th>Mẫu</th><th>Đã lưu GPU BF16 (Acc / EOT F1)</th><th>Tính lại CPU FP32 (Acc / EOT F1)</th><th>Ngưỡng semantic dev</th></tr>",
        summaries,
        "</table><label>Lọc tiếng <select id='lang'><option value=''>Tất cả</option><option>vi</option><option>en</option><option>zh</option></select></label>",
        "<label><input id='errors' type='checkbox'>Chỉ mẫu dự đoán khác nhãn</label>",
        "<div class='wrap'><table><thead><tr><th>Lang</th><th>Case</th><th>Nhãn floor</th><th>Dự đoán</th><th>P(EOT)</th><th>Ngưỡng</th><th>Khớp</th><th>Context</th><th>Text</th><th>Nguồn nhãn</th></tr></thead><tbody>",
        "".join(body),
        "</tbody></table></div>",
        "<script>function f(){let l=document.querySelector('#lang').value,e=document.querySelector('#errors').checked;for(let r of document.querySelectorAll('tbody tr'))r.hidden=!!((l&&r.dataset.lang!==l)||(e&&r.dataset.ok==='true'))}document.querySelector('#lang').onchange=f;document.querySelector('#errors').onchange=f;</script>",
    ])
    (out / "review.html").write_text(page, encoding="utf-8")
    print(json.dumps({"output_dir": str(out.resolve()), "case_rows": len(rows),
                      "files": sorted(p.name for p in out.iterdir())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
