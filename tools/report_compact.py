"""Build a readable compression review from scored dev records and receipts."""
from __future__ import annotations

import html
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turn_v2.common import read_jsonl, write_jsonl


def main():
    root = Path('outputs/compression_20261024')
    report = json.loads((root/'weight_only_full_dev_cached/report.json').read_text())
    manifest = json.loads(Path('models/turn_v2_compact_20261024/model_manifest.json').read_text())
    negative = json.loads((root/'int8_quint8_full_dev/report.json').read_text())
    bench = json.loads((root/'benchmark.json').read_text())
    compressed = np.load(root/'weight_only_full_dev_cached/dev_predictions.npz')
    teacher = np.load('outputs/training_trial_20261024/dev_predictions.npz')
    rows = list(read_jsonl(Path('data_v2/frozen/dev.jsonl')))
    assert np.array_equal(compressed['indices'], np.arange(len(rows)))
    assert np.array_equal(compressed['labels'], teacher['labels'])
    probs, before, labels = compressed['prob_EOT'], teacher['prob_EOT'], compressed['labels']
    threshold = manifest['dev_metrics']['threshold']
    changes = np.flatnonzero((probs>=threshold)!=(before>=.5))
    selected = set(np.random.default_rng(20261024).choice(len(rows),64,replace=False).tolist())
    selected.update(changes.tolist())
    records = []
    for index in sorted(selected):
        records.append({'dev_index':index,'text':rows[index]['text'],
            'reference_label':rows[index]['label'],'teacher_probability':float(before[index]),
            'compact_probability':float(probs[index]),
            'teacher_prediction':'EOT' if before[index]>=.5 else 'NEOT',
            'compact_prediction':'EOT' if probs[index]>=threshold else 'NEOT',
            'selection':'prediction_changed' if index in changes else 'uniform_sample',
            'compact_correct':bool((probs[index]>=threshold)==labels[index])})
    write_jsonl(root/'review_cases.jsonl',records)
    escape = html.escape
    metric_rows = []
    comparisons = [('Model gốc', report['reference_model_bytes'],report['teacher_dev_matched']),
        ('QUInt8 động — loại',negative['model_bytes'],negative['dev_selected_threshold']),
        ('INT8 trọng số, activation FP32 — chọn',report['model_bytes'],manifest['dev_metrics'])]
    for name,size,metrics in comparisons:
        metric_rows.append(f'<tr><td>{escape(name)}</td><td>{size/1e6:.3f}</td>'
            f'<td>{metrics["accuracy"]*100:.3f}%</td><td>{metrics["eot_f1"]*100:.3f}%</td>'
            f'<td>{metrics["threshold"]:.3f}</td></tr>')
    cases = []
    for row in records:
        status = 'đúng theo nhãn dữ liệu' if row['compact_correct'] else 'khác nhãn dữ liệu'
        cases.append(f'<tr data-changed="{str(row["selection"]=="prediction_changed").lower()}">'
            f'<td>{row["dev_index"]}</td><td>{escape(row["text"])}</td><td>{row["reference_label"]}</td>'
            f'<td>{row["teacher_prediction"]} ({row["teacher_probability"]:.4f})</td>'
            f'<td>{row["compact_prediction"]} ({row["compact_probability"]:.4f})</td><td>{status}</td></tr>')
    timings = []
    for profile, label in [('matched_dynamic8','Cùng padding bội số 8'),('deployment_padding','Padding khi triển khai (gốc 256, nén bội số 8)')]:
        a = bench['results']['teacher']['profiles'][profile]
        b = bench['results']['compact']['profiles'][profile]
        timings.append(f'<tr><td>{label}</td><td>{a["p50_ms"]:.2f}</td><td>{b["p50_ms"]:.2f}</td>'
            f'<td>{a["p95_ms"]:.2f}</td><td>{b["p95_ms"]:.2f}</td></tr>')
    document = f'''<!doctype html><html lang="vi"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Kết quả nén turn detector</title>
<style>body{{font:16px/1.55 system-ui;margin:32px auto;padding:0 24px;max-width:1300px;color:#1c2c40}}table{{width:100%;border-collapse:collapse;margin:20px 0}}td,th{{border:1px solid #dce4ec;padding:10px;text-align:left}}th{{background:#eef4f8}}.notice{{padding:16px;background:#fff4db;border-left:4px solid #bf8300}}.result{{padding:20px;background:#e6f6ef;border-radius:10px}}code{{overflow-wrap:anywhere}}button{{padding:8px 16px;cursor:pointer}}</style>
<h1>Turn detector: nén hơn 10 lần theo dung lượng file</h1>
<div class="result">ONNX <b>{report['model_bytes']/1e6:.3f} MB</b>; cả gói <b>{manifest['deployment_bytes']/1e6:.3f} MB</b>, bằng <b>{100*manifest['deployment_bytes']/report['reference_model_bytes']:.3f}%</b> ONNX gốc.
Accuracy <b>{manifest['dev_metrics']['accuracy']*100:.3f}%</b>, F1 EOT <b>{manifest['dev_metrics']['eot_f1']*100:.3f}%</b> trên <b>17.833 mẫu dev</b>.</div>
<p class="notice">Đây là kết quả chọn bản nén trên dev. Holdout của lần train trước đã dùng và không được chấm lại; chưa có điểm holdout độc lập mới cho bản nén. Nhãn dữ liệu cũ và nhóm nguồn heuristic chưa chứng minh chất lượng hội thoại thực.</p>
<h2>Dung lượng và chất lượng trên cùng dev</h2><table><thead><tr><th>Bản</th><th>ONNX (MB thập phân)</th><th>Accuracy</th><th>F1 EOT</th><th>Ngưỡng</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table>
<p>Giữ 12 lớp, hidden 768; vocab 251.000 → 23.078. Vocab chỉ dùng train; 0/315.174 mẫu train đổi cách tách token. Trên dev có {report['segmentation_changed_rows']} mẫu đổi tách token. Không train thêm, không distillation hay RL. Trọng số tuyến tính INT8 được giải lượng tử một lần khi nạp, phép tính và activation FP32. Dung lượng RAM và tốc độ không giảm 10 lần theo kích thước file.</p>
<h2>CPU trên Windows</h2><p>128 mẫu dev cố định, batch 1, 4 thread, warmup 8; thời gian gồm chuẩn hóa, tokenizer và suy luận. Hai profile được báo riêng vì padding ảnh hưởng tốc độ.</p>
<table><thead><tr><th>Profile</th><th>Gốc p50 ms</th><th>Nén p50 ms</th><th>Gốc p95 ms</th><th>Nén p95 ms</th></tr></thead><tbody>{''.join(timings)}</tbody></table>
<h2>Đối chiếu từng mẫu dev</h2><p>{len(changes)} mẫu đổi nhãn ở ngưỡng đã khóa. Bảng gồm toàn bộ các mẫu đổi nhãn và 64 mẫu chọn đều với seed 20261024. Xác suất chưa phải độ tin cậy đã hiệu chỉnh.</p>
<button id="filter">Chỉ xem mẫu đổi nhãn</button><table id="cases"><thead><tr><th>Index dev</th><th>Văn bản</th><th>Nhãn dữ liệu</th><th>Model gốc</th><th>Bản nén</th><th>Đối chiếu</th></tr></thead><tbody>{''.join(cases)}</tbody></table>
<p><a href="weight_only_full_dev_cached/report.json">Full dev report</a> · <a href="benchmark.json">Benchmark chi tiết</a> · <a href="review_cases.jsonl">Dự đoán từng mẫu</a> · <a href="../../models/turn_v2_compact_20261024/model_manifest.json">Model manifest và hashes</a> · <a href="crawl_smoke/results.html">Crawl và dự đoán</a></p>
<p>SHA256 ONNX: <code>{report['model_sha256']}</code></p>
<script>let filtered=false;document.getElementById('filter').onclick=function(){{filtered=!filtered;for(const row of document.querySelectorAll('#cases tbody tr'))row.hidden=filtered&&row.dataset.changed!=='true';this.textContent=filtered?'Hiện tất cả':'Chỉ xem mẫu đổi nhãn';}};</script></html>'''
    (root/'review.html').write_text(document,encoding='utf-8')
    print(json.dumps({'review':str(root/'review.html'),'cases':len(records),'prediction_changes':len(changes)}))


if __name__ == '__main__':
    main()
