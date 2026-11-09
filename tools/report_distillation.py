"""Write a model card and readable comparison from all 18 scored variants."""
from __future__ import annotations
import html
import json
from pathlib import Path


def main():
    root=Path('outputs/distillation_20261024');model=Path('models/turn_v2_distilled_min_20261024')
    manifest=json.loads((model/'model_manifest.json').read_text());selection=json.loads((root/'selection.json').read_text());bench=json.loads((root/'benchmark.json').read_text())
    metrics=manifest['dev_metrics'];all_rows=selection['candidates'];rows=sorted([r for r in all_rows if r['precision']=='int8'],key=lambda v:(v['parameters'],v['run']))
    table=['| Lượt train | Model + tokenizer, KB | Accuracy dev | F1 EOT dev |','| --- | ---: | ---: | ---: |']
    for row in rows:table.append(f"| {row['run'].replace('distill_gru_','').replace('_20261024','')} | {row['model_core_bytes']/1000:.3f} | {row['dev']['accuracy']*100:.3f}% | {row['dev']['eot_f1']*100:.3f}% |")
    before=bench['results']['teacher']['profiles']['matched_dynamic8'];after=bench['results']['compact']['profiles']['matched_dynamic8']
    training=json.loads((root/'runs'/manifest['source_run']/'training_report.json').read_text())
    history=json.loads((root/'runs'/manifest['source_run']/'history.json').read_text())
    readme=f'''# Student turn detector nhỏ nhất đạt trong lần thử

**GRU 96, embedding 64, 1 lớp; {manifest['parameters_deployment']:,} tham số triển khai.** Đã train/distill thật trên NVIDIA L4 từ nhãn gốc, logits và đặc trưng teacher cố định. Đã kiểm tra cả Windows và Colab.

- ONNX **{manifest['onnx_bytes']:,} byte** ({manifest['onnx_bytes']/1000:.3f} KB).
- Cả `onnx/` gồm model, tokenizer, cấu hình, runner Python và requirements: **{manifest['deployment_bytes']:,} byte** ({manifest['deployment_bytes']/1e6:.6f} MB), nhỏ hơn **{manifest['deployment_reduction_factor']:.1f} lần** gói teacher 109,609 MB. Chưa tính thư viện runtime đã cài.
- Dev **17.833 mẫu**: accuracy **{metrics['accuracy']*100:.3f}%**, F1 EOT **{metrics['eot_f1']*100:.3f}%**, precision **{metrics['eot_precision']*100:.3f}%**, recall **{metrics['eot_recall']*100:.3f}%**. Ngưỡng **{metrics['threshold']:.3f}**.
- So với teacher đã chọn trên dev, F1 giảm khoảng **0,45 điểm phần trăm**. Đây là đánh đổi dung lượng/chất lượng, không phải giữ nguyên điểm.

## Dùng ngay

Chạy từ project Windows, không cần PyTorch hay Transformers để suy luận:

```powershell
python models/turn_v2_distilled_min_20261024/onnx/run_model.py --text "tôi muốn đặt vé đi hà nội"
```

Hoặc copy nguyên thư mục `onnx/` sang máy khác, cài dependencies rồi chạy:

```bash
python -m pip install -r requirements.txt
python run_model.py --text "tôi muốn đặt vé đi hà nội" --context ""
```

Với project hiện tại vẫn dùng được API chung:

```python
from turn_v2.predict import TurnDetectorV2
detector = TurnDetectorV2("models/turn_v2_distilled_min_20261024/onnx")
print(detector.predict("cảm ơn bạn"))
```

Ô cuối `a.ipynb`, **Distillation**, xác minh hashes và suy luận bằng runner độc lập. Trên Colab model ở `/content/turn_detect_v2/models/turn_v2_distilled_min_20261024/onnx`. Cần giữ cả 7 file; tokenizer mới có ID riêng, không ghép với tokenizer teacher. `student.pt` để train tiếp nằm trong `outputs/distillation_20261024/runs/{manifest['source_run']}/`, chứa cả projection chỉ dùng lúc train.

## Các bản đã thử

Đã train/refine **6 lượt**, chấm **18** biến thể FP32/FP16/INT8 trên toàn bộ dev. Bảng dưới so các bản INT8, gồm model và tokenizer trước khi thêm cùng runner độc lập. Đây là bản nhỏ nhất **trong tìm kiếm hữu hạn này**, không chứng minh mức nhỏ nhất tuyệt đối.

{chr(10).join(table)}

Bản được chọn chạy {len(history)} epoch thực tế (tối đa 16, early stopping), checkpoint tốt nhất epoch **{training['best_epoch']}**, {training['optimizer_steps']:,} bước optimizer đã thực hiện. Loss kết hợp CE + KL temperature 2 + đặc trưng pooler teacher. Vocab 3.313 từ raw train, giữ fallback Latin/ASCII/byte/special; toàn bộ 315.174 mẫu train không đổi segmentation.

## Độ trễ và giới hạn

Trên 128 mẫu dev cố định, batch 1, CPU Windows **2 thread**, cả hai dùng padding bội số 8; thời gian gồm chuẩn hóa/tokenizer/suy luận: teacher p50 **{before['p50_ms']:.3f} ms**, student p50 **{after['p50_ms']:.3f} ms**, student p95 **{after['p95_ms']:.3f} ms**. Phép đo dùng API chung `TurnDetectorV2`; không phải cam kết trên phần cứng khác. Runner độc lập đã đối chiếu input trên toàn bộ dev và 132 lượt suy luận, xác suất khớp API chung.

**Dev được dùng chọn checkpoint, cấu hình, kiểu lưu trọng số và ngưỡng. Chưa kiểm chứng trên holdout độc lập mới/hội thoại thực.** F1 chỉ hơn mốc 90% khoảng 0,07 điểm phần trăm, nên không coi đây là đảm bảo ≥90% cho dữ liệu mới. Holdout đã dùng của teacher không chấm lại; điểm holdout teacher không phải điểm của student.

Trong demo UTF-8 thật, “hà nội là thủ đô và là một trong những trung tâm văn hóa lớn của việt nam”, “tôi muốn đặt vé đi”, “cảm ơn bạn” đều được dự đoán NEOT. Các ví dụ này cho thấy cần kiểm tra câu đã đủ ý/câu ngắn đúng miền sử dụng. Nhãn nguồn còn phụ thuộc prefix tiếp tục, thiếu ID hội thoại; vocab fallback đa ngôn ngữ đã thu hẹp. Không gọi Qwen, không RL, không tự biến dự đoán crawl thành nhãn train trong đợt này.

[Model manifest và hashes](model_manifest.json) · [Review HTML](../../outputs/distillation_20261024/review.html) · [Hồ sơ đầy đủ](../../EXPERIMENT_DISTILLATION_20261024.md) · [Dự đoán dev đã chấm](../../{manifest['selected_report']}) · [Notebook đã chạy](../../outputs/distillation_20261024/notebook_verification.json)
'''
    (model/'README.md').write_text(readme,encoding='utf-8')
    cells=[]
    for row in sorted(all_rows,key=lambda r:r['model_core_bytes']):
        chosen=row['run']==manifest['source_run'] and row['precision']==manifest['storage_precision']
        cells.append('<tr'+(' class="selected"' if chosen else '')+'><td>'+html.escape(row['run'].replace('distill_gru_','').replace('_20261024',''))+'</td><td>'+row['precision']+f"</td><td>{row['model_core_bytes']/1000:.3f}</td><td>{row['dev']['accuracy']*100:.3f}%</td><td>{row['dev']['eot_f1']*100:.3f}%</td><td>"+('đạt' if row['passes'] else 'chưa đạt')+'</td><td><a href="'+html.escape(str(Path(row['report']).relative_to(root).as_posix()))+'">report</a></td></tr>')
    text=f'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Distillation turn detector</title>
<style>body{{font:16px/1.6 system-ui;max-width:1250px;margin:32px auto;padding:0 24px;color:#182b40}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d8e2ec;padding:10px;text-align:left}}th{{background:#edf4fb}}.selected{{background:#daf3e8}}.notice{{background:#fff3d7;padding:16px;border-left:4px solid #b77b00}}code{{overflow-wrap:anywhere}}</style>
<h1>Student nhỏ nhất đạt trong các bản đã thử</h1>
<p><b>GRU 96, embedding 64, 1 lớp · {manifest['parameters_deployment']:,} tham số · ONNX {manifest['onnx_bytes']/1000:.3f} KB · cả gói {manifest['deployment_bytes']/1000:.3f} KB.</b></p>
<p>Accuracy <b>{metrics['accuracy']*100:.3f}%</b>, F1 EOT <b>{metrics['eot_f1']*100:.3f}%</b> trên 17.833 mẫu dev; threshold {metrics['threshold']:.3f}. Gói nhỏ hơn teacher {manifest['deployment_reduction_factor']:.1f} lần; F1 giảm khoảng 0,45 điểm phần trăm so với teacher.</p>
<p class="notice">Đây là chọn model trên dev, đã nhìn dev nhiều lần. Chưa có holdout độc lập mới. Mức F1 sát 90% không bảo đảm chất lượng hội thoại thực. Demo câu ngắn và câu đủ ý vẫn có thể ra NEOT; xem output thực thi trong notebook.</p>
<h2>18 biến thể đã chấm</h2><p>Dòng xanh là bản chọn. KB trong bảng gồm model/tokenizer/cấu hình; gói cuối thêm runner và requirements. Không tính thư viện runtime đã cài.</p>
<table><thead><tr><th>Lượt train</th><th>Lưu trọng số</th><th>KB</th><th>Accuracy dev</th><th>F1 EOT dev</th><th>Hai mốc 90%</th><th>Bằng chứng</th></tr></thead><tbody>{''.join(cells)}</tbody></table>
<h2>Benchmark Windows</h2><p>128 mẫu dev cố định, batch 1, 2 thread, cùng padding bội số 8, warmup 8. Tính cả chuẩn hóa, tokenizer và suy luận. Teacher p50 {before['p50_ms']:.3f} ms; student p50 {after['p50_ms']:.3f} ms, p95 {after['p95_ms']:.3f} ms. Không suy ra tốc độ trên máy khác.</p>
<p><a href="benchmark.json">Benchmark từng mẫu</a> · <a href="portable_runtime_parity.json">Parity runner độc lập</a> · <a href="notebook_verification.json">Ô Colab đã thực thi</a> · <a href="../../models/turn_v2_distilled_min_20261024/README.md">Hướng dẫn</a> · <a href="../../models/turn_v2_distilled_min_20261024/model_manifest.json">Manifest</a> · <a href="../../EXPERIMENT_DISTILLATION_20261024.md">Hồ sơ</a></p>
<p>SHA256 ONNX: <code>{manifest['files']['model.onnx']['sha256']}</code></p></html>'''
    (root/'review.html').write_text(text,encoding='utf-8')
    print(json.dumps({'review':str(root/'review.html'),'model_card':str(model/'README.md'),'variants':len(all_rows)}))


if __name__=='__main__':main()
