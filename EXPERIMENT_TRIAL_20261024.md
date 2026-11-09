# Bản train thử ngày 24/10/2026

Đã đạt **accuracy 93,81% và F1 EOT 90,62%** trên 17.099 mẫu holdout đã khóa. Ngưỡng 0,5 và checkpoint được chọn trên dev trước khi mở holdout. Đây là kết quả trên nhãn dữ liệu cung cấp, với cách gom nhóm nguồn bằng heuristic; chưa phải kết quả kiểm định hội thoại thực do người rà soát.

| Tập đánh giá | Số mẫu | Accuracy | F1 EOT | Precision EOT | Recall EOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dev | 17.833 | 93,78% | 90,55% | 86,72% | 94,73% |
| Holdout cuối | 17.099 | 93,81% | 90,62% | 86,64% | 94,99% |
| Test cũ, kiểm tra hồi quy | 4.183 | 93,47% | 88,68% | 89,98% | 87,41% |

**F1 EOT trên bộ test cũ chưa đạt 90%.** Bộ này được đo sau khi đã khóa model/ngưỡng; không dùng kết quả để đổi checkpoint hay ngưỡng. Model cũ trên cùng test đạt accuracy 65,03%, F1 EOT 58,94%; lỗi cắt sớm giảm từ 1.290 xuống 119, EOT bị bỏ lỡ giảm từ 173 xuống 154. Đây là kiểm tra hồi quy, vì test cũ đã được xem trước đợt train này. [Receipt và dự đoán từng mẫu](outputs/training_trial_20261024/legacy_regression/report.json).

Holdout có 789 lần cắt sớm và 270 EOT bị bỏ lỡ. Holdout đã được đánh dấu sử dụng ở `data_v2/frozen/holdout_consumed.json`; không dùng tập này để chọn tiếp model/ngưỡng khác.

## Đợt train đã thực hiện

- Run: `trial_gold_20261024_093324`, Colab NVIDIA L4.
- Backbone: `microsoft/mdeberta-v3-base`, revision `a0484667b22365f84929a935b5e50a51f71f159d`.
- Chạy 2.000 bước optimizer, batch 16 × tích lũy 2 = 32, khoảng 64.000 lượt mẫu. Pool train có 315.174 mẫu; đợt này tương đương 0,203 epoch, **không phải 3 epoch đầy đủ**.
- Đánh giá toàn bộ dev mỗi 250 bước. Chọn checkpoint bằng `min(accuracy, F1 EOT)` tại ngưỡng 0,5; checkpoint tốt nhất ở **bước 1.750**. Bước 2.000 giảm nhẹ F1 nên không được chọn.
- AdamW, encoder LR 2e-5, head LR gấp 5, warmup 6%, cosine decay, label smoothing 0,02, gradient clipping 1, FP32/TF32, seed 42.
- Dùng nhãn sẵn; không thêm nhãn Qwen/silver và không có bước RL trong đợt train này.
- Tổng job khoảng 19 phút 50 giây, bao gồm chuẩn bị và xuất model; phần train/eval theo Trainer khoảng 17 phút 39 giây.

Đợt thử đạt mục tiêu hiện tại nhưng chưa xác định trần tối ưu của backbone hoặc của dữ liệu. Điểm dev từ bước 1.000 đến 2.000 dao động; một seed và 0,203 epoch chưa đủ để kết luận đã hội tụ hoàn toàn.

## So model cũ trên cùng dữ liệu

Đã chọn cố định 2.048 mẫu dev bằng seed 20261024 trước khi xem dự đoán bản mới. Cả hai dùng cùng text/context, quyết định tương đương ngưỡng 0,5.

| Model | Accuracy | F1 EOT | Cắt sớm | Bỏ lỡ EOT |
| --- | ---: | ---: | ---: | ---: |
| Model ONNX INT8 cũ | 55,86% | 53,69% | 804 | 100 |
| Bản mDeBERTa mới | 93,80% | 90,28% | 93 | 34 |

Đây là so sánh mô tả giữa hai model. Tập train của checkpoint cũ chưa được xác minh và backbone cũng khác, nên không quy toàn bộ chênh lệch cho riêng một kỹ thuật train.

## Model và bằng chứng

- [Báo cáo trực quan, biểu đồ học và các lỗi dev](outputs/training_trial_20261024/review.html).
- [Kiểm tra hoàn tất: 37 kiểm tra đạt, hash model/dữ liệu/code và chấm lại từng dự đoán](outputs/training_trial_20261024/completion_validation.json).
- [Kết quả holdout gốc](outputs/training_trial_20261024/holdout_report.json).
- [Kết quả train và dev](outputs/training_trial_20261024/training_report.json).
- [Định danh dữ liệu, code, model và cấu hình](outputs/training_trial_20261024/run_identity.json).
- [So sánh trên cùng 2.048 mẫu](outputs/training_trial_20261024/matched_dev_comparison.json).
- [Kiểm tra trên test cũ](outputs/training_trial_20261024/legacy_regression/report.json).
- Model ONNX trên Windows: `models/turn_v2_trial_20261024/onnx/`.
- Checkpoint PyTorch đã chọn được lưu trên Windows: `models/turn_v2_trial_20261024/best/` để dùng fine-tune trong run mới.
- Model PyTorch trên Colab: `/content/turn_detect_v2/runs/trial_gold_20261024_093324/best/`.
- Model ONNX trên Colab: `/content/turn_detect_v2/runs/trial_gold_20261024_093324/onnx/`.

ONNX FP32 đã qua checker và đối chiếu 8 mẫu dev với PyTorch: cùng argmax, độ lệch xác suất EOT lớn nhất `1,70225e-6`. Chưa lượng tử hóa INT8; kiểm tra này không chứng minh độ trễ triển khai hay parity trên mọi đầu vào. Model và bộ chia dữ liệu có hash trong các receipt; file tải về được kiểm tra SHA256.

Chạy dự đoán với bản mới trên Windows:

```powershell
python -m turn_v2.predict --model models/turn_v2_trial_20261024/onnx --text "Tôi muốn đặt một chuyến bay từ"
```

Chạy crawl rồi dự đoán bằng bản mới:

```powershell
python -u -m turn_v2.crawl_predict --model models/turn_v2_trial_20261024/onnx --limit 12 --review-budget 0 --output outputs/crawl_trial_new
```

Đã chạy [12 mẫu crawl bằng bản mới](outputs/training_trial_20261024/crawl_new_model/results.html) trên Windows. Luồng tạo đủ 12 dự đoán; một câu hoàn chỉnh về Hà Nội được dự đoán NEOT, cho thấy cần kiểm định thêm miền dữ liệu thực. Dự đoán crawl chưa có nhãn tham chiếu độc lập nên không tính accuracy cho luồng này.

Đối chiếu 3 câu demo giữa ONNX Windows và PyTorch Colab cho cùng nhãn, độ lệch xác suất lớn nhất khoảng `3,7e-6`. Thời gian 3 câu khoảng 172–188 ms/câu, chỉ là phép thử chức năng.

Mỗi lần chọn output mới. Đường dẫn `/content` của Colab sẽ mất nếu runtime bị xóa; các bản ONNX và PyTorch tải về Windows được giữ trong dự án.
