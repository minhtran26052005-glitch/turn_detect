# Chạy model phát hiện kết thúc lượt nói

Đã chạy [pilot Việt–Anh–Trung](README_TRILINGUAL_V3.md) theo teacher→distill→QAT, **chưa đạt90% cả ba tiếng và có regression Việt**; giữ bản Việt làm mặc định. [Kết quả](outputs/trilingual_20261024/review.html) · [Player streaming từng prefix](outputs/trilingual_stream_20261024/review.html). Ô `trilingual_result` cuối notebook dùng bản thử riêng.

Đã chạy [QAT INT4](models/turn_v2_distilled_qat4_20261024/README.md) trên Colab: ONNX **177.1 KB**, cả gói **404.0 KB**, dev accuracy **93.55%**, F1 EOT **90.19%**, ngưỡng **0.470**. Trọng số4-bit và activations/phép tính FP32; chưa có holdout mới. [ZIP](models/turn_v2_distilled_qat4_20261024.zip) · ô `qat_result` cuối `a.ipynb`.

Đã kiểm tra thêm [student INT4](models/turn_v2_distilled_int4_20261024/README.md): **ONNX 186 KB, cả gói 413 KB**, dev accuracy **93,43%**, F1 EOT **90,06%**, ngưỡng mới **0,430**. Lưu trọng số 4-bit, phép tính FP32; dev dùng chọn bản/ngưỡng, chưa có holdout độc lập mới. [ZIP](models/turn_v2_distilled_int4_20261024.zip) · ô `int4_result` cuối `a.ipynb`.

Student mới đã distill: [model và hướng dẫn](models/turn_v2_distilled_min_20261024/README.md), **ONNX 329 KB**, cả gói **555 KB**, dev accuracy **93,56%**, F1 EOT **90,07%**. Nhỏ nhất đạt trong 18 biến thể đã đo; chưa kiểm chứng trên hội thoại/holdout mới. [ZIP](models/turn_v2_distilled_min_20261024.zip). Chạy trực tiếp `onnx/run_model.py` hoặc ô cuối `a.ipynb` đã kiểm tra trên Colab.

Bản nén mới: [hướng dẫn và model](models/turn_v2_compact_20261024/README.md), ONNX **108,2 MB**, cả gói **109,6 MB**, giảm hơn 10 lần so với ONNX đã train. Accuracy **93,79%**, F1 EOT **90,52%** trên **dev**, chưa kiểm chứng trên holdout độc lập mới. Dùng `turn_v2.predict` hoặc ô cuối `a.ipynb` đã kiểm tra trên Colab. [Review chi tiết](outputs/compression_20261024/review.html).

Đã có [bản train thử mới](EXPERIMENT_TRIAL_20261024.md): accuracy **93,81%**, F1 EOT **90,62%** trên holdout nhãn sẵn. Trên test cũ, accuracy **93,47%**, F1 EOT **88,68%**. Model mới ở `models/turn_v2_trial_20261024/onnx/`; phần hướng dẫn model gốc bên dưới vẫn dùng `checkpoint-23000/`.

Để train phiên bản mới trên Colab, mở `a.ipynb` và đọc [README_TRAIN_V2.md](README_TRAIN_V2.md).

Model nhận **văn bản tiếng Việt**, xuất dự đoán `EOT` (đã kết thúc lượt nói) hoặc `NEOT` (chưa kết thúc lượt nói). Đầu vào gồm `text` là câu đang nói và `context` là hội thoại trước đó. Model không nhận âm thanh trực tiếp; trong hệ thống thoại, văn bản đầu vào cần đến từ ASR.

## Thành phần đã cung cấp

| File | Vai trò |
| --- | --- |
| `checkpoint-23000/` | Model và tokenizer đã giải nén từ gói ban đầu |
| `checkpoint-23000/model_quantized.onnx` | Model ONNX, 87.193.678 byte, khoảng 83,15 MiB |
| `checkpoint-23000/config.json` | Cấu hình DeBERTa V2, 12 lớp, hidden size 384, 6 attention heads, vocabulary 128.000 |
| `checkpoint-23000/tokenizer*.json` | Tokenizer đi kèm model |
| `text_based.py` | Mã suy luận gốc; giữ nguyên |
| `train.jsonl` | 352.838 mẫu: 111.284 EOT, 241.554 NEOT |
| `test.jsonl` | 4.183 mẫu: 1.223 EOT, 2.960 NEOT |
| `run_turn_detector.py` | Lệnh chạy một câu hoặc JSONL, lưu dự đoán và đánh giá |
| `run_test.ps1` | Chạy lại toàn bộ test trên CPU |

Checkpoint chỉ chứa bản ONNX và tokenizer, không có trọng số PyTorch, optimizer hoặc trạng thái huấn luyện. Có thể dùng ngay để suy luận; không thể tiếp tục huấn luyện trực tiếp bằng checkpoint PyTorch từ gói này. Tên `checkpoint-23000` không đủ để xác minh lịch sử huấn luyện.

Ở lần kiểm tra cuối, file ZIP ban đầu không còn trong thư mục; thư mục model đã giải nén vẫn đầy đủ. Không cần ZIP để chạy các lệnh bên dưới.

## Kết quả đã chạy ngày 23/10/2026

Đã chạy toàn bộ `test.jsonl` bằng mã suy luận gốc, CPU, `max_length=256`, `batch_size=1`, quyết định bằng argmax. Không đổi ngưỡng hoặc chọn cấu hình theo điểm test.

| Chỉ số | Toàn bộ test | Loại đầu vào trùng train |
| --- | ---: | ---: |
| Số mẫu | 4.183 | 3.538 |
| Accuracy | 65,03% | 63,11% |
| Precision EOT | 44,87% | 46,79% |
| Recall EOT | 85,85% | 86,29% |
| F1 EOT | 58,94% | 60,68% |
| Kết thúc quá sớm: NEOT → EOT | 1.290 | 1.145 |
| Bỏ lỡ kết thúc: EOT → NEOT | 173 | 160 |

Ma trận toàn tập: `[[1670, 1290], [173, 1050]]`, hàng/cột theo thứ tự `[NEOT, EOT]`. Model nhận ra nhiều câu đã kết thúc nhưng cũng báo kết thúc sớm nhiều; việc chạy thành công không đồng nghĩa chất lượng đã đạt yêu cầu triển khai. Để tham chiếu mức lệch nhãn, luôn đoán NEOT sẽ đạt accuracy 70,76% và F1 EOT bằng 0.

Thời gian suy luận toàn tập khoảng **230 giây**, tương đương 18,19 mẫu/giây. Độ trễ từng mẫu sau warmup, có tính tokenize: trung bình **54,90 ms**, trung vị 51,28 ms, p95 80,55 ms. Số đo trên máy Windows đang sử dụng, không phải benchmark máy chuyên dụng; không tính thời gian import thư viện, nạp model và warmup.

Ví dụ đã chạy riêng: `tôi muốn đặt vé đi Hà Nội` → `EOT`, `prob_EOT=0.9449346`. Kết quả nằm trong `outputs/demo/`.

## Cách model xử lý đầu vào

1. Xóa khoảng trắng hai đầu, chuyển `context` và `text` thành chữ thường.
2. Tokenize riêng hai phần rồi ghép `[CLS] context [SEP] text [SEP]`.
3. Dùng cửa sổ 256 token, tính cả 3 token đặc biệt. Khi thiếu chỗ, bỏ token cũ ở đầu `context`; nếu riêng `text` dài hơn 253 token thì giữ 253 token đầu.
4. Pad bên phải, tạo `input_ids` và `attention_mask` kiểu int64.
5. ONNX trả hai logits. Softmax cho xác suất hai lớp; lấy lớp có xác suất lớn nhất. Theo mã gốc, lớp 0 là NEOT, lớp 1 là EOT. `prob_EOT` là đầu ra softmax, chưa có bằng chứng đã được hiệu chỉnh xác suất.

Model là DeBERTa V2 cho phân loại chuỗi, xuất ONNX opset 18. Theo `ort_config.json`, đây là lượng tử hóa động: trọng số QInt8, activation QUInt8; không phải toàn bộ mọi toán tử đều dùng int8. Metadata không xác minh được model nền cụ thể hoặc nguồn dữ liệu gốc.

## Chạy ngay trên máy hiện tại

Mở PowerShell trong thư mục này và chạy một câu:

```powershell
python .\run_turn_detector.py --text "tôi muốn đặt vé đi Hà Nội"
```

Có ngữ cảnh:

```powershell
python .\run_turn_detector.py --context "Bạn muốn đi đâu?" --text "tôi muốn đi Hà Nội"
```

Chạy toàn bộ test và đối chiếu đầu vào với train:

```powershell
python -u .\run_turn_detector.py --input .\test.jsonl --train .\train.jsonl --batch-size 1 --output-dir .\outputs\test_cpu_single
```

Hoặc chạy `powershell -ExecutionPolicy Bypass -File .\run_test.ps1`; tùy chọn này chỉ áp dụng cho tiến trình PowerShell được mở, không đổi chính sách hệ thống. Lần chạy lại sẽ ghi đè các file kết quả trong thư mục đích.

Môi trường Python hiện tại đã có thư viện nên không cần cài thêm. Khi dựng môi trường khác, có thể dùng `python -m pip install -r requirements.txt`. Phiên bản thực tế đã dùng và đường dẫn Python được ghi trong `metrics.json`; file requirements không phải bản khóa toàn bộ môi trường.

Model và tokenizer được nạp từ máy, wrapper bật `HF_HUB_OFFLINE=1` mặc định. Máy có RTX 3050 4 GB, nhưng ONNX Runtime đang dùng chỉ công bố `AzureExecutionProvider` và `CPUExecutionProvider`, nên lượt đánh giá dùng CPU Intel Core i5-12500H. Chưa có kết quả GPU. Wrapper từ chối yêu cầu `--device cuda` nếu môi trường không có CUDA provider.

## Đọc kết quả

- `outputs/test_cpu_single/metrics.json`: kết quả toàn tập, phần trùng/không trùng train, ma trận nhầm lẫn, thời gian, phiên bản thư viện và SHA-256.
- `outputs/test_cpu_single/predictions.jsonl`: mỗi dòng tương ứng một dòng dữ liệu; có nhãn gốc, dự đoán, `prob_EOT`, thời gian và cờ `exact_input_in_train`.
- `outputs/test_cpu_single/errors.jsonl`: chỉ các mẫu dự đoán khác nhãn, để xem các trường hợp kết thúc quá sớm hoặc bỏ lỡ kết thúc.
- `outputs/dataset_audit.json`: thống kê dữ liệu và số mẫu trùng đầu vào.
- `outputs/batch_consistency.json`: đối chiếu batch 1 và 32 trên 64 mẫu đầu; 2 mẫu đổi lớp, xác suất EOT chênh lớn nhất 0,12618. Chạy lại từng mẫu trên 8 mẫu đầu cho kết quả giống lần đánh giá.
- `outputs/completion_validation.json`: kiểm tra số dòng, thứ tự, miền xác suất và SHA-256 của các file đang dùng.

Trong ma trận nhầm lẫn, hàng là nhãn thật, cột là dự đoán; thứ tự hai lớp đều là `[NEOT, EOT]`. `false_eot` đếm mẫu NEOT bị dự đoán EOT, `missed_eot` đếm mẫu EOT bị dự đoán NEOT. Các chỉ số precision/recall/F1 được tính với EOT là lớp dương; metric không xác định do mẫu số bằng 0 sẽ là JSON `null`.

## Giới hạn đã quan sát

- Cả train và test đều có `context` rỗng. Điểm trên test chưa kiểm chứng khả năng dùng lịch sử hội thoại.
- Có **645/4.183 mẫu test trùng đầu vào với train (15,42%)**, so sánh chính xác cặp `(context.strip().lower(), text.strip().lower())`. Báo cáo tính riêng 3.538 mẫu không trùng theo tiêu chí này. Điều đó không chứng minh phần còn lại độc lập theo hội thoại hoặc câu nguồn.
- Bản ONNX cho xác suất khác nhau khi thay đổi nhóm mẫu trong batch. Vì vậy mặc định `--batch-size 1` để khớp cách gọi từng câu trong hệ thống thoại. Nếu đổi batch size, cần đánh giá lại theo đúng cách triển khai đó.
- Câu dài nhất trong test có 40 token sau tokenizer; không có mẫu test nào bị cắt do giới hạn 256. Chưa đánh giá chất lượng trên đầu vào dài vượt cửa sổ.
- Chưa kiểm chứng trên âm thanh/ASR trực tiếp, nhiễu thực tế hoặc hội thoại nhiều lượt; kết quả hiện tại chỉ mô tả dữ liệu JSONL được cung cấp.
- File gốc dùng tên trường khác nhau giữa hai API: `predict()` trả `predicted_class`/`prob_EOT`, còn `predict_batch()` trả `predicted`/`prob`. Wrapper chuẩn hóa đầu ra khi ghi file.
