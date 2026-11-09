## Material Passport

- Origin Skill: academic-research-suite/experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-24
- Verification Status: VERIFIED
- Version Label: turn_v2_compact_v1
- repro_lock: outputs/compression_20261024/protocol.json

## Mục tiêu và phép đo

Người dùng chọn ưu tiên giữ **accuracy và F1 EOT ≥90%**, với file triển khai tối đa **1/10 ONNX hiện tại**. File gốc là 1.116.285.156 byte; ngân sách là 111.628.515 byte. Cả model và toàn bộ thư mục triển khai đều được báo dung lượng sau nén.

Model gốc: `models/turn_v2_trial_20261024/onnx/`, SHA256 ONNX `3049addd73b26cb84d49141f807b13aac435b7203c68959dc6338021a6785a8c`.

Vocab được xây từ **train đã khóa**, không lấy token từ dev/holdout. Dev được dùng đo chất lượng và chọn ngưỡng. Holdout đã mở ở đợt train trước được giữ nguyên, không chấm lại để chọn bản nén. Điểm của bản nén vì vậy mang phạm vi **dev**, chưa phải kiểm chứng trên một holdout độc lập mới.

## Kết quả đã hoàn tất

Đã tạo `models/turn_v2_compact_20261024/onnx/`: ONNX **108.201.936 byte**, toàn bộ thư mục triển khai **109.609.065 byte**, bằng **9,819%** ONNX đã train. Nhỏ hơn lần lượt **10,317 lần** và **10,184 lần**. Ngưỡng cuối **0,545**, chọn trên dev.

| Model / ngưỡng | Dev accuracy | F1 EOT | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| Teacher, 0,5 | 93,7756% | 90,5484% | 814 | 296 |
| Weight-only, cùng 0,5 | 93,7363% | 90,5074% | 829 | 288 |
| Weight-only, ngưỡng dev 0,545 | **93,7924%** | **90,5231%** | 781 | 326 |

Đã chấm toàn bộ **17.833 mẫu dev**, CPU Windows batch 1, 4 thread, dynamic padding bội số 8. Thời gian suy luận đầy đủ 805,5 giây; toàn bộ job 832,75 giây. Tại cùng ngưỡng 0,5, 99,826% nhãn trùng teacher. Sau hiệu chỉnh ngưỡng riêng, 71 mẫu đổi nhãn so với teacher ở ngưỡng của teacher. Chênh lệch nhỏ không được diễn giải là cải thiện chất lượng có ý nghĩa thống kê.

Không mở lại holdout đã dùng và không chấm lại test cũ để chọn bản nén. Điểm ở đây là **dev selection**, chưa phải kiểm chứng ≥90% trên holdout độc lập mới hay hội thoại thật. Demo Windows/Colab vẫn dự đoán NEOT cho “cảm ơn bạn” và “tôi muốn đặt vé đi hà nội”; cần tập câu ngắn/hội thoại đúng miền để đánh giá và cải thiện điểm yếu này.

[Model card và cách dùng](models/turn_v2_compact_20261024/README.md) · [Review HTML](outputs/compression_20261024/review.html) · [Manifest và hashes](models/turn_v2_compact_20261024/model_manifest.json)

Kiểm tra hoàn tất đạt **49/49**: hashes/dung lượng model, dữ liệu và marker holdout giữ nguyên, metrics tính lại từ dự đoán, ONNX checker, source thực thi, test Windows/Colab, ô notebook, parity, benchmark và crawl. Receipt: `outputs/compression_20261024/completion_validation.json`.

## Cách nén đã chạy

Ở bước thu gọn vocab FP32, giữ nguyên 12 lớp encoder, hidden size 768 và các trọng số encoder đã train. Thu gọn vocab Unigram, chọn đúng hàng embedding ứng với ID mới. Giữ toàn bộ token từng dùng trong train, token ký tự, byte và token đặc biệt. Token đa ký tự bị bỏ có thể được tách lại thành các token nhỏ hơn còn giữ. Bước INT8 tiếp theo xấp xỉ các trọng số theo thang lượng tử, không giữ chính xác giá trị FP32.

- Vocab embedding: 251.000 → 23.078 hàng.
- Token xuất hiện trong train: 2.142 ID.
- Đã kiểm tra 315.174 mẫu train: 0 mẫu đổi cách tách token sau khi ánh xạ ngược ID.
- Số tham số sau thu gọn: 103.766.786, so với 278.810.882 ban đầu. Mục tiêu 1/10 ở đây là **dung lượng file**, không phải số tham số hay số lớp.
- FP32 sau thu gọn: 416.108.770 byte.
- Kiểm tra 1.024 mẫu dev: 100% cùng nhãn với bản gốc; 4 mẫu thay đổi cách tách token. Đây là kiểm tra trên một phần dev, không dùng thay điểm dev đầy đủ.

Đã thử INT8 đối xứng theo từng hàng embedding, QUInt8 theo từng kênh của trọng số MatMul và lượng tử hóa động activation. Dùng ORT BASIC để loại alias Identity; tắt ConstantFolding khi lưu model vì bước này tạo các bảng relative position FP32 lớn. Sửa thứ tự node theo cả phụ thuộc ngầm trong nhánh `If`, rồi chạy ONNX checker. Bản này đạt dung lượng nhưng không đạt chất lượng (xem bảng dưới).

Bản được chọn giữ activation FP32: 72 tensor trọng số tuyến tính encoder được lượng tử hóa INT8 đối xứng theo kênh rồi giải lượng tử bằng `DequantizeLinear`. Relative embedding, pooler và classifier giữ FP32; word embedding dùng INT8 theo hàng. Cấu hình/thông tin đánh giá bổ sung khiến cả gói cuối là 109.609.065 byte; file ONNX không đổi. Đây là giảm dung lượng lưu trữ; không suy ra RAM hay tốc độ giảm 10 lần vì phép tính vẫn FP32 và giữ 12 lớp.

Runtime dùng `session.disable_quant_qdq=1` để ORT 1.23.2 constant-fold các trọng số `DequantizeLinear` một lần khi nạp, phù hợp vì activation ở FP32. Tùy chọn có trong [source ORT 1.23.2](https://github.com/microsoft/onnxruntime/blob/v1.23.2/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h). So với mặc định trên 128 mẫu dev, batch 16 và cùng input: 100% cùng nhãn, chênh xác suất tối đa 4,3095×10⁻⁶. Lượt full dev mặc định đã được dừng trước 1.000 mẫu để sửa cấu hình hiệu năng; receipt `weight_only_full_dev/abandoned_reason.json`. Lượt đầy đủ được ghi riêng tại `weight_only_full_dev_cached/`. Đây là kiểm tra runtime trên dev, không dùng holdout.

Trên cùng 1.024 mẫu dev đã chọn sẵn, bản này cùng nhãn teacher ở 1.023 mẫu (99,902%). Accuracy 93,359%, F1 EOT 89,971%; teacher tương ứng 93,262% và 89,808%. Mẫu nhỏ không đạt mốc F1 nhưng không cho thấy mất chất lượng rõ so với teacher; sau đó đã chấm **toàn bộ dev** và đạt hai mốc như bảng trên.

[ONNX Runtime](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) mô tả lượng tử hóa động cho transformer và yêu cầu đo lại chất lượng sau chuyển đổi. Quy trình ở đây bổ sung xử lý vocab và embedding theo cấu trúc cụ thể của model này.

## Bằng chứng đã ghi

- `outputs/compression_20261024/work/vocabulary_report.json`
- `outputs/compression_20261024/work/pruning_report.json`
- `outputs/compression_20261024/fp32_parity_1024/report.json`
- `outputs/compression_20261024/work/int8_quint8_no_constfold_size.json`
- `outputs/compression_20261024/quantization_job.json`
- `outputs/compression_20261024/int8_quint8_full_dev/progress.json`
- `outputs/compression_20261024/weight_only_full_dev_cached/report.json`
- `outputs/compression_20261024/weight_only_full_dev_cached/dev_predictions.npz`
- `outputs/compression_20261024/weight_only_amendment.json`
- `outputs/compression_20261024/benchmark.json`
- `outputs/compression_20261024/notebook_verification.json`
- `outputs/compression_20261024/windows_colab_parity.json`
- `outputs/compression_20261024/crawl_smoke/results.html`
- `outputs/compression_20261024/completion_validation.json`

30 unit test đã chạy lại đạt sau thay đổi runner (15,90 giây), gồm kiểm tra thực thi hai nhánh của graph `If` sau sửa thứ tự và phát hiện phụ thuộc vòng/thiếu input. Chuyển đổi và đánh giá nén chạy trên Windows; chưa thực hiện distillation hay cập nhật trọng số bằng gradient.

Sau khi người dùng reconnect, runtime mới `Colab GPU L4 (1)` có hai notebook session `a-jvsc-*`, đều idle. Connector được bổ sung `--session-path` để chọn chính xác, không tắt hay khởi động lại kernel nào. Phiên được chọn: `a-jvsc-f2543605-ce4e-4c3d-a99f-688dd99e9b33d84c1b1f-d7f4-4eb7-9690-6a3514d8ec38.ipynb`. Đã chạy phép nhân ma trận CUDA 256×256 trên NVIDIA L4, kết quả hữu hạn (`colab_reconnected.json`). Runtime mới không có project cũ; code và dữ liệu đã đồng bộ lại, có marker holdout đã dùng. Không tải lại holdout để đánh giá.

Đã upload bản nén theo chunk 4 MiB, đối chiếu SHA256 từng chunk và toàn bộ file, rồi sao chép sang thư mục model cuối sau khi full dev đạt. Cả 5 file trên Colab khớp manifest Windows. Ô `compression_result` cuối `a.ipynb` thực thi thật trên Colab, xác minh hashes và chạy ba demo. Xác suất trên ba demo giống Windows (max delta 0). Runner ONNX dùng CPU; phép thử CUDA xác minh GPU cho runtime, không biến kết quả này thành benchmark suy luận GPU. 30/30 test đạt trên source cuối ở Colab (7,16 giây). Notebook trước thay đổi được giữ tại `outputs/compression_20261024/a_before_compression.ipynb`.

Benchmark 128 mẫu dev cố định, seed 20261024, batch 1, 4 thread, warmup 8, tính cả normalization/tokenizer/inference:

| Profile | Teacher p50 / p95 | Compact p50 / p95 |
| --- | ---: | ---: |
| Cùng dynamic padding bội số 8 | 28,76 / 41,86 ms | 28,24 / 40,48 ms |
| Padding mặc định (teacher 256, compact bội số 8) | 194,24 / 257,52 ms | 28,68 / 43,63 ms |

Khi cùng padding, chênh lệch nhỏ, không kết luận tăng tốc đáng kể. Profile mặc định chênh p50 khoảng 6,77 lần nhưng đồng thời đổi padding; không gán mức tăng đó cho lượng tử hóa. Lượt crawl thật 12 mẫu từ Wikipedia đạt (4 EOT, 8 NEOT), đã lưu URL/hash nguồn và HTML; chưa có nhãn tham chiếu để tính accuracy. Không dùng dự đoán crawl làm nhãn train, không gọi Qwen hay RL trong đợt nén.

## Các lần chuyển graph chưa được chọn

- Bản lượng tử hóa đầu tiên lỗi ONNX checker vì sorter của quantizer chưa tính tensor được nhánh `If` lấy từ scope cha.
- Bản dùng ORT BASIC mặc định qua checker nhưng 142,8 MB, vượt ngân sách: ConstantFolding tạo thêm các bảng relative position FP32. Bản này không được dùng đánh giá chất lượng hay triển khai.
- Bản tắt ConstantFolding qua checker: ONNX 106.785.546 byte, thư mục triển khai khoảng 108,2 MB. Toàn bộ dev ở batch 1, dynamic padding bội số 8, CPU 4 thread: accuracy **89,480%**, F1 **82,289%** ở ngưỡng teacher 0,5; sau chọn ngưỡng trên dev (0,2), accuracy **90,961%**, F1 **86,605%**. Không đạt chất lượng, không được chọn triển khai. Receipt: `outputs/compression_20261024/int8_quint8_full_dev/report.json`.

Mã thực thi bản QUInt8 được giữ ở `outputs/compression_20261024/source/`; mã bản weight-only được chụp riêng tại `source_weight_only/`, có hashes. Không gán phiên bản source mới cho lần chạy cũ.
