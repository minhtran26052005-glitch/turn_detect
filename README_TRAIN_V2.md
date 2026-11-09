# Train turn detector V2 trên Colab

Đã chạy [pilot Việt–Anh–Trung](README_TRILINGUAL_V3.md) theo teacher→distill→QAT, **chưa đạt90% cả ba tiếng và có regression Việt**; giữ bản Việt làm mặc định. [Kết quả](outputs/trilingual_20261024/review.html) · [Player streaming từng prefix](outputs/trilingual_stream_20261024/review.html). Ô `trilingual_result` cuối notebook dùng bản thử riêng.

Đã chạy [QAT INT4](models/turn_v2_distilled_qat4_20261024/README.md) trên Colab: ONNX **177.1 KB**, cả gói **404.0 KB**, dev accuracy **93.55%**, F1 EOT **90.19%**, ngưỡng **0.470**. Trọng số4-bit và activations/phép tính FP32; chưa có holdout mới. [ZIP](models/turn_v2_distilled_qat4_20261024.zip) · ô `qat_result` cuối `a.ipynb`.

Đã kiểm tra thêm [student INT4](models/turn_v2_distilled_int4_20261024/README.md): **ONNX 186 KB, cả gói 413 KB**, dev accuracy **93,43%**, F1 EOT **90,06%**, ngưỡng mới **0,430**. Lưu trọng số 4-bit, phép tính FP32; dev dùng chọn bản/ngưỡng, chưa có holdout độc lập mới. [ZIP](models/turn_v2_distilled_int4_20261024.zip) · ô `int4_result` cuối `a.ipynb`.

Đã có [student distillation nhỏ nhất trong các bản đã thử](models/turn_v2_distilled_min_20261024/README.md): **ONNX 329 KB, cả model/tokenizer/runner 555 KB**, dev accuracy **93,56%**, F1 EOT **90,07%**. Đã train trên Colab L4, chấm 18 biến thể và chạy ô cuối `a.ipynb`. F1 sát ngưỡng 90%, chưa có holdout độc lập mới. [ZIP dùng ngay](models/turn_v2_distilled_min_20261024.zip) · [Review](outputs/distillation_20261024/review.html).

Đã có [bản nén dùng được](models/turn_v2_compact_20261024/README.md): ONNX **108,2 MB**, cả gói **109,6 MB**, nhỏ hơn 10 lần ONNX đã train. Trên **17.833 mẫu dev**, accuracy **93,79%**, F1 EOT **90,52%**; chưa có holdout độc lập mới cho bản nén. Ô cuối `a.ipynb` đã chạy trên Colab mới, kiểm tra hashes và suy luận. Xem [báo cáo nén](outputs/compression_20261024/review.html).

Bản train thử 24/10 đã đạt **holdout accuracy 93,81%, F1 EOT 90,62%** trên bộ nhãn sẵn đã khóa. Test cũ đạt accuracy 93,47%, F1 EOT 88,68%, còn dưới mốc F1 90%. Xem [kết quả, giới hạn và cách dùng model](EXPERIMENT_TRIAL_20261024.md) hoặc [báo cáo trực quan](outputs/training_trial_20261024/review.html). `a.ipynb` có thêm kết quả và ưu tiên model mới trong ô crawl nếu checkpoint còn trên runtime.

Để crawl rồi dự đoán từng mẫu ngay, xem [README_CRAWL_PREDICT.md](README_CRAWL_PREDICT.md) và ô mới cuối `a.ipynb`.

Mở **`a.ipynb`**, chọn kernel **Colab GPU L4**, chạy từ trên xuống. Notebook đã nhúng mã nguồn để dùng lại trên runtime mới. Bản notebook trước khi sửa nằm ở `backups/a.before_v2.ipynb`.

Mục tiêu được chốt là **cả accuracy và F1 EOT ≥90%**. Smoke test chỉ kiểm chứng kết nối, gán nhãn, backprop, lưu/nạp và export; không phải kết quả chất lượng. Bản đạt mục tiêu bên trên là đợt train thật 2.000 bước, checkpoint chọn ở bước 1.750; chưa chạy hết 3 epoch.

## Dùng notebook

Mặc định notebook crawl và gán nhãn **4 ứng viên**, sau đó train **3 bước** trên một tập nhỏ. Dùng cấu hình này để kiểm tra hệ thống trước.

Để train đầy đủ bằng dữ liệu nhãn sẵn, sửa ô cấu hình:

```python
RUN_SMOKE = False
RUN_DATA_ROUTE = False
RUN_FULL_TRAIN = True
RUN_FINAL_HOLDOUT = False
EXPERIMENTS = ['gold']
```

Để so sánh cả hướng tăng dữ liệu và hướng train trên cùng split, model revision, seed và cấu hình:

```python
RUN_SMOKE = False
RUN_DATA_ROUTE = True
QWEN_LIMIT = 1000  # Tăng theo ngân sách GPU; không phải cam kết 1.000 nhãn được chấp nhận.
RUN_FULL_TRAIN = True
EXPERIMENTS = ['gold', 'gold_silver']
RUN_FINAL_HOLDOUT = False
```

Trong tên biến/run, `gold` nghĩa là **nhãn tham chiếu đang có trong train.jsonl**, chưa xác minh là nhãn do người duyệt. Hai run so sánh trên dev; chưa mở holdout. Qwen xử lý trước encoder, không giữ hai model trên GPU cùng lúc.

Sau khi chọn model bằng dev, bật `RUN_FINAL_HOLDOUT=True` rồi chạy riêng ô đánh giá cuối. Code chỉ cho mở holdout nếu dev đạt cả hai mốc 90%. Nếu đã reconnect, có thể đặt `EXISTING_RUN_DIRS` trỏ các thư mục run cũ và chạy các ô chọn model/đánh giá, không cần train lại.

## Hai hướng đã code

**Hướng train:** khởi tạo từ [microsoft/mdeberta-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base), không cố tiếp tục train từ bản ONNX cũ. Encoder LR `2e-5`, classifier/pooler LR gấp 5, AdamW, cosine, warmup 6%, clipping 1, label smoothing 0,02, early stopping, gradient checkpointing. Batch mặc định 16 × tích lũy 2, 3 epoch. Mặc định FP32/TF32 trên L4; một lượt BF16 đã phát sinh logits không hữu hạn ở đánh giá nên không dùng BF16 làm mặc định.

Checkpoint tốt nhất được chọn bằng `min(accuracy, F1 EOT)` trên dev tại ngưỡng 0,5; sau đó quét một lưới ngưỡng cố định trên dev để tối đa hóa cùng chỉ số. Ngưỡng được lưu cùng model; không chọn ngưỡng từ test/holdout.

**Hướng data:** crawl một số bài Wikipedia tiếng Việt có giới hạn → giữ URL/page ID/hash nguồn → tạo câu và prefix chưa gán nhãn → [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) gán nhãn bằng hai cách hỏi. Chỉ nhận khi hai cách hỏi cùng trả EOT/NEOT và cùng đủ mức tự tin cấu hình. Bất đồng, mơ hồ hoặc JSON lỗi đều được loại. Hai cách hỏi cùng model không độc lập; confidence tự khai chưa được hiệu chỉnh. Nhãn nhận vào vẫn là **silver**, weight 0,5; không biến thành nhãn người duyệt.

Wikipedia là nguồn văn xuôi khởi đầu. Để cải thiện đúng nhu cầu hội thoại/ASR, cần thêm nguồn cùng miền và một tập hội thoại mới do người duyệt, có `source_id`/ID cuộc hội thoại. Không dùng điểm trên văn xuôi để kết luận đã đạt chất lượng hội thoại thật.

## Dữ liệu và split cố định

| Phần | Số mẫu |
| --- | ---: |
| Train | 315.174 |
| Dev | 17.833 |
| Holdout | 17.099 |
| Cách ly vì nhóm liên quan test cũ | 2.731 |
| Cách ly vì trùng sau chuẩn hóa | 1 |

`data_v2/frozen/manifest.json` ghi hashes, seed và cách nhóm. Code ưu tiên `source_id`; với dữ liệu cũ, suy nhóm từ EOT và các prefix đi sau, rồi nối nhóm có prefix dài giống nhau. Có 7.217 NEOT ban đầu không khớp EOT ngay trước; thiếu ID gốc nên cách nhóm chỉ là heuristic, không chứng minh độc lập theo hội thoại.

Các split bất biến: đổi input, seed hoặc sửa split sẽ bị phát hiện. Chọn thư mục phiên bản dữ liệu mới nếu chủ động thay protocol. Silver chỉ được thêm vào train sau khi lọc trùng exact/source/prefix với tập bảo vệ. Bộ lọc này không chứng minh không có paraphrase hoặc trùng nội dung ngữ nghĩa.

`test.jsonl` cũ đã được xem ở baseline, chỉ dùng làm tập tham chiếu đã lộ; không đổi nhãn nó và không dùng để chọn model mới. Final holdout lưu `holdout_consumed.json`; sau khi mở, không đánh giá một candidate khác để tiếp tục chọn model trên cùng holdout.

## File và lệnh chính

| File/module | Công dụng |
| --- | --- |
| `a.ipynb` | Notebook đầy đủ có output chạy thật |
| `turn_v2/prepare.py` | Nhóm, cách ly, chia train/dev/holdout, khóa hash |
| `turn_v2/crawl.py` | Crawl có provenance, không tự gán nhãn prefix |
| `turn_v2/label_qwen.py` | Qwen trên GPU; lưu audit, silver và revision |
| `turn_v2/train.py` | Train, resume, chọn checkpoint/ngưỡng từ dev |
| `turn_v2/evaluate.py` | Đánh giá final holdout với model/ngưỡng cố định |
| `turn_v2/export.py` | Export ONNX FP32, kiểm tra parity |
| `turn_v2/predict.py` | Suy luận theo chuẩn tiền xử lý và ngưỡng mới |
| `tools/colab_runtime.py` | Dùng phiên Colab đã đăng nhập trong extension VS Code |
| `tools/sync_colab.py` | Đóng gói và upload code/data vào runtime đang kết nối |
| `tools/run_notebook_colab.py` | Chạy notebook qua kết nối hiện tại, lưu output vào notebook |
| `requirements-colab.txt` | Phiên bản thư viện đã kiểm tra trên runtime này |

Trong terminal **Colab**, sau khi đã có code/data:

```bash
cd /content/turn_detect_v2
python -m turn_v2.train --data data_v2/frozen --output runs/gold_v1 --epochs 3 --export
python -m turn_v2.train --data data_v2/frozen --output runs/gold_silver_v1 --silver data_v2/qwen_RUN/silver.jsonl --epochs 3 --export
python -m turn_v2.predict --model runs/gold_v1/onnx --text "tôi muốn đặt vé đi hà nội"
```

Resume một run đã có:

```bash
python -m turn_v2.train --data data_v2/frozen --output runs/gold_v1 --epochs 3 --export --resume runs/gold_v1/checkpoints/checkpoint-500
```

Giữ đúng model/revision, dữ liệu, seed và cấu hình gốc khi resume. Không dùng thư mục smoke để resume train đầy đủ. Giảm batch nếu gặp OOM và tăng accumulation tương ứng; nếu đổi cấu hình, bắt đầu run mới.

Trong PowerShell **Windows**, kiểm tra hoặc đồng bộ lại Colab đang được extension kết nối:

```powershell
python .\tools\colab_runtime.py --receipt .\outputs\colab_connection_test.json
python .\tools\sync_colab.py
python .\tools\run_notebook_colab.py
```

Nếu một runtime có nhiều phiên notebook, các lệnh trên hỗ trợ `--session-path` với đường dẫn Jupyter của phiên cần dùng và `--server-index` để chọn runtime. Không tự chọn ngẫu nhiên hay tắt kernel khác. Trên runtime đã đồng bộ hiện tại, bạn chỉ cần chạy ô cuối notebook để kiểm tra bản nén; ô này không phụ thuộc các ô train/smoke trước đó.

Connector chỉ đọc credential của assigned runtime trong extension `google.colab`, giữ trong bộ nhớ và dùng HTTPS/WSS; không lưu token vào notebook, bundle hay report. Không tạo runtime mới hoặc đổi tài khoản. Khi credential hết hạn, reconnect bằng extension rồi chạy lại. Windows cần `requests`, `websocket-client`, `cryptography`; máy hiện tại đã có.

Các model nền là public; notebook không cần đưa HF/API key vào chat. Code ghi revision model và hash dữ liệu/source vào report. Kernel và dữ liệu gốc không bị tắt/xóa. `/content` là bộ nhớ tạm: lưu checkpoint, manifest và marker holdout vào Drive đã mount hoặc tải về trước khi xóa runtime.

## Kết quả và phạm vi kiểm chứng

Xem `outputs/colab_notebook_test.json` cho kết quả từng ô và `outputs/colab_v2_smoke/` cho các report tải từ Colab. Model smoke đánh dấu `smoke_only=true`, không được dùng để xác nhận mốc 90%. Model mới có chuẩn chuẩn hóa/ngưỡng riêng, dùng `turn_v2.predict` thay cho runner bản cũ.

Đã chạy qua 11 ô code, 18/18 test trên Colab; smoke dùng 34 mẫu train (gồm 2 silver), 16 dev, 3 bước optimizer. Parity ONNX FP32 trên 8 mẫu chênh xác suất EOT lớn nhất `1,51×10⁻⁷`, các lớp argmax giống nhau. Kết nối WebSocket được kiểm tra tái sử dụng 3 lần. Chi tiết và giới hạn nằm ở `EXPERIMENT_V2.md`.

Model nền mới lớn hơn checkpoint ONNX hiện tại; chưa có benchmark độ trễ của model train đầy đủ. Không suy ra khả năng chạy realtime từ smoke test.

Nguồn API: [Transformers Trainer 5.20](https://huggingface.co/docs/transformers/v5.20.0/main_classes/trainer), [Colab extension chính thức](https://github.com/googlecolab/colab-vscode).
