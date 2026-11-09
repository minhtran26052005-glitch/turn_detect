# Turn Detection Service

Phân loại ranh giới câu cho ba model riêng `vi`, `en`, `zh`. Input gồm đoạn văn bản bên trái và **1 hoặc 2 từ tiếp theo**; text được chuẩn hóa chữ thường và bỏ dấu câu. Output là `EOT` hoặc `NEOT` và xác suất `EOT`.

Nhãn được tạo từ dấu câu nguồn: dấu cuối câu là `EOT`, dấu ngắt trong câu là `NEOT`. Mỗi model dùng tokenizer/weights và ngưỡng chọn trên DEV của chính nó. Với tiếng Trung, các từ lookahead được phân tách bằng dấu cách.

## Kết quả test tạm thời

| Model | Số mẫu test | Accuracy | EOT F1 |
|---|---:|---:|---:|
| Việt | 39.167 | 81,14% | 79,08% |
| Anh | 28.155 | 84,85% | 85,02% |
| Trung | 51.871 | 84,20% | 67,18% |

Đây là kết quả của checkpoint **cycle 75** trên văn bản có nhãn suy từ dấu câu, không phải test hội thoại/audio do người gán nhãn. Các chỉ số chi tiết nằm trong [`results/c75.json`](results/c75.json).

Model Việt INT4 nhánh cũ đạt **93,55% accuracy / 90,19% EOT F1 trên DEV cũ**; đây là tập đã dùng để chọn model, chưa phải kết quả test của student trên cycle 75. Tiếng Anh cycle 84 sau một epoch đạt **86,85% accuracy / 82,27% EOT F1 trên DEV**.

## Ví dụ đầu vào và kết quả

Các ví dụ dưới đây được lấy từ `data/test/samples.jsonl` và là kết quả đã chấm của model cycle 75. Model dự đoán ranh giới ngay sau phần **ngữ cảnh**, sử dụng thêm 1–2 từ ở cột lookahead.

| Ngôn ngữ | Ngữ cảnh | Lookahead | Nhãn test | Model dự đoán | Xác suất EOT |
|---|---|---|---|---|---:|
| Việt | cô nắm tay định đẩy cửa vào nhưng vội thu lại | cô | EOT | EOT | 0.8317 |
| Việt | tôi thấy rằng sau 37 năm | những | NEOT | NEOT | 0.0173 |
| Anh | we rent a field from a farmer | the farmer | EOT | EOT | 0.6960 |
| Anh | after the war | he returned | NEOT | NEOT | 0.0066 |
| Trung | 强制 减支 会 过于 激进 让 人 难以承受 并且 影响 到 一系列 部门 和 计划 | 五角大楼 | EOT | EOT | 0.9310 |
| Trung | 有 了 这个 独特 的 能力 | 在 | NEOT | NEOT | 0.0028 |

`EOT` là kết thúc câu; `NEOT` là chưa kết thúc. Nhãn test được suy từ dấu câu nguồn. Xác suất là điểm EOT của model, dùng ngưỡng chọn trên DEV để ra dự đoán.

## Mã chính

- `turn_detection/train.py`, `infer.py`, `distill.py`: train, suy luận và distill model hiện tại.
- `legacy/turn_v2/`: distill, QAT và export model Việt INT4 của nhánh cũ.
- `tests/`: kiểm tra các chức năng; các file còn lại trong `turn_detection/` là thành phần dùng chung.

## Cài đặt

Chạy train, distill và kiểm thử model trên Colab GPU. Mount Drive có bản sao lưu, sau đó:

```bash
pip install -r requirements.txt
```

Colab cần có PyTorch CUDA. Bộ dữ liệu và checkpoint được lưu trên Drive, không nằm trong Git.

## Chuẩn bị dữ liệu và checkpoint

```bash
DRIVE=/content/drive/MyDrive/turn_detect_backups/translation_segmenter_v15d_20261026
RUN=/content/turn-detection-work

python -m turn_detection.assets training \
  --manifest "$DRIVE/latest_manifest.json" --run-dir "$RUN" --language vi
```

Lệnh kiểm SHA-256, khôi phục dữ liệu, model tốt nhất và checkpoint có optimizer/RNG. Lặp lại với `en` hoặc `zh` để khôi phục model tương ứng. Cần đủ dung lượng cho SQLite giải nén, checkpoint, cache và ít nhất 25 GiB trống. Không dùng cùng thư mục làm việc với worker đang train.

## Train

```bash
python -m turn_detection.train --run-dir "$RUN" --language vi \
  --generation cycle000084_full_epoch_contiguous_v1 --epoch 1
```

Thay `--generation` và `--epoch` theo kết quả từ lệnh khôi phục. Một epoch đi qua toàn bộ mẫu TRAIN hợp lệ; checkpoint được lưu định kỳ. Khi hết epoch, model được đánh giá và chọn ngưỡng trên DEV. `--budget 500` giới hạn một lượt ở 500 bước và lưu checkpoint để chạy tiếp.

## Infer

```bash
python -m turn_detection.infer --model "$RUN/teachers/vi/best" \
  --context "hôm nay trời đẹp" --future "chúng ta"
```

Ngưỡng được đọc từ `best_report.json` cạnh thư mục model; có thể truyền `--threshold` nếu dùng checkpoint khác. Dùng `--device cpu` để chạy trên CPU.

## Distill

```bash
python -m turn_detection.distill --run-dir "$RUN" --language vi \
  --output "$RUN/students/vi" --epochs 3

python -m turn_detection.infer --model "$RUN/students/vi" \
  --context "hôm nay trời đẹp" --future "chúng ta"
```

Student GRU dùng tokenizer của teacher; loss kết hợp nhãn TRAIN và logits teacher. Checkpoint lưu định kỳ, thêm `--resume` để tiếp tục. Chọn model và ngưỡng trên DEV. Student này cần đánh giá riêng; đây không phải bản INT4 lịch sử trong `legacy/`.

## Test và kết quả

- `data/test/samples.jsonl`: 360 mẫu đã chấm, mỗi ngôn ngữ 120 mẫu. Bộ đầy đủ 119.193 mẫu được giữ trên Drive.
- `results/c75.json`: điểm tạm thời của ba checkpoint cycle 75. Đây là test văn bản với nhãn từ dấu câu, không phải đánh giá hội thoại/audio được gán nhãn bởi người.
- `results/full_epoch_dev.json`: kết quả DEV của tiếng Anh sau vòng đầu trên dữ liệu cycle 84.
- `results/vi_student_legacy.json`: kết quả DEV của student Việt INT4 lịch sử, không phải kết quả student mới hoặc checkpoint cycle 75.

Để lấy trace gốc và kiểm SHA-256 trên Colab:

```bash
python -m turn_detection.assets test --diagnostic-dir "$DRIVE/diagnostic_test_c75_v1" \
  --output "$RUN/test/c75_raw"
python -m turn_detection.summarize_test --input "$RUN/test/c75_raw"
```

Các trace chứa cả mẫu bị loại; `summarize_test` chỉ tính mẫu `included_in_clean_test=true`. Lệnh tổng hợp dự đoán đã lưu, không chạy lại model. Không dùng test đã xem để chọn checkpoint hoặc ngưỡng.

Nguồn test: FineWeb/FineWeb-2, Wikipedia và BAAI IndustryCorpus2 news/media. Tập chưa chấm được giữ trên Drive.

## Student INT4 lịch sử

Mã distill, QAT và export gốc nằm trong `legacy/turn_v2`. Quy trình này dùng teacher, tokenizer và dữ liệu của nhánh cũ; cần các artifact gốc tương ứng. Chi tiết lệnh: [legacy/README.md](legacy/README.md).

## Kiểm tra

```bash
python -m pytest -q
```
