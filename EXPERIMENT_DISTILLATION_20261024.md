## Material Passport

- Origin Skill: academic-research-suite/experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-24
- Verification Status: VERIFIED
- Version Label: tiny_turn_gru_v1
- repro_lock: outputs/distillation_20261024/protocol.json

## Mục tiêu

Tìm gói triển khai nhỏ nhất **trong các cấu hình đã thử** vẫn đạt accuracy và F1 EOT ≥90% trên toàn bộ dev. Đây là tìm kiếm hữu hạn, không chứng minh kích thước nhỏ nhất có thể trên mọi kiến trúc. Giữ bản đối chứng `models/turn_v2_compact_20261024/onnx/` (109.609.065 byte, dev accuracy 93,7924%, F1 90,5231%).

Chỉ train trên 315.174 mẫu train đã khóa. Dev 17.833 mẫu dùng chọn checkpoint, cấu hình, kiểu lưu trọng số và ngưỡng. Holdout đã dùng ở đợt train trước không được đánh giá lại. Kết quả được báo là **dev selection**, không phải kiểm chứng holdout độc lập mới/hội thoại thực.

## Kết quả hoàn tất

Đã chọn **GRU 96, embedding 64, 1 lớp**, **293.090 tham số triển khai**, checkpoint epoch 9. ONNX **328.761 byte**; cả thư mục `onnx/` gồm 7 file model/tokenizer/configuration/runner/requirements **555.479 byte**, bằng khoảng **1/197,3** gói teacher 109.609.065 byte. Không tính Python và thư viện runtime đã cài. ZIP đóng gói kèm README/manifest: **333.829 byte**; hashes tại `outputs/distillation_20261024/completion_validation.json`.

Dev sau INT8 và chọn ngưỡng **0,545**: accuracy **93,5625%**, F1 EOT **90,0744%**, precision **87,5021%**, recall **92,8024%**, FP 744, FN 404. So với teacher ở ngưỡng đã chọn, F1 giảm **0,4487 điểm phần trăm**; đây là đánh đổi để giảm file và tính toán, không phải giữ nguyên chất lượng. F1 student sát mốc 90%, không bảo đảm trên dữ liệu mới.

| Lượt train | Model + tokenizer INT8, KB | Accuracy dev | F1 EOT dev |
| --- | ---: | ---: | ---: |
| GRU 64, lượt đầu | 510,796 | 92,9849% | 89,4830% |
| GRU 64, train tiếp | 510,796 | 93,1363% | 89,5277% |
| **GRU 96, chọn** | **551,857** | **93,5625%** | **90,0744%** |
| GRU 128, lượt đầu | 605,210 | 93,3382% | 89,9151% |
| GRU 128, train tiếp | 605,210 | 93,4728% | 90,0632% |
| GRU 256, 2 lớp | 1.489,310 | 93,6690% | 90,3082% |

Mỗi lượt có ba kiểu lưu FP32/FP16/INT8 được chấm đầy đủ, tổng **18 biến thể**. Dung lượng bảng gồm 5 file model/tokenizer/cấu hình trước khi thêm cùng overhead runner và requirements; gói cuối 555.479 byte. GRU 96 là nhỏ nhất đạt cả hai mốc **trong các bản đã thử**, không phải cực tiểu tuyệt đối của mọi kiến trúc/hyperparameter.

Benchmark Windows 128 mẫu dev cố định, batch 1, **2 thread**, cùng padding bội số 8, warmup 8: teacher p50 **38,196 ms**, student p50 **0,328 ms**, student p95 **0,480 ms**. Đo bằng API chung, tính cả chuẩn hóa/tokenizer/inference; không suy ra độ trễ trên máy khác. Runner độc lập không import PyTorch/Transformers, input khớp API chung trên toàn bộ **17.833 mẫu dev** và bốn ca UTF-8/biên; **132** lượt so suy luận có max chênh xác suất **0**.

Đã đồng bộ 7 file và manifest lên Colab, thực thi thật ô `distillation_result` cuối `a.ipynb`, xác minh hash và chạy demo tiếng Việt. Demo Windows và Colab khớp xác suất. Cả câu đủ ý về Hà Nội và “cảm ơn bạn” vẫn ra NEOT; giữ nguyên kết quả để thể hiện giới hạn ngoài dữ liệu tham chiếu. Không dùng demo hay crawl làm nhãn train.

Kiểm tra hoàn tất **51/51 đạt**: files/hashes/ZIP, đối chứng và split giữ nguyên, cache train, 6 job hoàn tất, 18 biến thể dev, source thực thi, metrics tính lại từ dự đoán, ONNX checker, runner độc lập, Unicode, notebook/Colab parity, 33 unit test và benchmark. [Model card](models/turn_v2_distilled_min_20261024/README.md) · [Gói ZIP](models/turn_v2_distilled_min_20261024.zip) · [Review](outputs/distillation_20261024/review.html) · [Validation](outputs/distillation_20261024/completion_validation.json).

## Teacher và mục tiêu distillation

Teacher là ONNX INT8 trọng số/activation FP32 đã khóa, SHA256 `6e1a59a3f6d28afa0ed5d08cb0b77587c077d4928f3a2fbab2e4031aa66a8aac`. Khôi phục đúng các trọng số đã giải lượng tử sang PyTorch trên Colab L4 để tạo mục tiêu nhanh trên GPU. Kiểm tra 16 mẫu dev, FP32 không TF32: nhãn trùng 100%, max chênh xác suất 4,5913×10⁻⁷ so với ONNX CPU. Không đưa nhãn dev vào train.

Cache chỉ chứa input, nhãn, logits và đầu ra pooler 768 chiều cho **train**. Logits FP32, đặc trưng FP16; bật TF32 khi tạo mục tiêu train. Teacher không nhận gradient và không cập nhật trọng số. `cache/cache_report.json` ghi hashes/tổng mẫu, `cache/teacher_parity.json` ghi kiểm tra khôi phục. Teacher embedding được dùng khởi tạo embedding student qua PCA.

Loss: `0.5 * CE(label_smoothing=0.02) + 0.5 * T² * KL(student/T, teacher/T) + 0.05 * MSE(layer_norm(student_projection), layer_norm(teacher_pooler))`, với `T=2`. Projection sang 768 chiều chỉ dùng lúc train và bị loại khỏi ONNX triển khai. Đây là logits/feature distillation cho tác vụ; không phải tái lập đầy đủ quy trình attention/layer distillation của TinyBERT. Tham khảo gốc: [Hinton et al.](https://arxiv.org/abs/1503.02531), [TinyBERT](https://aclanthology.org/2020.findings-emnlp.372/).

## Student và thiết kế tìm kiếm

GRU một chiều, LayerNorm embedding, pooling last/mean/max, embedding của piece cuối và hai đặc trưng độ dài; head GELU hai lớp. Không dùng tương lai ngoài text đầu vào. Right padding không ảnh hưởng timestep hợp lệ và bị loại khỏi pooling. Ba test mới kiểm tra padding, batch khác độ dài và gradient hữu hạn đều đạt. Tổng bộ test hiện tại: 33 passed.

Tokenizer Unigram mới chỉ giữ token đã xuất hiện trong raw train, fallback Latin/ASCII/byte và special token. Vocab 23.078 → **3.313**, tokenizer **219.195 byte**; kiểm tra mọi input train: **0 thay đổi segmentation**. Giảm fallback đa ngôn ngữ là một giới hạn: không hứa giữ chất lượng ngoài miền tiếng Việt đang kiểm tra.

Lượt đầu: hidden/embedding/layers `(64,64,1)`, `(128,64,1)`, `(256,96,2)`, tối đa 8 epoch, LR 0,001, batch 256, seed 42, AdamW, warmup 100 bước + cosine, clip 1; chọn checkpoint bằng min(accuracy, F1) ở ngưỡng 0,5. Early stop sau 3 epoch không tăng, sớm nhất từ epoch 4. Sau đó chọn ngưỡng trên lưới dev cố định 0,05–0,95 (181 điểm).

Refinement có receipt riêng: GRU 128 tốt nhất ở epoch cuối nên train tiếp từ checkpoint, 8 epoch ở LR 0,0002. Kiểm tra thêm GRU 64 cùng cách train tiếp và GRU 96/embedding64 một lớp, tối đa 16 epoch. Các lượt trước được giữ nguyên, không ghi đè. Không có ablation CE-only; không kết luận lợi ích nhân quả của KD so với chỉ supervision.

## Export và lựa chọn

Mỗi student được export FP32 và hai dạng lưu trọng số FP16/INT8; activation và phép tính vẫn FP32. Với INT8, dùng scale theo hàng embedding/output channel, giữ bias ở FP32; ORT constant-fold các trọng số một lần khi nạp. ONNX checker, so PyTorch ở batch 1/3/8 và kiểm tra tăng right padding đều chạy trước đánh giá chất lượng. INT8 được chấm lại full dev, không lấy parity 8 mẫu làm bằng chứng ≥90%.

Các report đánh giá ban đầu dùng source `source_evaluation_initial/`: trường cũ `dev_at_teacher_threshold` thực tế là tại ngưỡng input contract của student, không phải ngưỡng teacher. Chỉ `dev_selected_threshold` được dùng chọn model. Đã sửa evaluator để tách `dev_at_contract_threshold` và ngưỡng teacher thực cho các lượt sau; dữ liệu dự đoán và điểm chọn của các lượt trước không thay đổi.

## Bằng chứng

- `outputs/distillation_20261024/protocol.json`, `refinement_protocol.json`
- `source_initial/`, `source_refinement/`, `source_evaluation_initial/`
- `cache/cache_report.json`, `cache/teacher_parity.json`
- `vocabulary/vocabulary_report.json`
- `runs/*/run_identity.json`, `training_report.json`, `history.json`, `dev_predictions.npz`
- `runs/*/export_report.json`, `runs/*/quality_*/report.json`
- `jobs/*/status.json`, `stdout.log`: PID, heartbeat, timeout và exit code thực

Kiểm tra khôi phục teacher trên Windows bị chặn bởi cặp torch/torchvision sẵn có (`torchvision::nms`); không sửa/xóa môi trường của người dùng. Khôi phục và kiểm chứng thật đã chạy đạt trên Colab. Các student/ONNX được đánh giá độc lập trên Windows.

Lệnh demo đầu tiên qua stdin PowerShell làm dấu tiếng Việt thành `?`; đã giữ receipt `_initial.json` và thay bằng script UTF-8 chạy trực tiếp từ file. Các file dev đều được đọc UTF-8 nên điểm chất lượng không bị ảnh hưởng. Validator cũng được sửa để đọc UTF-8 rõ ràng. Khi import runner từ file, Python sinh bytecode cache trong thư mục model; đã xóa đúng cache do lượt kiểm tra tạo, tạm tắt ghi bytecode khi kiểm tra notebook và xác minh lại toàn bộ gói. Model trọng số/dự đoán không thay đổi trong các sửa kiểm tra này.
