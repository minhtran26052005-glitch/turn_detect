# QAT INT4 student — 2026-10-24

Material Passport: **VERIFIED: TRAINED, PACKED FULL DEV, WINDOWS, COLAB, ZIP**.

Đã fine-tune **QAT trọng số INT4** trên Colab L4 với toàn bộ315.174 mẫu train: giả lập lượng tử ở mỗi forward, gradient clipped STE đến cả năm ma trận. Nhóm32 và64, tối đa8epoch, LR0,0001, seed43, batch256. Loss gồm nhãn train và distillation logits/đặc trưng teacher cố định. Scale FP16 khởi tạo bằng MSE trọng số rồi giữ nguyên cả lúc train và export; không tái fit scale sau QAT.

Chọn **nhóm64**, 8,624 bước, best epoch4. **ONNX177,124byte**, cả model/tokenizer/runner/requirements **403,973byte**, không tính thư viện đã cài. Accuracy **93.5457%**, F1 EOT **90.1934%**, ngưỡng **0.470** trên đủ17.833 frozen dev bằng chính ONNX INT4 đã đóng gói. GPU fake-quant so với CPU packed lệch xác suất tối đa1.37e-05.

| Bản | ONNX KB | Accuracy dev | F1 EOT dev | Ngưỡng |
| --- | ---: | ---: | ---: | ---: |
| INT8 | 328.761 | 93.5625% | 90.0744% | 0.545 |
| PTQ_INT4 | 185.933 | 93.4335% | 90.0620% | 0.430 |
| PTQ_INT4 g64 | 176.999 | 93.3606% | 89.9183% | 0.490 |
| QAT g32 | 186.058 | 93.5793% | 90.2229% | 0.500 |
| QAT g64 | 177.124 | 93.5457% | 90.1934% | 0.470 |

Trọng số ma trận lưu4-bit thực sự, hai mã mỗi byte. Bias/norm, activations và phép tính GRU vẫn FP32; QAT không tự tạo kernel INT4. Mục tiêu lượt này là dung lượng và điểm dev. Dev dùng chọn checkpoint/ngưỡng, không phải đánh giá độc lập; không chấm lại consumed holdout. Nhiều NEOT là tiền tố EOT và các câu ngắn như “cảm ơn bạn” có thể sai. Không quy toàn bộ thay đổi điểm cho QAT vì chưa có đối chứng fine-tune FP32 với cùng ngân sách.

Lần khởi động nhóm64 đầu lỗi cuDNN vì stride từ nhóm cuối chưa đầy trước bước optimizer nào; sửa output fakequant `.contiguous()`, thêm test CUDA forward/backward lặp lại, rồi chạy `r2` riêng. Giữ log lỗi và mã nguồn trước sửa. Không tính lần khởi động lỗi là mô hình QAT hoàn tất. Source/receipts/controls trong `outputs/qat_20261024/`.

Hoàn tất kiểm tra **94/94**: hai job QAT completed/exit0, fakequant có gradient qua năm ma trận, scale giữ nguyên từ lúc khởi tạo, packed decode đúng trọng số QAT, full dev hai nền tảng, identities frozen/bản cũ không đổi, notebook đã chạy Colab, **39 tests pass** gồm CUDA partial-group forward/backward. ZIP187,305byte, SHA256 `4dd234efbb9079ac81759902eff98b62af4f7c941bfdccf545aeca807a97e0fa`.
