# Chẩn đoán học mALBERT Việt Anh

Đã hoàn thành và kiểm chứng độc lập.

Phép thử học thuộc dùng 1.000 input TRAIN cố định và duy nhất, chia 500 mỗi tiếng, gồm 125 EOT và 375 NEOT mỗi tiếng. Model đạt accuracy, F1 và recall 100% sau 200 cập nhật với AdamW mới, LR 3e-5, batch hiệu dụng 32. Sau lỗi giới hạn VRAM ở lần đầu, microbatch 4 và tích lũy gradient được dùng để tiếp tục từ cùng sample và pin.

Hai nhánh LR kế thừa cùng checkpoint 2585080, model, 27 trạng thái AdamW, RNG và thứ tự nguồn. Mỗi nhánh có 3.000 cập nhật và 47.122 điểm được tính loss. Các dự đoán khác nhau có thể dẫn đến ngữ cảnh reset khác nhau về sau.

Ngưỡng được chọn riêng trên calibration bằng mục tiêu precision tối thiểu 50% mỗi tiếng rồi tối đa hóa recall của tiếng yếu hơn. Cả ba checkpoint đạt điều kiện trên calibration. Precision thực tế trên monitor được báo riêng trong bảng.

| Nhánh | Tiếng | Ngưỡng | F1 monitor | Precision monitor | Recall monitor |
| --- | --- | --- | --- | --- | --- |
| baseline | vi | 0.2 | 23.00% | 47.34% | 15.19% |
| baseline | en | 0.2 | 46.66% | 57.87% | 39.08% |
| lr4e6 | vi | 0.4 | 27.70% | 48.10% | 19.45% |
| lr4e6 | en | 0.4 | 47.06% | 54.55% | 41.38% |
| lr1e5 | vi | 0.9 | 23.27% | 56.21% | 14.68% |
| lr1e5 | en | 0.9 | 42.71% | 52.07% | 36.21% |

Nhánh 4e-6 có recall cao hơn 1e-5 ở cả hai tiếng trong phép thử ngắn này. Nhãn dựa trên dấu câu và tập học thuộc được phân tầng chủ động, vì vậy kết quả học thuộc chỉ chứng minh khả năng fit tập nhỏ. DEV vẫn còn nhiều EOT bị bỏ sót.

Toàn bộ 105.810 inference monitor đã tái hiện chính xác và được kiểm lại input, nhãn, hash token, future cùng turn và reset dự đoán. Kiểm 48 trường hợp canonical batch đơn có chênh xác suất tối đa 2,84e-7. Checkpoint, optimizer, RNG, source và code fingerprint đã kiểm chứng. TEST được giữ ngoài thí nghiệm. Trainer chính tiếp tục hoạt động.

Lần kiểm chứng đầu gặp thiếu trường split trong schema TRAIN đã chuẩn bị. Validator đã sửa để xác minh provenance qua SHA256 file TRAIN và manifest của checkpoint cố định. Metadata và source trước sửa giữ trong history/verification_schema_repair.

Báo cáo đầy đủ ở report.json, bằng chứng ở completion_verification.json và bảng xem nhanh ở review.html. Full checkpoint, sample, weights và trace giữ trên SSH.

SHA256 báo cáo `e1f7da5819cafc78b1834206ea179cbdf3257673d5165dacd5119654f1261e28`
