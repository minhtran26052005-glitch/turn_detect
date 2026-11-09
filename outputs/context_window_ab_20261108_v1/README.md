## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run and descriptive validation
- Origin Date: 2026-11-08
- Verification Status: VERIFIED for declared technical checks; CAUTION for generalization
- Version Label: context_window_ab_20261108_v1

# Đối chứng ngữ cảnh A/B

Hai nhánh mới cùng xuất phát từ checkpoint v7 3.731.173, giữ nguyên model 11.941.122 tham số, model/27 trạng thái AdamW/RNG/thứ tự nguồn ban đầu. Mỗi nhánh học đủ 333.315 điểm ASR tự nhiên trong một lượt; effective batch 16 mẫu được chấm, 20.833 lần cập nhật. LR 4e-6, loss NEOT1/EOT2 và future 1–2 từ cùng lượt nguồn giữ nguyên.

- **A:** encoder nhìn từ lần predicted EOT gần nhất.
- **B:** encoder nhìn tối đa 256 từ nguồn gần nhất, độc lập với quyết định cắt; cả hai được encode tối đa 256 token tổng. Quyết định cắt cho decoder vẫn được theo dõi.
- Chỉ thay chính sách ngữ cảnh trong TRAIN và DEV. Hai nhánh fit xong trước khi chấm monitor. Không chạy TEST/translator hoặc thay model mặc định.

## Kết quả DEV

| Nguồn/ngôn ngữ | A: F1 / P / R (%) | B: F1 / P / R (%) | ΔF1 (điểm %) |
|---|---:|---:|---:|
| Toàn DEV VI | 31.09 / 25.32 / 40.27 | 27.28 / 24.43 / 30.89 | -3.81 |
| Toàn DEV EN | 44.85 / 32.97 / 70.11 | 43.75 / 34.23 / 60.63 | -1.10 |
| asr EN | 48.46 / 39.52 / 62.63 | 50.23 / 44.62 / 57.44 | +1.77 |
| asr VI | 31.85 / 29.52 / 34.57 | 28.99 / 28.27 / 29.74 | -2.86 |
| daily EN | 49.90 / 36.50 / 78.85 | 45.54 / 34.93 / 65.38 | -4.36 |
| daily VI | 36.36 / 27.89 / 52.24 | 27.33 / 23.40 / 32.84 | -9.03 |
| factory EN | 45.32 / 32.61 / 74.26 | 39.60 / 29.70 / 59.41 | -5.71 |
| factory VI | 34.35 / 28.05 / 44.29 | 30.09 / 26.82 / 34.29 | -4.25 |
| numeric/dialogue EN | 24.82 / 15.11 / 69.39 | 31.78 / 20.61 / 69.39 | +6.96 |
| numeric/dialogue VI | 11.70 / 7.59 / 25.58 | 13.43 / 9.89 / 20.93 | +1.73 |

Ngưỡng calibration chọn riêng: A=0.02, B=0.03. Đây là so sánh hai pipeline đã calibration, không phải weights-only ở cùng ngưỡng.

Chênh lệch chỉ số chính (F1 ngôn ngữ yếu hơn trên toàn DEV): **-3.81 điểm %**, CI95% bootstrap theo 211 family: **[-6.28; -1.50]**, 2.000 lượt lấy mẫu ghép cặp, seed 20261108.

Điều kiện thăm dò khóa trước: **CHƯA ĐẠT**. Đây không phải gate chấp nhận sản phẩm. Không tự nhận/adopt hay mở thêm epoch.

## Kiểm chứng và phạm vi

- Fixture mới của cả hai chính sách: ngắt giữa gradient window rồi resume cho kết quả bitwise bằng chạy liền.
- Audit 100 cập nhật đầu mỗi nhánh: 1.600 mẫu được chấm; kiểm source, nhãn, mask, future, reset và encoded-token hash; model hữu hạn và đã thay đổi.
- Kiểm đủ một lượt 333.315 điểm/420 task mỗi nhánh; 27 AdamW, toàn bộ RNG/order/cursor/lanes và trạng thái accumulator; weights khớp full resume.
- Audit 606,460 bản ghi calibration/monitor; 569 ca đơn đối chiếu, sai lệch xác suất lớn nhất 1.13e-06. Không audit lại mọi training forward.
- B dùng lại xác suất calibration chỉ khi source-anchored input giống hệt. Số bản ghi audit không phải số forward GPU mới; số forward riêng nằm trong selection_receipt.json.
- Checkpoint cha v7 giữ nguyên SHA, vẫn PAUSED và STOP_REQUESTED. Không phục hồi checkpoint fail đã xóa.

Padding, lượng tính toán và lịch sử reset/dropout có thể khác sau khi đổi context. Cùng RNG ban đầu không có nghĩa mọi dropout mask sau đó giống nhau. Kết quả không tách riêng tác động context lúc train khỏi lúc suy luận.

## Diễn giải thống kê — 11/11 loại sai lệch đã xem xét

1. Simpson: báo cả toàn bộ và từng nguồn/ngôn ngữ; không suy từ điểm gộp rằng mọi nguồn tốt hơn.
2. Ecological: CI theo family, không suy thành độ đúng ngữ nghĩa từng câu.
3. Selection/Berkson: đây là ASR TRAIN đã lọc và DEV cố định; kết quả có điều kiện trên các nguồn này.
4. Collider: không lọc mẫu theo dự đoán/model-score; cùng denominator ở hai nhánh.
5. Base rate: báo TP/FP/FN và số EOT trong report; không dùng accuracy tổng làm thành công.
6. Regression to mean: có control cùng điểm khởi đầu, không chỉ so với một điểm thấp trước đây.
7. Survivorship: cả hai hoàn tất ngân sách; báo cả kết quả âm, không giữ riêng nhánh tốt.
8. Look-elsewhere: chỉ một chỉ số chính khóa trước; chỉ số nguồn/ngôn ngữ là mô tả/guard, không gán significance hàng loạt.
9. Forking paths: protocol được hash trước fit; DEV đã xem nhiều lần nên vẫn là thăm dò, không phải đăng ký trước trên tập kín mới.
10. Causation: một seed, can thiệp chính sách context cả TRAIN+DEV; không tách riêng train-context, eval-context hay chứng minh nguyên nhân ngữ nghĩa duy nhất.
11. Reverse causality: chính sách khóa trước kết quả; không sửa ngưỡng bằng monitor/TEST.

CI percentile là độ bất định lấy mẫu có điều kiện trên outputs hiện tại và giả định family là đơn vị độc lập. Không bao gồm phương sai seed training, lỗi nhãn punctuation-silver, chồng lặp lineage chưa biết hoặc lệch domain. Không diễn giải CI như xác suất tham số nằm trong khoảng; không suy từ CI chứa 0 rằng hai phương pháp tương đương.

Không có human/audio gold, latency đầu cuối hoặc điểm dịch mới. Giới hạn nguồn nhãn/future và population được giữ nguyên. Xem locked_protocol.json, report.json, completion_verification.json và paired_dev_cases.json để kiểm tra chi tiết.

## Các điều kiện thăm dò khóa trước

- FAIL: primary_gain_at_least_3pp
- FAIL: primary_paired_family_ci_above_zero
- FAIL: vi_f1_guard
- PASS: vi_precision_guard
- PASS: vi_false_cut_guard
- PASS: en_f1_guard
- PASS: en_precision_guard
- PASS: en_false_cut_guard
- PASS: asr:en:domain_f1_guard
- PASS: asr:vi:domain_f1_guard
- FAIL: daily:en:domain_f1_guard
- FAIL: daily:vi:domain_f1_guard
- FAIL: factory:en:domain_f1_guard
- FAIL: factory:vi:domain_f1_guard
