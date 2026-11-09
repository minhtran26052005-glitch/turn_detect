# Curriculum Việt Anh v6

Nhánh đang train thật từ checkpoint warm-start mALBERT 11.941.122 tham số. Chuyển từ v5 bằng checkpoint đầy đủ tại global step 2624058. Model, 27 trạng thái AdamW, Torch/CUDA/NumPy/Python RNG và ngưỡng ban đầu đã được đối chiếu chính xác. Thứ tự nguồn, cursor và các lane hội thoại mới được version hóa theo lịch curriculum. Thứ tự, cursor, lane và sampler cũ nằm trong frozen/parent_full_resume.pt.

## Lịch mở dữ liệu

Tỷ lệ chỉ số family hội thoại, gồm mọi stream và bản ngôn ngữ liên quan. Các pool lồng nhau và trộn lại dữ liệu của mức trước. Xếp hạng dùng tỷ lệ vị trí bị loại do mơ hồ/xung đột, độ dài turn và hash cố định trên TRAIN, phân tầng theo nguồn và thành phần ngôn ngữ của family. Tỷ lệ nhãn EOT/NEOT theo dữ liệu tự nhiên trong pool.

| Mức | Family mục tiêu | Số family | Số stream | Điểm hợp lệ mỗi lượt | Ngân sách |
| --- | --- | --- | --- | --- | --- |
| 1 | 10% | 874 | 1725 | 463539 | 10000 |
| 2 | 25% | 2174 | 4363 | 1624700 | 20000 |
| 3 | 50% | 4343 | 8765 | 3302616 | 30000 |
| 4 | 100% | 8681 | 17301 | 6163255 | Ít nhất 6 epoch toàn bộ, sau đó quality gate |

Ba mức đầu mở rộng sau số cập nhật cố định. Một mức có thể kết thúc giữa lượt duyệt pool. Chỉ coverage receipt của mức 100% xác nhận đã qua trọn 6.163.255 điểm trong một epoch đầy đủ. Bộ raw và toàn bộ TRAIN hợp lệ đều được lưu.

## Huấn luyện và DEV

AdamW có LR 4e-6 và weight decay 0,01. Weighted cross entropy có NEOT 1, EOT 2. Batch 16, 16 lane hội thoại, 2 CPU thread và GPU cap 50%. Context lowercase không dấu câu, future 1 hoặc 2 từ cùng turn nguồn, reset theo dự đoán. Nhãn là dấu câu silver. Input model được tạo từ source_raw.

DEV calibration và monitor có cùng SHA256 với v5. Ngưỡng chọn bằng rollout nhân quả trên calibration và báo monitor riêng mỗi 10.000 cập nhật. Epoch count ở ba mức đầu có trường full_filtered_epochs_complete=0 để phân biệt lượt subset với epoch đầy đủ. Best selection được version hóa theo mỗi mức, các best trước giữ ở runs/finetune/curriculum_stage_*_best.pt và report tương ứng. Các kiểm chứng CPU, QAT và package dùng quality gate hiện có sau quá trình train mức 100%.

## Bằng chứng đã chạy

Fixture CPU thực thi chính hàm fit qua cả bốn mức, gặp batch không có vị trí scored, ngắt giữa lane đang hoạt động rồi resume. Model, optimizer, RNG, sampler, order, cursor, lane, stage counters và full epoch count khớp bitwise với chạy liền. Đây là fixture kiểm kỹ thuật.

Bản train thực đã được kiểm ở branch 500 / global 2624558, stage 10%, có 7928 điểm được học. Audit 1582 input thực qua 100 optimizer updates kiểm source, nhãn, conflict mask, future và reset. Checkpoint mới có 27 AdamW entries, weights hữu hạn và đã thay đổi sau cập nhật. Worker cũ đã thoát và có STOP_REQUESTED cho handover.

Các kết quả sau train sẽ thể hiện tiến triển trên DEV cố định. Việc xác định riêng lợi ích nhân quả của curriculum cần đối chứng cùng checkpoint và ngân sách cập nhật.

Xem curriculum_plan.json, migration_receipt.json, technical_fixture_verification.json, independent_actual_curriculum_training_verification.json và review.html. Full weights/checkpoints/raw/trace lưu trên SSH. Lịch theo dõi ACTIVE 300 phút.

Manifest SHA256 `1df05f4c3f2e88cea19cdd8c79fa0cf3af5a09ec6a03f8c1b6b555103159c51f`

Plan SHA256 `2014b6233eb4ca74467314f66bbb495d7f405342fe48228be7bd8c0ec5af90ee`


## Kiểm tra ngày 06/11 lúc 13 giờ 20 Việt Nam

Đã mở mức 100% family sau khi hoàn thành ngân sách 10.000 + 20.000 + 30.000 cập nhật của ba mức đầu. Checkpoint branch 70.000 ghi stage 3, 157.029 điểm ở lượt toàn bộ đầu tiên và full_filtered_epochs_complete=0. DEV 70.000 đang hiệu chuẩn.

DEV hoàn tất gần nhất branch 60.000, global 2.684.058, đo cuối pool 50%, ngưỡng 0,1. Việt có F1 30,23%, precision 40,63%, recall 24,06%. Anh có F1 49,69%, precision 54,66%, recall 45,55%. Retention gate chưa đạt. Nhãn DEV là dấu câu silver. Báo cáo bước đã kiểm SHA256 1902580fe03ddb1f256129080f59dcd00348d48f73f8bba8d8e0618cd37ed432. Trainer tiếp tục theo lịch.


## So sánh DEV trực tiếp ngày 06/11 lúc 16 giờ 29 Việt Nam

DEV trước chuyển curriculum tại global 2.615.080 có F1 Việt 25,97% và Anh 47,99%. Latest curriculum DEV branch 150.000/global 2.774.058 có F1 Việt 27,76% và Anh 49,60%, cùng ngưỡng 0,1 và cùng SHA256 bộ DEV. Mức tăng lần lượt 1,79 và 1,61 điểm phần trăm. Recall Việt 18,26% lên 20,14%, Anh 40,23% lên 44,40%. Precision Anh từ 59,45% còn 56,18%. Đây so sánh trước/sau, có thêm bước train và thay đổi lịch dữ liệu.

So với báo cáo branch 60.000, F1 Việt từ 30,23% còn 27,76%, Anh 49,69% còn 49,60%. Điểm vẫn dao động và quality gate chưa đạt. Stage 100% mới qua khoảng 24% lượt đầu, full_filtered_epochs_complete=0. Worker hiện khỏe và tiếp tục chạy theo lịch. ReportSHA 0cfa0dbb9519886304512a4ca2ac02467a21ada6e9f4e83fc23ef361dc14bb30.


## DEV ngày 06/11 lúc 18 giờ 17 Việt Nam

Latest COMPLETE branch 200.000/global 2.824.058, shared threshold 0,1. Việt có F1 25,81%, precision 43,03%, recall 18,43%. Anh có F1 47,14%, precision 56,91%, recall 40,23%. Retention gate chưa đạt. Report SHA256 90ebb00f410ff699db3c9ac2d865d4f797906f5ed423420c5b6e7d7226f5ed8e. Đã so đúng báo cáo per-step và bản local chuẩn hóa LF.

Worker 873860/ticks 16570173/boot651 được kiểm đúng một process, CPU ticks tăng và branch 207.750 lên 207.850, source/code/plan fingerprints khớp. Mức full 100% mới qua 2.322.712 điểm của lượt đầu, full_filtered_epochs_complete=0. Không có lỗi đang hoạt động. Dung lượng SSD 41,48 GiB, HDD 23,01 GiB mounted rw, Windows 18,86 GiB. Trainer tiếp tục theo lịch.


## DEV ngày 06/11 lúc 23 giờ 17 Việt Nam

Latest COMPLETE branch 340.000/global 2.964.058, shared threshold 0,1. Việt có F1 27,83%, precision 45,04%, recall 20,14%. Anh có F1 50,24%, precision 55,79%, recall 45,69%. Retention gate chưa đạt. Report SHA256 a61935fd6a2a40e2057113d1636ab4ee03d8517da68d19e44dfc6f7c9056d240. Đã so đúng báo cáo per-step và bản local chuẩn hóa LF.

Worker 873860/ticks 16570173/boot651 được kiểm đúng một process, CPU ticks tăng và branch 346.800 lên 346.850, source/code/plan/TRAIN fingerprints khớp. Mức full 100% mới qua 4.506.340 điểm của lượt đầu, full_filtered_epochs_complete=0. Không có lỗi đang hoạt động. Dung lượng SSD 41,40 GiB, HDD 23,01 GiB mounted rw, Windows 17,51 GiB. Trainer tiếp tục theo lịch.


## Kiểm tra ngày 07/11 lúc 08 giờ Việt Nam

Đã kiểm chứng epoch toàn bộ đầu tiên của curriculum, đủ 6.163.255/6.163.255 điểm TRAIN hợp lệ. Coverage SHA256 bc49abba4d75c961d4e905f0c134c8bad0f1e8b2f086158633f252c068811d47. Full checkpoint branch 598.000 có epoch=1, 27 AdamW entries, order 34.602 task duy nhất, toàn bộ RNG và lane, weights hữu hạn. Đang học epoch thứ hai, khoảng 37% lượt. Đây mốc 1/6 epoch tối thiểu.

Latest COMPLETE DEV branch 590.000/global 3.214.058, shared threshold 0,1. Việt F1 29,43%, precision 45,07%, recall 21,84%. Anh F1 50,76%, precision 57,56%, recall 45,40%. Retention gate chưa đạt. Report SHA256 2e874de849f20e309dd9db78de7710cd768822f559fb54c790b3f6886f366d9f. Đã so bản remote per-step và local chuẩn hóa LF.

Worker 873860/ticks 16570173/boot651 đúng một process, CPU và branch tăng 598.400 lên 598.450. Source/code/plan/TRAIN SHA khớp, không lỗi đang hoạt động. SSD 41,34 GiB, HDD 23,01 GiB rw. Windows C dao động 12,63 đến 15.11 GiB trong lượt kiểm, lần đo cuối 15.11 GiB, reserve 10 GiB. Trainer tiếp tục theo lịch.


## Audit cấu trúc input ngày 07/11

Toàn bộ TRAIN hiện tại có 167.888 turn nguồn trong 17.301 streams. Khi đếm các phần theo nhãn fullstop, 82.021 turn có một phần (48,85%), 85.867 turn có từ hai phần (51,15%), trong đó 23.101 turn có từ ba phần (13,76%). Median turn 15 từ, p90 34 từ, 2.841 turn có ít nhất 50 từ. Median stream 161 từ và 10 turn. Dữ liệu có cả turn ngắn và đoạn nhiều câu.

232.059 fullstop EOT anchors gồm 106.442 ở cuối turn (45,87%). Các EOT cuối turn bị loại khỏi scoring theo hợp đồng future 1/2 từ cùng turn, vì không đủ future. Turn có một phần vì vậy có thể chỉ đóng góp NEOT. K1/K2 còn 124.470/123.866 EOT points sau mask, tổng số điểm 6.163.255 khớp manifest. Đây khoảng trống supervision tại cuối turn, cần phân biệt với EOT nội bộ đoạn dài. Tác động lên recall cần đối chứng để xác định.

Xem input_sentence_structure_audit_20261107.json cho mọi thống kê theo nguồn/tiếng và ví dụ source được phục dựng từ từ chuẩn hóa cùng nhãn dấu chấm. Source SHA trước/sau audit khớp. Audit không đổi data, trainer hay input contract.


## Chuyển sang sentence mix v7 ngày 07/11

V6 đã checkpoint và thoát tại global 3231046. Nhánh kế tiếp multilingual_malbert_vi_en_sentence_mix_20261107_v7 đã được kiểm chứng train thật trên đoạn ghép cân bằng 1–10 câu. Full checkpoint và toàn bộ trạng thái cũ giữ trong parent pin của v7.
