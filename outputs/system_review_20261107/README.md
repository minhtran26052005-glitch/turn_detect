# Rà soát tổng thể detector phục vụ dịch — 07/11/2026

## Kết luận điều hành

Chưa đủ bằng chứng để kết luận model không học được, dữ liệu là nguyên nhân duy nhất, hoặc tăng epoch sẽ giải quyết vấn đề. Đã chứng minh có cập nhật trọng số và kết quả chấm nhất quán trong phạm vi kiểm chứng; chưa chứng minh hệ thống đáp ứng mục tiêu cắt để dịch sớm, đúng nghĩa và xử lý được lúc người nói dừng.

Các nhánh nối tiếp đã thay nhiều yếu tố, thường kế thừa cả model và AdamW. Vì vậy kết quả trước/sau không phải thí nghiệm độc lập để xác định nguyên nhân. Tiếp tục đổi data/loss rồi train dài sẽ không giải quyết được thiếu hụt bằng chứng này.

Rà soát này chỉ đọc code, dữ liệu thứ tự batch, báo cáo, receipt và trạng thái SSH. Không train, không chạy inference mới, không chấm lại TEST, không đổi model/ngưỡng/default. Parent v7 vẫn PAUSED; ASR trial COMPLETE và không được áp dụng.

## Bằng chứng đã kiểm lại trực tiếp

- Ngày 07/11/2026 lúc 16:03 UTC / 23:03 UTC+7: không có process v7 hoặc ASR trial; parent có STOP_REQUESTED và STOPPED_WITH_CHECKPOINT. ASR trial COMPLETE.
- Code SHA của 6 thành phần trial/data/encoder/model/evaluator khớp manifest SSH và bản local (chuẩn hóa CRLF khi cần). Report SHA khớp receipt.
- Lập lại lịch ASR bằng thuật toán riêng từ segment, nhãn, conflict mask và order trong checkpoint; không dùng production StreamOrder để tính lịch. Khớp chính xác 333.315 điểm / 22.147 cập nhật.
- Dung lượng lúc kiểm: Windows 16,47 GiB, SSD 39,32 GiB, HDD 23,01 GiB; HDD /dev/sda1 thực sự mounted rw. Không thay file dữ liệu hoặc job khác.
- Bằng chứng máy đọc: `independent_schedule_and_state.json`; script: `tools/audit_system_review_20261107.py`.

Các kiểm chứng đã có được đọc và đối chiếu lại, không chạy lại: all-monitor trace audit, canonical predictor parity, full checkpoint optimizer/RNG, source coverage, migration receipts. Chúng xác nhận triển khai thực hiện giao thức đã chọn; không chứng minh giao thức đó đúng cho mục tiêu sản phẩm, cũng không loại trừ mọi lỗi cùng tồn tại giữa trainer và verifier.

## Các vấn đề xác nhận được

### 1. Quy tắc nhãn và mục tiêu sản phẩm chưa đồng nhất

Giữ dấu chấm làm EOT là một định nghĩa hợp lệ, không cần tự ý đổi dấu phẩy thành EOT. Nhưng F1 khôi phục dấu chấm không đồng nghĩa chất lượng cắt để dịch. Một số điểm cắt dấu phẩy vẫn dịch được; một số bỏ sót dấu chấm làm dịch chậm nhưng không nhất thiết làm bản dịch sai nghĩa.

Review định tính trước đó có 10/16 FP được đánh giá là điểm cắt dịch hợp lý, 4/16 cắt cụm từ quá sớm; 11/16 FN là cơ hội cắt sớm rõ ràng. Đây là mẫu nhỏ phân tầng, đánh giá của assistant, không phải tỷ lệ lỗi toàn bộ hay human gold. Không được lấy nó để sửa toàn bộ nhãn hoặc tuyên bố điểm F1 thấp không quan trọng.

Nguồn: parent v7 `semantic_boundary_review_20261107.json`.

### 2. Hệ thống hiện chỉ xử lý ranh giới có từ phía sau

Code chỉ tạo điểm khi còn đủ 1 hoặc 2 từ sau ranh giới trong turn nguồn; ranh giới cuối turn không đủ future bị bỏ. Do đó detector hiện không giải quyết được terminal endpoint khi người nói dừng hẳn mà chưa có từ tiếp theo. Phần flush cuối message trong test dịch là cơ chế bên ngoài, không phải dự đoán EOT của detector.

Cần phân biệt hai phép đo: cắt bên trong dòng chữ để dịch sớm, và chốt phần còn lại khi nguồn kết thúc/dừng. Không thể dùng tỷ lệ message được flush để chứng minh detector tự biết kết thúc. Hiện chưa có benchmark audio/timestamp để xác nhận độ trễ kết thúc thực tế.

DEV hiện giữ future trong turn nguồn nhưng giữ context sau predicted reset xuyên qua turn; đây là giao thức đã chọn, không phải mô phỏng hoàn chỉnh mọi hành vi ASR live.

Nguồn: `uploaded_vi_en_data_v4.py` StreamOrder; `translation_turn_test_20261105_v1/report.json`; `sentence_mix_end_only_test_20261107_v1/user_summary.json`.

### 3. Tỷ lệ EOT thấp là tự nhiên; cách lấy mẫu vẫn làm mất một phần điểm quan trọng

Không có yêu cầu EOT/NEOT phải 50/50. ASR trial có 15.005/333.315 điểm EOT = 4,50%; tỷ lệ này tự nó không chứng minh data sai.

Tuy nhiên các turn một câu thường chỉ đóng góp NEOT vì dấu chấm cuối thiếu future. DEV monitor có 1.183/1.763 turn một span (67,10%); TEST có 10.554/15.000 (70,36%). Các số này đếm span dấu câu, không phải mọi mệnh đề ngữ pháp. Đã có báo cáo riêng cho multi-span; không nên dùng toàn bộ message làm mẫu số để đánh giá nhu cầu cắt nội bộ.

### 4. Ghép câu đã tăng positive anchors nhưng tạo khác biệt TRAIN/DEV

V7 dùng 49.260 pack tổng hợp, cho future đi qua chỗ nối câu/turn/speaker gốc; 160.095 internal joins không liền kề theo nguồn chuẩn. DEV vẫn dùng turn tự nhiên. Đây là thay đổi được user yêu cầu và đã làm đúng, nhưng không chứng minh model học được quan hệ diễn ngôn tự nhiên chỉ nhờ cân bằng số câu.

V7 không dùng 61.446 phần đuôi không kết thúc fullstop / 696.846 từ; raw vẫn giữ. Không thể nói v7 đã học toàn bộ mọi phần raw. Trial ASR lại quay về thứ tự tự nhiên và chỉ ASR: hai yếu tố cùng thay đổi, không có nhánh general-data cùng ngân sách làm đối chứng.

Nguồn: v7 `sentence_pack_audit.json`, `independent_pack_verification.json`, `manifest_clarification.json`; ASR manifest.

### 5. Loss đúng công thức, nhưng batch hiệu dụng không ổn định

Code dùng weighted CE mean: L = sum(w_y * CE) / sum(w_y), w_NEOT=1, w_EOT=2, sau đó clip gradient và AdamW mỗi batch có ít nhất một mẫu scored. Không KD. Trọng số 2 không có nghĩa mọi update positive luôn có gradient gấp đôi: nếu batch chỉ cùng một nhãn, trọng số chung triệt tiêu ở mẫu số.

Lịch ASR độc lập xác nhận:

- 22.147 cập nhật; 11.258 batch toàn NEOT (50,83%), 10.874 mixed, 15 toàn EOT.
- 16.579 cập nhật có đủ 16 mẫu; 446 cập nhật chỉ có một mẫu scored.
- 1.000 cập nhật cuối chỉ có 3.316 mẫu (3,316 mẫu/cập nhật), phần lớn là VI.
- Tổng hệ số EOT sau chuẩn hóa từng batch chiếm 8,00% tổng hệ số loss. Đây là hệ số, không phải tỷ lệ gradient thực tế, vì gradient còn phụ thuộc logits, AdamW và clipping.

Đây là rủi ro cần kiểm soát về trọng lượng cập nhật theo batch nhỏ và thứ tự nguồn. Chưa chứng minh nó là nguyên nhân duy nhất gây tụt VI; không được tự suy ra phải ép cân bằng 50/50 hoặc tăng weight tùy ý.

Nguồn: `asr_only_adaptation_20261107_v1.py:step_batch`, audit lịch mới trong folder này.

### 6. Calibration chưa khảo sát hết miền ngưỡng

Grid chỉ có 0,1…0,9 và chọn 0,1 ở biên dưới trong trial. Sau ASR, median p(EOT) tại positive VI giảm 0,005073 xuống 0,000449; AUROC từ 0,744 xuống 0,727 trên context reset khác nhau. Điểm F1 rất thấp ở ngưỡng hiện tại không đủ chứng minh model mất hết thông tin phân biệt.

Chưa có kết quả causal calibration ở dưới 0,1; không được hứa hạ ngưỡng sẽ sửa được precision/recall. Nếu kiểm, phải dùng calibration đã tách, chạy lại rollout vì đổi ngưỡng đổi context; không chọn ngưỡng trên monitor/TEST hoặc chỉ threshold lại trace cũ.

### 7. Gate hiện tại không chứng nhận chất lượng sử dụng

Gate là giữ F1/P/R tương đối với teacher135M vốn yếu trên DEV mới (F1 VI22,34%, EN38,43%). Một checkpoint pass gate vẫn có thể bỏ sót nhiều EOT. Sáu epoch là ngân sách, không phải bằng chứng đạt chất lượng. Chưa có accepted INT4 hoặc detector được chứng nhận dùng thực tế.

Không tự ý sửa gate đang lưu; muốn nghiệm thu cho dịch cần tiêu chí tuyệt đối được xác định trước: cắt vỡ nghĩa, tỷ lệ tìm được ranh giới hữu ích, độ trễ và chất lượng dịch so với chờ cả đoạn.

### 8. Chuỗi thử nghiệm chưa xác định được nguyên nhân

V4→v5→v6→v7→ASR kế thừa weights/AdamW, không phải các model khởi tạo độc lập. Có lineage exposure cũ không thể xóa bằng thay dataset. Chuỗi có nhiều thay đổi về nguồn, thứ tự, reset, loss và calibration; trước/sau giúp theo dõi chứ không cô lập nguyên nhân.

Sanity 1.000 fixed TRAIN inputs fit100% chỉ loại trừ một số lỗi học cơ bản; không chứng minh generalization. Audit input/hash/parity chứng minh tính nhất quán, không thay thế đối chứng chất lượng. Các TEST đã xem nhiều lần không còn là final sealed confirmation cho quyết định phát triển tiếp theo.

## Những kết luận không được phép rút ra

- Không đủ căn cứ nói model 11,94M quá nhỏ hoặc không hiểu nghĩa.
- Không đủ căn cứ nói data sai toàn bộ, chỉ cần nhiều data/epoch là sẽ tốt.
- Không đủ căn cứ nói chỉ do threshold, batch nhỏ, class imbalance hoặc catastrophic forgetting. Đây là những giả thuyết cần phân biệt.
- Không có bằng chứng ASR-only đã cải thiện cả VI/EN; trial vừa rồi làm VI giảm mạnh, không được áp dụng.
- Không dùng kết quả học thuộc, pass retention yếu, hoặc final flush để quảng bá chất lượng detector.

## Cách tiếp tục có giới hạn, chưa triển khai

1. Giữ parent v7 và trial cố định, training tiếp tục paused. Chốt riêng bài toán internal segmentation và terminal flush; giữ label fullstop như yêu cầu, thêm báo cáo hữu ích cho dịch thay vì thay nhãn tùy ý.
2. Làm một chẩn đoán calibration có giới hạn trên cùng frozen weights, không train: mở miền ngưỡng trên calibration, giữ monitor tách biệt, không TEST. Mục đích duy nhất là xác định mức độ vấn đề do operating point; báo cả recall và số cắt nhầm, không chỉ F1. Không tự áp model/ngưỡng vào default.
3. Chỉ nếu còn cần training, thiết kế một đối chứng cùng checkpoint/ngân sách/source order, thay đúng một yếu tố có giả thuyết rõ. Điểm cần kiểm soát đầu tiên là số mẫu hiệu dụng mỗi optimizer update; không đồng thời đổi replay, LR, class weight, init và packing. Quy tắc tiếp tục/hủy phải định trước; không tự thêm epoch khi kết quả kém.
4. Chỉ sau khi có cải thiện DEV ổn định và không làm ngôn ngữ/nguồn khác suy giảm mới chạy dài. Muốn kết luận sản phẩm cần tập xác nhận chưa dùng để chọn phương án và dữ liệu có thời gian nếu đo độ trễ thực.

Các bước trên là đề xuất quy trình, không phải cam kết rằng một cách sửa cụ thể chắc chắn tăng chất lượng. Lượt này hoàn thành rà soát; không mở thêm nhánh thử nghiệm.
