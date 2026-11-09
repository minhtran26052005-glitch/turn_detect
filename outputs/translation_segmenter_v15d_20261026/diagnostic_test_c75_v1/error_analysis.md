# Kết quả test chẩn đoán cycle75 và hướng crawl

## Material Passport

- Dữ liệu: public prose, nhãn từ dấu câu; input chữ thường bỏ dấu câu, lookahead1/2 từ.
- Model: ba checkpoint cố định cycle75/100, mỗi model1.100 bước optimizer; ngưỡng chỉ chọn trên dev.
- Test: tài liệu thêm sau45 đến75,80% trang dùng chẩn đoán;20% trang giữ riêng chưa chấm.
- Phạm vi: phân tích khám phá để thiết kế dữ liệu train; không phải chứng nhận hội thoại/audio90%.
- Verification: scored records, frozen hashes, overlap audit and original-source alignment inspected; no performance rerun or causal intervention.

## Kết quả chấm trên dữ liệu test mới

| Ngôn ngữ | Mẫu sạch | Accuracy | EOT F1 | Balanced accuracy |95% CI accuracy theo trang|
|---|---:|---:|---:|---:|---|
|vi|39,167|81.14%|79.08%|81.40%|80.08%–82.17%|
|en|28,155|84.85%|85.02%|84.88%|83.15%–86.25%|
|zh|51,871|84.20%|67.18%|82.44%|82.57%–86.09%|

Tổng119,193 mẫu sạch. Không so trực tiếp với test cycle45 vì model và thành phần dữ liệu đều khác. Bộ cycle45 không được chấm lại.

## Phần sai và bằng chứng

### vi

- Cắt sớm false EOT:4,572; bỏ sót cuối câu:2,815; tổng sai:7,387.
- Recall cuối dấu chấm:83.36%; giữ NEOT tại dấu câu nội bộ:79.58%.
- Nhóm dấu câu góp nhiều lỗi nhất: ',':4,274/20,940 (20.4%); '.':2,676/16,085 (16.6%); ':':176/1,177 (15.0%); ';':122/273 (44.7%); '?':77/484 (15.9%).
- Lỗi lookahead không liền nguồn đã căn chỉnh:248/16,777 mẫu EOT; chưa căn chỉnh:0.
- Train-pool control2.000 mẫu: accuracy80.40%, F180.61%, balanced accuracy80.40%. Test balanced accuracy81.40%. Đây là các mẫu thuộc pool train, không chắc đã được optimizer lấy trúng.
- Volume học:70.400 lượt lấy mẫu /865,075 mẫu hiện có; số mẫu khác nhau đã thấy không thể vượt8.14% pool hiện tại. Lượt lặp và pool cũ khiến độ phủ thực còn thấp hơn cận trên này.

Lỗi trích xuất đã xác nhận bằng nguồn: câu ngắn bị loại trước khi ghép cặp, làm từ tương lai nhảy qua nội dung thật. Cần sửa dữ liệu mới và đánh giá lại trên dev sạch; chưa có thí nghiệm chứng minh mức tăng điểm sau sửa.
Train-pool control và test cùng thấp hoặc gần nhau phù hợp với giả thuyết tối ưu còn ít/nhãn khó; chưa đủ chứng minh chỉ thiếu dữ liệu.

|Nguồn|Mẫu|Sai|Tỷ lệ sai|
|---|---:|---:|---:|
|hf:fineweb2:vie_Latn|39,123|7,381|18.87%|
|wiki:vi|44|6|13.64%|

Lookahead1/2 trên cùng ranh giới: {"both_correct": 14718, "both_wrong": 2656, "lookahead2_rescues": 1204, "lookahead2_hurts": 866}.
Va chạm nhãn trên cùng text nhìn thấy: {"keys": 0, "rows": 0, "minimum_unavoidable_errors_with_identical_visible_text": 0}.

### en

- Cắt sớm false EOT:1,906; bỏ sót cuối câu:2,359; tổng sai:4,265.
- Recall cuối dấu chấm:83.83%; giữ NEOT tại dấu câu nội bộ:86.08%.
- Nhóm dấu câu góp nhiều lỗi nhất: '.':2,211/13,672 (16.2%); ',':1,580/12,772 (12.4%); ':':185/615 (30.1%); ';':141/310 (45.5%); '?':80/460 (17.4%).
- Lỗi lookahead không liền nguồn đã căn chỉnh:605/14,458 mẫu EOT; chưa căn chỉnh:0.
- Train-pool control2.000 mẫu: accuracy85.95%, F185.97%, balanced accuracy85.95%. Test balanced accuracy84.88%. Đây là các mẫu thuộc pool train, không chắc đã được optimizer lấy trúng.
- Volume học:70.400 lượt lấy mẫu /449,514 mẫu hiện có; số mẫu khác nhau đã thấy không thể vượt15.66% pool hiện tại. Lượt lặp và pool cũ khiến độ phủ thực còn thấp hơn cận trên này.

Lỗi trích xuất đã xác nhận bằng nguồn: câu ngắn bị loại trước khi ghép cặp, làm từ tương lai nhảy qua nội dung thật. Cần sửa dữ liệu mới và đánh giá lại trên dev sạch; chưa có thí nghiệm chứng minh mức tăng điểm sau sửa.
Train-pool control và test cùng thấp hoặc gần nhau phù hợp với giả thuyết tối ưu còn ít/nhãn khó; chưa đủ chứng minh chỉ thiếu dữ liệu.

|Nguồn|Mẫu|Sai|Tỷ lệ sai|
|---|---:|---:|---:|
|hf:fineweb:en|28,141|4,262|15.15%|
|wiki:en|14|3|21.43%|

Lookahead1/2 trên cùng ranh giới: {"both_correct": 11237, "both_wrong": 1491, "lookahead2_rescues": 761, "lookahead2_hurts": 499}.
Va chạm nhãn trên cùng text nhìn thấy: {"keys": 0, "rows": 0, "minimum_unavoidable_errors_with_identical_visible_text": 0}.

### zh

- Cắt sớm false EOT:6,027; bỏ sót cuối câu:2,168; tổng sai:8,195.
- Recall cuối dấu chấm:79.47%; giữ NEOT tại dấu câu nội bộ:85.41%.
- Nhóm dấu câu góp nhiều lỗi nhất: '，':3,740/18,874 (19.8%); ',':1,630/17,908 (9.1%); '。':1,533/8,023 (19.1%); '.':544/2,095 (26.0%); '、':205/2,410 (8.5%).
- Lỗi lookahead không liền nguồn đã căn chỉnh:1,305/8,767 mẫu EOT; chưa căn chỉnh:1,789.
- Train-pool control2.000 mẫu: accuracy82.90%, F182.09%, balanced accuracy82.90%. Test balanced accuracy82.44%. Đây là các mẫu thuộc pool train, không chắc đã được optimizer lấy trúng.
- Volume học:70.400 lượt lấy mẫu /1,128,617 mẫu hiện có; số mẫu khác nhau đã thấy không thể vượt6.24% pool hiện tại. Lượt lặp và pool cũ khiến độ phủ thực còn thấp hơn cận trên này.

Lỗi trích xuất đã xác nhận bằng nguồn: câu ngắn bị loại trước khi ghép cặp, làm từ tương lai nhảy qua nội dung thật. Cần sửa dữ liệu mới và đánh giá lại trên dev sạch; chưa có thí nghiệm chứng minh mức tăng điểm sau sửa.
Train-pool control và test cùng thấp hoặc gần nhau phù hợp với giả thuyết tối ưu còn ít/nhãn khó; chưa đủ chứng minh chỉ thiếu dữ liệu.

|Nguồn|Mẫu|Sai|Tỷ lệ sai|
|---|---:|---:|---:|
|hf:fineweb2:cmn_Hani|31,025|5,880|18.95%|
|hf:industrycorpus2_news_media:zh|20,786|2,299|11.06%|
|wiki:zh|60|16|26.67%|

Lookahead1/2 trên cùng ranh giới: {"both_correct": 21012, "lookahead2_hurts": 654, "both_wrong": 3291, "lookahead2_rescues": 938}.
Va chạm nhãn trên cùng text nhìn thấy: {"keys": 0, "rows": 0, "minimum_unavoidable_errors_with_identical_visible_text": 0}.

## Tốc độ thêm dữ liệu so với lượng học

|Ngôn ngữ|Mẫu TRAIN mới cycle73–75|Lượt lấy mẫu để học trong100 bước|
|---|---:|---:|
|en|19,855|6.400|
|vi|40,960|6.400|
|zh|57,313|6.400|

Pool tăng nhanh hơn lượng lấy mẫu học ở cả3 ngôn ngữ trong giai đoạn này. Mọi lượt lấy mẫu có thể lặp và lấy từ toàn pool, nên đây chỉ là so sánh volume, không phải số mẫu mới đã học.

## Kế hoạch crawl và train có thể kiểm chứng

1. Dùng extractor phiên bản mới giữ nguyên chuỗi nguồn: câu ngắn và đoạn cuối chưa có dấu chấm vẫn cung cấp đúng1/2 từ tương lai. Tách phiên bản dữ liệu; không ghi đè corpus/checkpoint đang chạy.
2. Với tài liệu mới, lấy cặp EOT và NEOT theo các nhóm dấu câu có nhiều lỗi bên trên, cân bằng riêng lookahead1/2; giữ đủ câu ngắn, liệt kê, mệnh đề nối và nội dung có số.
3. Lấy tài liệu TRAIN mới từ nguồn công khai đúng ngôn ngữ, kiểm tra SHA nguồn và actual256-token input với dev/test/reserve. Không đưa câu, URL hoặc nhãn từ diagnostic/test trở lại train; BAAI EN tiếp tục cách ly.
4. Tăng mức học/kiểm độ phủ trên pool đã có trước khi tăng tốc crawl thô.70.400 lượt lấy mẫu không phải đã học hết hàng trăm nghìn mẫu. Lượng draw bằng một pool cũng không bảo đảm nhìn hết khi lấy có hoàn lại.
5. Mọi lựa chọn model/ngưỡng/cấu hình mới dùng dev. Sau phát triển, chỉ mở bộ xác nhận mới một lần; giữ reserve hiện tại chưa chấm và loại trùng lại. Muốn kết luận hội thoại thật cần human/audio gold độc lập.

## Kiểm tra cách diễn giải thống kê

11/11 hạng mục đã xem: Simpson (tách nguồn/nhãn/lookahead); ecological (không suy sang hội thoại cá nhân); Berkson và collider (nêu rõ bộ lọc lấy mẫu/loại trùng); base rate (confusion và balanced accuracy); regression-to-mean (không chọn từ test); survivorship (báo số bị loại); look-elsewhere và forking paths (slice khám phá, không tuyên bố p-value); correlation/causation và reverse causality (không quy tương quan thành hiệu quả sửa lỗi).

Ngưỡng và weights không được chọn lại theo kết quả này. Confidence interval bootstrap theo trang chỉ phản ánh tập prose chẩn đoán và quy trình lấy mẫu hiện tại. Các nhóm ít mẫu và paraphrase chưa khử hết vẫn là giới hạn.

Report SHA256:fe4555399de148fe34c2e96f7d95d4edb2aa453caa618f0d69a333ebbb238650
