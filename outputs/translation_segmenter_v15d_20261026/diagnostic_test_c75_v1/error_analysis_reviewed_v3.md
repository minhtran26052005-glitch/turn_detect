# Cycle 75: kết quả test và hướng crawl

## Material Passport

- Ba checkpoint cycle 75/100 và ngưỡng dev được cố định trước khi chấm.
- 119.193 mẫu sạch, từ tài liệu mới trong cycle 46–75; loại 1.882 mẫu trùng.
- Đây là test chẩn đoán trên văn bản lấy nhãn từ dấu câu, không phải human/audio conversational gold.
- Phân tích do trợ lý kiểm tra từ report, trace receipts, mã nguồn và đối chiếu đoạn nguồn. Không sửa nhãn, ngưỡng hoặc chọn lại checkpoint để tăng điểm test.

## Kết quả

| Model | Mẫu sạch | Accuracy | EOT F1 | Cắt sớm FP | Bỏ sót EOT FN | 95% CI accuracy theo trang |
|---|---:|---:|---:|---:|---:|---|
| Việt | 39.167 | 81,14% | 79,08% | 4.572 | 2.815 | 80,08–82,17% |
| Anh | 28.155 | 84,85% | 85,02% | 1.906 | 2.359 | 83,15–86,25% |
| Trung | 51.871 | 84,20% | 67,18% | 6.027 | 2.168 | 82,57–86,09% |

Chưa đạt 90%. Trung có precision EOT 58,19% và recall 79,46%. EOT chỉ chiếm 20,35% tập Trung nên accuracy cao hơn đáng kể so với F1; cần giảm các lần cắt sớm, không chỉ đọc accuracy.

## Nguyên nhân có bằng chứng

### 1. Lookahead bị nhảy qua nội dung nguồn

Code cũ bỏ câu dưới 25 ký tự hoặc dưới 5 từ trước khi ghép current/next. Khi có câu ngắn ở giữa, 1–2 từ tương lai được lấy từ câu xa hơn.

| Ngôn ngữ | EOT căn chỉnh được | Lookahead không liền nguồn | Tỷ lệ trong phần căn chỉnh | Sai model trong nhóm này | Chưa căn chỉnh được |
|---|---:|---:|---:|---:|---:|
| Việt | 16.777 | 248 | 1,48% | 64 | 0 |
| Anh | 14.458 | 605 | 4,18% | 153 | 0 |
| Trung | 8.767 | 1.305 | 14,89% | 316 | 1.789 |

Đã tái hiện bằng kiểm tra và nguồn thật: sau “cuối cùng ông trở về”, record nhìn “và ông” thay vì “cả bốn” vì bỏ câu ngắn xen giữa. Các mẫu Trung chưa căn chỉnh được không tự bị kết luận sai nhãn. Đây là tần suất lỗi input đã xác nhận, không phải mức tăng accuracy đã chứng minh sau khi sửa.

### 2. Dấu chấm trong viết tắt bị coi là hết câu

Các ví dụ nguồn đã kiểm tra gồm “TP. Hồ Chí Minh” và “PGS.TS.”. Bộ nhãn tạo 52 ứng viên Việt thuộc TP/TS/GS; 47 dự đoán NEOT bị silver target tính sai. Đây là số mẫu lookahead ứng viên, không phải 52 câu độc lập đã được người gán nhãn xác nhận. Điểm test chính giữ nguyên.

### 3. Độ phủ học còn thấp và model chưa fit cao trên mẫu đã học

| Ngôn ngữ | Accuracy trên 2.000 mẫu thực sự được optimizer lấy | Trên 2.000 mẫu ngẫu nhiên thuộc pool train | Cận trên độ phủ pool hiện tại |
|---|---:|---:|---:|
| Việt | 84,10% | 80,40% | 8,14% |
| Anh | 88,85% | 85,95% | 15,66% |
| Trung | 86,60% | 82,90% | 6,24% |

Hai control train đều cân bằng EOT/NEOT nên accuracy cũng là balanced accuracy. Khi đối chiếu với test có tỷ lệ nhãn khác nhau, xem thêm balanced accuracy trong JSON; không dùng khoảng cách F1 thô để kết luận overfit.

Sampler cycle 75 được tái hiện chính xác: fingerprint dữ liệu và trạng thái RNG NumPy cuối vòng khớp cả ba checkpoint. Cột đầu thực sự là các mẫu đã được optimizer lấy. Mỗi control có 500 mẫu cho mỗi tổ hợp label/lookahead.

Mỗi model mới có 1.100 bước × 32 microbatch × 2 accumulation = 70.400 lượt lấy mẫu cho pool 449.514–1.128.617 mẫu. Lượt lấy mẫu có lặp và gồm các pool cũ, nên độ phủ khác nhau thực tế còn thấp hơn cận trên. Đây không phải số epoch đã hoàn tất.

Trong cycle 73–75, mẫu TRAIN mới tăng: Việt 40.960, Anh 19.855, Trung 57.313. Mỗi model chỉ có 6.400 lượt lấy mẫu trong lượt train 100 bước tương ứng, lại lấy trên toàn pool. Dữ liệu tăng nhanh hơn lượng học.

Các kết quả phù hợp với mức tối ưu/độ phủ còn hạn chế và lỗi input/target. Chưa đủ để chứng minh chỉ tăng bước sẽ đạt 90%, hoặc quy hết lỗi cho thiếu crawl.

### 4. Nhóm cần crawl có chọn lọc

- Việt: dấu phẩy NEOT và dấu chấm EOT đóng góp phần lớn lỗi; dấu chấm phẩy sai 122/273 mẫu, 44,69%.
- Anh: dấu chấm EOT và dấu phẩy NEOT đóng góp phần lớn lỗi; dấu chấm phẩy sai 141/310 mẫu, 45,48%.
- Lookahead 2 cải thiện accuracy gần 2 điểm ở Việt/Anh; số ranh giới được cứu nhiều hơn số bị làm hỏng. Giữ báo cáo tách cả lookahead 1 và 2.
- Wikipedia chỉ có 14 mẫu Anh và 44 mẫu Việt trong test này; không xếp hạng chất lượng nguồn từ các nhóm nhỏ đó.

## Kế hoạch tiếp theo đã chuẩn bị, chưa kích hoạt

1. Dùng recipe mới giữ câu ngắn và đoạn cuối chưa có dấu chấm làm future thật, đồng thời xử lý viết tắt theo ngôn ngữ. Mã chuẩn bị: `turn_v2/translation_segmenter_causal_extraction_v2.py`. Có 20 kiểm tra protocol/safety đã qua; chưa thay worker hoặc corpus đang chạy.
2. Với TRAIN mới, tăng nhóm khó: comma/semicolon/colon sau mệnh đề hoàn chỉnh; EOT câu ngắn, hỏi, cảm thán; phân biệt dấu chấm câu với viết tắt/số. Cân bằng label × lookahead và chú ý nhóm dấu câu hiếm.
3. Thử tăng budget học trên pool cố định, kiểm fit/coverage và chọn mọi thay đổi bằng dev. Điều chỉnh nhịp crawl theo lượng học; không tối ưu ngưỡng hoặc chọn checkpoint theo test này.
4. Dùng tài liệu TRAIN mới, kiểm trùng nguồn và input 256 token với dev/test/reserve; không copy câu hoặc URL test vào train. BAAI EN tiếp tục cách ly. Giữ ba model, input chữ thường không dấu câu, lookahead 1/2, không Qwen, model execution trên Colab và artifacts lớn trên Drive.
5. Tạo recipe và receipt mới tại ranh giới an toàn, giữ corpus/HF state/log/checkpoint cũ; không sửa dataset giữa một cycle đang resume.

## Bộ giữ riêng và giới hạn

Đã loại cả trang reserve nếu có bất kỳ input 256 token trùng với toàn bộ trace đã suy luận, kể cả raw rows bị loại khỏi metric sạch. Không dùng nhãn reserve để chọn model/ngưỡng.

| Ngôn ngữ | Trang reserve còn lại | Mẫu chưa chấm còn lại |
|---|---:|---:|
| Việt | 88 | 6.654 |
| Anh | 125 | 5.767 |
| Trung | 203 | 22.468 |

Cần kiểm trùng lại với train/dev mới trước khi mở reserve. Exact overlap đã kiểm; paraphrase/near-duplicate không được chứng minh loại hết. Muốn xác nhận hội thoại/audio thật cần gold độc lập.

Frozen SQLite có thêm nguồn sau cycle 75 nhưng không dùng chúng cho cohort test 75. Counter `old_test_rows_not_rescored` trong partition gộp nguồn ngoài khoảng 46–75, không chỉ test cycle 45.

Campaign đã tự tiếp tục sau cửa sổ ưu tiên test. Các bản sửa và kế hoạch mới hiện là chuẩn bị để review, chưa triển khai vào crawler đang chạy.

Report SHA256: `fe4555399de148fe34c2e96f7d95d4edb2aa453caa618f0d69a333ebbb238650`.

Raw report, ba full prediction traces và ba frozen weights đã khớp SHA với manifest Drive. Windows chỉ nhận khoảng 1,24 MiB reports/review; không có weights, SQLite hoặc ZIP mới được tải về.
