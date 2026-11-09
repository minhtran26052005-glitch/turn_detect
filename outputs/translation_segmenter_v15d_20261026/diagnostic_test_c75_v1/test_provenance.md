# Tập test checkpoint 75 được tạo như thế nào?

Đây là bộ **test chẩn đoán do campaign tự tách từ văn bản công khai**, không phải benchmark test chính thức do Hugging Face cung cấp, cũng không phải hội thoại được người nghe audio gán nhãn. Kết quả dùng để tìm lỗi dữ liệu và định hướng TRAIN sau này. Sau khi đã xem lỗi, bộ này không còn là một phép xác nhận cuối hoàn toàn kín cho các thay đổi được thiết kế từ nó.

## 1. Dữ liệu lấy từ đâu?

Crawler stream trực tiếp trên Colab. Bảng dưới là **số mẫu thực sự còn lại trong metric sạch**, không phải số mẫu dự kiến hoặc tỷ lệ trộn cấu hình.

| Ngôn ngữ | Nguồn ghi trong provenance | Mẫu sạch |
|---|---|---:|
| Việt | `HuggingFaceFW/fineweb-2`, config `vie_Latn` | 39.123 |
| Việt | Wikipedia API tiếng Việt | 44 |
| Anh | `HuggingFaceFW/fineweb` | 28.141 |
| Anh | Wikipedia API tiếng Anh | 14 |
| Trung | `HuggingFaceFW/fineweb-2`, config `cmn_Hani` | 31.025 |
| Trung | `BAAI/IndustryCorpus2_news_media`, luồng tiếng Trung | 20.786 |
| Trung | Wikipedia API tiếng Trung | 60 |

Địa chỉ dataset được lưu trong cấu hình crawler: [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), [BAAI news/media](https://huggingface.co/datasets/BAAI/IndustryCorpus2_news_media). Wikipedia lấy qua API theo ngôn ngữ. BAAI tiếng Anh đã bị cách ly do sai ngôn ngữ và không có trong test này; nguồn Việt yêu cầu đăng nhập cũng không được dùng. Không có Qwen hoặc HF token trong quy trình tạo nhãn này.

Các HF stream được đọc từ upstream split tên `train`. **Tên đó thuộc dataset nguồn**, khác với split TRAIN/DEV/TEST riêng của campaign. Campaign tự chia lại theo tài liệu trước khi tạo ví dụ; đây không phải việc lấy các ví dụ đã train trong campaign để gọi là test. Tuy vậy, không xác minh được các trang web này đã xuất hiện trong pretraining của model nền hay chưa.

Mỗi record có `source_id`, `source_url`, `source_title`, chỉ số đoạn/câu, vị trí cắt, dấu câu target và `example_id`. Với FineWeb, URL thường là trang gốc được dataset ghi lại. Một số record BAAI không có URL bài gốc nên `source_url` là trang dataset thay thế; không gọi đó là URL bài báo đã xác minh. Ví dụ nguồn và số trang theo nguồn nằm trong `test_provenance_audit_v1.json`; dữ liệu gốc đầy đủ ở SQLite đã khóa trên Drive, không tải về Windows.

Audit lần này còn thấy record Wikipedia có URL thay thế sai dạng `https://huggingface.co/datasets/Wikipedia API`. Do đó **không dùng trường URL đó để khẳng định đã truy xuất bài gốc**. ID và tiêu đề vẫn truy vết được: ví dụ `wiki:vi:8570` / “Trí tuệ nhân tạo”, `wiki:en:21634` / “Neoclassical economics”, `wiki:zh:42471` / “城市轨道交通”. Có thể dựng link tra cứu `https://<language>.wikipedia.org/?curid=<page_id>` từ ID; link dựng lại không phải URL gốc đã lưu hoặc nội dung đã kiểm tra trực tuyến. Lỗi metadata URL này được ghi vào kế hoạch sửa, không sửa record/điểm test đã khóa.

Theo trace sạch, Việt có 430 trang FineWeb-2 và 2 trang Wikipedia; Anh có 399 trang FineWeb và 1 trang Wikipedia; Trung có 276 trang FineWeb-2, 109 trang BAAI và 1 trang Wikipedia. Các số Wikipedia rất nhỏ, chưa đủ để xếp hạng nguồn theo chất lượng.

## 2. Chia train/dev/test theo tài liệu

Mỗi tài liệu được gán split cố định bằng:

```text
bucket = int(SHA256(language + ':' + page_id)[:8], 16) % 100
0..79  -> train
80..89 -> dev
90..99 -> test
```

Vì vậy toàn bộ các ranh giới và cả lookahead 1/2 của cùng một `source_id` thuộc cùng split. Đây là tỷ lệ hash kỳ vọng 80/10/10 theo tài liệu, không bảo đảm đúng 80/10/10 theo số mẫu. Audit SQLite đã khóa tìm thấy **0 nhóm nguồn xuất hiện ở nhiều split**. Các bản sao nội dung giữa ID khác nhau được kiểm thêm ở bước loại trùng bên dưới.

Test chẩn đoán lần này chỉ lấy tài liệu đang ở split `test`, **được thêm vào campaign sau cycle 45 đến hết cycle 75**. Cycle là lần crawl, không phải ngày xuất bản bài. Không chấm lại bộ cycle 45 cũ. SQLite được backup khi campaign đã có thêm cycle 76; tài liệu ngoài khoảng 46–75 không đi vào cohort chấm lần này.

Trong cohort đủ điều kiện, khoảng 20% trang được giữ riêng bằng hash:

```text
int(SHA256('c75-unscored-reserve-v1:' + source_id)[:16], 16) % 5 == 0
```

Các trang còn lại tạo bộ chẩn đoán khoảng 80%. Reserve được tách theo cả trang trước khi suy luận; không dùng nhãn reserve để chọn model hoặc ngưỡng. Các số lượng sau lọc nằm ở mục 5.

## 3. Tạo input và nhãn ra sao?

Nhãn được tạo tự động từ dấu câu của văn bản gốc:

- **EOT**: vị trí hết câu tại `. ! ? 。！？`; phần trái là nội dung câu, phần phải là 1 hoặc 2 từ sau ranh giới.
- **NEOT**: vị trí dấu câu nội bộ `, ， ; ； : ： 、`; phần trái là tiền tố trong câu, phần phải là 1 hoặc 2 từ ngay sau dấu câu theo extractor.
- Tiếng Việt/Anh dùng các chuỗi chữ/số Unicode làm đơn vị; tiếng Trung dùng **Jieba, HMM=False**. Lookahead Trung là 1/2 từ theo Jieba, không phải 1/2 ký tự hay token SentencePiece.

Trước khi vào model: chuẩn hóa Unicode NFKC, chuyển **chữ thường**, thay ký tự thuộc nhóm dấu câu Unicode bằng khoảng trắng và gom khoảng trắng. Dấu câu gốc, chữ hoa, URL, tiêu đề, label và đoạn nguồn để review **không được đưa vào model**. Dấu tiếng Việt vẫn giữ nguyên.

Input thực tế:

```text
[CLS] phần trái [SEP] 1 hoặc 2 từ tương lai [SEP]
```

Giới hạn là **256 tokenizer tokens tính cả token đặc biệt**. Giữ phần future trước, rồi lấy phần đuôi của context vừa ngân sách. Lookahead là số từ trước tokenize, không phải số subword tokens. Một ranh giới thường tạo hai record cho k=1 và k=2; nhiều ranh giới cùng trang không độc lập với nhau.

**Giới hạn của recipe đã chấm:** extractor cũ chỉ giữ câu kết thúc bằng dấu câu, dài ít nhất 25 ký tự và ít nhất 5 đơn vị từ; đoạn phải có ít nhất hai câu hợp lệ. Nó tạo EOT và NEOT từ các câu trước câu hợp lệ cuối đoạn. Vì vậy test thiên về prose/câu dài, không phủ đầy đủ câu ngắn hoặc mọi vị trí bất kỳ giữa hai từ. Quan trọng hơn, lọc câu ngắn trước khi ghép next có thể làm future nhảy qua nội dung thật. Đã đo lỗi đó trong `parser_audit_v1.json`; đây là lỗi của input, không được che đi bằng mô tả “future luôn liền nguồn”. Viết tắt như `TP.` hoặc `PGS.TS.` cũng có thể tạo EOT sai. Giữ nguyên nhãn và điểm test đã khóa; sửa bằng recipe mới cho thí nghiệm TRAIN tiếp theo.

## 4. Cố định checkpoint, kiểm trùng và chấm

Ba checkpoint **cycle 75, step 100/100, tổng 1.100 optimizer steps/model** được pin trước khi campaign có thể ghi đè; weights, tokenizer và SQLite được khóa bằng hash. Đây là `latest_resume` ở mốc 75, không tuyên bố là checkpoint tốt nhất toàn bộ campaign.

Ngưỡng đã chọn từ DEV cycle 75: Việt **0,660**, Anh **0,580**, Trung **0,755**. Không đổi weights/ngưỡng theo test, không thử nhiều checkpoint trên bộ này để chọn điểm cao nhất. Mọi model inference chạy trên Colab.

Trước khi tính metric sạch:

1. Loại tài liệu có hash text giống train/dev, test ngoài cohort hoặc reserve.
2. Encode đúng input model sau normalize và cắt 256 token; loại record có input token IDs trùng train/dev, test ngoài cohort hoặc reserve. Audit train/dev dùng cả dữ liệu cycle 76 trong snapshot nên chặt hơn việc chỉ xét đến 75.
3. Điểm chính chỉ tính trên các record còn lại. Số key trùng không nhất thiết bằng số record bị loại vì một input có thể có nhiều record hoặc trùng nhiều nhóm.

| Ngôn ngữ | Trước audit input | Loại trùng | Chấm sạch | Trang nguồn sạch | EOT | NEOT |
|---|---:|---:|---:|---:|---:|---:|
| Việt | 40.234 | 1.067 | 39.167 | 432 | 16.777 | 22.390 |
| Anh | 28.313 | 158 | 28.155 | 400 | 14.458 | 13.697 |
| Trung | 52.528 | 657 | 51.871 | 386 | 10.556 | 41.315 |
| Tổng | 121.075 | 1.882 | 119.193 | 1.218 | 41.791 | 77.402 |

Khoảng tin cậy 95% được bootstrap **500 lần theo cả trang nguồn**, không coi 119.193 record là 119.193 quan sát độc lập. Accuracy, EOT precision/recall/F1, confusion, từng lookahead và từng nguồn đều có trong `report.json`. Không loại hết paraphrase hoặc near-duplicate; không tuyên bố không có mọi dạng leakage.

## 5. Phần giữ riêng chưa chấm

Scorer hiện tại đã suy luận trên raw diagnostic rows rồi lọc để tính metric sạch. Để bảo vệ reserve, audit riêng loại **cả trang reserve** nếu bất kỳ input 256-token của trang trùng với bất kỳ raw input đã suy luận, kể cả record bị loại khỏi metric. Đây là kiểm input trên CPU, không chấm model trên reserve.

| Ngôn ngữ | Reserve ban đầu (trang/mẫu) | Reserve sau audit (trang/mẫu chưa chấm) |
|---|---:|---:|
| Việt | 107 / 9.717 | 88 / 6.654 |
| Anh | 126 / 6.159 | 125 / 5.767 |
| Trung | 205 / 23.279 | 203 / 22.468 |
| Tổng | 438 / 39.155 | 416 / 34.889 |

Cần audit lại với train/dev mới trước khi mở reserve một lần để xác nhận. Tập chẩn đoán đã xem có thể định hướng nhóm lỗi cần thu thập, nhưng không copy chính câu/URL test vào TRAIN và không dùng nó làm vòng lặp chọn checkpoint/ngưỡng. Muốn xác nhận chất lượng hội thoại thực cần tập human/audio gold độc lập theo đúng input triển khai.

## 6. Bằng chứng và nơi lưu

- Kết quả bất biến: `report.json`, SHA256 `fe4555399de148fe34c2e96f7d95d4edb2aa453caa618f0d69a333ebbb238650`.
- Model/data pin và protocol: `manifest.json`; nguồn và số lượng kiểm từ trace sạch: `test_provenance_audit_v1.json`.
- Các trace đầy đủ `test_predictions_{vi,en,zh}.jsonl.gz`, SQLite/ZIP đã khóa, weights/tokenizer và mã chạy: lưu trực tiếp trên mounted Drive tại `MyDrive/turn_detect_backups/translation_segmenter_v15d_20261026/diagnostic_test_c75_v1`.
- `latest_manifest.json` trên Drive ánh xạ logical path sang content object SHA và cờ readback; cần manifest commit hoàn tất mới gọi là đã lưu đủ. Windows chỉ có báo cáo/receipt và mẫu review nhỏ.
- Phiên bản sửa tiếp diễn v2 xử lý lỗi kỹ thuật publisher trước khi có scoring, giữ nguyên frozen protocol; không chấm lại ngôn ngữ đã hoàn thành. Các lỗi kỹ thuật và receipt cũ được giữ để audit.

Test này đo khả năng khôi phục ranh giới dấu câu từ văn bản không dấu câu. Nó chưa chứng minh đạt 90% hội thoại thật và không thể quy mọi dự đoán khác nhãn dấu câu thành lỗi ngữ nghĩa của model.
