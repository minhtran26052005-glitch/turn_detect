# Streaming V4 — train và kiểm thử 25/10/2026

**Chưa đạt accuracy và F1 EOT >90% đồng thời ở cả ba ngôn ngữ.** Colab L4 đã chạy xong 3 vòng ×600 bước và 300 bước cân bằng nhãn; nhánh local chạy thêm 3 vòng ×200 bước. Vòng cân bằng dừng do ba lần dev liên tiếp không cải thiện. Checkpoint chọn bằng dev là vòng 2, bước 200; không chọn lại sau khi xem test.

**Pilot hội thoại mới, nhãn tự động từ audio**

| Ngôn ngữ | Mẫu | Accuracy | F1 EOT | EOT nhầm | Bỏ lỡ EOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| vi | 30 | 50.00% | 57.14% | 11 | 4 |
| en | 20 | 70.00% | 66.67% | 2 | 4 |
| zh | 20 | 55.00% | 66.67% | 8 | 1 |

Việt: 30 khoảng dừng từ 2 cuộc gọi DataHive. Anh/Trung: mỗi tiếng 20 khoảng dừng từ 10 turn CallHome thuộc official-test, một hold và một EOT mỗi turn; các turn được lấy đầu file nên chỉ thuộc **1 cuộc gọi mỗi tiếng**. ASR chỉ nghe tối đa15 giây trước thời điểm quyết định `silence.start+0,2s`. Không đưa tiếng nói/phần chữ tương lai vào model. Nhãn do VAD và hành vi đổi người nói suy ra, chưa được người chấm xác nhận; mẫu nhỏ và các điểm trong cùng cuộc gọi không độc lập. Đây là chẩn đoán ban đầu, chưa là benchmark gold đại diện.

**Teacher gốc trên đúng các mẫu mới và cùng cách mã hóa causal**

| Ngôn ngữ | Mẫu | Accuracy | F1 EOT | EOT nhầm | Bỏ lỡ EOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| en | 20 | 75.00% | 70.59% | 1 | 4 |
| vi | 30 | 46.67% | 42.86% | 8 | 8 |
| zh | 20 | 65.00% | 69.57% | 5 | 2 |

Ngưỡng của mỗi model được chọn riêng từ dev và khóa trước pilot. Sau train, Việt tốt hơn đối chứng trên pilot nhỏ, Anh/Trung thấp hơn; không có bằng chứng để thay model triển khai hoặc tuyên bố đạt mục tiêu. Khác nguồn và nhãn giữa dev cũ với hội thoại tự nhiên là vấn đề cần giải quyết; chưa thể quy toàn bộ lỗi cho một nguyên nhân từ 70 mẫu tự động.

**Dev Việt cũ đủ17.833 mẫu**

| Ngôn ngữ | Mẫu | Accuracy | F1 EOT | EOT nhầm | Bỏ lỡ EOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| vi | 17833 | 93.38% | 90.09% | 936 | 245 |

Dev này là nhãn prefix suy ra. Điểm >90% ở đây không đồng nghĩa đã >90% trên hội thoại thật. Bản Việt INT4 đang có đạt accuracy46,67% / F1 EOT38,46% trên cùng30mẫu Việt mới; con số này cũng chỉ là pilot tự động. Nhánh local mới đạt56,67% /60,61% trên30mẫu đó và chưa được chọn làm bản triển khai.

**Dữ liệu và cách train**

Train104.143 mẫu, dev5.992 mẫu; cân bằng mức xuất hiện ba tiếng. Giữ riêng1.855 mẫu EN/ZH đọc sẵn trước lượt kiểm tra cuối. LiveKit đã được xem ở V3 nên từ V4 là train/dev, chia theo toàn cuộc hội thoại; không gọi là test độc lập. Việt rehearsal lấy từ frozen train/dev cũ, không đọc lại holdout đã dùng. Loại toàn câu hiện tại khỏi `messages` của Scicom đọc sẵn để tránh lộ phần lời tương lai. Đầu vào dài giữ phần mới nhất của current text, tối đa256tokens gồm ngữ cảnh và special tokens; dấu câu được chuẩn hóa khỏi đầu vào.

Colab: full teacher mDeBERTa12 lớp, FP32/TF32 tắt, AdamW/warmup/cosine/gradient checkpointing. Ba vòng LR1e-5,5e-6,2e-6; microbatch16 ×2. Vòng cân bằng lấy EOT/NEOT đều trong từng nguồn, LR2e-6; checkpoint trước vẫn là đối chứng hợp lệ. Local RTX3050: chỉ học3lớp cuối/pooler/head, microbatch4 ×2, lưu delta kèm hash base.

Tập đọc sẵn tiếng Trung bị lệch nhãn mạnh: lượt held có498EOT,36NEOT. Accuracy93,07% / F196,38% vẫn đi cùng31/36hold bị kết thúc nhầm; không dùng điểm này để tuyên bố hội thoại Trung đã tốt.

**Crawl streaming**

204 prefix tại dấu phẩy/chấm từ11bài mới, đủ Việt/Anh/Trung, đã dự đoán và giữ từng xác suất. Cả204mẫu vẫn `label:null`, vì dấu câu không phải nhãn nhường lượt. Accuracy/F1 crawl hiện là `null`. [Player và bảng kết quả](outputs/stream_v4_20261025/review.html), [chấm semantic-completeness độc lập](outputs/stream_v4_20261025/fresh_crawl_review.html), [nghe30điểm dừng Việt](outputs/stream_v4_20261025/vi_natural_review.html).

**Nguồn và giấy phép**

- [Scicom semantic-vad-eot](https://huggingface.co/datasets/Scicom-intl/semantic-vad-eot/tree/98d1445eab71e4084ccb838dbd87a205f6f2e293): read speech CC BY4.0. Subset CallHome hội thoại giữ giấy phép nguồn CC BY-NC-SA4.0, chỉ dùng kiểm thử nghiên cứu trong vòng này.
- [DataHive conversations](https://huggingface.co/datasets/datahiveai/multilingual-conversations-sample/tree/1120cd7533b16ba8dd7a7c07e6d316b74b60f72a): CC BY4.0, attribution DataHive, không kèm transcript.
- [LiveKit EOT data](https://huggingface.co/datasets/livekit/eot-bench-data/tree/ca9d98a9686b920a2d8c9eb984224ba9be74e4dd): dữ liệu CC BY4.0; giấy phép Apache2.0 của harness không phải giấy phép dataset. Sửa nhầm lẫn trong README V3.
- [Whisper small CTranslate2](https://huggingface.co/Systran/faster-whisper-small/tree/536b0662742c02347bc0e980a01041f333bce120): snapshot ASR được khóa trong pilot EN/ZH. Nguồn Wikipedia, URL và hash từng bài nằm trong `data_v4/fresh_crawl_20261025/sources.jsonl`.

**File đã lưu và cách kiểm tra**

Teacher FP32: `outputs/stream_v4_20261025/runs/stream_v4_campaign_20261025/round_02/best/model.safetensors` (1.115.268.200byte), SHA256 `89ebf4e8c8b99e4ea1ad2eac19f9132b0ce8506ee0836943dba343fc849d6509`. Đây là teacher lớn; chưa xuất một bản distill/INT4 V4 đạt tiêu chuẩn. Bản Việt QAT mặc định và các checkpoint cũ giữ nguyên.

`outputs/stream_v4_20261025/summary.json` chứa bảng tổng; `colab_final/report.json` và NPZ/JSONL chứa kết quả tính; `completion_validation.json` kiểm tra kỹ thuật. Đã chạy45unit tests pass. Notebook `a.ipynb` có ô `stream_v4_result` cuối file để xác minh hash và đọc đúng kết quả này. Nguồn và hash dữ liệu, source launch/final, receipts, audio và dự đoán được giữ lại để kiểm tra.

Mốc >90% vẫn chưa đạt. Các test vừa xem chuyển thành regression; vòng sau cần nguồn hội thoại train bổ sung và một cohort kiểm tra mới, hoặc thay đầu vào theo lựa chọn text-only/âm thanh. Không lặp tối ưu trên chính70mẫu này để biến chúng thành điểm chấp nhận.
