# Data mở rộng, crawl Qwen và thử reward — V10

## Material Passport

Experiment type: CODE_TRAINING_ABLATION. Ba model riêng. Input text-only. GPU Colab L4 23.034MiB; không tải teacher lớn về Windows. Mục tiêu >90% accuracy/F1 EOT trên hội thoại độc lập chưa được chứng nhận.

## Thiết kế khóa trước khi chạy

Mỗi ngôn ngữ khởi tạo từ đúng student QAT riêng đã có. Vocabulary và scale INT4 giữ nguyên; cập nhật latent weights có fake quant mỗi forward. Codec giữ token hiện tại mới nhất trong256token. Không có dữ liệu người nói tương lai/timing/audio trong input.

`data_sft`:450bước với dữ liệu mở rộng, balanced nguồn/nhãn. Sau đó `ce_control`, `cost_ce`, `reinforce` cùng bắt đầu từ checkpoint data_sft được chọn bằng dev,225bước/nhánh, batch256, cùng seed sinh chỉ số batch và LR. `cost_ce` tăng trọng số NEOT để kiểm tra lợi ích có chỉ đến từ việc ưu tiên tránh ngắt sớm hay không.

RL là one-step REINFORCE: hành động wait/stop, đúng+1, stop trên NEOT−2, wait trên EOT−0,5. Baseline reward kỳ vọng được detach để giảm phương sai; thêm CE và KL tới data_sft để tránh quên. Vì có nhãn binary, reward của cả hai hành động có thể suy ra; đây không phải bằng chứng rằng RL tốt hơn phân loại theo cost. Không có simulator thời gian hoặc latency thật, nên−0,5 là sở thích cost cố định, không phải số mili giây. Crawl silver không nhận policy reward; chỉ vào CE với weight0,25. Nhãn floor tự động và prefix proxy vẫn có lỗi.

## Nguồn dữ liệu và crawl

Tận dụng thêm dialogue official train trong MultiWOZ2.2 và CrossWOZ đã tải, giữ revision/provenance. Cuối lượt và prefix có tiếp lời chỉ là nhãn cấu trúc proxy, không tự là human semantic EOT. Nhánh Việt thêm628mẫu natural train trước lọc và24.000mẫu replay train. Dữ liệu dev, toàn bộ test/holdout đã xem và204prefix cũ được bảo vệ khỏi training theo exact visible-input hash; official dialogue train/dev khác nhau. Đây chưa chứng minh không có trùng paraphrase hoặc pretraining contamination.

Crawl12bài Wikipedia mới có giới hạn, loại source của204prefix review cũ, giữ page ID/revision/URL/hash. Thu110prefix (Việt43, Anh47, Trung20); hai request Trung bị429, không lặp request dồn dập. Dấu câu chỉ chọn vị trí quan sát, không tạo nhãn.

Qwen/Qwen3-4B-Instruct-2507 revision `cdbee75f17c01a7cc42f958dc650907174af0554` chạy BF16, tối đa24prompt/batch và16.384token/batch. Chỉ nhận hai view cùng EOT/NEOT và min relative A/B/C score≥0,8. Hai view cùng model có tương quan, score chưa hiệu chỉnh; nhãn vẫn silver. Qwen chỉ thấy text/context hiện tại, không thấy suffix. Không dùng nhãn Qwen làm test gold.

## Phạm vi đánh giá

Dev natural Việt145/Anh79/Trung69; các slice replay/legacy báo riêng và cùng ràng buộc ngưỡng/chọn checkpoint. Tất cả nhánh được chấm lại trên cùng dev của vòng này. Các số cũ dùng codec/split khác không là đối chứng nhân quả. Không chấm lại384mẫu regression để tune; không gọi dữ liệu crawl không có nhãn độc lập là accuracy/F1 thực.

| Tiếng | Nhánh | Accuracy natural dev | F1 EOT natural dev | EOT nhầm/hold | Min dev slice score |
|---|---|---:|---:|---:|---:|
| vi | baseline | 46.90% | 45.39% | 62.50% | 45.39% |
| vi | data_sft | 45.52% | 46.98% | 67.71% | 45.52% |
| vi | ce_control | 45.52% | 46.98% | 67.71% | 45.52% |
| vi | cost_ce | 45.52% | 46.98% | 67.71% | 45.52% |
| vi | reinforce | 45.52% | 46.98% | 67.71% | 45.52% |
| en | baseline | 64.56% | 75.00% | 65.38% | 64.56% |
| en | data_sft | 64.56% | 75.00% | 65.38% | 64.56% |
| en | ce_control | 64.56% | 75.00% | 65.38% | 64.56% |
| en | cost_ce | 64.56% | 75.00% | 65.38% | 64.56% |
| en | reinforce | 64.56% | 75.00% | 65.38% | 64.56% |
| zh | baseline | 76.81% | 86.89% | 100.00% | 47.53% |
| zh | data_sft | 71.01% | 82.76% | 93.75% | 53.33% |
| zh | ce_control | 71.01% | 82.76% | 93.75% | 53.33% |
| zh | cost_ce | 71.01% | 82.76% | 93.75% | 53.33% |
| zh | reinforce | 71.01% | 82.76% | 93.75% | 53.33% |

Qwen accepted silver: {"vi": 40, "en": 44, "zh": 20}. GPU peak 8.09GiB.

## So sánh reward với đối chứng cùng ngân sách

- vi: chọn dev `data_sft`; REINFORCE thắng CE theo min-slice score: False; thắng cost-CE: False.
- en: chọn dev `data_sft`; REINFORCE thắng CE theo min-slice score: False; thắng cost-CE: False.
- zh: chọn dev `data_sft`; REINFORCE thắng CE theo min-slice score: False; thắng cost-CE: False.


Kích thước/model và pipeline cũ giữ nguyên. Không tự promote nhánh chưa vượt tiêu chí. [Kết quả](outputs/reward_v10_20261025/review.html) · [Số liệu JSON](outputs/reward_v10_20261025/summary.json).

Nguồn: [MultiWOZ upstream](https://github.com/budzianowski/multiwoz/tree/fe0c8e65cfcd8462bd33c86e35f21addc84ca82b), dữ liệu CC BY4 theo Cambridge, code MIT; [CrossWOZ upstream](https://github.com/thu-coai/CrossWOZ/tree/df82c9fdff91b9b130f2d6b89110d3870ba6260e), Apache2. [Qwen model](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554), Apache2. Wikipedia CC BY-SA, URL/revision từng bài trong `data_v10/reward_20261025/crawl/sources.jsonl`.

[Contextual Bandit Algorithms with Supervised Learning Guarantees](https://proceedings.mlr.press/v15/beygelzimer11a/beygelzimer11a.pdf) mô tả partial-feedback bandit. V10 khác ở chỗ reward suy từ nhãn nên có full-feedback giả định; code không triển khai thuật toán Exp4.P hay tuyên bố các bảo đảm của bài báo áp dụng.
