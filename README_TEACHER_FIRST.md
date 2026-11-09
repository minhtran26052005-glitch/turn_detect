# Teacher lớn trước rồi mới distill

Chiến dịch `teacher_first_v14_20261026` làm theo yêu cầu đổi hướng ngày 26/10/2026: ba teacher pretrained lớn riêng Việt / Anh / Trung, đánh giá đạt mục tiêu trước khi distill. Không tiếp tục tối ưu GRU nhỏ trong pha này.

## Model và cách chạy

Mỗi tiếng fine-tune toàn bộ `microsoft/mdeberta-v3-base`, revision `a0484667b22365f84929a935b5e50a51f71f159d`. Đây là encoder pretrained 12 lớp, hidden768, khoảng278M params; số chính xác được đếm trong `teachers/<lang>/identity.json`. Head phân loại hai nhãn EOT/NEOT được khởi tạo mới, không tuyên bố pretrained encoder đã đạt90% EOT.

Worker `python -u -m turn_v2.teacher_first_v14 --steps 100 --microbatch 16 --candidates 384` lặp không giới hạn. Mỗi vòng Qwen3-8B thu thập/chấm text riêng rồi thoát child để giải phóng GPU; sau đó ba child fine-tune teacher tuần tự. Dùng FP32 master weights, BF16 autocast, gradient checkpointing, gradient accumulation2, AdamW và cân bằng hai lớp trong batch. Dữ liệu train tích lũy qua các vòng. Không dừng khi dev plateau hoặc silver90.

V13 đã được dừng nhẹ theo yêu cầu chuyển hướng sau vòng8; snapshot/hash, checkpoint, dữ liệu và source/log vẫn được giữ. V14 dùng SQLite backup để kế thừa dữ liệu và rào chắn nguồn/dev, có database riêng. Các gói ONNX INT4 và registry cũ không bị thay đổi.

## Đánh giá và điều kiện distill

Giữ frozen semantic Qwen silver dev VI31 / EN52 / ZH22; đây là tập nhỏ chọn checkpoint, không phải test độc lập. Natural observed-floor dev145 /79 /69 được báo riêng tại ngưỡng semantic. Không đổi tên nhãn floor thành semantic gold và không tune384regression.

Code chỉ nhận nhãn Qwen hai view đồng thuận, loại mơ hồ; proxy prefix/natural cũ được chấm lại trước khi bổ sung vào train. Qwen không được xem nhãn cũ, đáp án tương lai hay dự đoán của teacher. Các đoạn quan sát được giới hạn trước khi gán nhãn để prompt nằm trong budget.

`distillation_gate.json` giữ distillation khóa cho tới khi cả ba teacher đạt accuracy và F1 EOT≥90% trên dev cùng kiểm chứng độc lập đúng nhiệm vụ và đúng hash checkpoint. Quy tắc thử nghiệm yêu cầu ít nhất200mẫu độc lập,40mẫu mỗi lớp và fingerprint data/prediction. Các ngưỡng cỡ mẫu này là quy tắc chiến dịch, không phải chứng minh thống kê. Hiện chưa có bộ human/publisher semantic EOT gold độc lập cho cả ba tiếng; không chứng nhận90% từ silver. Tiếp tục phát triển trên train/dev; một bộ test đã xem không được tái dùng làm holdout mới.

## Lưu, theo dõi và disk

Teacher best được giữ riêng mỗi tiếng, optimizer/model/RNG lưu trong `latest_resume.pt` mỗi25bước. Phiên bị gián đoạn giữa pha chỉ resume khi fingerprint dữ liệu trùng; pha completed được ghi marker để không chạy lại. Checkpoint đầy đủ hiện nằm trên Colab. `metadata_snapshot.zip` chỉ chứa database/config/report/code, không chứa teacher weights hoặc optimizer; không đủ phục hồi teacher khi VM mất. Chưa xác nhận backup đầy đủ ra ngoài VM.

Không tải Qwen/teacher lớn về Windows. Guard giữ10GiB local,25GiB Colab; thiếu disk thì chờ, không tự xóa. Không có keepalive/lách giới hạn Colab.

- `python tools/teacher_first_colab.py status`: đọc runtime mới, xác minh parent và child theo cmdline/start ticks, GPU/disk, data và teacher.
- `python tools/teacher_first_colab.py launch`: chỉ khởi chạy khi không còn worker v13/v14; không nhân đôi tiến trình.
- `python tools/teacher_first_colab.py snapshot`: lấy metadata dưới30MiB, kiểm size/SHA256, không lấy trọng số.
- `python tools/report_teacher_first.py --notebook`: dashboard và ô đọc trạng thái cuối `a.ipynb`, có receipt thực thi Colab.

Lịch theo dõi15phút đã chuyển sang v14. Nếu HTTP404/503 hoặc kernel mất kết nối, trạng thái remote là chưa biết; cần kết nối lại kernel rồi kiểm checkpoint hiện có. Không dùng PID/log cũ để khẳng định còn chạy. Chỉ dừng worker/lịch theo yêu cầu người dùng.

## Mốc kiểm tra ngày26/10/2026

Lúc01:31 giờ Bangkok, teacher Anh sau300bước đạt94,23% accuracy /91,89% F1 trên52mẫu semantic Qwen silver dev. Đã kiểm SHA256 checkpoint và tính lại confusion matrix từ prediction lưu: `en_silver90_technical_receipt.json`. Checkpoint SHA256 `903088db0cb011314d385199f87012afb63427c2596c975b7f4ec39f24815ed4`. Đây chỉ là mốc silver dev; observed-floor dev79mẫu là49,37% /57,45%, chưa có chứng nhận độc lập. Worker vẫn chạy; không mở distillation hoặc đổi registry serving vì mốc này.

Lúc01:49 giờ Bangkok, cả ba teacher được giữ đã chạm ngưỡng90% trên silver dev nhỏ: VI96,77%/92,31% (31mẫu,400bước), EN94,23%/91,89% (52mẫu,300bước), ZH90,91%/90,00% (22mẫu,300bước). Đã kiểm SHA256 trọng số, số params, rào chắn train/dev và tính lại confusion matrix từ prediction lưu (`all_three_silver90_predictions_receipt.json`). Observed-floor dev vẫn dưới ngưỡng, không có kiểm chứng semantic EOT độc lập và distillation vẫn khóa. Worker tiếp tục vòng5.

Khi xuất hiện hai notebook `a-jvsc-*`, bộ giám sát dùng đúng session cũ còn tồn tại trong `runtime_selection.json`; không tạo hoặc restart kernel/worker. Notebook được người dùng chạy lại làm thay đổi outputs/execution_count, nên validator bảo vệ id/thứ tự/source các ô gốc và kiểm receipt mới của ô trạng thái. Bản notebook và báo cáo validator trước cập nhật vẫn được lưu; không hoàn tác kết quả người dùng chạy.

Lúc02:01 giờ Bangkok, đã hoàn tất vòng6 (600bước/tiếng). Champion Trung ở500bước đạt95,45%/95,24% trên22mẫu silver, cải thiện đúng một mẫu EOT so với champion trước; observed-floor69mẫu đạt55,07%/69,31%. Hash/prediction được xác minh trong `zh_step500_technical_receipt.json`. Đây là cập nhật dev nhỏ, không thay đổi kết luận về kiểm chứng độc lập hoặc mở distillation. Metadata cycle6 đã tải và kiểm SHA; teacher/optimizer đầy đủ vẫn chỉ trên Colab.


## Progress receipt — 2026-10-25 UTC

Cycle 35 is active on the Colab L4 with all three independent mDeBERTa-v3-base teachers (278,810,882 trainable parameters each). Freshly verified EN champion improved on the inherited 52-case Qwen silver set to 96.15% accuracy and 94.44% EOT F1. VI and ZH silver champions remain 96.77% / 93.33% F1 and 95.45% / 95.24% F1. These are small silver-dev results, not independent conversational quality. Natural observed-floor results remain materially lower (best retained floor F1: VI 38.94%, EN 57.45%, ZH 69.31%); no independent human/publisher EOT gold is available, so quality certification and distillation remain disabled. Latest snapshot is metadata-only (cycle 34, SHA-256 8b868ad2a2de8414d5e4c902092429eb01311918bc5fcd3363e13eb332dbfac4); teacher weights and optimizer are still remote-only.


## Natural-floor case review — 2026-10-26

The source is on Colab at /content/turn_detect_v2/data_v10/reward_20261025/{vi,en,zh}/dev.jsonl; the review slice filters valuation_slice == natural (VI 145, EN 79, ZH 69 rows). Local input-plus-output JSONL and the filterable HTML are in outputs/teacher_first_v14_20261026/natural_floor_review/. The review keeps the full input row, observed-floor label/source, selected checkpoint hash, frozen semantic-dev threshold, P(EOT), predicted label, and per-case match. Labels are VAD/observed-floor proxies, is_human_gold=false; they are not semantic human EOT gold.

The selected best teacher was re-run read-only on Colab CPU FP32 to expose per-case probabilities without competing for the training GPU. Recomputed (accuracy / EOT F1) versus the original saved GPU BF16 metrics: VI 50.34% / 36.84% vs 52.41% / 38.94%; EN 45.57% / 53.76% vs 48.10% / 56.84%; ZH 55.07% / 69.31% in both. JSON summary records source and checkpoint SHA-256 plus both metric sets; the small precision-path differences explain the non-identical VI/EN outputs. These scores do not certify the 90% conversational target.


## Live check — 2026-10-26 02:34 UTC

Cycle 44 is collecting teacher data. The pinned Colab L4 sampled at 100% utilization with 16,530 / 23,034 MiB allocated; parent and child process identities both verified and no runtime error is recorded. Current train-pool counts: VI 5,519 (475 EOT / 5,044 NEOT), EN 10,338 (3,472 / 6,866), ZH 6,001 (2,856 / 3,145). The latest verified snapshot is metadata-only, cycle 43, SHA-256 e28d821dbda4d9b1017985f0c98bebcbd02d8f975ffe2010f3a41d12ac64c4ea; weights and optimizer remain remote-only. Semantic champions did not change and remain small Qwen-silver results; independent EOT gold is still unavailable, so quality certification/distillation remain disabled.

## Đánh giá streaming theo dấu chấm cuối câu — 26/10/2026

Đã thêm bộ chấm riêng theo quy tắc mới: mọi prefix trước dấu chấm cuối câu là NEOT, kể cả khi đoạn đó đã có nghĩa hoặc kết thúc bằng dấu phẩy; chỉ prefix cuối có `.` (tiếng Việt/Anh) hoặc `。` (tiếng Trung) mới là EOT. Bộ dữ liệu lấy từ 39 đoạn văn crawl hiện có, đóng băng 157 câu và 6.136 prefix (VI 25/1.007, EN 71/1.957, ZH 61/3.172). Đây là kiểm tra quy tắc trên văn xuôi, không phải hội thoại thật hay human gold; không thay thế natural observed-floor hoặc semantic silver dev.

Kết quả trên best checkpoint, ngưỡng giữ nguyên từ semantic silver dev, suy luận CPU FP32 trên Colab: VI accuracy 54,62%, EOT F1 9,86%, EOT sớm trước dấu chấm 457/982 prefix, END tại dấu chấm 25/25, strict pass 0/25 câu; EN 80,63%, 22,49%, EOT sớm 363/1.886, END cuối 55/71, strict pass 0/71; ZH 71,34%, 10,97%, EOT sớm 904/3.111, END cuối 56/61, strict pass 0/61. **Không tiếng nào đạt điều kiện nghiêm ngặt.**

Nguyên nhân trực tiếp: `turn_v2.common.normalize` xóa dấu câu trước tokenizer; thử cặp prefix có/không có dấu chấm cho thấy token IDs giống hệt ở cả 157 câu. Vì vậy model hiện tại không thể dùng dấu chấm làm tín hiệu riêng. Muốn đáp ứng quy tắc này cần giữ terminal punctuation trong input codec, đưa prefix trước dấu chấm vào NEOT và prefix có dấu chấm vào EOT khi train, rồi đánh giá lại bằng tập khóa độc lập. Lượt train v14 đang chạy không bị thay code hay gián đoạn; bài này không mở distillation và không chứng nhận chất lượng hội thoại.

Review từng câu: `outputs/teacher_first_v14_20261026/terminal_punctuation_eval_20261026T025016Z/review.html`; input/nhãn: `sentence_prefix_manifest.jsonl`; dự đoán: `trace.jsonl`; receipt/metrics: `report.json`. Hash nguồn crawl: `62a523511c3fec4ead0fcf8e7a6dc20d7ab63141c44fed9edd1bc498f73a9697`.

## Live check — 2026-10-26 03:02 UTC

Colab L4 is running cycle 45, training EN step 55/100 (4,455 steps total). Parent and child command/start-tick checks passed. GPU sample: 100%, 5,788/23,034 MiB; Colab disk free: 153.9 GiB. VI completed 4,500 steps; its latest candidate report is 93.55% accuracy / 83.33% EOT F1 on 31 Qwen semantic-silver cases and 66.21% / 22.22% on 145 observed-floor cases. This is a candidate-cycle receipt, not independent gold and not a branch-selection signal by itself. The previously selected champions and the punctuation-rule results remain separately recorded.

The metadata snapshot advanced to cycle 44 and was downloaded locally with SHA-256 `4461fe650b25b5cafe8515dcc79986b4bcb96c9e6b9c3f86088967a8dd8a6f61` (16,985,791 bytes). Local receipt verified size and hash. It contains metadata only; teacher weights and optimizer remain remote-only. Colab reports Drive not mounted, so no external Drive backup is claimed.
