# Turn detector Việt–Anh — hồ sơ đầy đủ các hướng đã thử

Cập nhật ngày **09/11/2026, 10 giờ 43, Asia/Bangkok**. Số liệu trong tài liệu được tổng hợp từ report JSON, manifest, receipt và trạng thái SSH đọc trực tiếp. Mỗi bảng ghi rõ bộ dữ liệu, checkpoint và cách chọn ngưỡng liên quan.

## Mục lục

- [Hiện trạng và hạ tầng](#hien-trang)
- [Dữ liệu, conflict filter, curriculum và packing](#du-lieu)
- [Chuỗi phiên bản v1–v7](#phien-ban)
- [Học thuộc và đối chứng learning rate](#chan-doan)
- [TEST v7 và tỷ lệ end-only](#end-only)
- [Trial train riêng ASR](#asr-only)
- [Audit calibration mở rộng](#calibration)
- [Trial effective16](#effective16)
- [Đối chứng context A/B](#context-ab)
- [Thí nghiệm dịch Qwen3](#dich)
- [Kiến trúc và nén](#nen)
- [Các bộ test lịch sử](#test-cu)
- [Paper và phạm vi áp dụng](#paper)
- [Vấn đề đã xác nhận và câu hỏi còn mở](#van-de)
- [Checkpoint, cleanup và điều kiện package](#checkpoint)
- [Chỉ mục mọi thư mục thí nghiệm](#chi-muc)
- [SHA nguồn và phạm vi bản tổng hợp](#bang-chung)

<a id="hien-trang"></a>

## 1. Trạng thái hiện tại

| Thành phần | Trạng thái đã kiểm tra |
|---|---|
| Campaign chính sentence-mix v7 | `PAUSED_BY_USER_FOR_TEST`, full checkpoint tại global **3.731.173**, branch **500.127** |
| Worker Python của project trên SSH | **0 worker** tại lần kiểm tra ngày 09/11 |
| ASR-only, calibration mở rộng, effective16 và context A/B | Đều hoàn tất và đã báo kết quả |
| Model chính | mALBERT **11.941.122 tham số**, trọng số FP32, huấn luyện BF16 |
| INT4 cho model Việt–Anh hiện tại | Chưa có package được chấp nhận |
| Theo dõi định kỳ | Lịch 300 phút, theo ledger của từng nhánh |
| Boot hiện tại của máy SSH | `bd7525e3-4324-4733-94cb-b0dc4861abc1` |
| Boot của lần train và đối chứng trước | `65117834-0b91-465e-a099-805d6100f317` |
| HDD dùng cho project | Phân vùng NTFS `/dev/sda1`, UUID `37622A5A7A5A4B57`, **hiện chưa mount tại `/mnt/hdd500`** |
| SSD trống lúc đồng bộ 10 giờ 43 | Khoảng **40,53 GiB** |
| Windows C trống lúc đồng bộ 10 giờ 43 | Khoảng **27,00 GiB**, reserve **10 GiB + 5 MiB** |

Máy SSH đã reboot sau các thí nghiệm. Cần mount lại đúng phân vùng HDD rồi kiểm tra chế độ đọc/ghi và dung lượng trước lần train tiếp theo. Dung lượng đọc từ đường dẫn `/mnt/hdd500` khi chưa mount thuộc filesystem gốc, vì vậy trường dung lượng HDD trong snapshot được ghi `null`.

Full checkpoint v7 hiện vẫn khớp SHA256 đã đóng băng. Chi tiết kiểm tra hiện trạng nằm trong snapshot nguồn. [Snapshot ngày 09/11](outputs/experiment_overview_20261109/source_snapshot.json)

Snapshot lúc 10 giờ 26 ghi nhận ổ C còn 11,94 GiB. Khi đồng bộ lúc 10 giờ 43, dung lượng đo lại là 27,00 GiB và HDD vẫn chưa mount. Hai lần đo được giữ theo thời điểm tương ứng. [Biên bản đồng bộ](outputs/experiment_overview_20261109/publication_verification.json)

```text
Checkpoint v7
/home/edabk/hoangbpm/turn_detect/outputs/
  multilingual_malbert_vi_en_sentence_mix_20261107_v7/
  runs/finetune/latest_resume.pt

SHA256
feb4bc2e6cbd4d8c7fa3148f49edb86500c898045a33cc1db8a334bcc03255a8
```

## 2. Bài toán và tiêu chí cần giải quyết

Đầu vào là luồng từ ASR Việt hoặc Anh đã chuẩn hóa chữ thường và bỏ dấu câu. Detector quyết định ranh giới đủ để chuyển một phần văn bản sang bộ dịch. Mỗi quyết định được phép nhìn thêm một hoặc hai từ ở phía trước. Mục tiêu kích thước detector là dưới 20 triệu tham số.

Trong hồ sơ thí nghiệm hiện tại, EOT lấy từ dấu chấm rõ ràng có sẵn trong văn bản nguồn. Các khoảng từ nội bộ được gán NEOT. Câu hỏi, dấu chấm than, ellipsis, viết tắt mơ hồ và các điểm unknown tuân theo chính sách loại khỏi scoring của từng bộ chuẩn bị. Việc đối chiếu nhãn dấu câu đo mức khớp với các nhãn này. Mức hoàn chỉnh ngữ nghĩa cho dịch và thời điểm kết thúc lượt nói theo audio còn cần đánh giá riêng.

Các đại lượng đang dùng được hiểu như sau.

| Đại lượng | Ý nghĩa trong project |
|---|---|
| Precision EOT | Trong các lần model dự đoán cắt, tỷ lệ khớp nhãn EOT |
| Recall EOT | Trong các EOT được phép chấm, tỷ lệ model phát hiện |
| F1 EOT | Trung bình điều hòa của precision và recall |
| FP | Dự đoán cắt tại điểm được gán NEOT |
| FN | Bỏ sót một điểm được gán EOT |
| Accuracy | Đúng trên toàn bộ điểm, bao gồm số lượng NEOT lớn |
| End-only | Một lượt nguồn nhiều phần câu có 0 lần model cắt ở bên trong |
| Forced flush | Bộ gọi bên ngoài gửi nốt phần cuối message hoặc turn |
| Full epoch | Đã đi qua đủ số điểm hợp lệ của chính bộ TRAIN đang xét, có coverage receipt |
| Branch step | Số optimizer updates kể từ khi mở nhánh đó |
| Global step | Bộ đếm kế thừa từ các nhánh cha |

## 3. Hợp đồng đầu vào và model

| Thuộc tính | Cấu hình dùng trong các nhánh gần đây |
|---|---|
| Backbone | `cservan/multilingual-albert-base-cased-32k` |
| Revision | `d3f3c1addfec3420d3f306f674ed2087b34e0411` |
| Số tham số deployment | **11.941.122** |
| Input | `source_raw` đã chuẩn hóa, context bên trái và future 1–2 từ |
| Độ dài encoding | Tối đa **256 token tổng**, giữ phần future trước rồi lấy suffix context vừa budget |
| Reset thông thường | Theo predicted EOT |
| Metadata ngôn ngữ và người nói | Dùng cho chuẩn bị dữ liệu, giới hạn future và báo cáo |
| Model input được phép | Từ nguồn đã quan sát cùng 1–2 từ future theo hợp đồng của bộ dữ liệu |
| LR | **4e-6** trong v5, v6, v7 và các ASR trial |
| Optimizer | AdamW, weight decay **0,01**, **27 trạng thái tham số** |
| Loss từ v5 | Weighted CE, NEOT **1**, EOT **2**, reduction weighted mean |
| Batch chính | 16 causal lanes |
| Effective16 trial | Tích lũy đủ 16 mẫu được chấm cho mỗi optimizer window |
| CPU / GPU | 2 CPU threads, giới hạn bộ nhớ GPU 50% |

`instruction`, `target`, bản dịch tham chiếu và `rest_of_sentence` trong dữ liệu được xử lý như trường dữ liệu nguồn. Input model được kiểm bằng encoded-token hash và các quy tắc causal. TEST được tách khỏi model selection và threshold selection trong các protocol được ghi bên dưới.

<a id="du-lieu"></a>

## 4. Các bộ dữ liệu và thay đổi qua phiên bản

### 4.1. Uploaded TRAIN tự nhiên từ v4

Sau chuẩn bị có **198.509 raw rows được đưa vào**, **17.301 streams**, **3.373.257 từ** và **6.169.873 điểm hợp lệ** cho hai chế độ future k1/k2 trước conflict mask v5.

| Nguồn raw đã kiểm | Số raw rows của nguồn |
|---|---:|
| Daily / `public_checked` | 189.122 |
| Factory / gen500 / `factory_checked` | 8.198 |
| ASR / run2b | 12.415 |
| Factory / gen4000 | 64.947 |

Các số trong bảng mô tả raw đầu vào trước các loại trừ và chống trùng. Tổng rows TRAIN cuối là 198.509 sau chuẩn bị. Member gen500 trùng bản `factory_checked` được nhận diện trong audit. Nguồn DEV/TEST gốc, source family bị giữ lại, chunk không hợp lệ, trùng turn dài và substantial shingles overlap đều được ghi trong manifest. Metadata lịch sử giữ toàn bộ raw/held/rejected trên SSH. [Hồ sơ dữ liệu v4](outputs/multilingual_malbert_vi_en_uploaded_20261104_v4/README.md)

### 4.2. Conflict filter và weighted CE v5

TRAIN được quét theo khóa gồm tối đa tám từ context bên trái và cùng future một hoặc hai từ. Khi cùng khóa có cả nhãn NEOT và EOT, mọi occurrence của cả hai lớp tại khóa đó được mask khỏi loss.

| Mục | Kết quả |
|---|---:|
| Khóa đối nghịch | 697 |
| Điểm bị mask | 6.618 |
| NEOT bị mask | 4.797 |
| EOT bị mask | 1.821 |
| Điểm còn lại mỗi full epoch | **6.163.255** |

Quy tắc cửa sổ tám từ có thể loại cả trường hợp phân giải được bằng lịch sử dài hơn. Phạm vi audit của nó bao phủ khóa đã định nghĩa, còn các va chạm do WordPiece, paraphrase hoặc mọi lịch sử reset động vẫn là giới hạn của kiểm tra. Exposure từ các nhánh cha đã nằm trong trọng số kế thừa. [Hồ sơ filter và loss v5](outputs/multilingual_malbert_vi_en_conflict_weighted_20261106_v5/README.md)

### 4.3. Curriculum v6

Curriculum dùng nested whole-family pools. Mọi stream/bản Việt–Anh trong một family được chọn cùng nhau. Xếp hạng dùng thống kê TRAIN về gap unknown/conflict, độ dài actual turn, wordcount và hash cố định, có phân tầng theo nguồn và thành phần ngôn ngữ của family.

| Stage | Tỷ lệ family mục tiêu | Families | Điểm hợp lệ mỗi lượt pool | Ngân sách stage |
|---|---:|---:|---:|---:|
| 0 | 10% | 874 | 463.539 | 10.000 updates |
| 1 | 25% | 2.174 | 1.624.700 | 20.000 updates riêng |
| 2 | 50% | 4.343 | 3.302.616 | 30.000 updates riêng |
| 3 | 100% | 8.681 | 6.163.255 | Kế hoạch ít nhất 6 full epochs rồi gate |

Các tỷ lệ tính theo số family. Ba stage đầu kết thúc theo budget updates, nên việc mở stage mới có thể xảy ra giữa lượt đi qua pool. Coverage receipt đã xác nhận **một full epoch 6.163.255 điểm** ở stage 3 trước chuyển v7. V6 được dừng để thực hiện yêu cầu đổi dữ liệu sang sentence-mix. [Curriculum v6](outputs/multilingual_malbert_vi_en_curriculum_20261106_v6/README.md)

### 4.4. Sentence-mix v7

Yêu cầu là ghép các câu thành đoạn và giảm tỷ lệ đoạn một câu. Triển khai chọn tỷ lệ theo số pack, áp dụng riêng cho mỗi ngôn ngữ.

| Số câu trong pack | Tỷ lệ | Số pack cộng hai tiếng |
|---|---:|---:|
| 1 | 15% | 7.389 |
| 2 | 15% | 7.389 |
| 3 | 15% | 7.389 |
| 4 | 15% | 7.389 |
| 5 | 15% | 7.389 |
| 6 | 5% | 2.463 |
| 7 | 5% | 2.463 |
| 8 | 5% | 2.463 |
| 9 | 5% | 2.463 |
| 10 | 5% | 2.463 |

Tổng **49.260 packs**, gồm **23.340 Việt** và **25.920 Anh**. Có **208.906 đơn vị fullstop hoàn chỉnh duy nhất**, **449 lượt dùng thêm** để lấp block cuối, **209.355 lượt dùng câu**. Mọi pack text hash là duy nhất. Các câu được ghép cùng tiếng và cùng nguồn, có thể đến từ các actual turn hoặc speaker khác nhau.

Trong TRAIN v7, cả pack là một segment tổng hợp, future được phép qua điểm nối giữa các câu nguồn trong pack. DEV vẫn giới hạn future trong actual source turn. Đây là khác biệt về phân phối TRAIN/DEV được ghi rõ trong protocol.

| Mục | Kết quả v7 |
|---|---:|
| Điểm hợp lệ mỗi full epoch | **5.040.023** |
| EOT points | 319.993, khoảng 6,35% |
| Fullstop anchors cuối đoạn thiếu future | 23,529% |
| Conflict keys hiện hành | **75** |
| Điểm mask hiện hành | **276** |
| Raw tail elements được giữ ở parent, ngoài pool pack | 61.446, tổng 696.846 từ |
| Held source occurrences bị loại | 112 |
| Exact duplicate occurrences được gom provenance | 23.041 |

Field `conflict_filter` kế thừa trong manifest có thống kê v5/v6. Số hiện hành v7 lấy từ `sentence_pack_audit` và `independent_pack_verification`. `manifest_clarification.json` giải thích sự kế thừa này. Full epoch đầu **5.040.023 điểm** đã được kiểm chứng. V7 dừng theo yêu cầu test tại stage_seen **7.773.883**, tương đương một full epoch và phần lượt tiếp theo. [Sentence-mix v7](outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7/README.md)

## 5. DEV và TEST gồm bao nhiêu câu

Audit cấu trúc đếm các span theo fullstop rõ ràng cộng phần đuôi còn từ. Đối tượng là actual source turn có dấu câu và ít nhất ba từ để có khả năng sinh input k2. “Một span” trong bảng mô tả cách đếm này, còn cấu trúc mệnh đề/ngữ nghĩa cần kiểm riêng.

| Cohort trong audit | Turn đủ điều kiện | Một span | Tỷ lệ một span | Từ hai span |
|---|---|---|---|---|
| DEV_monitor | 1.763 | 1.183 | 67,10% | 580 |
| DEV_calibration | 545 | 339 | 62,20% | 206 |
| DEV_full | 14.952 | 10.470 | 70,02% | 4.482 |
| TEST_last_evaluated | 15.000 | 10.554 | 70,36% | 4.446 |

Bộ full DEV lưu 5.911 streams. Census cấu trúc chỉ thống kê các turn đáp ứng điều kiện audit, thuộc 4.137 streams. Cần dùng đúng denominator của từng bảng. [Audit cấu trúc DEV/TEST](outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7/dev_test_sentence_structure_audit_20261107.json)

Audit TRAIN tự nhiên trước v7 có **167.888 turn**, trong đó **82.021 turn một span**, **85.867 turn từ hai span**, và **23.101 turn từ ba span**. Có **106.442/232.059 fullstop anchors nằm ở cuối actual turn**, tương đương **45,87%**, chịu giới hạn thiếu future. Một turn một span có thể chỉ đóng góp các điểm NEOT. [Audit TRAIN trước packing](outputs/multilingual_malbert_vi_en_curriculum_20261106_v6/input_sentence_structure_audit_20261107.json)

DEV hiện tại có **119 calibration streams**, **13.275 scored k2 points**, rồi **394 monitor streams**, **34.987 scored k2 points**. Toàn monitor có **586 EOT Việt** và **696 EOT Anh**. Phần ASR monitor chỉ gồm **7 turn/7 streams mỗi tiếng**, với **269 EOT Việt** và **289 EOT Anh**. Calibration ASR có **3 turn mỗi tiếng**. Số điểm từ lớn cần được đọc cùng số nguồn độc lập nhỏ này.

<a id="phien-ban"></a>

## 6. Chuỗi phiên bản và điểm DEV

### 6.1. Các phiên bản dùng DEV cũ

| Phiên bản | Global step report cuối | Ngưỡng | F1 Việt (%) | F1 Anh (%) | F1 Trung (%) |
|---|---|---|---|---|---|
| [v1](outputs/multilingual_malbert_pretrained_20261103_v1/README.md) | 14.000 | 0,28 | 76,92 | 73,54 | 43,36 |
| [v2](outputs/multilingual_malbert_vi_en_20261103_v2/README.md) | 85.000 | 0,82 | 79,01 | 86,78 | — |
| [v3](outputs/multilingual_malbert_vi_en_checked_20261103_v3/README.md) | 1.465.000 | 0,13 | 81,61 | 83,40 | — |

Các mốc dưới đây là report DEV định kỳ hoàn tất cuối được tìm thấy khi lập hồ sơ. Cohort cũ có 2.058 điểm Việt và 2.304 điểm Anh, cùng nhãn synthetic punctuation. V1 còn có cohort Trung lịch sử. [Danh sách nguồn report](outputs/experiment_overview_20261109/latest_training_devs.json)

### 6.2. Các phiên bản dùng DEV rộng từ v4

| Phiên bản | Branch report cuối | Global step | Ngưỡng | F1 Việt (%) | F1 Anh (%) |
|---|---|---|---|---|---|
| [v4](outputs/multilingual_malbert_vi_en_uploaded_20261104_v4/README.md) | 890.000 | 2.355.080 | 0,1 | 22,44 | 44,65 |
| [v5](outputs/multilingual_malbert_vi_en_conflict_weighted_20261106_v5/README.md) | 260.000 | 2.615.080 | 0,1 | 25,97 | 47,99 |
| [v6](outputs/multilingual_malbert_vi_en_curriculum_20261106_v6/README.md) | 600.000 | 3.224.058 | 0,1 | 26,51 | 48,89 |
| [v7](outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7/README.md) | 500.000 | 3.731.046 | 0,1 | 29,65 | 47,65 |

V4 chuyển sang uploaded-only TRAIN và bộ DEV rộng. V5 đồng thời đổi filter và class weighting. V6 đồng thời thêm updates và curriculum. V7 đổi pool thành synthetic sentence packs và tiếp tục từ v6. Các thay đổi này tạo chuỗi warm-start trước/sau. Muốn phân giải riêng tác dụng của từng thành phần cần đối chứng phù hợp.

### 6.3. Diễn biến của v7

- Trong 48 report đầu đã phân tích, best được giữ ở branch **70.000**, F1 Việt **35,06%**, Anh **49,94%**.
- Trung bình 10 report đầu là **33,11% Việt / 49,80% Anh**, 10 report cuối của đợt phân tích là **30,45% / 48,59%**.
- Branch **380.000** đạt retention gate tương đối với teacher trong lần chấm đó, F1 **31,21% / 47,82%**. Recall và precision là các điều kiện của gate.
- Ngưỡng ở một số report khác nhau, gồm mốc branch 400.000 chọn 0,4. Các report còn lại phần lớn chọn 0,1.
- Điểm dao động và giảm nhẹ về cuối đợt đã đọc. Các kết quả này chưa phân giải riêng overfitting, thay đổi ngưỡng và các reset history phát sinh.

Ngân sách sáu full epochs là kế hoạch huấn luyện của campaign. Lệnh tạm dừng của user được thực hiện sau một full epoch v7 và phần epoch tiếp theo. [Phân tích trend đã lưu](outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7/direct_dev_trend_20261107.json)

<a id="chan-doan"></a>

## 7. Các chẩn đoán khả năng học và learning rate

### 7.1. Học thuộc 1.000 input TRAIN cố định

| Thiết lập | Kết quả |
|---|---|
| Số input | 1.000 input duy nhất, 500 mỗi tiếng |
| Phân bố mỗi tiếng | 125 EOT và 375 NEOT, được phân tầng chủ động |
| Optimizer / LR | AdamW mới, 3e-5 |
| Batch | Effective32, microbatch4 |
| Ngân sách thực tế | 200 updates |
| TRAIN accuracy / F1 / recall cuối | **100% / 100% / 100%** |

Phép thử xác nhận pipeline có khả năng fit một bộ TRAIN nhỏ cố định. Tập này được phân tầng 25% EOT và cố định input, khác phân bố tự nhiên cùng ngữ cảnh reset động của campaign. [Report sanity](outputs/malbert_learning_diagnostic_20261106_v1/sanity/report.json)

### 7.2. Đối chứng LR 4e-6 và 1e-5

Hai nhánh bắt đầu cùng full checkpoint **2.585.080**, model, 27 AdamW, RNG và order. Mỗi nhánh chạy **3.000 updates**, **47.122 điểm được chấm**. Calibration dùng mục tiêu precision ít nhất 50% mỗi tiếng rồi tối đa recall tiếng yếu hơn.

| Nhánh | Ngưỡng | Việt F1 / P / R (%) | Anh F1 / P / R (%) |
|---|---:|---:|---:|
| Baseline | 0,2 | 23,00 / 47,34 / 15,19 | 46,66 / 57,87 / 39,08 |
| LR 4e-6 | 0,4 | 27,70 / 48,10 / 19,45 | 47,06 / 54,55 / 41,38 |
| LR 1e-5 | 0,9 | 23,27 / 56,21 / 14,68 | 42,71 / 52,07 / 36,21 |

Trong trial này, 4e-6 cho recall cao hơn 1e-5 ở cả hai tiếng. Precision trên calibration và monitor được báo riêng vì chúng là hai cohort khác nhau. Đã audit 105.810 inference monitor và 48 ca canonical. [Chẩn đoán học và LR](outputs/malbert_learning_diagnostic_20261106_v1/README.md)

<a id="end-only"></a>

## 8. Tạm dừng v7 và kiểm tra end-only trên TEST

V7 được tạm dừng tại global **3.731.173**, branch **500.127**. Cùng checkpoint này được chọn ngưỡng **0,1** trên DEV trước khi chấm TEST. Tổng **579.685 inferences** gồm **297.397 k1** và **282.288 k2**, có audit nguồn, future, reset, encoding, confusion và word conservation.

| Kết quả TEST k2 của v7 đã tạm dừng | Việt | Anh |
|---|---:|---:|
| F1 | 36,45% | 46,66% |
| Precision | 42,37% | 44,43% |
| Recall | 31,97% | 49,13% |

### 8.1. Loại turn một span khỏi denominator end-only

| Cohort | Turn end-only / turn nhiều span | Tỷ lệ |
|---|---:|---:|
| Toàn TEST k2 | 1.190 / 4.446 | **26,77%** |
| Việt | 687 / 2.069 | **33,20%** |
| Anh | 503 / 2.377 | **21,16%** |
| Strict ít nhất hai fullstop hoàn chỉnh | 863 / 3.238 | **26,65%** |
| Sau loại known legacy overlap | 1.130 / 4.233 | **26,69%** |

Có **10.554 turn một span** được loại khỏi denominator chính. Thêm **645 turn nhiều span** có cắt nhưng mọi cut đều lệch các fullstop được chấm. Một số điểm cắt khác dấu chấm vẫn có thể phù hợp ngữ nghĩa, nên việc phân loại theo nhãn dấu câu cần được đọc đúng mục tiêu.

Metric turn end-only giả định caller sẽ flush khi turn nguồn kết thúc. Evaluator detector giữ rolling context qua source turn và speaker. Trong replay message, harness flush cuối message như một thao tác bên ngoài.

### 8.2. Replay detector trên 310 message dịch

| Cách đếm message nhiều câu | End-only / mẫu nhiều câu | Tỷ lệ |
|---|---:|---:|
| Đếm span theo period raw | 9 / 21 | 42,86% |
| Sau loại ba numbered headings / No. giả đa câu | 6 / 18 | **33,33%** |
| Thêm một message mở bằng câu hỏi | 6 / 19 | **31,58%** |

Bộ 310 message có **289 message một span** theo cách đếm raw. Replay này thực hiện **3.900 quyết định detector**. Translator và điểm BLEU/chrF của thí nghiệm dịch cũ là kết quả riêng ở mục 13. [TEST v7 và end-only](outputs/sentence_mix_end_only_test_20261107_v1/README.md)

<a id="asr-only"></a>

## 9. Train riêng ASR một full epoch

Mở một nhánh mới từ v7 đã tạm dừng, dùng **210 streams / 81 canonical video families / 169.011 từ**, tổng **333.315 điểm ASR**. Nhánh đi theo thứ tự actual source turn tự nhiên, có cùng model/27 AdamW/RNG ban đầu. Full epoch cần **22.147 updates** khi chấm theo batch/lane cũ.

Thay đổi so với parent gồm domain ASR và thứ tự tự nhiên thay cho synthetic packs. Kết quả trên cùng ngưỡng 0,1 được ghi dưới đây.

| Nguồn | Parent F1 Việt / Anh (%) | Sau ASR-only F1 Việt / Anh (%) |
|---|---:|---:|
| ASR | 10,43 / 39,05 | **0,74 / 42,13** |
| Daily | 44,85 / 57,44 | 6,76 / 47,06 |
| Factory | 45,24 / 54,81 | 12,50 / 39,58 |
| Numeric dialogue | 14,14 / 29,87 | 0,00 / 26,00 |

ASR Việt còn **1 TP / 269 EOT**, **0 FP**, recall **0,37%**. Precision 100% đi cùng đúng một cut. ASR Anh giảm FP từ 82 xuống 37 nhưng TP từ 90 xuống 87.

Trial hoàn tất đúng một full epoch và có đầy đủ proof về coverage, model thay đổi, optimizer/RNG/order và monitor. Sau đó, mở rộng calibration trên chính trọng số này cho kết quả khác đáng kể, được ghi ở mục 11. [Trial ASR-only](outputs/asr_only_adaptation_20261107_v1/README.md)

<a id="calibration"></a>

## 10. Calibration mở rộng trên đúng checkpoint v7 đã tạm dừng

Grid cũ 0,1–0,9 giới hạn miền ngưỡng được khảo sát. Audit mới dùng **20 ngưỡng cố định từ 1e-5 đến 0,9**, chọn maximum worst-language F1 rồi sum F1 trên calibration. Ngưỡng được chọn cho checkpoint v7 là **0,03**. Trong audit này có 12 rollout calibration mới đến 0,1, phần trên 0,1 được tái dùng từ hồ sơ khớp trọng số, và một monitor mới tại ngưỡng đã chọn.

| Chỉ số toàn DEV | V7 ngưỡng 0,1 | Cùng weights ngưỡng 0,03 |
|---|---:|---:|
| Việt F1 / P / R (%) | 29,93 / 39,12 / 24,23 | **34,88 / 30,66 / 40,44** |
| Anh F1 / P / R (%) | 47,61 / 47,24 / 47,99 | **48,24 / 38,42 / 64,80** |
| Việt FP / FN | 221 / 444 | 536 / 349 |
| Anh FP / FN | 373 / 362 | 723 / 245 |
| ASR Việt F1 / recall (%) | 10,43 / 6,32 | 22,88 / 20,07 |
| ASR Anh F1 / recall (%) | 39,05 / 31,14 | 44,69 / 49,48 |

Ngưỡng thấp tăng recall và số cut lệch nhãn dấu câu. Trong pipeline predicted-reset, ngưỡng còn thay đổi context của các input phía sau. Audit đã kiểm **160.776 bản ghi calibration mới**, **35.270 bản ghi monitor**, **158 ca canonical**. [Audit calibration](outputs/malbert_calibration_audit_20261107_v1/README.md)

<a id="effective16"></a>

## 11. Effective batch 16 và hiệu ứng ngưỡng trên ASR-only

Hai nhánh bắt đầu cùng initial model, 27 AdamW, RNG và order của ASR-only. Cùng **333.315 điểm** và một full epoch. Control cũ có **22.147 updates**, treatment có **20.833 updates**, gồm 20.832 window đủ 16 và window cuối 3 mẫu. Cách gom mẫu thay đổi update count và dropout grouping, các reset history về sau có thể khác nhau.

### 11.1. Những model/policy cần phân biệt

| Model và cách chọn ngưỡng | Ngưỡng | Toàn DEV Việt / Anh F1 (%) | ASR Việt / Anh F1 (%) |
|---|---:|---:|---:|
| Parent v7, calibration mở rộng | 0,03 | 34,88 / 48,24 | 22,88 / 44,69 |
| ASR-only cũ, grid cũ | 0,1 | 5,13 / 41,10 | 0,74 / 42,13 |
| Cùng weights ASR-only cũ, calibration mở rộng | 0,003 | 30,74 / 47,87 | **27,47 / 53,58** |
| ASR effective16, calibration mở rộng | 0,03 | 30,88 / 48,29 | **31,20 / 52,65** |

Cùng weights ASR-only cũ có F1 ASR Việt **0,74% tại 0,1** và **27,47% tại 0,003**. Nhận định mất khả năng hiểu từ điểm 0,74% đơn lẻ vượt quá điều phép đo đó xác nhận.

### 11.2. Control cũ đã calibration so với effective16

| Chỉ số ASR | Control cũ | Effective16 |
|---|---:|---:|
| Việt F1 / P / R (%) | 27,47 / 30,09 / 25,28 | 31,20 / 33,77 / 29,00 |
| Việt TP / FP / FN | 68 / 158 / 201 | 78 / 153 / 191 |
| Anh F1 / P / R (%) | 53,58 / 47,83 / 60,90 | 52,65 / 47,88 / 58,48 |
| Anh TP / FP / FN | 176 / 192 / 113 | 169 / 184 / 120 |

So với parent v7 đã calibration, effective16 tăng ASR nhưng toàn DEV Việt giảm **34,88 → 30,88%**, daily Việt **46,20 → 34,03%**, factory Việt **48,00 → 34,25%**. Trial cho thấy sự đánh đổi giữa domain. Full coverage, optimizer/RNG/accumulator và **606.460 bản ghi calibration/monitor** đã được audit, với **560 ca canonical**. [Trial effective16](outputs/asr_effective_batch16_20261107_v1/README.md)

<a id="context-ab"></a>

## 12. Đối chứng giữ ngữ cảnh A/B mới nhất

Cả hai nhánh mới bắt đầu từ v7 global **3.731.173**, cùng initial model/27 AdamW/all RNG/order. Mỗi nhánh có **333.315 điểm ASR**, **20.833 updates**, cùng LR, class weights và effective16. A dùng context từ predicted EOT gần nhất. B giữ tối đa 256 từ nguồn bên trái, độc lập với quyết định cắt, rồi encode tối đa 256 token tổng.

Hai nhánh đều fit xong trước khi chấm monitor. Cùng grid 20 ngưỡng và mục tiêu calibration, A chọn **0,02**, B chọn **0,03**.

| Cohort | A F1 (%) | B F1 (%) | B − A (điểm phần trăm) |
|---|---|---|---|
| Toàn DEV VI | 31,09 | 27,28 | -3,81 |
| Toàn DEV EN | 44,85 | 43,75 | -1,10 |
| asr EN | 48,46 | 50,23 | 1,77 |
| asr VI | 31,85 | 28,99 | -2,86 |
| daily EN | 49,90 | 45,54 | -4,36 |
| daily VI | 36,36 | 27,33 | -9,03 |
| factory EN | 45,32 | 39,60 | -5,71 |
| factory VI | 34,35 | 30,09 | -4,25 |
| numeric/dialogue EN | 24,82 | 31,78 | 6,96 |
| numeric/dialogue VI | 11,70 | 13,43 | 1,73 |

Trên toàn DEV, recall Việt **40,27 → 30,89%**, Anh **70,11 → 60,63%**. FP giảm **696 → 560** ở Việt, **992 → 811** ở Anh. TP giảm **236 → 181** và **488 → 422** tương ứng.

Chỉ số chính là thay đổi F1 ngôn ngữ yếu hơn trên toàn DEV. Kết quả **−3,81 điểm phần trăm**, bootstrap ghép cặp **2.000 lần / 211 family**, CI95% **[−6,28, −1,50] điểm phần trăm**. Điều kiện thăm dò khóa trước chưa đạt. Hai threshold được chọn riêng, nên số liệu mô tả hai pipeline đã calibration.

Đây là một lần train mới cho control A. Trọng số cuối control A có SHA khác model effective16 cũ. Cần so A/B trong đúng trial này. Cùng RNG khởi đầu chưa xác nhận bitwise GPU replay giữa các lần chạy, và CI bootstrap theo family chỉ mô tả độ bất định lấy mẫu trên các outputs hiện tại.

Fixture CPU kiểm resume giữa gradient window khớp bitwise ở cả hai chính sách. Audit 100 updates đầu mỗi nhánh có **1.622 encoded inputs / 1.600 scored samples**. B có **1.060 input history khác decoder history**. Đã kiểm final full epoch/27 AdamW/RNG/order/lanes/accumulator và **606.460 bản ghi DEV**, **569 ca canonical**. Có **16 ví dụ DEV ghép cặp** để xem khác biệt quyết định tại cùng source boundary. [Báo cáo A/B đầy đủ](outputs/context_window_ab_20261108_v1/review.html)

<a id="dich"></a>

## 13. Thí nghiệm dịch với Qwen3

Detector và translator là hai model riêng. Translator do user cung cấp có **596.049.920 tham số**, checkpoint **15.640**. Thí nghiệm dịch ngày 05/11 dùng detector lịch sử checkpoint **1.945.080**, threshold **0,1**, cùng 310 message và template B.

| Nhánh dịch | chrF toàn bộ | chrF Việt→Anh | chrF Anh→Việt | BLEU toàn bộ |
|---|---:|---:|---:|---:|
| Whole written | 41,06 | 40,95 | 41,15 | 17,52 |
| Whole ASR | 39,62 | 40,23 | 39,06 | 14,25 |
| Detector ASR | 39,32 | 40,10 | 38,59 | 13,91 |

Có **33 internal cuts trong 29 message**, **343 translation chunks**, và **310 final message flushes** do harness. Tổng 930 message records, 963 translation prompts. Có 48 message cho output khác giữa hai cách chạy. Năm lần sinh whole-message chạm token cap và có lặp, được giữ trong metric.

chrF và BLEU là điểm đối chiếu tham chiếu văn bản. Dòng thử nghiệm này chưa đo thời gian endpoint theo audio. Phép replay detector v7 ở mục 8 dùng lại bộ message để thống kê segmentation, còn các điểm dịch trong bảng thuộc lần chạy ngày 05/11. [Báo cáo dịch](outputs/translation_turn_test_20261105_v1/README.md)

<a id="nen"></a>

## 14. Những hướng kiến trúc và nén đã thử

### 14.1. TinyGRU, custom dưới 20M và transformer INT8 lịch sử

| Candidate và report | Model MB | F1 Việt (%) | F1 Anh (%) | F1 Trung (%) | Retention trong report |
|---|---|---|---|---|---|
| [gru128_fp32](outputs/multilingual_tiny_compression_20261102_v1/final_report.json) | 2,640 | 89,02 | 91,36 | 42,52 | Chưa đạt |
| [gru128_int8](outputs/multilingual_tiny_compression_20261102_v1/final_report.json) | 0,699 | 89,02 | 90,98 | 42,52 | Chưa đạt |
| [gru128_int4_g64](outputs/multilingual_tiny_compression_20261102_v1/final_report.json) | 0,371 | 88,37 | 89,88 | 45,62 | Chưa đạt |
| [gru128_int4_g32](outputs/multilingual_tiny_compression_20261102_v1/final_report.json) | 0,392 | 90,17 | 89,88 | 47,89 | Chưa đạt |
| [gru192_fp32](outputs/multilingual_tiny_compression_20261102_v1/final_report.json) | 3,200 | 88,52 | 94,02 | 50,30 | Chưa đạt |
| [gru192_int8](outputs/multilingual_tiny_compression_20261102_v1/final_report.json) | 0,845 | 88,52 | 94,02 | 50,68 | Chưa đạt |
| [gru192_int4_g64](outputs/multilingual_tiny_compression_20261102_v1/final_report.json) | 0,447 | 88,14 | 94,74 | 49,77 | Chưa đạt |
| [gru192_int4_g32](outputs/multilingual_tiny_compression_20261102_v1/final_report.json) | 0,472 | 89,50 | 94,02 | 49,90 | Chưa đạt |
| [compact_fp32](outputs/multilingual_under20m_20261103_v1/final_report.json) | 74,735 | 1,90 | 1,48 | 37,70 | Chưa đạt |
| [compact_int8](outputs/multilingual_under20m_20261103_v1/final_report.json) | 19,446 | 1,90 | 1,48 | 37,11 | Chưa đạt |
| [compact_int4_g64](outputs/multilingual_under20m_20261103_v1/final_report.json) | 10,127 | 4,07 | 1,54 | 9,91 | Chưa đạt |
| [compact_int4_g32](outputs/multilingual_under20m_20261103_v1/final_report.json) | 10,709 | 0,00 | 0,00 | 7,92 | Chưa đạt |
| [int4_g32](outputs/multilingual_transformer_compression_20261102_v1/final_report.json) | 76,419 | 89,66 | 97,94 | 63,79 | Chưa đạt |
| [mixed_int4_int8_embedding_g32](outputs/multilingual_transformer_compression_20261102_v1/final_report.json) | 122,521 | 89,66 | 97,94 | 63,98 | Chưa đạt |
| [int8_g64](outputs/multilingual_transformer_compression_20261102_v1/final_report.json) | 139,824 | 91,62 | 95,58 | 65,39 | Đạt |

Các candidate trong bảng dùng bộ DEV cũ. Kết luận TinyGRU là trượt retention gate ba ngôn ngữ của chiến dịch đó. Các điểm Việt/Anh trên synthetic DEV cũ cần đi kèm cohort khi đọc. Model custom có **18.668.738 tham số**. Transformer INT8 lịch sử giữ backbone khoảng **135M tham số**, package trọng số khoảng **139,8 MB** và được chấp nhận theo gate cũ.

Mục tiêu dưới 20M hiện dùng số tham số deployment. Giảm storage bits của backbone 135M vẫn để số tham số ở quy mô đó. Hướng mALBERT 11,94M hiện tại mới đạt giới hạn kích thước tham số, còn chất lượng và package INT4 đang là các điều kiện cần giải quyết.

### 14.2. Teacher 135M trên DEV rộng hiện tại

Frozen teacher checkpoint **514.235**, threshold **0,1**, được kiểm trên cùng DEV calibration/monitor mới.

| Ngôn ngữ | F1 (%) | Precision (%) | Recall (%) |
|---|---:|---:|---:|
| Việt | 22,34 | 17,24 | 31,74 |
| Anh | 38,43 | 30,19 | 52,87 |

Retention gate cho phép giảm tối đa **2 điểm phần trăm F1**, **2 điểm precision**, **3 điểm recall** mỗi tiếng so với control này. Vì control trên DEV rộng có mức điểm như bảng, việc qua retention gate mô tả giữ chất lượng tương đối với control. [Kiểm lại teacher](outputs/multilingual_malbert_vi_en_uploaded_20261104_v4/teacher_new_dev_user_recheck_20261105.json)

### 14.3. Precision probe trên bộ cũ

Probe checkpoint lịch sử chọn threshold **0,75** thay **0,53** để nhắm precision 95%. Trên verification monolingual, precision Việt/Anh **96,25% / 97,33%**, recall **80,21% / 90,12%**. Trên mixed cohort, precision **100% / 96,63%**, recall **86,67% / 92,47%**. Cohort và model của probe này thuộc nhánh stream 135M cũ. [Precision probe](outputs/multilingual_stream_precision_dev_20261102_v1/report.json)

<a id="test-cu"></a>

## 15. Các bộ TEST và diagnostic lịch sử

### 15.1. Uploaded original TEST, fullstop, k2

| Checkpoint / report | Ngưỡng DEV | F1 Việt (%) | F1 Anh (%) | Nguồn |
|---|---|---|---|---|
| [827.000](outputs/malbert_eval_zip_20261104_v1/report.json) | 0,75 | 16,14 | 24,49 | Uploaded original TEST |
| [1.340.000](outputs/malbert_eval_zip_20261104_v2/report.json) | 0,73 | 14,65 | 27,25 | Uploaded original TEST |
| [1.945.080](outputs/malbert_uploaded_test_20261105_v1/report.json) | 0,1 | 26,00 | 41,14 | Uploaded original TEST |
| [3.731.173](outputs/sentence_mix_end_only_test_20261107_v1/report.json) | 0,1 | 36,45 | 46,66 | V7 đã tạm dừng |

Các hàng dùng cohort `all_requested`. Known legacy overlap được giữ nhãn và có báo cáo subgroup loại overlap riêng. Checkpoint và threshold được khóa từ DEV trước lần TEST tương ứng. Những TEST đã xem giới hạn khả năng dùng lại chúng làm xác nhận độc lập cho các quyết định thiết kế phát sinh sau đó.

### 15.2. C75 và hội thoại synthetic lịch sử

| Checkpoint / report | Cohort | F1 Việt (%) | F1 Anh (%) | F1 Trung (%) |
|---|---|---|---|---|
| [106.000](outputs/multilingual_student_test_20261031_v2/report.json) | C75 exposed diagnostic | 77,33 | 84,31 | 61,85 |
| [211.200](outputs/multilingual_student_test_20261101_v1/report.json) | C75 sau audit | 75,27 | 83,35 | 63,32 |
| [542.135](outputs/multilingual_stream_c75_test_20261102_v1/report.json) | C75 sau audit | 52,35 | 70,61 | 58,36 |
| [515.435](outputs/multilingual_stream_test_20261102_v1/report.json) | Synthetic paragraphs mono / mixed | 86,73 / 91,18 | 96,99 / 96,99 | — |
| [741.935](outputs/multilingual_stream_test_20261103_v1/report.json) | Synthetic paragraphs mono / mixed | 89,74 / 93,36 | 97,62 / 97,62 | — |

Các cohort này gồm dữ liệu punctuation-derived và dữ liệu đã từng được expose trong quá trình phát triển. Những điểm F1 cao ở đây thuộc đúng loại nguồn và protocol ghi trong report. [Diagnostic stream checkpoint 741.935](outputs/multilingual_stream_test_20261103_v1/README.md)

<a id="paper"></a>

## 16. Paper đã đọc và cách liên hệ với thí nghiệm

Rà soát ngày 08/11 là targeted narrative review, gồm các phần phương pháp/bảng kết quả liên quan. Danh sách sau ghi nội dung đã dùng và giới hạn áp dụng. [Bản review đầy đủ](outputs/turn_failure_literature_20261108/README.md)

| Paper | Nội dung đã dùng | Giới hạn áp dụng | Nguồn |
|---|---|---|---|
| P1 — Dynamic Sentence Boundary Detection for Simultaneous Translation | Prefix và multi-position boundary prediction. F-score 74% với một future word, 77% với hai | ERNIE, En–De, quy tắc punctuation khác project | [Primary source](https://aclanthology.org/2020.autosimtrans-1.1.pdf) |
| P2 — Online Punctuation Restoration using ELECTRA Model for streaming ASR Systems | ELECTRA-Small 13,6M, ghép 1–10 câu, delay ba từ. Average punctuation F1 71,2/69,4% trên hai tập ASR | Czech, bộ fine-tune 335M tokens. Weight dấu câu áp vào xác suất inference | [Primary source](https://www.isca-archive.org/interspeech_2023/polacek23_interspeech.pdf) |
| P3 — Improved Training for End-to-End Streaming ASR Model with Punctuation | Acoustic CTC 18M và concat/chunk loss. Period F1 35,8 → 46,6% | Có audio, future theo giây. Ablation cho thấy trade-off với WER | [Primary source](https://www.isca-archive.org/interspeech_2023/kim23_interspeech.pdf) |
| P4 — Efficient Automatic Punctuation Restoration Using Bidirectional Transformers with Robust Inference | Sequence prediction và overlapping-window aggregation. RoBERTa-base combined F1 83,9% | Bidirectional/offline, context và kích thước khác detector hiện tại | [Primary source](https://aclanthology.org/2020.iwslt-1.33.pdf) |
| P5 — Improving Punctuation Restoration for Speech Transcripts via External Data | Chọn external text gần domain rồi fine-tune in-domain. F1 68,48 → 69,60% | English noisy support transcripts, nhãn được người sửa, protocol khác future2 | [Primary source](https://aclanthology.org/2021.wnut-1.19.pdf) |
| P6 — On Calibration of Modern Neural Networks | Đánh giá calibration và temperature scaling trên validation | Cải thiện thang xác suất cần phân biệt với chất lượng xếp hạng và reset context | [Primary source](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) |
| P7 — Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution | Full fine-tuning có thể làm giảm OOD performance, LP→FT được khảo sát | Các benchmark ảnh và mô hình lý thuyết, cơ chế ở mALBERT cần kiểm riêng | [Primary source](https://arxiv.org/pdf/2202.10054) |
| P8 — Fine-Tuning is Fine, if Calibrated | Feature có thể được giữ trong khi biased logits làm điểm giảm | Subset-class image transfer, hai lớp EOT/NEOT vẫn cùng có mặt trong TRAIN hiện tại | [Primary source](https://arxiv.org/html/2409.16223v3) |
| P9 — Segmentation-Free Streaming Machine Translation | Kết hợp segmentation vào streaming MT và memory policy | Cần thay đổi huấn luyện translator và pipeline | [Primary source](https://aclanthology.org/2024.tacl-1.61.pdf) |
| P10 — Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks | DAPT/TAPT trên bốn domains, tám classification tasks | MLM adaptation khác supervised EOT fine-tune, corpus ASR hiện khá nhỏ | [Primary source](https://aclanthology.org/2020.acl-main.740/) |

Nguồn 2025 trên ScienceDirect về lightweight online punctuation/capitalization gặp lỗi truy cập toàn văn. Review giữ bản ghi trong search log và dùng các primary source truy cập được cho số liệu chi tiết. [Search log](outputs/turn_failure_literature_20261108/search_log.json)

<a id="van-de"></a>

## 17. Vấn đề đã xác nhận và câu hỏi còn mở

| Vấn đề | Điều đã xác nhận | Phạm vi còn cần giải quyết |
|---|---|---|
| Bỏ sót ranh giới | Recall thấp trên nhiều cohort, end-only đáng kể trên turn nhiều span | Những điểm cắt nào phù hợp ngữ nghĩa cho dịch cần bộ nhãn tương ứng |
| Miền calibration cũ | Cùng ASR-only weights đạt 0,74% và 27,47% F1 Việt ASR ở hai operating policies | Calibration đánh đổi FP/FN và làm đổi reset histories ở pipeline A |
| Thiếu supervision cuối actual turn | 45,87% fullstop anchors tự nhiên ở cuối turn chịu thiếu future | Cơ chế giao việc cho caller khi hết stream/timeout/audio cần đặc tả và đo riêng |
| Sentence packing | Tăng EOT points và giảm phần anchors cuối đoạn bị bỏ | Chất lượng chuyển từ synthetic joins sang source turn tự nhiên chưa được giải quyết bằng packing |
| Domain adaptation | ASR có thể tăng đồng thời daily/factory giảm | Tỷ lệ nguồn và mục tiêu tối ưu cho workload thực cần bằng chứng riêng |
| Batch/lane tail | Effective16 có coverage và gradient normalization đúng, thay đổi tổng thể nhỏ | Trial một epoch chưa định lượng phương sai nhiều seed |
| Reset context | Source-window B giảm F1 tổng thể trong A/B đã chạy | Kết quả giới hạn ở policy, backbone, budget và dữ liệu của trial |
| Chất lượng nhãn | Nhãn lấy dấu câu, có unknown và nguồn có noise | Human/audio gold và nhãn meaningful split cho dịch còn thiếu |
| Độ đại diện DEV ASR | Monitor chỉ có 7 turn mỗi tiếng | Nhiều điểm từ trong cùng clip tạo tương quan, cần đọc CI theo family |
| Khả năng học | Fit 1.000 TRAIN input cố định đạt 100% | Tổng quát hóa và mục tiêu ngữ nghĩa cần phép đo riêng |
| Trùng nguồn | Whole-family/turn/shingle exclusions có audit | Paraphrase, WordPiece và toàn bộ lineage cũ còn giới hạn audit |
| Model nhỏ | mALBERT đạt 11.941.122 tham số | Mục tiêu chất lượng tuyệt đối và INT4 deployment còn chưa đạt |
| Hạ tầng hiện tại | Host reboot, HDD chưa mount, main checkpoint SHA khớp | Xác nhận lại mount/rw/reserve và identity trước mọi lần tiếp tục |

Các mức tỷ lệ pack 15%/5%, cửa sổ conflict tám từ, subset ASR một epoch, cửa sổ context B 256 từ, grid calibration và ngưỡng thăm dò +3 điểm F1 là lựa chọn triển khai của các protocol đã lưu. Mỗi mục được ghi lại để người đọc phân biệt yêu cầu user với quyết định thiết kế.

<a id="checkpoint"></a>

## 18. Checkpoint, nén và dữ liệu còn ở đâu

```text
C:/Users/ASUS/Documents/turn_detect/
  README.md                         hồ sơ trung tâm này
  turn_v2/                          implementation các nhánh
  tools/                            collector, verifier và công cụ báo cáo
  outputs/                          report, manifest và review nhỏ trên Windows

/home/edabk/hoangbpm/turn_detect/
  outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7/
    runs/finetune/latest_resume.pt   full checkpoint v7 đã tạm dừng
    runs/finetune/                   periodic DEV và các checkpoint liên quan
    frozen/                         parent pin và bằng chứng chuyển nhánh
    tokenizer/                      tokenizer được đóng băng
    pretrained/                     config và pretrained assets
  outputs/context_window_ab_20261108_v1/
    predicted_reset/                arm A, weights/full state/DEV traces
    source_window/                  arm B, weights/full state/DEV traces
    report.json                     kết quả so sánh
    locked_protocol.json            protocol trước fit
    completion_verification.json    scope audit
    paired_family_bootstrap.json    khoảng bất định theo family
    paired_dev_cases.json            16 ca ghép cặp
  data/checked_dialogues_vi_en_20261103_v1/raw/
                                    raw uploaded sources
```

Full checkpoint gồm model, optimizer, RNG Torch/CUDA/NumPy/Python/sampler, order/cursor, causal lanes và counters. Effective16 còn lưu gradients, sample count, weight sum và pending queue. Thay đổi boot hoặc source fingerprint đòi hỏi kiểm tra checkpoint và metadata trước resume.

### 18.1. Cleanup đã thực hiện theo yêu cầu

Đã xóa **77 checkpoint/export files trên SSH** thuộc các nhánh bị loại và **16 files trên Windows**, giải phóng thực đo **3,2198 GiB SSH** và **1,3126 GiB C** tại thời điểm cleanup. Mỗi file có xác minh path, size/mtime/inode theo nền tảng, SHA và kiểm tra đã vắng mặt sau xóa.

Các báo cáo, source, traces và raw được giữ theo receipt. Những trọng số bị xóa có tombstone `CHECKPOINTS_REMOVED_BY_USER`. Các proof trước cleanup là bằng chứng lịch sử của lần kiểm đó. Muốn chạy lại phép kiểm cần trọng số đã xóa phải xét khả năng phục hồi hoặc mở một thí nghiệm mới được cho phép. [Biên bản cleanup](outputs/failed_checkpoint_cleanup_20261108/deletion_receipt.json)

### 18.2. Điều kiện package

Lộ trình lưu trong campaign gồm FP32 CPU DEV gate, các control/INT8/QAT được phép sau gate, parity/padding check, benchmark fresh process với 2 CPU threads/batch1, đóng gói và ZIP SHA readback. Candidate cuối của model dưới 20M và packed INT4 cần đủ các receipt tương ứng.

<a id="chi-muc"></a>

## 19. Chỉ mục toàn bộ thư mục thí nghiệm

Các trạng thái trong phụ lục lấy từ file tổng hợp hoặc ledger có sẵn. Thư mục cũ có thể giữ `status.json` ở bước cuối từng chạy. Hiện trạng worker mới nhất của project nằm ở mục 1 và snapshot SSH. Các thư mục Colab/Drive là hồ sơ lịch sử của nhánh đã được user bỏ, trạng thái physical worker Colab cũ từng chưa xác minh được.

Folder TinyGRU còn một ledger phase mang tên control transformer INT8. `final_report.json` của TinyGRU ghi `accepted=false`, còn transformer INT8 được chấp nhận có folder và final report riêng. Bảng nén ở mục 14 lấy quyết định từ đúng final report của từng nhóm.

| Thư mục / tài liệu gốc | Trạng thái trong hồ sơ | Kết quả khai báo |
|---|---|---|
| [asr_effective_batch16_20261107_v1](outputs/asr_effective_batch16_20261107_v1/README.md) | COMPLETE | Hoàn tất theo ledger |
| [asr_only_adaptation_20261107_v1](outputs/asr_only_adaptation_20261107_v1/README.md) | BOUNDED_TRIAL_COMPLETE | Hoàn tất theo ledger |
| [auto_rl_v11_20261025](outputs/auto_rl_v11_20261025/report.json) | JUDGE_PROBE_FAILED_NO_POLICY_UPDATES | quality_target_certified=false |
| [auto_rl_v12_20261025](outputs/auto_rl_v12_20261025/report.json) | BOUNDED_AUTOMATIC_CRAWL_RL_COMPLETED_TARGET_UNMET | quality_target_certified=false |
| [colab_v2_smoke](outputs/colab_v2_smoke) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |
| [compression_20261024](outputs/compression_20261024) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |
| [context_window_ab_20261108_v1](outputs/context_window_ab_20261108_v1/README.md) | COMPLETE | adopted=false |
| [continuous_v13_20261026](outputs/continuous_v13_20261026/summary.json) | SUPERSEDED_BY_TEACHER_FIRST_USER_REQUEST | quality_target_certified=false |
| [crawl_predict_20261024_colab](outputs/crawl_predict_20261024_colab/report.json) | SMOKE_INFERENCE_ONLY | Xem hồ sơ của folder |
| [crawl_predict_20261024_legacy](outputs/crawl_predict_20261024_legacy/report.json) | INFERENCE_COMPLETED | Xem hồ sơ của folder |
| [data_role_audit_20261104](outputs/data_role_audit_20261104) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |
| [demo](outputs/demo) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |
| [distillation_20261024](outputs/distillation_20261024) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |
| [experiment_overview_20261109](outputs/experiment_overview_20261109) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |
| [factory_dialogue_campaign_v6](outputs/factory_dialogue_campaign_v6/README.md) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |
| [factory_dialogue_pilot_20261030](outputs/factory_dialogue_pilot_20261030/README.md) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |
| [factory_spoken_campaign_v10](outputs/factory_spoken_campaign_v10/README.md) | DISCARDED_BY_USER | Xem hồ sơ của folder |
| [factory_spoken_campaign_v9](outputs/factory_spoken_campaign_v9/README.md) | STOPPED_BY_REQUEST_SUPERSEDED_BY_BILINGUAL_V10 | Xem hồ sơ của folder |
| [failed_checkpoint_cleanup_20261108](outputs/failed_checkpoint_cleanup_20261108/README.md) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |
| [malbert_calibration_audit_20261107_v1](outputs/malbert_calibration_audit_20261107_v1/README.md) | COMPLETE | Hoàn tất theo ledger |
| [malbert_eval_zip_20261104_v1](outputs/malbert_eval_zip_20261104_v1/README.md) | COMPLETE | Hoàn tất theo ledger |
| [malbert_eval_zip_20261104_v2](outputs/malbert_eval_zip_20261104_v2/README.md) | COMPLETE | Hoàn tất theo ledger |
| [malbert_learning_diagnostic_20261106_v1](outputs/malbert_learning_diagnostic_20261106_v1/README.md) | COMPLETE | Hoàn tất theo ledger |
| [malbert_uploaded_test_20261105_v1](outputs/malbert_uploaded_test_20261105_v1/README.md) | COMPLETE | Hoàn tất theo ledger |
| [monolingual_20261025](outputs/monolingual_20261025/report.json) | THREE_SEPARATE_MODELS_TRAINED_PACKAGED_AND_VERIFIED_QUALITY_TARGET_UNMET | Xem hồ sơ của folder |
| [multilingual_distill_20261030](outputs/multilingual_distill_20261030/README.md) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |
| [multilingual_malbert_pretrained_20261103_v1](outputs/multilingual_malbert_pretrained_20261103_v1/README.md) | STOPPED_BY_REQUEST | Xem hồ sơ của folder |
| [multilingual_malbert_vi_en_20261103_v2](outputs/multilingual_malbert_vi_en_20261103_v2/README.md) | STOPPED_BY_REQUEST_FOR_ADDITIONAL_TRAIN_DATA | Xem hồ sơ của folder |
| [multilingual_malbert_vi_en_checked_20261103_v3](outputs/multilingual_malbert_vi_en_checked_20261103_v3/README.md) | STOPPED_BY_REQUEST | Xem hồ sơ của folder |
| [multilingual_malbert_vi_en_conflict_weighted_20261106_v5](outputs/multilingual_malbert_vi_en_conflict_weighted_20261106_v5/README.md) | STOPPED_BY_REQUEST_FOR_CURRICULUM_V6 | Xem hồ sơ của folder |
| [multilingual_malbert_vi_en_curriculum_20261106_v6](outputs/multilingual_malbert_vi_en_curriculum_20261106_v6/README.md) | STOPPED_BY_REQUEST_FOR_SENTENCE_MIX_V7 | Xem hồ sơ của folder |
| [multilingual_malbert_vi_en_sentence_mix_20261107_v7](outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7/README.md) | PAUSED_BY_USER_FOR_TEST | Xem hồ sơ của folder |
| [multilingual_malbert_vi_en_uploaded_20261104_v4](outputs/multilingual_malbert_vi_en_uploaded_20261104_v4/README.md) | SUPERSEDED_BY_V5 | Xem hồ sơ của folder |
| [multilingual_stream_c75_test_20261102_v1](outputs/multilingual_stream_c75_test_20261102_v1/README.md) | COMPLETE | quality_target_certified=false |
| [multilingual_stream_precision_dev_20261102_v1](outputs/multilingual_stream_precision_dev_20261102_v1/report.json) | COMPLETE | target_met_on_verification=true |
| [multilingual_stream_test_20261102_v1](outputs/multilingual_stream_test_20261102_v1/report.json) | COMPLETE | Report hoàn tất, xem scope/gate |
| [multilingual_stream_test_20261103_v1](outputs/multilingual_stream_test_20261103_v1/README.md) | COMPLETE | Report hoàn tất, xem scope/gate |
| [multilingual_stream_v2_20261102](outputs/multilingual_stream_v2_20261102/README.md) | STOPPED_BY_REQUEST | Xem hồ sơ của folder |
| [multilingual_student_test_20261031_v2](outputs/multilingual_student_test_20261031_v2/report.json) | COMPLETE | quality_target_certified=false |
| [multilingual_student_test_20261101_v1](outputs/multilingual_student_test_20261101_v1/report.json) | COMPLETE | quality_target_certified=false |
| [multilingual_tiny_compression_20261102_v1](outputs/multilingual_tiny_compression_20261102_v1/README.md) | COMPLETE_ACCEPTED_TRANSFORMER_INT8_CONTROL | accepted=false |
| [multilingual_transformer_compression_20261102_v1](outputs/multilingual_transformer_compression_20261102_v1/final_report.json) | COMPLETE | accepted=true |
| [multilingual_under20m_20261103_v1](outputs/multilingual_under20m_20261103_v1/README.md) | COMPLETE_NO_ACCEPTED_UNDER20M_MODEL | accepted=false |
| [multilingual_vi_en_stream_probe_20261102_v1](outputs/multilingual_vi_en_stream_probe_20261102_v1/report.json) | COMPLETE | Report hoàn tất, xem scope/gate |
| [multilingual_vi_en_stream_probe_20261102_v2](outputs/multilingual_vi_en_stream_probe_20261102_v2/README.md) | COMPLETE | Report hoàn tất, xem scope/gate |
| [natural_v5_20261025](outputs/natural_v5_20261025/status.json) | ok | Xem hồ sơ của folder |
| [natural_v6_20261025](outputs/natural_v6_20261025/status.json) | ok | Xem hồ sơ của folder |
| [natural_v7_20261025](outputs/natural_v7_20261025/summary.json) | CURRENT_CAMPAIGNS_AND_FROZEN_FINAL_TEST_COMPLETE_TARGET_UNMET | user_target_achieved=false |
| [natural_v8_20261025](outputs/natural_v8_20261025) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |
| [qat_20261024](outputs/qat_20261024) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |
| [quant4_20261024](outputs/quant4_20261024) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |
| [reward_v10_20261025](outputs/reward_v10_20261025/summary.json) | DEV_ABLATION_COMPLETE | quality_target_certified=false |
| [sentence_mix_end_only_test_20261107_v1](outputs/sentence_mix_end_only_test_20261107_v1/README.md) | COMPLETE | Hoàn tất theo ledger |
| [storage_audit_20261025](outputs/storage_audit_20261025/task_state.json) | AWAITING_USER_APPROVAL_FOR_VERIFIED_DUPLICATE_PARTS | Xem hồ sơ của folder |
| [stream_v4_20261025](outputs/stream_v4_20261025/summary.json) | TRAINED_AND_EVALUATED_REAL_TARGET_NOT_ACHIEVED | user_target_achieved=false |
| [system_review_20261107](outputs/system_review_20261107/README.md) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |
| [teacher_first_v14_20261026](outputs/teacher_first_v14_20261026/summary.json) | ACTIVE_TEACHER_FIRST_COLAB | quality_target_certified=false |
| [test_cpu_single](outputs/test_cpu_single) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |
| [training_trial_20261024](outputs/training_trial_20261024/status.json) | COMPLETED | Xem hồ sơ của folder |
| [translation_segmenter_v15_20261026](outputs/translation_segmenter_v15_20261026/README.md) | RUNNING_COLAB_CRAWL_WAITING_FOR_MINIMUM_DATA | Xem hồ sơ của folder |
| [translation_segmenter_v15b_20261026](outputs/translation_segmenter_v15b_20261026) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |
| [translation_segmenter_v15c_20261026](outputs/translation_segmenter_v15c_20261026/README.md) | STOPPED_BY_REQUEST_DATA_SNAPSHOT_REFRESHED | Xem hồ sơ của folder |
| [translation_segmenter_v15d_20261026](outputs/translation_segmenter_v15d_20261026/README.md) | RECOVERING | Xem hồ sơ của folder |
| [translation_turn_test_20261105_v1](outputs/translation_turn_test_20261105_v1/README.md) | COMPLETE | Hoàn tất theo ledger |
| [trilingual_20261024](outputs/trilingual_20261024) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |
| [trilingual_stream_20261024](outputs/trilingual_stream_20261024) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |
| [turn_failure_literature_20261108](outputs/turn_failure_literature_20261108/README.md) | Chỉ có artifact hoặc báo cáo con | Xem hồ sơ của folder |

<a id="bang-chung"></a>

## 20. Bằng chứng nguồn và giới hạn của bản tổng hợp

Các phần định lượng sử dụng report JSON gắn với cùng checkpoint và protocol trong từng nhóm bảng. Latest periodic DEV được đọc trực tiếp trên SSH. Các report cũ được đọc từ bản report/review đã lưu, không thực hiện lại phép chấm trong quá trình tổng hợp tài liệu. Bản overview nguồn được lưu cạnh snapshot để đối chiếu. Các report cũ có thông báo trainer đang chạy phản ánh thời điểm tạo report đó, còn mục 1 là trạng thái ngày cập nhật tài liệu này.

| Nguồn đã đọc | SHA256 chuẩn hóa LF |
|---|---|
| [outputs/context_window_ab_20261108_v1/report.json](outputs/context_window_ab_20261108_v1/report.json) | `2afae76f7c1972df405f26f90a313e5f1e54f67dcef5939300fdf079e32792f6` |
| [outputs/context_window_ab_20261108_v1/completion_verification.json](outputs/context_window_ab_20261108_v1/completion_verification.json) | `c3d04ae68137ebf8e706425736f981fddc5b97ba27aa6021acdbd9c5d8cf8cff` |
| [outputs/asr_effective_batch16_20261107_v1/report.json](outputs/asr_effective_batch16_20261107_v1/report.json) | `30b7de7efb9ba825ee37b93d5e636a9c2fa700ad1697987f5c6066e3ec78db3e` |
| [outputs/asr_only_adaptation_20261107_v1/dev_report.json](outputs/asr_only_adaptation_20261107_v1/dev_report.json) | `557573fb176aff658eef47b4feeb28305334e6932c17ddecb1e85de3c5b1375c` |
| [outputs/malbert_calibration_audit_20261107_v1/report.json](outputs/malbert_calibration_audit_20261107_v1/report.json) | `4330c69784d51e522ff70fb76a0ddab0d60cd9120a3871f13c1c46094cc784ca` |
| [outputs/malbert_learning_diagnostic_20261106_v1/report.json](outputs/malbert_learning_diagnostic_20261106_v1/report.json) | `e1f7da5819cafc78b1834206ea179cbdf3257673d5165dacd5119654f1261e28` |
| [outputs/sentence_mix_end_only_test_20261107_v1/report.json](outputs/sentence_mix_end_only_test_20261107_v1/report.json) | `97f7a9ab3abdae67591e0c51b1fb836a064457af5da521b91c2981ce86638482` |
| [outputs/translation_turn_test_20261105_v1/report.json](outputs/translation_turn_test_20261105_v1/report.json) | `e05c84065922f312ee85b221d64c3e1b8ea4cc5bb0b1d2e95160c50fe07b47af` |
| [outputs/malbert_uploaded_test_20261105_v1/report.json](outputs/malbert_uploaded_test_20261105_v1/report.json) | `8745e359b9e286742b5da89b01e9dceb3b742e6ecc4a87f566f3855f95a40d96` |
| [outputs/failed_checkpoint_cleanup_20261108/deletion_receipt.json](outputs/failed_checkpoint_cleanup_20261108/deletion_receipt.json) | `52403265a898a9288494e872ebd652e32d16e22d6dfffe492cbb78ce4b10c88e` |
| [outputs/turn_failure_literature_20261108/references.json](outputs/turn_failure_literature_20261108/references.json) | `1860637149692a8647510054a2d893def224ba35b17b2670fa8c45be2d031dfb` |
| [outputs/experiment_overview_20261109/latest_training_devs.json](outputs/experiment_overview_20261109/latest_training_devs.json) | `8293d4872830509f4bbc8ce6ec3c8d5d38d28b0387dd8e09f77d36aff9963adb` |
| [outputs/experiment_overview_20261109/source_snapshot.json](outputs/experiment_overview_20261109/source_snapshot.json) | `dae9b678780a87aafaa7799ee46fb3e1414a1f04042a48b554547793b5e22f40` |

README này tổng hợp sâu chuỗi mALBERT v1–v7, các trial calibration/ASR/context, test và compression liên quan. Chỉ mục thư mục bao phủ các campaign cũ còn trong workspace. Những folder thiếu report quyết định trong bản lưu được đánh dấu theo hồ sơ tìm thấy, và liên kết dẫn đến tài liệu gốc của folder đó.
