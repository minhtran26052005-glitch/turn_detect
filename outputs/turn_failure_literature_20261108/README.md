# Nghiên cứu lỗi detector để dịch — 08/11/2026

Đã hoàn thành rà soát có mục tiêu trên nguồn sơ cấp và đối chiếu thực nghiệm của project. Đây là narrative review, không phải systematic review/PRISMA hoặc meta-analysis. Không chạy thêm TRAIN/TEST trong yêu cầu này. Model v7 vẫn PAUSED; các checkpoint fail được xóa theo yêu cầu user, báo cáo và traces còn giữ.

## Kết luận chính

1. **Có lỗi giới hạn miền calibration trong quy trình cũ.** Cùng weights ASR-only, ngưỡng0,1 cho ASR VI F1=0,74%; mở grid và chọn0,003 trên calibration đưa F1 monitor lên27,47%. Vì thế nhận định “fine-tune ASR làm model mất khả năng hiểu” từ điểm0,74 là quá mạnh. Đây là thay đổi ngưỡng/policy reset, không phải model học thêm.
2. **Calibration không sửa hết chất lượng.** Ngay sau mở grid, precision/recall ASR VI vẫn chỉ30,09/25,28%. Trong pipeline stateful, đổi ngưỡng còn đổi ngữ cảnh sau reset, nên không thể suy ra đường cong mới bằng threshold lại trace cũ.
3. **Đối chứng gom16 chưa cho cải thiện toàn diện.** Cùng333.315điểm, cùng initialmodel/AdamW/RNG/order: ASR VI F1=27,47→31,20; EN53,58→52,65; overallVI30,74→30,88, EN47,87→48,29. Hai ngưỡng được chọn riêng0,003/0,03, nên đây là comparison của calibrated pipelines, không cô lập riêng hiệu ứng weights hay phép chia loss.
4. **Có đánh đổi giữa nguồn dữ liệu.** So với parentv7 trướcASR đãcalibration, ASR tăng22,88/44,69→31,20/52,65, nhưng overallVI34,88→30,88, dailyVI46,20→34,03 và factoryVI48,00→34,25. Điều này phù hợp với domain specialization/regression; chưa chứng minh feature forgetting là nguyên nhân duy nhất.
5. **Ghép1–10câu không tự nó sai.** Có công trình thành công cũng ghép1–10câu; điểm khác là cách gán nhãn từng token, cửa sổ streaming và dữ liệu/tiền huấn luyện. Không thể lấy việc v7 ghép câu làm nguyên nhân đã được chứng minh.
6. **Chưa có căn cứ bắt buộc tăng kích thước trên20M.** Các công trình nhỏ đạt kết quả tốt trong điều kiện riêng, nhưng số liệu của họ không chuyển trực tiếp sang VI/EN, nhãn silver, dữ liệu nhỏ và giới hạn future của project.

## Phạm vi và cách tìm nguồn

RQ: Với detector text-only dưới20M, lấy dấu chấm làm EOT và chỉ1/2futurewords, bằng chứng nào giải thích ít EOT, lệch xác suất sau ASR adaptation và suy giảm ngoài domain; thay đổi nào đáng kiểm tra trước khi train dài?

Tìm ngày08/11/2026: ACL Anthology, ISCA Archive, PMLR, arXiv và trang xuất bản. Nhóm truy vấn: streaming punctuation/right context; dynamic sentence boundary simultaneous translation; domain adaptation punctuation noisy transcripts; fine-tuning feature distortion versus calibration; segmentation-free MT. Chỉ sử dụng nguồn sơ cấp. Đọc các phần phương pháp, bảng thực nghiệm hoặc abstract được ghi rõ dưới đây. Không khẳng định đã đọc mọi phụ lục của từng paper.

Loại bài không đúng chủ đề emergent communication; loại trang ACL2024.tacl-1.58 vì thực tế là summarization; không dùng mirror/blog để xác nhận kết quả. Trang nhà xuất bản2025 về APCR trả403 khi mở toàn văn: chỉ ghi trong search log, không dùng các bảng đó làm bằng chứng định lượng. Không gán số lượng paper sàng lọc giả hoặc khẳng định bao quát toàn bộ literature2026.

## Ma trận paper và thực nghiệm đã xác minh

### P1 — Zhang & Zhang, AutoSimTrans2020

[Dynamic Sentence Boundary Detection for Simultaneous Translation](https://aclanthology.org/2020.autosimtrans-1.1.pdf), bảng2/3/5, §3. M=1:F-score0,74/BLEU22,01; M=2:0,77/22,23. Train WMT thay IWSLT: F-score0,74→0,68 ởM=1. Method dự đoán nhiều vị trí từprefix và voting. **Giới hạn:** ERNIE, En-De, quy tắc nhãn gộp? ! ; thành period khác project; không chứng minh đạt cùng điểm với11,94MVIEN. Liên quan trực tiếp tới segmentation cho MT.

### P2 — Poláček et al., Interspeech2023

[Online Punctuation Restoration using ELECTRA Model for streaming ASR Systems](https://www.isca-archive.org/interspeech_2023/polacek23_interspeech.pdf), §3.3–3.5/bảng1/abstract. ELECTRA-Small13,6M, tokenhead, ghép1–10câu; online delay3words; averageF1 trên hai tập ASR Czech71,2/69,4. **Giới hạn:** gộp ba dấu câu, không binaryEOT; fine-tune335Mtokens, lớn hơn dữ liệu project nhiều. Weight period2 trong paper áp vào xác suất lúc inference, không phải bằng chứng CEweight2 là tối ưu.

### P3 — Kim et al., Interspeech2023

[Improved Training for End-to-End Streaming ASR Model with Punctuation](https://www.isca-archive.org/interspeech_2023/kim23_interspeech.pdf), bảng1/3. PeriodF1 dài35,8→46,6; bỏconcat còn41,0. Bỏchunkloss làm WER15,8→21,3 dù periodF1 tăng52,9. **Giới hạn:** acoustic CTC18M, MuST-CEnglish, future theo giây, không text-only. Chứng minh concat cần thiết kế loss/evaluation cùng hệ thống, không bảo đảm ghéptext là đủ.

### P4 — Courtland et al., IWSLT2020

[Efficient Automatic Punctuation Restoration Using Bidirectional Transformers with Robust Inference](https://aclanthology.org/2020.iwslt-1.33.pdf), abstract/bảng1. Sequence prediction và ghép dự đoán từ cửa sổ chồng lấp; RoBERTabase combinedF1=83,9. **Giới hạn:** context bidirectional/offline, backbone lớn; không thể dùng điểm này làm tiêu chuẩn1–2futurewords hoặc khẳng định bản dưới20M sẽ tương đương.

### P5 — Fu et al., W-NUT2021

[Improving Punctuation Restoration for Speech Transcripts via External Data](https://aclanthology.org/2021.wnut-1.19.pdf), bảng3/4/5. In-domainF168,48; sampled-external61,68; external→in-domain69,60; periodF175,14. Six-layer68,48 so12-layer68,36 trong setting tương ứng. **Giới hạn:** English customer-support, nhãn có sửa bởi annotators, không giao thức future2 của project. Bằng chứng chọn nguồn phù hợp hữu ích hơn cứ trộn thêm text chung.

### P6 — Guo et al., ICML2017

[On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), §4–5. Temperature scaling trênvalidation cải thiện calibration; argmax class không đổi. **Giới hạn:** calibration không chứng minh hiểu ngữ nghĩa hoặc sửa feature ranking. Với binary scores trên input cố định, phép biến đổi đơn điệu không cải thiện thứ tựPR/AUC; đây là hệ quả toán học, không phải kết quả MT của paper.

### P7 — Kumar et al., ICLR2022

[Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://arxiv.org/pdf/2202.10054), abstract/§1. Trên10shiftbenchmarks ảnh, fullFT hơn linearprobe khoảng2ppID nhưng kém7ppOOD; LP→FT giảm vấn đề trong thực nghiệm. **Giới hạn:** vision và mô hình lý thuyết đơn giản, không xác nhận cơ chế trong mALBERT/EOT; chỉ hỗ trợ giả thuyết và thiết kế probe.

### P8 — Mai et al.,2024

[Fine-Tuning is Fine, if Calibrated](https://arxiv.org/html/2409.16223v3), §4–6. Trong subset-class image transfer, chất lượng feature có thể được giữ/tăng, trong khi biased logits làm điểm toànmodel giảm; calibration khôi phục một phần. **Giới hạn:** project vẫn có cả NEOT/EOT khi train, không phải absent-class setup; không được coi đây là lời giải trực tiếp. Phản chứng hữu ích cho kết luận tự động “F1 tụt = mất kiến thức”.

### P9 — Iranzo-Sánchez et al., TACL2024

[Segmentation-Free Streaming Machine Translation](https://aclanthology.org/2024.tacl-1.61.pdf), abstract/§1–3. MT sinh bản dịch và delimiter, lùi quyết định segmentation tới sau translation; báo trade-off quality/latency tốt hơn các segmenter cascade so sánh. **Giới hạn:** phải đổi/huấn luyện translator và memorypolicy, không cắm trực tiếp vào Qwen checkpoint user hoặc bảo đảm phù hợp hạn20M của detector.

### P10 — Gururangan et al., ACL2020

[Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks](https://aclanthology.org/2020.acl-main.740/), abstract và bảng19 truy cập qua bản xuất bản. DAPT/TAPT được khảo sát trên4domains/8classificationtasks, có lợi trong các setting báo cáo. **Giới hạn:** MLM adaptivepretraining khác supervisedEOTfine-tune; không chứng minh ASR-onlyCE hay tiếp tục từ optimizer cũ sẽ cải thiện. Đây là hướng thứ cấp, chưa nên triển khai mặc định từ corpus ASR nhỏ.

## Điều gì thực sự suy ra cho project?

**Chắc chắn từ thực nghiệm local:** optimizer có cập nhật, sourcecoverage đủ, phép chấm tái lập trong phạm vi audit; gridcũ bỏ qua vùng ngưỡng hữu ích; gom16 không tạo bước nhảy chất lượng tổng thể trong một lượt; adaptation giúp ASR nhưng giảm một số nguồn khác. Những điều này không cần paper để xác nhận.

**Giả thuyết có căn cứ nhưng chưa được xác nhận:** cách cắt context theo predictedEOT tạo vòng phản hồi: một cut sai thay đổi mọi input phía sau; cùng dữ liệu nhãn nhưng train dùng dropout/BF16 và evaldeterministic có thể gặp các reset histories khác nhau. Sampling prefix/window không gắn với quyết định cắt là ứng viên đáng kiểm tra. Các paper streaming dùng thiết kế context/decoding rõ, không chỉ tăngEOTweight.

**Điều literature không cho phép kết luận:** chỉ cần thay mALBERT thànhELECTRA sẽ đạt70%; chỉ cần focal loss; phải cân bằng50/50; model135M chắc chắn tốt; mọi dấu phẩy là điểm cắt sai nghĩa; ghép ngẫu nhiên chắc chắn gây lỗi; feature forgetting đã được chứng minh. Paper P7 vàP8 có kết quả khác do thiết lập khác nhau, không thể chọn một paper thuận giả thuyết rồi coi là nguyên nhân.

### Bảng bằng chứng local đã dùng — không chạy model mới

| Model/policy | Ngưỡng | F1ASRVI | F1ASREN | F1overallVI | F1overallEN |
|---|---:|---:|---:|---:|---:|
| Parentv7, calibration mở rộng |0,03|22,88|44,69|34,88|48,24|
| ASR-only, gridcũ |0,1|0,74|42,13|5,13|41,10|
| Cùng ASR-onlyweights, calibration mở rộng |0,003|27,47|53,58|30,74|47,87|
| ASReffective16, calibration mở rộng |0,03|31,20|52,65|30,88|48,29|

Tất cả là DEVpunctuation silver. Mỗi lựa chọn dựa calibration trước monitor; hai weights có thể chọn ngưỡng khác. Không phải human/audio gold, không đo dịch hay latencythực. AUROC trước/sauASR trong report cũ cũng dùng resetcontexts khác, nên không phải feature-rankingprobe trên inputmatched.

Local evidence: `../asr_only_adaptation_20261107_v1/dev_report.json`, `../malbert_calibration_audit_20261107_v1/report.json`, `../asr_effective_batch16_20261107_v1/report.json`; SHA và scope trong `local_evidence.json`. Trọng số hai nhánh fail hiện đã xóa theo yêu cầu, báo cáo/traces vẫn còn; không được chạy lại verifier cầnweights rồi coi missingfile là lỗi campaign mới.

## Đề xuất thực nghiệm tiếp theo — thiết kế, chưa chạy

**Ưu tiên1: kiểm tra một thay đổi về context, không đồng thời đổi backbone/loss/data.** A giữ context từ predictedEOT như hiện tại. B dùng leftwindow theo vị trí nguồn, độc lập với quyết định cắt; decoder cắt phần dịch nhưng encoder vẫn nhìn lịch sử nguồn đã thấy. Cùng mALBERT11,94M, cùng initialbackbone/head/AdamW/RNG/order, cùng eligiblepoints, classweights1/2/LR/batch16. Giữ1/2futurewords đúng cùngturn như user yêu cầu; leftwindow/tokenbudget định trước. Đây là suy luận từ evidence, không phải bảo đảm paperP2 đã kiểm đúng mALBERT này.

Trước fit, compile và kiểm mọi input prefix: không punctuation/languageID/target/rest hoặc future vượt giới hạn, không goldreset. Ranh giới cuối thiếu future vẫn được thống kê riêng, không tự thêmEOS để biến thành positive dễ. Validation/monitor giữ family riêng và cùng population, threshold chọn calibrationonly. Có thể mở rộng nguồn DEVchưachấm nếu đủ family, chọn trước modelscoring; không lấyTRAIN đãhọc làmtest mới.

Ngân sách đề xuất một lượt cùng snapshot/mỗiarm; số points/tokens/update được báo riêng. Gate thực nghiệm phải khóa trước: cải thiện tiếng yếu hơn đủ lớn, không tăng FP quá mức và không giảm nguồn khác; không lấy gate tương đối teacher yếu làm chứng nhận sản phẩm. Ví dụ +3ppF1 kèm kiểmP/R mỗi nguồn là một ngưỡng thiết kế có thể cân nhắc, **chưa thay gate đang lưu và không phải con số paper chứng minh tối ưu**. Báo paired bootstrap theo video/hội thoại; ít family thì kết luận chưa chắc, không tự train thêm vô hạn.

**Ưu tiên2 chỉ khi contexttest cho thấy còn cần đổi formulation:** tái hiện hướng multi-position/prefix hoặc tokenhead củaP1/P2 với backbone dưới20M. Nếu giữ đúngfuture1/2 thì offsetcandidate chỉ −1/−2+NONE; paperP1 còn có classcurrent0 nên bản giữ contract này là biến thể, không tuyên bố exactreplication. Vanilla bidirectionaltagger trên cả đoạn có thể lộ tương lai; maskloss không chặn encoder nhìn xa. Cần audit dependency xuyên mọi layer/chunk, không chỉ gánlocalmaskr=2 rồi mặc định tổngfuture=2.

**Ưu tiên3 cho mục tiêu dịch:** đánh giá trên dòng nhiều câu liên tục, đo meaningfulsplit/wordconservation/độ trễ và chất lượng dịch; so với whole-message+forcedflush. Giữ kết thúc dòng/timeout/VADflush riêng với internalEOT. Bộ310message hiện chỉ có18multis phù hợp định nghĩa period sau kiểm nguồn; không đủ để chứng nhận online endpoint. SegFreeP9 là phương án kiến trúc dài hạn, cần thay translatortraining nên không coi là bản vá nhanh.

Không khởi động các bước này trong yêu cầu research hiện tại. Các thử nghiệm cũ COMPLETE/reported không rerun; dataset/threshold/default không đổi. Mainv7fullresume vẫn nguyênSHA.

## Cleanup đi kèm

93exactcheckpoint/exportfiles đãxóa theouser:77SSH,16Windows. Giải phóng thực đo3,2198GiBSSH và1,3126GiBC. Caches/raw/reports/traces và checkpoint chính/teachercontrols được giữ. Chi tiết [receipt](../failed_checkpoint_cleanup_20261108/deletion_receipt.json). Không coi model research-only/pendingeval hoặc checkpoint dừng dohandover là fail nếu không có bằng chứng.
