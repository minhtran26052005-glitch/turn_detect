# Sentence mix Việt Anh v7

Đã chuyển sang TRAIN đoạn tổng hợp 1–10 câu và kiểm chứng cập nhật weights thực. Model mALBERT 11.941.122 tham số, LR 4e-6, AdamW 27 entries, weighted CE NEOT 1 / EOT 2, batch 16. Warm-start tại global 3231046. Model/optimizer/Torch/CUDA/NumPy/Python RNG được đối chiếu chính xác. Thứ tự nguồn, sampler, cursor và lane mới được version hóa theo bộ data mới, toàn bộ trạng thái cũ giữ trong frozen/parent_full_resume.pt.

## Phân bố do triển khai chọn

Tỷ lệ theo số đoạn, riêng mỗi ngôn ngữ. Các nhóm 1, 2, 3, 4, 5 câu mỗi nhóm 15%. Các nhóm 6, 7, 8, 9, 10 câu mỗi nhóm 5%, cộng 25%. Tổng 49.260 đoạn, gồm 23.340 Việt và 25.920 Anh. Mỗi nhóm 1–5 có 7.389 đoạn, mỗi nhóm 6–10 có 2.463 đoạn khi cộng hai tiếng.

## Tạo đoạn và nguồn nhãn

Tách câu theo fullstop EOT đã có trong nguồn, ghép cùng tiếng/cùng nguồn. Đây là đoạn TRAIN tổng hợp, các câu có thể khác lượt hoặc người nói gốc. Unit references lưu stream/segment/start/end của từng câu nguồn. Unknown boundaries được hợp nhất bảo thủ khi câu trùng. Có 208.906 câu hoàn chỉnh duy nhất sau lọc nguồn held và exact dedup. Toàn bộ các câu đủ điều kiện được dùng. Thêm 449 lượt dùng lại câu để lấp block cân bằng cuối, khoảng 0,21%. Mọi đoạn ghép có hash văn bản duy nhất.

Chỉ dùng phần đã khép bằng dấu chấm có sẵn. Có 61.446 phần đuôi sau dấu chấm cuối hoặc kết thúc bằng dấu khác, tổng 696.846 từ, được giữ trong dataset/raw parent. Những phần đuôi này không vào pool ghép hiện tại. Có 112 sentence occurrences trùng nguồn held bị loại và 23.041 occurrences exact trùng được gom vào provenance.

Point nối giữ EOT từ dấu chấm gốc. TRAIN cho phép future 1/2 từ qua điểm nối câu nguồn trong cùng đoạn tổng hợp. Context lowercase không dấu câu và reset theo dự đoán. Điểm cuối đoạn mới vẫn chịu quy tắc thiếu future thì bỏ. Tỷ lệ fullstop anchors cuối đoạn từ khoảng 45,87% nguồn cũ còn 23,53% trong bộ ghép. Dữ liệu là punctuation silver.

## Audit và chống trùng

Kiểm source words, dấu chấm, unknown masks và mọi đoạn tái ghép. Loại exact held sentence/turn từ 8 từ và kiểm 32-word shingles trên nguồn DEV/TEST. Chưa exhaust chuỗi chung ngắn hơn 8 từ, WordPiece-only hay paraphrase. Mọi điểm nối tổng hợp được ghi rõ. Không coi các điểm nối này là bằng chứng same-speaker tự nhiên.

Tính lại conflict rule max8leftwords + 1/2future trên TRAIN mới. Có 75 conflicting keys, mask toàn bộ 276 điểm hai lớp. Thống kê mask hiện hành ở sentence_pack_audit.json và independent_pack_verification.json. Field conflict_filter kế thừa trong manifest cơ sở là metadata lịch sử của parent, được giải nghĩa trong manifest_clarification.json. File TRAIN và mọi count hiện hành được hash/kiểm độc lập.

Có 5.040.023 điểm hợp lệ mỗi full epoch, gồm 319.993 EOT points, khoảng 6,35%. Tỷ lệ 15%/5% là tỷ lệ đoạn, các tỷ lệ theo token/boundary thay đổi theo độ dài câu.

## Train và theo dõi

Curriculum cũ đã đến full stage. V7 bắt đầu trực tiếp stage 3 với toàn bộ đoạn cân bằng, epoch mới từ 0, ít nhất 6 full epochs 5.040.023 điểm rồi quality gate hiện có. DEV calibration/monitor là bộ cố định từ v6, dùng future trong actual source turn. DEV đầu tiên sau 10.000 cập nhật v7. TEST giữ ngoài selection. Chất lượng trên DEV sẽ cho biết mức chuyển giao từ đoạn ghép sang luồng nguồn thật.

Fixture CPU chạy chính hàm fit qua stage/partial lane và resume, model/optimizer/RNG/order/stage counters khớp bitwise với chạy liền. Audit độc lập toàn bộ units/packs/conflict masks và count theo k/language/label khớp manifest. Checkpoint branch 500 / global 3231546 đã học 7780 điểm, weights hữu hạn và thay đổi. Audit 1567 input actual qua 100 updates kiểm đúng label/mask/future trong đoạn tổng hợp và predicted resets. Parent v6 đã checkpoint và thoát. Full checkpoints/provenance/weights/raw giữ SSH. Windows chỉ report/review/source nhỏ.

Manifest SHA256 `b5732ca1acaa317fff66981beda0cff7502258b8d747f6af0ed9a67b5c5e2f7f`

Plan SHA256 `fe83ccd9e57f798c2822f97e16cf2630adfb860ee03f5c1bc684b776d2331d3d`


## ETA và DEV trực tiếp ngày 07/11 lúc 09 giờ Việt Nam

Bộ mới có 5.040.023 điểm hợp lệ mỗi epoch, ít hơn 18,22% so với 6.163.255 trước đó. Đã đo 713 block 50 cập nhật, khoảng 283,47 giây train/10.000 cập nhật. Ba lượt DEV hết 572,49 / 541,32 / 570,17 giây. Kể cả DEV mỗi 10.000 bước, ước tính 7–9 giờ/epoch, epoch đầu khoảng 16–18 giờ ngày 07/11 và 6 epoch tối thiểu khoảng 09–10/11 nếu máy chạy liên tục. Package INT4 phụ thuộc gate và các lượt kiểm chứng/QAT sau đó.

Latest COMPLETE DEV 30.000/global 3.261.046, threshold 0,1. Việt F1 32,87%, precision 38,84%, recall 28,50%. Anh F1 49,39%, precision 46,90%, recall 52,16%. Retention gate chưa đạt. Report SHA256 77ab31529e43337b5718be2f40d2412b64d70470217b24d72033479fccf7b8e2. Các dự báo thời gian là ngoại suy tốc độ đo ban đầu, có thể đổi khi độ dài batch/tải GPU đổi.


## DEV ngày 07/11 lúc 13 giờ Việt Nam

Latest COMPLETE branch 180.000/global 3.411.046, threshold 0,1. Việt F1 30,45%, precision 39,14%, recall 24,91%. Anh F1 48,92%, precision 51,01%, recall 46,98%. Retention gate chưa đạt. SHA256 e3e88c3d0fbf41461a161d3371c7181dbf8c3038988e06cf0c7bf7813ff8d1a2. Báo cáo remote per-step khớp bản latest và local chuẩn hóa LF.

Worker 1121366/ticks 24252420/boot651 đúng một process, CPU ticks tăng. Đang hiệu chuẩn DEV 190.000, các điểm trong bước hiệu chuẩn chưa là báo cáo monitor hoàn tất. Full epoch mới 0, stage_seen 2.953.618/5.040.023 điểm lượt đầu, khoảng 58,6%. Source/code/plan/TRAIN SHA đã đối chiếu. SSD 40,01 GiB, HDD 23,01 GiB rw, Windows 16.29 GiB. Không lỗi đang hoạt động. Trainer tiếp tục theo lịch.


## DEV và epoch ngày 07/11 lúc 18 giờ 06 Việt Nam

Đã kiểm chứng epoch đầu tiên của TRAIN ghép, đủ 5.040.023/5.040.023 điểm hợp lệ. Coverage SHA256 72123a7e381098a32c33da39c6aae28260a136b0d3fb6c9e24981b94695ae6eb. Checkpoint branch 390.000/global 3.621.046 có epoch=1, 27 AdamW entries đúng step, đủ RNG/order/cursor/lanes, 98.520 task duy nhất và weights hữu hạn. Epoch thứ hai đã qua 1.022.173 điểm, khoảng 20,3%.

Latest COMPLETE DEV branch 380.000/global 3.611.046, threshold 0,1. Việt F1 31,21%, precision 34,00%, recall 28,84%. Anh F1 47,82%, precision 45,47%, recall 50,43%. Gate giữ chất lượng tương đối so với frozen teacher trên cùng DEV đạt ở checkpoint này, đã tính lại từng điều kiện F1/precision/recall mỗi tiếng. Đây tiêu chí tương đối trên punctuation silver. Yêu cầu tổng cộng tối thiểu 6 full epochs, hiện hoàn thành 1/6, trước các bước CPU/QAT/package. Report SHA256 7229475422a479d6cc272dcce0621bdcc29f920d702d2e6f609ff4545fc87c6d.

Worker 1121366/ticks 24252420/boot651 đúng một process, CPU ticks tăng. DEV 390.000 đang hiệu chuẩn, chưa hoàn tất. Source/code/plan/TRAIN SHA khớp. SSD 39,94 GiB, HDD 23,01 GiB rw, Windows 17.28 GiB. Trainer tiếp tục theo lịch.


## Direct DEV trend check 07 November

All 48 completed v7 DEV reports through branch 480000 were read without rescoring. Scores fluctuate; later quality is weaker than the early best, rather than declining at every checkpoint. Selected best remains branch 70000 (F1 VI 35.06%, EN 49.94%). Latest complete branch 480000/global 3711046 at threshold 0.1: F1 VI 29.02%, EN 45.92%; precision 43.24/47.06%, recall 21.84/44.83%. Report SHA256 b7fdb91b201810275ee5b8c6ab92bc5fc73caee32333abcc9341b7a6fc6240a4. Trend and process/hash evidence: direct_dev_trend_20261107.json and direct_dev_trend_verification_20261107.json. No matched TRAIN-fit trend proves overfitting; six epochs is a training budget, not evidence of eventual improvement. Keep current authorized campaign active and preserve the selected best.


## Semantic boundary reading review 07 November

Saved branch480000 DEV trace was verified against every scored decision/confusion count and its recorded digest, then 2 FP + 2 FN + 1 TP per source/language were hash-selected (32 errors + 8 controls). Original punctuated text was reconstructed from preserved source display and aligned word-for-word to current monitor. Of 16 FP: 10 plausible translation boundaries, 4 premature semantic splits, 2 unclear-source/borderline. Of 16 FN: 11 clear missed early translation opportunities, 5 unclear-source/borderline. All 8 TP controls plausible. These are subjective assistant reading judgments, not a representative population estimate, human gold, audio review, or measured MT preservation. FP by fullstop is not equivalent to broken translation, and FN can mean delayed output rather than wrong meaning. No training, labels, threshold, gates, or selected model changed. Inspect semantic_boundary_review_20261107.html/json.


## User-requested pause and new end-only TEST 07 November

Trainer actually stopped with a full checkpoint at global 3731173/branch 500127. STOP_REQUESTED retained; do not auto-resume. The model, 27 optimizer states, RNG, order/cursor/lanes are pinned and verified. New job outputs/sentence_mix_end_only_test_20261107_v1 uses this actual paused model with same-checkpoint DEV-selected threshold 0.1, then original TEST and detector-only replay of 310 translation messages. Exclude one-sentence turns from end-only denominators. No translator rerun or MT quality score, no training/label/loss/source changes. Final results in the new job review/report after verification.


## End-only TEST completed while TRAIN remains paused

New TEST sentence_mix_end_only_test_20261107_v1@3731173 is COMPLETE/independentlyverified/reported. After excluding one-span turns, originalTEST has1190/4446turns with no internal detectorcut (26.77percent), VI687/2069/EN503/2377. Detector replay on translationtest inputs: after source-only format review,6/18multi-sentenceperiodinputs end-only (33.33percent); includequestion-first ->6/19 (31.58percent). Qwen translator notrerun. Fullcheckpoint remainsunchanged; optimizer/RNG/order/lanes kept. DesiredstatePAUSED_BY_USER_FOR_TEST/no auto resume. Details/newreview in outputs/sentence_mix_end_only_test_20261107_v1.


## ASR-only trial authorized 07 November

The user requested a separate ASR training trial. Original v7 remains paused and immutable at3731173; new active module turn_v2.asr_only_adaptation_20261107_v1, output outputs/asr_only_adaptation_20261107_v1. One additional full natural-ASR pass (210streams/81video families/169011words/333315scoredpoints) then sameoriginalDEV before/after all domains and fixedbaseline.1. Fullmodel27AdamW/RNG inherited exactly, newversionedASRorder/freshlanes, no base reset or TESTscoring. Source ASR focus and naturalordering both differfromv7syntheticpack input; no single-factorcausalclaim. Newtrial does not restart the paused v7 trainer.


## ASR-only trial completed, original v7 remains paused

outputs/asr_only_adaptation_20261107_v1@3753320 completed one fullASRepoch and allDEV verification. Same0.1threshold: ASRVI F110.43->0.74, EN39.05->42.13; VIrecall6.32->0.37,EN31.14->30.10. Daily/factory declined. Trial not adopted/default unchanged; its weights/optimizer/RNG retained for review. Parentv7 still paused3731173; do not auto-resume it or autoextend the one-epoch experiment. New reportSHA 7ef181e9a10c69a022f646782e3c241b3a67f2dc1d6a4ca19cdeb7dcece9d196.


## USER_FAILED_CHECKPOINT_CLEANUP_AND_LITERATURE_COMPLETE_20261108
Theo yêu cầu user: đã xóa checkpoint/export của các nhánh được xác nhận fail;
receipt ở ../failed_checkpoint_cleanup_20261108/deletion_receipt.json.
Các file trong receipt bị xóa có chủ đích, không phải lỗi mới và không tự phục hồi/retrain.
Main full checkpoint, teacher và dữ liệu/báo cáo/traces được giữ.
Rà soát 10 paper cùng thực nghiệm đã lưu hoàn thành tại
../turn_failure_literature_20261108/README.md và review.html.
Thiết kế A/B context là đề xuất, chưa chạy. Main vẫn PAUSED; research không mở lại TRAIN/TEST.


## CONTEXT_WINDOW_AB_COMPLETE_20261108
Hai nhánh mới đều hoàn tất333315points/20833updates, fullstate/source/DEVtrace/metric/audit đã kiểm chứng. Kết quả và ca DEV ghép cặp: ../context_window_ab_20261108_v1/review.html. Không tự adopt/mở thêm epoch, không TEST/translator/default change. Main vẫn PAUSED3731173/STOP_REQUESTED; fullSHA nguyên.
