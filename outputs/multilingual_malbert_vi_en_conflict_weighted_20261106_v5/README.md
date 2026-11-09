# Filtered weighted-CE VI/EN training v5

The user requested on 2026-11-06 Bangkok to filter conflicting training labels, change the loss and continue training. The new branch resumes the complete v4 state from global2,355,080 / branch890,000. Parent205061 exited STOPPED_WITH_CHECKPOINT. Active module turn_v2.multilingual_malbert_vi_en_conflict_weighted_train_v5, output multilingual_malbert_vi_en_conflict_weighted_20261106_v5. Fresh identity at verification was754535/start_ticks12833606/boot65117834-0b91-465e-a099-805d6100f317. Verify the current launcher/proc on each check.

## TRAIN and filtering

All198,509 source rows/17,301 streams/3,373,257 source words remain. The four uploaded origins are daily, factory/gen500, asr/run2b and factory/gen4000. Eligibility and source-family exclusions carry forward. Fullstop defines EOT. TRAIN-only conflict detection uses an exact normalized suffix of up to8left words plus1/2same-turn future words. For each key observed with opposing labels, both classes at every matching source gap are excluded from loss. This conservative window was chosen for the implementation and can exclude gaps that longer history could resolve. It does not exhaust arbitrary dynamic reset windows, tokenization-only collisions or paraphrases. Earlier exposure remains in the inherited model.

697 conflicting keys were identified.6,618 scored gaps are masked, comprising4,797 NEOT and1,821 EOT. Remaining scored points per full filtered epoch6,163,255. Source text and punctuation stay available for causal context. Every original6,169,873gap, both-label membership and surviving-key consistency passed independent verification. DEV/calibration/monitor/diagnostics and frozen teacher files have identical bytes to v4. No DEV/TEST labels were used to construct the mask.

## Loss and continuation

Weighted CE is calculated with weights NEOT1/EOT2 and weighted-mean reduction. Manual loss-value and relative-gradient fixtures passed. AdamW27states and inherited LR4e-6 continue, along with all model tensors, torch/CUDA/NumPy/Python/samplerRNG, source order/cursor and16conversation lanes. Initial child state was independently compared field-by-field to the parent. New branch counters and best-model selection are versioned for the loss change. The unfinished parent-epoch tail is completed first under the filter. It has4,520,132eligible remaining points. Its receipt is cutover_partial_tail_coverage.json. Later epoch_*_coverage receipts each prove6,163,255points per full filtered epoch. Minimum6full filtered epochs are scheduled, followed by continued training if retention remains below the existing gate. The partial tail is separate from that count.

DEV continues on the same119calibration and394monitor streams. The existing0.1..0.9candidate policy and per-language retention thresholds continue. First post-weighted-training periodic DEV is scheduled at branch10,000. The latest observed branch7,650 had no completed new DEV. Future improvements must be established on DEV. CE-weighted and previous CE results include the filter and continued training differences.

## Verified execution

Controller conflict_weighted_cutover_v5_v2 completed and exited after training verification. The first controller had a syntax error before any handover and was replaced with a compiled v2. Its log remains retained. A prelaunch guard was clarified to accept the actual v4 STOPPED_WITH_CHECKPOINT stage together with explicit STOP_REQUESTED and process absence. Data masks and model state were preserved through that metadata-only clarification. These completed setup issues are retained for audit.

The new checkpoint at branch500/global2,355,580 proved7,849new scored exposures and changed finite weights. Independent audit of1,583actual inputs at branch1500/global2,356,580 passed all labels, masks, encoded sequences, same-turn futures and predicted-only resets.27optimizer entries and all resume fields were verified. Parent fullSHAf020f5b0cecf5576ca594a92e300a4ea94b2810ec7973c463cf625d09e5d8b8c. Initial childSHA19ed9f7c3a638a9e10bdc79f7b1f1d0c272719bd9ec0d5a4e9b8e00210660c71. ManifestSHAb13b27a95e96f69c9b15f9c80a5d64d810f284db89acdb6e5a2821c27bd2b842.

Fresh handover check global2,362,730/branch7,650 had increasing CPUticks, sole trainer754535/GPU7154MiB/no active error, SSD43.54GiB/HDD23.01GiB rw/Windows19.36GiB free. These values require fresh verification. Model11,941,122 parameters, FP32 weights/BF16 training. Compression remains gated on CPU FP32 DEV and validated exported artifacts. Keep storage guards and all historical raw/checkpoints/optimizer/RNG/rejected candidates. Checkpoints are retained on SSH. Windows receives bounded small reports only. Drive is discarded from the campaign. Automation continues every300minutes.

Collector tools/collect_conflict_weighted_v5.py. Independent evidence is in independent_preparation_verification.json, migration_receipt.json, independent_weighted_training_verification.json, independent_actual_weighted_training_audit.json and handover_liveness_verification.json. Completed ZIP/dialogue/translation tests stay reported. New tests require a new request.


## Direct check, 2026-11-06 08:25 Bangkok

Same sole worker754535/start_ticks12833606/boot65117834 matched actual proc identity and advancing CPU ticks. Branch215250->215650/global2570730, weightedCE NEOT1/EOT2, GPU5020MiB, no active error. A brief SSH connection failure recovered on retry. SSD43.43GiB/HDD23.01GiB rw/Windows20.68GiB remain above reserves. These are snapshots.

Only latest complete post-weighted-training DEV branch210000/global2565080 is reported at threshold0.1. VI F1 26.75%,precision50.00%,recall18.26%,479misses/586EOT. EN F1 46.49%,precision55.60%,recall39.94%,418misses/696EOT. ReportSHA708e3e9dc37d62d34d81543d62074675acca6ded8547247b2a7e66d6088b5707 matched exact remote per-step record and LF-normalized Windows JSON. Same394stream punctuation-silver DEV. Retention false. Earlier10000..200000 reports skipped. Resume still completes the inherited unfinished tail, with no filtered full-epoch coverage receipt yet. Keep training and the existing data/loss/optimizer/threshold policy. No new TEST or compression acceptance.


## Chẩn đoán học 06/11 đã hoàn thành

Phép thử 1.000 mẫu cố định đạt 100% sau 200 bước. Hai nhánh LR 4e-6 và 1e-5 chạy 3.000 cập nhật từ cùng full checkpoint 2585080. Đã kiểm chứng toàn bộ 105.810 inference monitor và 27 trạng thái AdamW. Kết quả và protocol ở ../malbert_learning_diagnostic_20261106_v1/README.md. Trainer v5 tiếp tục hoạt động.


## Chuyển sang curriculum v6 ngày 06/11

V5 đã checkpoint và thoát theo yêu cầu mở dữ liệu dần. Full checkpoint cuối global 2624058 được giữ trong nhánh v6. Nhánh kế tiếp outputs/multilingual_malbert_vi_en_curriculum_20261106_v6 đã được kiểm chứng train thật.
