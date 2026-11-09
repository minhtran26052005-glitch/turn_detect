# ASR effective16: bounded controlled experiment

Kết luận: gom đủ16 chỉ cho thay đổi nhỏ trong phép thử này, chưa giải quyết chất lượng. ASRVI F127,47→31,20/P30,09→33,77/R25,28→29,00; ASREN F153,58→52,65/P47,83→47,88/R60,90→58,48. VI vẫn FN191/269 và FP153. OverallF1VI30,74→30,88/EN47,87→48,29 gần như không đổi. Hai ngưỡng chọn khác nhau(.003/.03), vì vậy đây là so sánh pipeline đãcalibration chứ không phải weights-only ở cùng một ngưỡng. So với parent chưaASRđãcalibration, ASR tăng nhưng overallVI34,88→30,88, dailyVI46,20→34,03 và factoryVI48,00→34,25: không có cải thiện toàn diện. Không áp model/default hoặc tự train thêm. Không chứng minh batch nhỏ là nguyên nhân duy nhất hoặc ASR specialization nói chung không thể hiệu quả.

Same source/order/initial model+27AdamW+allRNG from3731173. Same333315eligiblepoints. Control22147updates; treatment20833 (20832windows16 + final3). WeightedCE1/2, LR4e-6 unchanged. Effective-window grouping changes update count, dropout grouping and potentially predicted-reset contexts; not normalization-only causal proof. Each arm uses the same locked20-threshold calibration policy; selected thresholds 0.003/0.03. Thresholds may differ, so selected-monitor comparison measures the calibrated training pipeline, not weights alone at a universal fixed threshold. No TEST/default replacement. Silver punctuation DEV, not human/audio gold or translation-quality/latency acceptance. Parentv7 remains paused; no autoextend.

Parent v7 trước ASR, cùng DEV, ngưỡng0.03 chọn theo cùng mục tiêu calibration, dùng lại báo cáo đã lưu không chấm lại: en ASR F1 44.69 / P 40.74 / R 49.48, vi ASR F1 22.88 / P 26.60 / R 20.07. Cần đọc mốc này cùng hai nhánh thử nghiệm; tốt hơn control ASR cũ không tự động có nghĩa tốt hơn parent.

| Nguồn | Tiếng | F1 cũ → mới | Precision cũ → mới | Recall cũ → mới | FP cũ → mới | FN cũ → mới |
|---|---|---|---|---|---|---|
| ALL | vi | 30.74 → 30.88 | 28.70 → 28.07 | 33.11 → 34.30 | 482 → 515 | 392 → 385 |
| ALL | en | 47.87 → 48.29 | 38.01 → 38.75 | 64.66 → 64.08 | 734 → 705 | 246 → 250 |
| asr | en | 53.58 → 52.65 | 47.83 → 47.88 | 60.90 → 58.48 | 192 → 184 | 113 → 120 |
| asr | vi | 27.47 → 31.20 | 30.09 → 33.77 | 25.28 → 29.00 | 158 → 153 | 201 → 191 |
| daily | en | 54.81 → 53.65 | 44.58 → 42.38 | 71.15 → 73.08 | 138 → 155 | 45 → 42 |
| daily | vi | 35.37 → 34.03 | 31.07 → 28.36 | 41.04 → 42.54 | 122 → 144 | 79 → 77 |
| factory | en | 44.56 → 47.54 | 33.67 → 36.89 | 65.84 → 66.83 | 262 → 231 | 69 → 67 |
| factory | vi | 38.12 → 34.25 | 33.89 → 29.95 | 43.57 → 40.00 | 119 → 131 | 79 → 84 |
| numeric/dialogue | en | 27.15 → 26.42 | 17.44 → 17.18 | 61.22 → 57.14 | 142 → 135 | 19 → 21 |
| numeric/dialogue | vi | 14.71 → 14.29 | 10.75 → 10.31 | 23.26 → 23.26 | 83 → 87 | 33 → 33 |

All new calibration/monitor traces audited: 606,460; canonical cases 560; maxdelta 2.47359276e-06. ReportSHA 30b7de7efb9ba825ee37b93d5e636a9c2fa700ad1697987f5c6066e3ec78db3e. All weights/optimizerRNG/traces stay SSH, no Drive backup.
