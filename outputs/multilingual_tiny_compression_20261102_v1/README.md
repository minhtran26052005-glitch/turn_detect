Compression campaign complete. Tiny GRU128/192 distillation on300000TRAIN rows (100000perlanguage) and matched FP32/QATg64/QATg32 controls were completed and exported to actual CPUONNX. No tiny candidate retained all three languages within the declared F1/precision/recall margins. All rejected artifacts, checkpoints, optimizer/RNG and reports are preserved on SSH.

The accepted compression is the existing six-layer shared multilingual DistilBERT, with INT8group64 stored matrices and FP32 computation. It is post-training storage quantization, not a successful tiny-GRU/QAT result. Teacher branch28000/global514235 weights541317336bytes; compressed model139823559bytes; deployment142753473bytes; ZIP132170620bytes. CPUONNX DEV F1: VI91.6201%, EN95.5823%, ZH65.3938%; reference91.6201/95.5823/65.5502%. VI/EN same confusion counts; ZH one additional false positive. Shared threshold0.26, causal predicted-EOT resets, no language ID or gold reset.

CPU batch1/two threads/24cases/120timed calls including tokenization: teacher median12.7114ms/p9534.077ms, compressed8.9661ms/p9530.5416ms. Shared host; no dedicated machine timing claim. Benchmarkv2 uses exec-local LinuxVmHWM; old resource high-water values were inherited and have been superseded, with evidence retained. Current final report/manifest/ZIP contain corrected measurements.

Source722 synthetic punctuation TRAIN plus legacyZH; new spokenv9 data NOT used. VI/EN causal synthetic DEV and ZH legacy DEV anchors differ in distribution. No independent human/audio/conversational test or90percent target certification. Original main training and Colab spoken generation continue, active model/default not replaced. New SSH package/checkpoints NOT backed up toDrive. No model or cache downloaded toWindows.

Accepted package: /home/edabk/hoangbpm/turn_detect/models/turn_multilingual_transformer_compressed_20261102_v1
ModelSHA256 fd1e6c64f32f3a681752c91b4d830d3ba9d0fcc90fd3e672eb3d01b14a25f62c
Final ZIP SHA256 9ca5c8734725d2e2ab9227a302843d9126017418b8d7325029409c0106ea0534
Manifest SHA256 b22ac82f48bf7765573298ea8676980128310820bd26094de527b966547f009b

Independent package readback verified every manifest file, ZIP member, packed matrix initializer types (40INT8/40FP16scale/64FP32vector tensors, no FP32matrix initializer), and portable CPU loading on VI/EN/ZH without importing PyTorch. See review.html/compression_summary.json and sibling multilingual_transformer_compression_20261102_v1/final_report.json/package_verification.json. All model weights remain onSSH.

No compression worker should be relaunched: PID3217492/ticks52069749 and control3391254/ticks52255374 both exited normally. Monitoring remains for main training and spoken generation, with this completion already reported. Do not repeat old failure/progress notifications or rerun completed compression/test from heartbeat.

User03/11 nowrequiresstrictly<20Mparameters. Thehistorical135MINT8control fails this new sizecriterion; it remains a preservedretentioncontrol, not the deliverable for thatrequirement. Newbranch multilingual_under20m_20261103_v1 trains a18668738parameterTransformer with originalmultilingualtokenizer, whilemaintrainingcontinues. Do notrestart thecompletedoldGRU/controlworkers.
