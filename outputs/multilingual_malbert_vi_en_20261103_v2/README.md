# mALBERT turn detector — Vietnamese and English only

User explicitly removed Chinese from every active workflow on 2026-11-03. Only this 11,941,122-parameter pretrained mALBERT trains on SSH; the old 135M trainer, failed custom 18.67M campaign, and trilingual mALBERT predecessor must not resume.

The continuation inherits step 14,082, all model tensors, AdamW (27 states), Torch/CUDA/NumPy/Python RNG, sampler RNG and the exact remaining VI/EN row order. The old 300,000-row permutation is projected onto 200,000 rows: 100,000 Vietnamese and 100,000 English. `migration_receipt.json` records parent and migrated checkpoint hashes. `independent_training_verification.json` independently verifies a later checkpoint, its optimizer step, 200,000-row permutation, bilingual cache/source hashes and DEV languages. Old data and full checkpoints are preserved.

TRAIN uses the existing frozen source722 VI/EN data, not new spoken generation. DEV, common threshold calibration, retention gates, QAT, export fixtures and CPU benchmarks are restricted to VI/EN. Inputs remain causal: context and at most one/two future words; no language ID or gold resets at inference. Data is synthetic punctuation silver, not human/audio gold. No test is used for selection.

Training continues from the inherited state through CE and distillation, with checkpoints and DEV. INT4 QAT and CPU packaging follow only after the FP32 quality gate passes; no final compressed model is accepted yet. Preserve all controls, optimizer/RNG checkpoints and rejected candidates. No automatic deployment or Drive backup is configured.

Monitor with `python -X utf8 tools/collect_malbert_vi_en_v2.py`; this downloads small reports only, never weights. SSH output: `/home/edabk/hoangbpm/turn_detect/outputs/multilingual_malbert_vi_en_20261103_v2`. SSD/HDD reserve guards remain 13 GiB and require the HDD mount read/write.
