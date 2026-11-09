# Current uploaded-only mALBERT TEST, 2026-11-05

New direct user request: run the current model on TEST. This is a new authorized run; previous completed evaluations remain intact. Freeze the first temporal v4 checkpoint with matching new DEV after launch, then lock its threshold before any TEST inference. The active trainer continues unchanged.

Only original TEST from the previously verified eval ZIP is scored, using source_raw, one/two-word same-turn future and predicted-EOT-only history resets. Primary punctuation EOT remains full stop; question/exclamation marks are supplemental diagnostic. Include keep=false and known legacy factory overlaps as in the previous requested population; retain the cohort excluding known overlaps. Prior overlap is not undone by dropping those families from new batches. Labels are punctuation silver, not independent human/audio turn gold.

The full model/27 AdamW/RNG/order/cursor/conversation-lane pin, input ZIP, prepared TEST and complete traces stay on SSH. Windows receives only bounded reports and review. Audit every causal encoded input, confusion count and conserved source word, plus canonical v4 predictor numerical parity, before reporting completion. No TEST tuning, new training source, optimizer reset, active/default replacement or INT4 acceptance. Checkpoints are not backed up to Drive.


## Verified current-model TEST, 2026-11-05

The direct request to test the current uploaded-only model is complete and reported. Frozen global step1,945,080 / branch480,000, mALBERT11,941,122 FP32 parameters, threshold0.1 locked from the matching new-DEV calibration before TEST. No training interruption, new test-derived threshold/model/loss selection or default replacement.

All579,685 inferences (297,397 lookahead1 /282,288 lookahead2) passed encoded-input/future/reset/confusion/word-conservation/SHA auditing. All probabilities were finite;62 canonical v4 predictor cases matched decisions with maximum probability delta8.94e-8. Report SHA `8745e359b9e286742b5da89b01e9dceb3b742e6ecc4a87f566f3855f95a40d96`.

Primary original-TEST lookahead2: VI F1 26.00%, precision52.14%, recall17.31%,647 false cuts/1,352 scored cuts,3,367 misses/4,072 EOT; EN F1 41.14%, precision50.02%, recall34.94%,1,570 false cuts/3,141 scored cuts,2,925 misses/4,496 EOT. Old1340000 results14.65%/27.25% are reused from saved reports, not rescored. New training and DEV calibration changed; thresholds0.73->0.1, so improvement is not attributed solely to weights. Excluding24 known legacy factory TEST overlaps gives VI F1 25.97% /EN41.00%; this does not exhaust every possible overlap. Punctuation silver, not independent human/audio turn gold.

The first whole uploaded epoch has a COMPLETE coverage receipt for6,169,873/6,169,873 eligible scored gaps, zero Drive replay. At the pin,7,544,867 exposures and epochdisplay2; six minimum epochs are not complete. Source files/core trainer hashes are unchanged. Main205061/ticks547555/boot65117834 remains the sole active trainer, healthy with no error; latest runtime proof shows GPU11168MiB, SSD46.27GiB/HDD23.01GiB rw and Windows23.94GiB (snapshots only). No accepted INT4. Fullpin/weights/raw/preparedTEST/fulltrace stay on SSH; Windows holds bounded reports/review only, no Drive backup. See `outputs/malbert_uploaded_test_20261105_v1/review.html`. Never rerun this completed test without a new request; continue the active training campaign.
