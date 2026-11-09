# Uploaded-only Vietnamese / English training, version 4

User request on 2026-11-04: replace narrow DEV, train on newly uploaded sources, discard the Drive data/generation branch. Preserve historical data and checkpoints. No new TEST scoring was requested for this cutover.

Model: pretrained mALBERT, **11,941,122 parameters**. The parent warm weights, all 27 AdamW states and Torch/CUDA/NumPy/Python RNG are preserved in `frozen/parent_full_resume.pt` on SSH. The source order is explicitly versioned for the new dataset. Earlier training exposure cannot be undone by removing those examples from future batches.

## Data

Four unique source groups: `daily` from public_checked.jsonl, `factory/gen500` from factory_checked.jsonl, newly supplied `asr/run2b`, and newly supplied `factory/gen4000`. The ZIP gen500 member duplicates factory_checked and is not imported twice. Raw uploads remain on SSH. Original DEV/TEST source families, invalid or noncontiguous chunks, duplicate turns and substantial source overlaps are excluded from TRAIN; exclusions are recorded in dataset_manifest.json.

After filtering: **198,509 source rows**, **17,301 conversation streams**, **3,373,257 words**, and **6,169,873 eligible scored gaps per complete epoch** across one/two-word lookahead. Every eligible uploaded TRAIN gap is scheduled; there is no 100,000-per-language cap and **zero Drive replay rows**. This describes the dataset, not completed training coverage. Per-epoch coverage receipts prove when all gaps have actually been visited.

Only `source_raw` supplies model input. Translation targets, row instructions and privileged rest_of_sentence are never model input. Source punctuation supplies silver labels, not human/audio EOT gold. Numbers in source text are retained. The labels and input normalization follow the existing fullstop protocol; ambiguous punctuation endpoints remain unscored.

## New DEV

Original DEV only. Thirty known factory DEV conversations exposed to the parent TRAIN are excluded. TEST remains outside threshold/model selection and is used only for source exclusion. Whole-family calibration and monitoring cohorts are fixed by source hash before model scoring. Full DEV and supplemental numeric diagnostics are preserved separately on SSH.

- Calibration: 119 streams, 13,275 scored two-word-lookahead gaps.
- Monitoring: 394 streams, 34,987 scored two-word-lookahead gaps across ASR, daily, factory and numeric dialogue.
- Full retained DEV: 5,911 streams. The fixed monitoring subset is not a claim that every full-DEV stream is scored each cycle.

Threshold selection now runs actual causal predicted-reset rollouts at every candidate threshold on calibration families, then scores disjoint monitoring families. It does not use reference/speaker/language resets. These new DEV metrics are not directly comparable with the former 72-stream DEV. Prior whole-lineage token-window and paraphrase contamination has not been exhausted.

## Training and resume

Module `turn_v2.multilingual_malbert_vi_en_uploaded_train_v4`; one CAMPAIGN_LOCK worker. The former v3 worker exited STOPPED_BY_REQUEST at its preserved full checkpoint. Never auto-resume v3 or completed predecessors.

Training walks conversations causally in lanes; every conversation is scheduled with one-word and two-word future. Only model predictions reset the visible history; source turn metadata limits future availability. All eligible scored gaps receive CE; Drive teacher replay/targets are absent. The inherited AdamW state and learning rate are preserved. Full checkpoints contain conversation lanes, source order, cursor, optimizer, RNG and quantization state for exact continuation. No plateau stop.

Only a DEV report with branch_step > 0 is after new-data training. Branch zero is a warm-start baseline. Frozen historical teacher inference on the same new DEV establishes the retention comparator; its training is never restarted. Original per-language retention tolerances remain unchanged. FP32/INT8/packed INT4 packaging still requires trained artifacts, CPU DEV, parity/padding checks, batch-one/two-thread benchmark and ZIP hash readback. No active/default model is replaced.

The initial teacher-token trace digest used uint16 despite the teacher's larger vocabulary. The metadata-only uint32 repair happened at branch step zero. Old source, error, launcher, manifest and full inherited checkpoint remain in `history/token_digest_uint32_repair01`; the repair receipt verifies unchanged source data, model, optimizer and RNG. No training data or state was discarded.

Collector: `python -X utf8 tools/collect_uploaded_vi_en_v4.py`. Windows receives only small source/status/manifests/reports/review, never models, caches or full traces. Keep SSD/HDD reserves of 13 GiB and Windows C reserve of 10 GiB. Do not delete data or touch unrelated jobs. Checkpoints and packages are not backed up to Drive.

The user discarded Colab/Drive generation. Its stop helper had no current a.ipynb runtime, so physical worker shutdown remains unverified; do not restart, recover, take ownership or request a new kernel. The automation now monitors only this uploaded-data training campaign.

## Verified cutover

Independent raw-source audit passed for all 198,509 included source rows and all 6,169,873 eligible scored gaps. Independent causal-training audit passed for 1,593 actual training inputs and the full checkpoint at global step 1,466,580 / new branch step 1,500. That checkpoint had trained 23,598 points, including 107 ASR/run2b and 6,919 gen4000 points. Model weights changed, remained finite, and all 27 optimizer states, RNG, source order and conversation lanes were present. This proves new-source training has started; it does not claim the first full epoch is complete. See independent_causal_training_verification.json and current heartbeat_verification.json for the evidence and fresh state.

## Host reboot and verified recovery, 2026-11-04

The host rebooted at approximately 12:57 Bangkok time, changing boot ID from `92e2e568-17ea-47cf-984f-ba547e7f76bb` to `65117834-0b91-465e-a099-805d6100f317`. The old worker disappeared during partial DEV calibration at branch step 40,000; that partial calibration is not a completed DEV report.

`/mnt/hdd500` was initially unmounted. The user restored the NTFS volume UUID `37622A5A7A5A4B57` on `/dev/sda1` to that exact mount point. The recovery controller verified rw mount, >=13 GiB reserves, all dataset hashes and the complete saved model/27 AdamW/RNG/sampler/conversation-lane state before launching exactly one replacement worker. No sudo password was read or stored; no source, loss, threshold policy, dataset or model initialization was changed.

Recovery retained an immutable hardlink of the branch-40,000 full checkpoint (SHA `b3c80475a124b345c76768280ef2f663ab1771a0fedc3e825fb6966a758bfe09`). A new full checkpoint at branch 40,500/global 1,505,580 advanced to 637,035 uploaded points; its SHA is `8447b1c47590a4c3751db8dd2a951348ad870a80e21e9df1edee8c194e776e85`. All optimizer steps, RNG/order/conversation fields, finite weights and unchanged fingerprints passed. See recovery_training_verification.json. Recovery controller exited after verification; keep the main worker active and do not launch duplicate recovery processes.

Latest fully completed DEV at this check was branch 30,000/global 1,495,080, shared threshold 0.1: VI F1 8.94%, precision 35.29%, recall 5.12%; EN F1 23.10%, precision 41.89%, recall 15.95%. This is the new 394-stream punctuation-silver DEV, not a TEST result. It is still weak; no retention acceptance, INT4 completion, default-model replacement or first full epoch completion is claimed. Those results and the recovered startup were directly reported once.

## Heartbeat, 2026-11-04 15:46 Bangkok

Same recovered worker PID205061/start_ticks547555/boot65117834 was freshly verified with advancing CPU ticks and an active GPU process. No active error; `/dev/sda1` remained rw at `/mnt/hdd500`, with 23.01 GiB free; SSD45.71 GiB and Windows15.29 GiB free. All resource values are snapshots and require fresh verification.

Newest complete post-training DEV was branch60,000/global1,525,080, threshold0.1: VI F1 **12.26%**, precision42.42%, recall7.17%; EN F1 **28.92%**, precision48.81%, recall20.55%. Report SHA `66b100d27569d07be9ae3e2e6d935ed8a35504aecf5c2ab4dab268df0f2c0f08` matched the exact per-step record. This report supersedes earlier periodic notifications; intermediate50,000 is not separately announced. Branch70,000 was still calibrating and was not reported as complete. Retention remains false; training continues without changing data, loss, optimizer or threshold policy. No TEST rerun or INT4 acceptance.

## Heartbeat, 2026-11-04 21:47 Bangkok

Worker PID205061/start_ticks547555/boot65117834 remained the sole active v4 worker. Identity, advancing CPU ticks and GPU11168MiB were freshly verified; observed branch218500->219000/global1684080. No active error. SSD46.55GiB/HDD23.01GiB and Windows22.39GiB were free; `/dev/sda1` remained mounted rw. Values are historical snapshots for future checks. The first epoch was not complete.

Only newest complete post-training DEV branch210,000/global1,675,080 was notified: shared threshold0.1; VI F1 **20.91%**, precision37.44%, recall14.51%; EN F1 **36.65%**, precision51.03%, recall28.59%. Remote report SHA `b1a415445b1f3148f41bd8a40e5efbce2f697da03c7f7bfdd9a9d06c0b7ad5c4` matched the exact remote per-step record and the local readback after Windows CRLF-to-LF normalization. Earlier intermediate reports were skipped. This is the same 394-stream/34,987-point punctuation-silver DEV, not TEST or human/audio gold. Retention remains false; preserve training and its full state. No dataset/loss/threshold-policy/batch change, compression acceptance or TEST rerun.

## Heartbeat, 2026-11-05 02:49 Bangkok

Same sole worker PID205061/start_ticks547555/boot65117834 was freshly verified with advancing CPU ticks, branch343100->343550/global1808630 and GPU11168MiB. No active error. SSD46.51GiB/HDD23.01GiB and Windows23.10GiB free; `/dev/sda1` remained rw at `/mnt/hdd500`. The first uploaded-data epoch still had no completion coverage receipt, so full-epoch coverage is not claimed.

Only latest completed post-training DEV branch340,000/global1,805,080 was notified: threshold0.1; VI F1 **20.47%**, precision44.32%, recall13.31%; EN F1 **33.14%**, precision52.83%, recall24.14%. Remote report SHA `b375a9b1439f3342023f256e40b29b691f142b00e3467e8eb4da022918d0d280` matched the exact per-step record and the LF-normalized Windows readback. Intermediate reports were skipped. Retention remains false; training continues with unchanged dataset/loss/optimizer/threshold policy/batch. No TEST rerun or INT4 acceptance.


## Verified current-model TEST, 2026-11-05

The direct request to test the current uploaded-only model is complete and reported. Frozen global step1,945,080 / branch480,000, mALBERT11,941,122 FP32 parameters, threshold0.1 locked from the matching new-DEV calibration before TEST. No training interruption, new test-derived threshold/model/loss selection or default replacement.

All579,685 inferences (297,397 lookahead1 /282,288 lookahead2) passed encoded-input/future/reset/confusion/word-conservation/SHA auditing. All probabilities were finite;62 canonical v4 predictor cases matched decisions with maximum probability delta8.94e-8. Report SHA `8745e359b9e286742b5da89b01e9dceb3b742e6ecc4a87f566f3855f95a40d96`.

Primary original-TEST lookahead2: VI F1 26.00%, precision52.14%, recall17.31%,647 false cuts/1,352 scored cuts,3,367 misses/4,072 EOT; EN F1 41.14%, precision50.02%, recall34.94%,1,570 false cuts/3,141 scored cuts,2,925 misses/4,496 EOT. Old1340000 results14.65%/27.25% are reused from saved reports, not rescored. New training and DEV calibration changed; thresholds0.73->0.1, so improvement is not attributed solely to weights. Excluding24 known legacy factory TEST overlaps gives VI F1 25.97% /EN41.00%; this does not exhaust every possible overlap. Punctuation silver, not independent human/audio turn gold.

The first whole uploaded epoch has a COMPLETE coverage receipt for6,169,873/6,169,873 eligible scored gaps, zero Drive replay. At the pin,7,544,867 exposures and epochdisplay2; six minimum epochs are not complete. Source files/core trainer hashes are unchanged. Main205061/ticks547555/boot65117834 remains the sole active trainer, healthy with no error; latest runtime proof shows GPU11168MiB, SSD46.27GiB/HDD23.01GiB rw and Windows23.94GiB (snapshots only). No accepted INT4. Fullpin/weights/raw/preparedTEST/fulltrace stay on SSH; Windows holds bounded reports/review only, no Drive backup. See `outputs/malbert_uploaded_test_20261105_v1/review.html`. Never rerun this completed test without a new request; continue the active training campaign.


## Heartbeat, 2026-11-05 09:10 Bangkok

Fresh sole worker205061/start_ticks547555/boot65117834 matched actual proc/cmdline/owner identity, with increasing CPU ticks and branch492050->492550/global1957630. GPU11168MiB, no active errors. SSD46.27GiB/HDD23.01GiB rw and Windows23.93GiB free are snapshots. First uploaded epoch coverage remains complete; second epoch is active. No restart, dataset/source/loss/optimizer/batch/threshold-policy change or INT4 acceptance.

Only newest complete post-training DEV branch490,000/global1,955,080 is notified: shared threshold0.1, VI F1 **22.19%**, precision43.94%, recall14.85%; EN F1 **42.21%**, precision55.50%, recall34.05%. Retention remains false; keep training. Report SHA `671071ed6b2c8f6cfaa7e19f8941e378788647fc731f30f1beab06f108c74f48` matched the exact remote per-step report and LF-normalized local JSON. Intermediate350,000..480,000 are not separately notified. Same394-stream/34,987-point punctuation-silver DEV, not TEST or human/audio gold. The directly reported TEST@1945080 remains COMPLETE/reported; no rerun or repeated notification. Automation remains active every300minutes.


## Direct check, 2026-11-05 14:07 Bangkok

Sole main205061/start_ticks547555/boot65117834 identity and advancing CPU ticks were freshly verified. Training advanced branch621100->621900/global2086980, uploaded9,776,570 scored exposures, epochdisplay2; one full6,169,873-gap uploaded epoch has a COMPLETE receipt. No active error. GPU88%, own process11172MiB. SSD46.48GiB/HDD23.01GiB rw and Windows19.68GiB free remain above reserves (snapshots). No restart, test rerun, model/source/loss/optimizer/batch or threshold-policy change.

Only latest complete post-training DEV branch620,000/global2,085,080 is reported: threshold0.1, VI F1 **23.05%**, precision46.15%, recall15.36%; EN F1 **44.08%**, precision61.07%, recall34.48%. Report SHA `7b9f7740b5ecced918cf253257b197c7ac59647a541aa9898a3516ad010e5d1a` matched the exact remote per-step record and LF-normalized local readback. Intermediate500,000..610,000 are skipped. Same394-stream punctuation-silver DEV, not TEST/human gold. Retention is false; training continues. Model11,941,122 parameters remains FP32 weights/BF16 training with no accepted INT4. Completed TEST@1945080 remains reported and is not rerun. Automation remains active every300minutes.


## Heartbeat, 2026-11-05 21:46 Bangkok

Same sole main205061/start_ticks547555/boot65117834 freshly matched proc/cmdline identity and increasing CPU ticks; GPU11198MiB. It is healthy calibrating DEV820,000, advancing threshold0.2->0.3 between observations. This incomplete calibration is not a completed monitor report. Full checkpoint branch820000/global2285080 has12,884,203 uploaded scored exposures. Two whole uploaded epochs have COMPLETE6,169,873/6,169,873 coverage receipts each; third epoch is active and six minimum epochs are not complete. No active error. SSD46.44GiB/HDD23.01GiB rw and Windows20.71GiB free are snapshots above reserves.

Only latest COMPLETE post-training DEV branch810,000/global2,275,080 is notified: threshold0.1, VI F1 **24.06%**, precision49.73%, recall15.87%; EN F1 **49.33%**, precision54.42%, recall45.11%. Report SHA `ab9c639c32e1f487fb0e591ef9e3bbdd9fee175c93cbad4b887c2c7e2f16985b` matched the exact remote per-step record and LF-normalized Windows readback. Intermediate630,000..800,000 are skipped. Same394-stream/34,987-point punctuation-silver DEV; not TEST/human gold. Retention remains false; keep training and its source/loss/optimizer/RNG/batch/threshold policy. No restart, TEST rerun, INT4 acceptance or active/default replacement; keep automation active300minutes.


## User-requested old135M new-DEV check, 2026-11-05 22:48 Bangkok

The old full-precision control is135,326,210 parameters, frozen checkpoint514235/branch28000 (weightSHA15d4ebfc7fcba7c3c842b7b7815ac309cf349f0436dca69522d9d529295954cc). It had already been scored on these exact new-DEV119 calibration/394 monitor streams during v4 setup. This direct request reused and independently verified that completed result instead of rescoring the entire DEV. All35,270 monitor inferences passed encoded-input/future/label/predicted-reset/confusion/digest verification;16 canonical GPU predictor cases matched every decision, maxdelta1.788e-7. Receipt teacher_new_dev_user_recheck_20261105.json; prior reportSHAb97bb82239422bab7d41a7842edbd9c9b50e553b1eae0292e73d47202ff2e6ce.

At new-DEV-selected threshold0.1: VI F1 22.34%,precision17.24%,recall31.74%,893 false cuts/1079 scored cuts82.76%; EN F1 38.43%,precision30.19%,recall52.87%,851 false cuts/1219 scored cuts69.81%. Labels are punctuation silver. Old135M has not been fine-tuned on the new uploaded training sources. Main11.94M trainer remained active; no old training resumed, TEST scored, loss/source/threshold-policy/default change. The direct old-control result was reported once. MainDEVnotification ledger is unchanged by this old-control check.


## Translation-purpose evaluation completed, 2026-11-05

User supplied Qwen3 checkpoint15640 ZIP, template B and CPU demo. The existing SSH checkpoint matched all eight archive files by SHA and was copied into an isolated project data folder; no other project/job was modified. Translation model596,049,920 parameters, FP32; this is separate from the11,941,122-parameter detector. CPU exact CLI demo returned "I'm doing well, and you?". GPU cached/uncached demo matched, and CPU/GPU demo text matched.

Full eval_b contains310VI/EN messages in37 groups (70tele+240MT). All three arms completed:whole_written,whole_asr,detector_asr;930 message records/963 translation prompts. The frozen turn detector is prior-test checkpoint1945080/t.1; no new detector or threshold selection from these tests. Whole_asr vsdetector_asr uses the same source normalization and references; chunking also changes the clipped source-history context, as in the real template B flow. Source references never enter prompts.

chrF all/VItoEN/ENtoVI:written41.06/40.95/41.15;ASRwhole39.62/40.23/39.06;ASRdetector39.32/40.10/38.59. BLEUall17.52/14.25/13.91. Metrics are text-reference scores, not correctness percentages. Full prompt/source/word-conservation/3900detector-decision/metric/SHA audit COMPLETE; reportSHAe05c84065922f312ee85b221d64c3e1b8ea4cc5bb0b1d2e95160c50fe07b47af.

Only33internal predicted cuts in29messages;343translationchunks;310 forced final-message flushes by harness. Thus this measures within-message segmentation impact and history changes, not true audio floor timing or cross-message endpoint latency.48 message outputs differed. Five whole-message generation runs reached512token cap with repetitive outputs; kept in metrics. References include questionable MT pairs and are not independently audited human gold; training/test overlap is not exhaustively audited. Examples selected by largest sentence-chrF differences for illustration, not random sampling or new selection.

Main trainer205061/ticks547555/boot651 remains active. No source/loss/optimizer/threshold-policy/default/QAT changes. No models, raw archives or full traces downloaded to Windows; only small reports/review. Artifacts and isolated translation model staySSH. No Drive backup. Test complete/reported; do not rerun automatically. Collector tools/collect_translation_turn_test_20261105_v1.py, verifier tools/verify_translation_turn_test_20261105_v1.py.
