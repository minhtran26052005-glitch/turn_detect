# Cycle75 diagnostic: COMPLETE

Current interpretation: the legacy-dev comparison is NOT a matched lookahead benchmark and must not rank the old and new models. The C75 encoder expects left=context and right=1/2 actual following words. All17,833 legacy contexts are empty; the previous run placed the entire current prefix in the future slot (17,385 right slots exceed2 words). The stored target is at the END of that prefix, so splitting its final words would move the target boundary. All17,833 encoded sequences match the official new encoder; frozen weight/tokenizer/config hashes, label map and threshold also match. This is a task/input-role mismatch, not a detected model-loading or label-index defect. Raw scores and traces are preserved; no new inference, threshold tuning, label edits or holdout scoring occurred.

The text-only data audit found0 source-copy/label mismatches,0 conflicting normalized inputs and0 exact normalized legacy train/dev overlaps.12,011/12,220 NEOT examples match strict prefixes of longer EOT examples within heuristic groups, but conversation/future-word provenance is absent. For example, a complete shorter question can be NEOT when the original speaker continues with another clause. This is ambiguity, not proof all labels are wrong. The label-source value provided_gold is assigned by the preparation code and does not certify independent human annotation. Old dev was used for student selection. C75 VI has865,075 eligible rows but only70,400 optimizer row draws, including repetitions; more corpus rows do not establish more training exposure or a causal quality drop. The new prose audit also retains248/16,777 VI EOT rows with noncontiguous future after short-sentence filtering and unresolved abbreviation candidates. See legacy_protocol_data_audit_v1.json, encoder_contract_audit_v1.json and legacy_comparison_validation_v1.json. The reader report prominently corrects the interpretation and keeps old raw numbers in an expandable reference.

The user-requested legacy-dev comparison is COMPLETE on Colab L4. Exact17,833-row dev SHA be8a3a8857a2628dda4fbca4ea7b44a6a4eebdee1acd2c8d60702326938d6c4e; frozen VI cycle75 weights SHA unchanged. New VI model at its fixed0.660 threshold:accuracy78.4837%, EOT F1 70.6539%, FP2,843/FN994. Archived packed-QAT probabilities on the same rows/order reproduce93.5457%/90.1934% at the old fixed0.470 threshold. No new training, calibration or holdout/reserve scoring occurred. All legacy context fields are empty; original text is preserved in the native context/text pair, with no future words invented. This is input-domain transfer on previously selected dev, not an isolated compression effect or a new blinded test. Report SHA168f3f507895f64110921406df99002c67117fc6b3d668a3e78f686c57aa9b92; full trace SHAaf2179c847f04ea480c8e9da2f14b7ff1e0616f29cf000059f7856b03ab55203. Input, source, full trace and result were saved directly to Drive under old_dev_c75_vi_v1 with SHA readback. Local files are compact report/protocol/manifest/samples only.

The reader-facing report now includes a “Bản Việt nhỏ” section. It describes the prior Vietnamese GRU/QAT INT4 package:293,090 deployment parameters,177,124-byte model and403,973-byte runtime package excluding installed libraries. All seven existing local package files match the saved manifest size/SHA; no inference ran for this report edit. The271.3x size ratio uses the109,609,065-byte teacher package from the SAME older compression branch, not the cycle75 teacher. Historical dev17,833 scores93.55%/90.19% and the seen210-point single-call automatic-label regression32.86%/22.10% are clearly separated from the unscored-for-this-student new39,167-row VI test. This edit does not turn the old student into a newly distilled cycle75 checkpoint or claim new-lookahead quality. See vi_compression_evidence_v1.json, plain_report_receipt_v2.json and report_qa_compression/qa_receipt.json.

Reader-facing result: [Báo cáo kết quả dễ đọc](bao_cao_ket_qua.html). The previously open review_complete_v3.html now shows this same nontechnical report: model inputs/outputs and architecture, interactive1/2-word lookahead, actual TRAIN examples, public TRAIN/TEST sources, final scores and saved correct/incorrect TEST outputs. It contains no execution diary. The original technical review is preserved byte-for-byte as review_technical_v3.html and as its existing Drive content object. All model metrics and raw report bytes remain unchanged. See plain_report_receipt_v1.json; the report was checked on desktop1440px and mobile390px with working lookahead/language buttons.

The one-shot test, random TRAIN-pool control, actual optimizer-drawn TRAIN control, source-parser audit, reserve exposure audit and Drive finalization have completed. Campaign launcher PID29096/start717109 automatically resumed at2026-10-27T04:37:09Z after the user-authorized test-priority window. A fresh check at04:53:30Z verified the original parent and its cycle81/VI child. The last committed full campaign backup was data cycle79 with full_recovery_available=true. Test results remain fixed to cycle75, not the newer training checkpoint.

119,193 clean rows after removing1,882 overlaps: VI accuracy81.14% /EOT F1 79.08%; EN84.85% /85.02%; ZH84.20% /67.18%. No90% certification. Raw report SHA256:fe4555399de148fe34c2e96f7d95d4edb2aa453caa618f0d69a333ebbb238650.

See [test provenance](test_provenance.md) for exact upstream datasets, source-page splits, punctuation targets, normalization,256-token overlap filtering and the416-page/34,889-row remaining unscored reserve. See [complete review](review_complete_v3.html) for results, reviewed causes, source origins and360 inspectable samples, or [crawl plan](crawl_plan_v3.json) for the prepared TRAIN-only changes. These changes are NOT ACTIVATED in the live crawler. Immutable reports, original failure evidence and versioned receipts are retained; do not relaunch or rescore.

## Material Passport

- Mode: experiment run and exploratory error analysis.
- Material: frozen cycle75 checkpoints, dev thresholds, punctuation-derived public prose.
- Scope: three separate lowercase, punctuation-free vi/en/zh models;1/2-word lookahead.
- Status: COMPLETE on Colab; no local model inference or bulk transfer.
- Limit: not independent human/audio conversational gold; no90% certification.

All3 cycle75/100 optimizer checkpoints were pinned before later training could replace them. Their weights are extracted on Colab. Thresholds come from immutable cycle75 dev reports (VI0.66, EN0.58, ZH0.755), not test optimization.

Only source pages first added after cycle45 through75 are eligible.20% of eligible source pages are deterministically reserved and remain unscored. The diagnostic partition excludes exact source copies and actual normalized/truncated256-token model inputs overlapping train/dev, prior test or the unscored reserve. Paraphrase/near-duplicate detection is not exhaustive. The original heldout_test_v2 remains unchanged and is not rescored.

Scoring is one-shot. Errors may inform exploratory TRAIN-only data collection. Frozen model inputs, source-page bootstrap95% intervals, per-lookahead results, full traces and label-audit paragraphs are retained. GPU inference yields to campaign training children; the campaign is not stopped or reset.

A separate one-shot control measures fit on2,000 frozen train-pool rows per language (500 per label/lookahead stratum), with the same weights and thresholds. This is a TRAIN-pool fit diagnostic, not proof the sampled rows were actually drawn by the optimizer. Total row draws are measured from checkpoint steps and the actual32x2 recipe.

Large artifacts are copied Colab-to-mounted-Drive under turn_detect_backups/translation_segmenter_v15d_20261026/diagnostic_test_c75_v1, with fsync/SHA readback and a separate manifest. Windows receives at most5MiB of reports/review. Preserve failed or partially scored state; use a versioned technical continuation rather than redoing scored evidence.

Controls:
python -X utf8 tools/translation_segmenter_diagnostic_colab.py status
python -X utf8 tools/translation_segmenter_diagnostic_colab.py fetch

Do not relaunch the evaluator or analysis companion. Current launch receipts identify both by PID/cmdline/start ticks.


## Technical continuation v2

Initial v1 publisher failed before scoring any diagnostic row: it hashed live status.json and then changed that same file during transfer. Frozen weights and SQLite were preserved. New v2 publisher pins mutable metadata bytes before status updates; a regression test reproducing the race passes. All12 protocol, leakage, causal-extraction and publisher tests pass. The new evaluator validates original frozen hashes and does not repeat any completed language. Original error/launcher/status evidence is retained. analysis_v2 combines source alignment and fixed train-pool controls; previous controls exited before scoring.

Measured optimizer exposure:70,400 draws per model (1100x32x2). Current eligible pools:VI865,075; EN449,514; ZH1,128,617. Upper bounds for unique exposure are8.14%,15.66%,6.24%; repetitions and older pools reduce actual coverage. No epoch completion or causal accuracy gain is claimed.

A causal extraction defect was reproduced without model inference: long sentence. Yes. Another long sentence. produced future=another under the old25-character/min5-token filtering. turn_v2/translation_segmenter_causal_extraction_v1.py preserves short sentences and unfinished tails. It has not been activated in the live crawler or used to rewrite old examples/checkpoints. The scored-source audit will measure the defect only for clean diagnostic records that align exactly to the original extraction.


## Execution history and user test-priority choice

The completed evaluator was v2 (PID46990/start1150946), and exited normally. A proposed v3 clean-only-inference revision was NOT launched: its conservative guard detected an existing CUDA context and refused before sending any signal. Existing scoring progress was preserved. The scorer inferred raw diagnostic rows and excluded overlaps from its official clean metrics. A separate CPU reserve_exposure_audit_v1 removed entire reserved source pages matching ANY raw inferred input, including filtered rows, before a future reserve test. Its remaining-source receipt is required; re-audit against future train/dev before any final test. No old test report was rescored.

The user explicitly selected: prioritize test after the currently running train, then automatically continue campaign. priority_window_v1 (supervisor52938, watchdog52939) stops only launcher PID29096/start717109 with SIGSTOP; current train child49468, cycle78/ZH, finishes normally. It never interrupts/resets the kernel or signals the train child. Existing evaluator, source audit/train controls, reserve audit and finalizer then get the GPU window. Supervisor finally sends SIGCONT to the same verified campaign launcher after finalizer and reserve audit COMPLETE, or on upstream error. An independent watchdog resumes if the supervisor exits unexpectedly. Do not create another campaign/evaluator/priority worker.

finalizer_v1 automatically creates error_analysis.md, crawl_plan_v1.json and review.html on Colab after scoring and controls, then publishes all large evidence and compact reports directly to Drive with SHA readback. Fetch only the compact artifacts after finalization; <=5MiB total. Crawl recipe and causal extractor are reviewable preparations, not an activated rewrite of the live corpus.
