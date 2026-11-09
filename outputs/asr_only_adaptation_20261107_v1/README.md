# Bounded ASR-only adaptation

User requested trying ASR-only training. One full epoch is the initial experiment budget, then the same frozen DEV protocol reports ASR and other domains. Original v7 stays paused, no automatic default replacement and no TEST evaluation.

ASR-only natural-source subset: 210 streams, 81 canonical video families, 169011 words, 6846 retained valid raw source rows, 333315 eligible points across one/two same-turn future words. This is not all12415rawrows. Source is the verified v5 TRAIN subset; inherited conflict masks and original source order retained, no v7 synthetic sentence repacking. Canonical video families disjoint DEV and TEST source-only metadata. No held labels, predictions, translations or rest_of_sentence as model input.

Model11941122FP32weights/BF16training, inherited model27AdamW/TorchCUDA_numpyPythonRNG exact from paused3731173. New420-task permutation/fresh16lanes explicit, old parentorder/cursor/lanes stored immutable. SameLR4e-6/wd.01/weightedCE NEOT1EOT2/batch16/CPU2/GPUcap50%. Full resume only own latest_resume.pt with fingerprint/model/optimizer/RNG/order/cursor/lanes/sampler, no fallbackbase. Technical CPU interrupted-active-lane fixture passed bitwise equality, technical only.

Baseline reuses exact frozen3731173 DEV: ASRVI F110.4294/P29.8246/R6.3197, ENF139.0456/P52.3256/R31.1419percent. Same119calibration/394monitor, selected sharedthresholdgrid.1..9 worst-languageF1; report all domains and fixedbaseline.1 for weight-only comparison. No TEST rerun or test-driven hyperparameter choice, no quality-gate relaxation, no INT4/deployment acceptance claim. Source labels punctuation silver, not human/audio gold.

Two changes (ASR focus and original ordering) differ from v7; no matched-budget general-domain control. One-epoch result is exploratory. Keep allweights/fulloptimizerRNG/raw/pins/traces SSH, not backed up to Drive. Windows only bounded source/status/report/review <=5MiB. Preserve mounts/reserves and other jobs.


## Verified one-epoch result 07 November

Budget completed exactly333315scoredpoints/420tasks/22147updates; global3753320. Sourceword/label/mask/gapcounts and parentmodel27AdamW/allRNG inheritance verified. Actual1600traininginputs/100updates replayed; 35270savedmonitorinferences independently audited with22canonicalcases, maxdelta5.960464477539063e-08/all decisions identical. Calibration curve selection recomputed; raw calibration predictions not separately saved. Final fullresume model/optimizer/RNG/order/cursor/lanes verified, parentv7paused3731173 unchanged. ReportSHA 7ef181e9a10c69a022f646782e3c241b3a67f2dc1d6a4ca19cdeb7dcece9d196.

Both predeclared threshold policies select0.1. ASR VI F110.4294->0.7407percent (17->1 TP outof269EOT), recall6.3197->0.3717percent. ASREN F139.0456->42.1308percent, precision52.3256->70.1613 but recall31.1419->30.1038. DailyVI F144.8529->6.7568, factoryVI45.2381->12.5; Englishdaily/factory also decline10.38/15.23pp. This setup is not adopted; originalv7 remains paused/default unchanged. One-full-epoch trial ended by its declared budget, not a plateau/quality stop. Preserve rejectedtrialweights/fulloptimizerRNG/source/proofs. No new TEST/translation scoring.

Posthoc diagnostics (not tuning): ASRVI median score at EOT0.005073->0.000449; rankingAUROC0.74383->0.72738 under different dynamicresetcontexts. No lowerthreshold selected/applied. Last1000updates contain3316points (VI2470/EN846) as lanes drain; schedule replay matches all22147updates/333315points. These observations do not establish a unique causal explanation. ASRfocus AND naturalordering changed fromv7; trial was requested after previously viewedTESTdiagnostics, so no claim of fresh sealedTESTconfirmation.

Allweights/checkpoints remainSSH, no Drivebackup, no Windowsmodel/fulltrace downloads. TrialCOMPLETE/reportedtrue; no autoextend/restart or general-v7resume without new user instruction. Review: review.html/user_summary.json.
