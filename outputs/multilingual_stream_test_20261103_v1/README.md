User-requested dialogue diagnostic completed on 2026-11-03.

Frozen main student global741935/branch255700, full optimizer104entries/RNG/sampler preserved on SSH. Shared threshold0.34 selected on the same frozen checkpoint rolling DEV before any test inference. Main training continued throughout; post-test correct PID2827094/ticks50410807 reached global742565/branch256330 with no error. Evaluator399406/ticks58647694 exited normally.

Inputs: the previously exposed100VI+100EN Qwen paragraphs plus100lockedVI-to-ENpairs. All source dialogue IDs excluded fromTRAIN/DEV; exact full-paragraph collisions0, family overlap49TRAIN/2DEV. This is a previously seen synthetic punctuation diagnostic, not independent human/audio or spokenv9 evaluation. Question/exclamation endpoints unscored. Test never selects threshold/model or changes training.

With lookahead2 and reset only on predictedEOT: monoVI TP140/FP17/FN15, wrongcuts17/157=10.83percent, EOTF189.74percent; monoEN TP246/FP9/FN3, wrongcuts9/255=3.53percent, F197.62percent. MixedVI TP218/FP16/FN15, F193.36percent; mixedEN same asmonoEN. Aggregate mixed25wrongcuts/489scoredmodelcuts,18misses/482referenceEOT. Language-switch predictions are not language-identification accuracy.

Compared with the verified previous test515435, ownDEVthreshold0.53: monoVI wrongcuts20/154->17/157 andFN21->15; ENwrongcuts8/250->9/255 andFN7->3. Same inputs/pair order; old traces reused without rescoring the old model. No selection from these results.

All35244inferences independently audited for reference labels/confusion counts, causal context, only1/2futurewords, predictedEOTresets and word conservation; all100mixedstreams conserve words. Fullpin/weights/traces SHA verified after completion. See completion_verification.json,post_test_remote_verification.json,report.json and review.html. Only small reports downloaded toWindows; frozen checkpoints/traces/optimizerRNG retained onSSH and not backed up toDrive.

ReportSHA256: 211ff9272be93f72b2baec194b033063749bfe32593b9c1a39cd11dcd7c80ce8
FullpinSHA256: 675592e8ea49bddb7166511e92d165703a66941185471932a81904171279d48b
WeightsSHA256: aeccca2ac98b3d2e5f3199d1b4e3a1165755d99285e8643808a2e993ba53f89b
EvaluatorSHA256: 1f1729f1f79e43ad7b1d03533e0e086d2b04ae2b968d17dde45a0ba472122c17
