# Paused v7 detector TEST and end-only delivery

User requested temporarily pause training, test the actual final model and exclude one-sentence inputs from any whole-at-end delivery denominator. Trainer is STOPPED_WITH_CHECKPOINT and remains paused until user requests resume. Full checkpoint/27 AdamW states/all RNG/order/cursor/active conversation lanes preserved. No base reset, weight update or current/default replacement.

Frozen global 3731173, branch 500127, 11,941,122 FP32 parameters; threshold 0.1 chosen on the exact paused checkpoint and existing separate DEV calibration families before TEST. Full checkpoint SHA feb4bc2e6cbd4d8c7fa3148f49edb86500c898045a33cc1db8a334bcc03255a8; weights SHA e9687f933d77d5a6ed63a9b0225ca29b5e5d17200526e7ac1d00e535ecadba52.

Run original TEST source-only causal inference with one/two future words; audit every saved input/hash/reset/confusion/word conservation plus canonical numerical parity. Measure zero internal cuts on multi-sentence actual turns and separately whole streams. One-span turns excluded; stricter >=2 completed fullstop subgroup included. Actual-turn no-cut metric assumes caller flushes at turn end; main evaluator keeps rolling context across source turn/speaker boundaries and does not do that flush.

Replay only the new detector on the same 310 ASR-style translation-test messages. Qwen translator is not loaded or rerun and no new BLEU/chrF/real audio latency is measured. Each message has external final flush; never label that as model EOT. Period-only count gives 21 multi-span inputs/289 singles; a source-only locked format review removes 3 numbered/No. cases, leaving 18; one question-first case yields 19 when including question/exclamation terminals. Preserve both definitions and all original labels.

Source/cohort refinement is for reporting only, not model/threshold/loss/TRAIN selection. Known old TRAIN-family overlaps retained/tagged as before, with separate excluded-overlap subgroup. Labels punctuation silver, not human/audio gold. Keep old tests, raw sources, all weights/optimizer/RNG and traces. No Colab, no Windows weights/caches/fulltraces. All model data stay on SSH and have not been backed up to Drive.


## Verified result 07 November

Complete originalTEST579685inferences (297397k1/282288k2), every encodedinput/causalfuture/predictedreset/confusion/wordconservation/SHA audited. Canonical62cases maximum probability delta6.332993507385254e-08, all decisions identical. Independent source-turn census16groups and310message segmentation census passed; actual trainer still absent and final fullcheckpoint SHA unchanged. ReportSHA 97f7a9ab3abdae67591e0c51b1fb836a064457af5da521b91c2981ce86638482.

OriginalTESTk2 excludes10554one-span turns, leaving4446multi-span turns:1190nointernalmodelcut (26.7656percent), VI687/2069 (33.2044percent), EN503/2377 (21.1611percent). Another645multi-span turns have cuts but none match an eligiblefullstop; comma cuts may still be semantically acceptable. Strict >=2completedfullstops:863/3238 (26.6523percent). Withoutknownlegacyoverlap:1130/4233 (26.6950percent). These actual-turn metrics imply whole-turn delivery only if caller flushes at source-turn end; the original evaluator itself retains context across source turns.

Translationinputreplay310messages/3900detector decisions, sameASRwordatoms/causalinputs/chunkwordconservation verified. Rawperiod multi21/endonly9; format-cleaned period multi18/endonly6 (33.3333percent): VI2/3, EN4/15. Includequestion-first message ->19multi/endonly6 (31.5789percent), VI2/4, EN4/15. Thus do not use all310messages as the multi-sentence end-only denominator. Old saved predictions recounted only, no old model rerun: same refined18oldendonly10, refined19oldendonly11. No new translator output/BLEU/chrF/real audio latency.

WholeTEST punctuationF1 VI36.4451percent/EN46.6624percent; these scores include all eligible test gaps and are separate from multi-sentence emission denominators. The model has not been accepted/quantized/replaced as active. Training remains PAUSED_BY_USER_FOR_TEST; no auto resume. No change to model/loss/labels/TRAIN/DEV/thresholdpolicy from test.
