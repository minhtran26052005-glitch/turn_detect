## Material Passport

- Origin Skill: academic-research-suite/experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-24
- Verification Status: UNVERIFIED
- Version Label: crawl_predict_v1.0
- repro_lock: null

The unverified item is the model-quality/RL-improvement claim. Execution checks passed; no RL update or accuracy claim was made.

## Execution result

Status: **CRAWL_PREDICT_PASSED_RL_NOT_RUN**.

- 27 tests passed on Windows and on the existing Colab L4 runtime.
- The original ONNX model predicted 12 newly crawled candidates locally: 7 EOT, 5 NEOT. These are prediction counts, not accuracy measurements.
- The appended notebook stage predicted 12 candidates using the previously tested V2 smoke checkpoint, explicitly marked as smoke inference.
- Qwen reviewed 4 selected candidates. Both prompt views returned NEOT for each, but confidence was 0.7 in one view and 0.95 in the other. All 4 were rejected by the unchanged two-view minimum-confidence rule of 0.9. No silver labels were forced into training.
- Existing notebook configuration and prior outputs were preserved. The new bootstrap, tests and crawl/predict cells were executed and their outputs saved.
- Original model, original inference code and raw train/test hashes remained unchanged. Holdout was not evaluated. No model weights were updated by this stage.

## Evidence

- `outputs/crawl_predict_20261024_legacy/results.html`: readable predictions from the original model.
- `outputs/crawl_predict_20261024_colab/results.html`: V2 smoke predictions and Qwen review status.
- `outputs/crawl_predict_20261024_colab/qwen/audit.jsonl`: both teacher outputs for the 4 cases.
- `outputs/colab_crawl_predict_notebook_test.json`: executed notebook cells.
- `outputs/crawl_predict_20261024_validation.json`: final checks and hashes.

The module provides streaming inference and a bounded review queue. A future RL experiment requires a separately specified reward and reference streaming endpoints; it has not been implemented or evaluated here.
