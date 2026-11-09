Kết quả train thật mới nhất: [bản thử 24/10/2026](EXPERIMENT_TRIAL_20261024.md), holdout accuracy **93,81%**, F1 EOT **90,62%**. Phần dưới là hồ sơ smoke ban đầu ngày 23/10, được giữ để đối chiếu lịch sử.

## Material Passport

- Origin Skill: academic-research-suite/experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-23T22:08:14+07:00
- Verification Status: UNVERIFIED
- Version Label: turn_detector_v2.0
- repro_lock: null

`UNVERIFIED` applies to the requested 90% quality target. The execution checks below passed; no full quality experiment or final holdout evaluation has been performed.

## Execution result

Status: **SMOKE_PASSED_TARGET_NOT_EVALUATED**.

- Colab GPU: NVIDIA L4, CUDA available, 22.03 GiB VRAM.
- Python 3.13.15, PyTorch 2.11.0+cu130, Transformers 5.20.1.
- Notebook: 11 code cells completed; outputs saved in `a.ipynb`.
- Tests: 18 passed locally and on Colab.
- Live crawl: 4 candidates with source provenance.
- Qwen: 4 candidates, 2 accepted silver, 2 rejected. Teacher revision recorded.
- Encoder smoke: 34 train records including 2 silver, 16 dev records, 3 optimizer steps. Gradient checkpointing and FP32 were enabled.
- Saved/reloaded checkpoint and ONNX inference completed.
- FP32 ONNX parity: 8 samples, maximum EOT probability difference `1.5056080071573774e-07`; argmax classes matched.
- Persistent authenticated WebSocket reused successfully for 3 successive cells.
- Final holdout remained unopened. Target achievement is explicitly false in the smoke report.
- Original model, original inference code, train.jsonl and test.jsonl hashes remained unchanged.

## Evidence

- `outputs/colab_notebook_test.json`: per-cell result and timing.
- `outputs/colab_v2_smoke/training_report.json`: training settings, source/data hashes and model revision.
- `outputs/colab_v2_smoke/qwen_labeling_report.json`: teacher revision and acceptance counts.
- `outputs/colab_v2_smoke/qwen_audit.jsonl`: both teacher responses per candidate.
- `outputs/colab_v2_smoke/onnx_parity_report.json`: export check.
- `outputs/colab_v2_smoke/runtime_summary.json`: remote paths and connection reuse.
- `outputs/colab_v2_completion_validation.json`: final artifact checks.

## Scope and remaining experiment

Run full training and select the candidate/threshold using dev only. A 90% result requires both accuracy and EOT F1 at least 0.90 on the frozen final holdout. Original labels and source grouping remain imperfect references; a stronger real-dialogue claim requires a separately reviewed dataset with original conversation IDs. Teacher labels remain silver. No quality gain is inferred from a model change or successful execution.
