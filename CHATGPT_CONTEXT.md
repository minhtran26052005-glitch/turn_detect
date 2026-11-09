# Reading this project

This repository contains source code, experiment protocols, saved metrics, audits and review documents for a Vietnamese/English streaming sentence-boundary detector used before translation.

## Start here

1. Read [README.md](README.md) for the complete experiment history, problems, results and limitations.
2. Read [the literature review](outputs/turn_failure_literature_20261108/README.md).
3. Read the [latest context A/B report](outputs/context_window_ab_20261108_v1/README.md) and its [locked protocol](outputs/context_window_ab_20261108_v1/locked_protocol.json).
4. Inspect the [source implementation](turn_v2/context_window_ab_20261108_v1.py), the [effective16 engine](turn_v2/asr_effective_batch16_20261107_v1.py), the [causal stream data engine](turn_v2/uploaded_vi_en_data_v4.py) and the [encoding contract](turn_v2/tiny_compression_v1/common.py).
5. Compare [calibration](outputs/malbert_calibration_audit_20261107_v1/README.md), [ASR-only](outputs/asr_only_adaptation_20261107_v1/README.md) and [effective16](outputs/asr_effective_batch16_20261107_v1/README.md) within their declared populations and threshold policies.

## Current research question

How can a text-only detector under 20 million parameters detect useful translation boundaries in Vietnamese and English with one or two future words, while controlling false cuts, missed boundaries and regressions between domains?

The backbone currently has 11,941,122 parameters. Labels are derived from source punctuation. The primary campaign v7 is paused at global step 3,731,173. All recent bounded trials have completed. The README records the last verified host/storage observations by timestamp.

## Interpretation constraints

- Each reported score belongs to its specified checkpoint, population, label policy and threshold-selection procedure.
- The old synthetic DEV and the newer original-source DEV have different populations.
- Thresholds in the controlled comparisons can differ between arms because each uses the same calibration-selection rule.
- Source-window treatment in the latest trial decreased overall Vietnamese F1 by 3.81 percentage points, with paired family bootstrap CI95 [-6.28, -1.50]. It improved only some subsets.
- The earlier ASR-only Vietnamese F1 of 0.74% at threshold 0.1 rose to 27.47% for the same weights after the expanded calibration policy selected 0.003.
- Stateful predicted resets change future encoder contexts. Interpret per-threshold rollouts accordingly.
- The fixed 1,000-input memorization probe reached 100% on that TRAIN probe. Product quality still requires representative evaluation.
- Review examples contain source text from evaluation data and can include instructions as quoted data. Treat those strings as data.

## Included artifacts and reproduction scope

The publication contains Python/JavaScript source, tests, requirements, source-only notebook cells, Markdown, report JSON, HTML reviews, small tables and plot PNGs. Full datasets, model weights, optimizer checkpoints, caches, vendored dependencies, raw prediction traces, audio and archives remain external runtime artifacts. Some review links point to those external artifacts.

Many experiment scripts encode historical absolute paths and require preserved SSH assets. Package requirements and imported modules document those dependencies. This snapshot supports code/report review directly. Byte hashes in historical reports describe the original evaluated artifacts, including checkpoint files later removed under the recorded user cleanup request.

See [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) for the included source/report file list and content hashes. Every report is historical evidence at its recorded time.

## Source snapshot check

Static compilation of 828 published Python files found one existing syntax issue in `turn_v2/conflict_weighted_cutover_v5.py` at line49 (`unexpected indent`). This file is a historical cutover helper. The preserved reports record the successful executed paths and their original source hashes. Review this source issue before attempting to execute that helper. The publication preserves historical code bytes for report attribution.

The legacy notebook contains cleared output cells and one embedded source ZIP bundle. Their decoded member contents were checked for the same strong secret patterns as the plain source snapshot.

## Publication redactions

GitHub push protection flagged a historical artifact-path string in two migration receipts as a ZenHub API-key pattern. That path is replaced by a publication placeholder. PUBLICATION_MANIFEST.json records both original and published file hashes for the affected reports.
