# Translation sentence boundary — v15c

## Input and target contract

The model receives **punctuation-free text only**: left context plus one or two following words. The punctuation from the crawled source is retained solely as supervision metadata. A sentence-final period is labeled EOT; other terminal marks are also EOT. Internal comma-like punctuation is labeled NEOT. Qwen is not used.

The period is therefore a strong training/evaluation target, not a hard-coded inference rule: punctuation is absent from model input, so the classifier can miss a period boundary. Reports must show terminal-period EOT recall separately for one-word and two-word lookahead, alongside accuracy and EOT F1. These crawl labels come from published prose and do not establish real conversation/audio performance or human gold.

## Runtime

- Colab campaign: `turn_v2.translation_segmenter_v15c`
- Parent PID/start ticks: `336572 / 8156693`
- Child PID/start ticks at last check: `336573 / 8156712`
- Source SHA-256: `7bd4dd908d114add5826b8c9ec06dcc0ca13a28f0fe56b0d260411cc2c0fbbea`
- Last checked: `2026-10-26T03:53:53Z`; cycle 8 collecting, no student training confirmed yet.
- GPU: NVIDIA L4; 0% utilization and 3 MiB used during the status sample. Remote free disk: about 154 GiB.
- Corpus was copied from the stopped v15b run using SQLite backup; see `carryover_receipt.json` when present.

The old v15 status file said RUNNING even though its recorded parent/child PID+start-tick identities were absent. The launch audit recorded this as stale status and required all old campaign PIDs to be inactive before launching v15c. The old status file itself was preserved.

## Data and review

The output folder will contain the metadata snapshot, SQLite corpus and dev prediction traces after a completed cycle. The review page lists punctuation as a separate label-only column; it is not a model feature. Current copied counts are provisional and the first new crawl cycle is still running. Training is deferred until each language has enough examples per label and lookahead horizon in both train and dev.

Open `review.html` after the first v15c snapshot. Do not interpret a punctuation-derived prose score as conversational EOT gold.


## Latest inspection snapshot

Cycle 11 snapshot SHA-256: `9ee164580d91443c28f1b9dbcd6d81d7885a01c0d5faa4a00d2e9e259bb94472` (334,620 bytes). It contains 1,872 examples: VI 320, EN 380, ZH 1,172. A Unicode punctuation-category audit found zero punctuation characters in every `context` and future-text field; target punctuation is stored separately. The review page is `review.html`.

At the latest live status, the worker was processing cycle 12; it added 4 pages per language (VI 100 examples, EN 97, ZH 394). Cycle-12 data is newer than the downloaded cycle-11 snapshot. Training remained deferred: the EOT/NEOT counts per lookahead horizon are not yet sufficient in dev, especially Vietnamese. No accuracy/F1 result or trained checkpoint exists yet.


## Latest verified state

Cycle 12 snapshot SHA-256: `053b515def8cc04e2c44c5b9ef6c3346e056c7648c608e2d439af6f3a68a7fa6` (429,763 bytes). It contains 2,463 examples: VI 420, EN 477, ZH 1,566. The review page was rebuilt from this snapshot. A complete Unicode punctuation audit found zero punctuation characters in the model `context` and future-text fields.

The live worker was PID 336572 (start ticks 8156693), with child PID 337961 (start ticks 8187352), both verified as v15c. It had advanced to cycle 13 crawling. Cycle 12 training was deferred in all three languages because at least one class/lookahead slice lacked enough dev samples: VI 360/42 total train/dev, EN 343/106, ZH 1,062/326. No accuracy/F1 or trained checkpoint is reported at this point.
