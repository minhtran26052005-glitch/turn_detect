# Text-only sentence segmentation for translation

This campaign replaces the v14 Qwen semantic-judge path. It trains separate Vietnamese, English, and Chinese sentence-boundary models with one- or two-word lookahead. All collection and training run in the connected Colab runtime.

## Input and labels

- The model input contains only words: the text before a candidate boundary and the next 1 or 2 words. Punctuation characters are stripped from both input fields.
- Sentence-final punctuation in the crawled source provides EOT supervision. The punctuation mark is stored as `target_punctuation` metadata, outside the model input.
- Internal punctuation such as commas provides NEOT examples. Evaluation separately reports terminal-period EOT recall for each lookahead length and internal-punctuation false-END behavior.
- For Chinese, the lookahead units are `jieba` word tokens; English and Vietnamese use the campaign's word tokenization rules.

Punctuation-derived web prose is a useful first training signal, but it is not conversational, speech, or independent human gold. Model accuracy on it does not certify translation quality in production. The period target is evaluated explicitly; because the model does not see punctuation, a model prediction is not a deterministic guarantee that every period will be emitted as END.

## Data and review

The Colab worker crawls public Wikipedia pages through the MediaWiki API with request spacing and retry/backoff. Each page URL, source text hash, and stable page-grouped 80/10/10 split are stored in `corpus.sqlite`. Test pages are never used for checkpoint selection. Review the current text inputs and target metadata in `review.html`; prediction fields remain pending until there are enough examples to train and score each language.

At the cycle-4 snapshot, there are 1,366 examples total. VI has 232 train and 18 test examples but no dev pages yet; EN has 268 train and 40 dev; ZH has 592 train, 156 dev, and 60 test. For every language, training waits until every label-by-lookahead group has at least 100 examples in both train and dev. Thus no model training or quality score is confirmed yet.

## Runtime and recovery

- Colab parent PID 331597, start ticks `8040478`; the latest verified child was the cycle-5 collector, PID 332796, start ticks `8067127`.
- The latest sample showed an NVIDIA L4 at 0% utilization, 3 MiB / 23,034 MiB, with about 154 GiB free on Colab disk.
- Source SHA-256: `f1f517e7f271a9455eaaa9617f0f08a6e22ef1fe0c17f768528c29569a4a7dbc`.
- Cycle-4 metadata snapshot SHA-256: `da7c22c4676448620c88a0726c3fca77b1541d88b9a123f581fd5f41ee21c787` (247,910 bytes). This snapshot contains metadata and data state only; it is not a full teacher/optimizer backup.
- The preceding v14 worker was stopped by request during cycle-46 Qwen collection; cycle-45 checkpoints remain historical artifacts and v14 is not restarted.

The filterable local review page is generated from the verified metadata snapshot and can be rebuilt with `python tools/report_translation_segmenter_v15.py`.
