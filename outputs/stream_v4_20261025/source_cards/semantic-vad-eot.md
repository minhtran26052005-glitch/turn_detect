---
license: cc-by-4.0
task_categories:
- voice-activity-detection
language:
- de
- en
- es
- fr
- it
- ja
- ko
- ms
- pl
- pt
- ru
- ta
- th
- tr
- zh
tags:
- end-of-turn
- semantic-vad
- eot
- turn-detection
- telephony
- conversational
pretty_name: Semantic-VAD EOT
configs:
- config_name: all
  default: true
  data_files:
  - split: train
    path:
    - data/*.parquet
    - data/train/*-*.parquet
  - split: validation
    path: data/validation/*-*.parquet
  - split: test
    path: data/test/*-*.parquet
- config_name: en
  data_files:
  - split: train
    path:
    - data/ml-english-*.parquet
    - data/train/ml-english-*.parquet
  - split: validation
    path: data/validation/ml-english-*.parquet
  - split: test
    path: data/test/ml-english-*.parquet
- config_name: fr
  data_files:
  - split: train
    path:
    - data/ml-french-*.parquet
    - data/train/ml-french-*.parquet
  - split: validation
    path: data/validation/ml-french-*.parquet
  - split: test
    path: data/test/ml-french-*.parquet
- config_name: de
  data_files:
  - split: train
    path:
    - data/ml-german-*.parquet
    - data/train/ml-german-*.parquet
  - split: validation
    path: data/validation/ml-german-*.parquet
  - split: test
    path: data/test/ml-german-*.parquet
- config_name: it
  data_files:
  - split: train
    path:
    - data/ml-italian-*.parquet
    - data/train/ml-italian-*.parquet
  - split: validation
    path: data/validation/ml-italian-*.parquet
  - split: test
    path: data/test/ml-italian-*.parquet
- config_name: ja
  data_files:
  - split: train
    path:
    - data/ml-japanese-*.parquet
    - data/train/ml-japanese-*.parquet
  - split: validation
    path: data/validation/ml-japanese-*.parquet
  - split: test
    path: data/test/ml-japanese-*.parquet
- config_name: ko
  data_files:
  - split: train
    path:
    - data/ml-korean-*.parquet
    - data/train/ml-korean-*.parquet
  - split: validation
    path: data/validation/ml-korean-*.parquet
  - split: test
    path: data/test/ml-korean-*.parquet
- config_name: zh
  data_files:
  - split: train
    path:
    - data/ml-mandarin-*.parquet
    - data/train/ml-mandarin-*.parquet
  - split: validation
    path: data/validation/ml-mandarin-*.parquet
  - split: test
    path: data/test/ml-mandarin-*.parquet
- config_name: pl
  data_files:
  - split: train
    path:
    - data/ml-polish-*.parquet
    - data/train/ml-polish-*.parquet
  - split: validation
    path: data/validation/ml-polish-*.parquet
  - split: test
    path: data/test/ml-polish-*.parquet
- config_name: pt
  data_files:
  - split: train
    path:
    - data/ml-portuguese-*.parquet
    - data/train/ml-portuguese-*.parquet
  - split: validation
    path: data/validation/ml-portuguese-*.parquet
  - split: test
    path: data/test/ml-portuguese-*.parquet
- config_name: ru
  data_files:
  - split: train
    path:
    - data/ml-russian-*.parquet
    - data/train/ml-russian-*.parquet
  - split: validation
    path: data/validation/ml-russian-*.parquet
  - split: test
    path: data/test/ml-russian-*.parquet
- config_name: es
  data_files:
  - split: train
    path:
    - data/ml-spanish-*.parquet
    - data/train/ml-spanish-*.parquet
  - split: validation
    path: data/validation/ml-spanish-*.parquet
  - split: test
    path: data/test/ml-spanish-*.parquet
- config_name: th
  data_files:
  - split: train
    path:
    - data/ml-thai-*.parquet
    - data/train/ml-thai-*.parquet
  - split: validation
    path: data/validation/ml-thai-*.parquet
  - split: test
    path: data/test/ml-thai-*.parquet
- config_name: tr
  data_files:
  - split: train
    path:
    - data/ml-turkish-*.parquet
    - data/train/ml-turkish-*.parquet
  - split: validation
    path: data/validation/ml-turkish-*.parquet
  - split: test
    path: data/test/ml-turkish-*.parquet
- config_name: all_telephony
  data_files:
  - split: train
    path:
    - telephony/*.parquet
    - telephony/train/*-*.parquet
  - split: validation
    path: telephony/validation/*-*.parquet
  - split: test
    path: telephony/test/*-*.parquet
- config_name: en_telephony
  data_files:
  - split: train
    path:
    - telephony/ml-english-*.parquet
    - telephony/train/ml-english-*.parquet
  - split: validation
    path: telephony/validation/ml-english-*.parquet
  - split: test
    path: telephony/test/ml-english-*.parquet
- config_name: fr_telephony
  data_files:
  - split: train
    path:
    - telephony/ml-french-*.parquet
    - telephony/train/ml-french-*.parquet
  - split: validation
    path: telephony/validation/ml-french-*.parquet
  - split: test
    path: telephony/test/ml-french-*.parquet
- config_name: de_telephony
  data_files:
  - split: train
    path:
    - telephony/ml-german-*.parquet
    - telephony/train/ml-german-*.parquet
  - split: validation
    path: telephony/validation/ml-german-*.parquet
  - split: test
    path: telephony/test/ml-german-*.parquet
- config_name: it_telephony
  data_files:
  - split: train
    path:
    - telephony/ml-italian-*.parquet
    - telephony/train/ml-italian-*.parquet
  - split: validation
    path: telephony/validation/ml-italian-*.parquet
  - split: test
    path: telephony/test/ml-italian-*.parquet
- config_name: ja_telephony
  data_files:
  - split: train
    path:
    - telephony/ml-japanese-*.parquet
    - telephony/train/ml-japanese-*.parquet
  - split: validation
    path: telephony/validation/ml-japanese-*.parquet
  - split: test
    path: telephony/test/ml-japanese-*.parquet
- config_name: ko_telephony
  data_files:
  - split: train
    path:
    - telephony/ml-korean-*.parquet
    - telephony/train/ml-korean-*.parquet
  - split: validation
    path: telephony/validation/ml-korean-*.parquet
  - split: test
    path: telephony/test/ml-korean-*.parquet
- config_name: zh_telephony
  data_files:
  - split: train
    path:
    - telephony/ml-mandarin-*.parquet
    - telephony/train/ml-mandarin-*.parquet
  - split: validation
    path: telephony/validation/ml-mandarin-*.parquet
  - split: test
    path: telephony/test/ml-mandarin-*.parquet
- config_name: pl_telephony
  data_files:
  - split: train
    path:
    - telephony/ml-polish-*.parquet
    - telephony/train/ml-polish-*.parquet
  - split: validation
    path: telephony/validation/ml-polish-*.parquet
  - split: test
    path: telephony/test/ml-polish-*.parquet
- config_name: pt_telephony
  data_files:
  - split: train
    path:
    - telephony/ml-portuguese-*.parquet
    - telephony/train/ml-portuguese-*.parquet
  - split: validation
    path: telephony/validation/ml-portuguese-*.parquet
  - split: test
    path: telephony/test/ml-portuguese-*.parquet
- config_name: ru_telephony
  data_files:
  - split: train
    path:
    - telephony/ml-russian-*.parquet
    - telephony/train/ml-russian-*.parquet
  - split: validation
    path: telephony/validation/ml-russian-*.parquet
  - split: test
    path: telephony/test/ml-russian-*.parquet
- config_name: es_telephony
  data_files:
  - split: train
    path:
    - telephony/ml-spanish-*.parquet
    - telephony/train/ml-spanish-*.parquet
  - split: validation
    path: telephony/validation/ml-spanish-*.parquet
  - split: test
    path: telephony/test/ml-spanish-*.parquet
- config_name: th_telephony
  data_files:
  - split: train
    path:
    - telephony/ml-thai-*.parquet
    - telephony/train/ml-thai-*.parquet
  - split: validation
    path: telephony/validation/ml-thai-*.parquet
  - split: test
    path: telephony/test/ml-thai-*.parquet
- config_name: tr_telephony
  data_files:
  - split: train
    path:
    - telephony/ml-turkish-*.parquet
    - telephony/train/ml-turkish-*.parquet
  - split: validation
    path: telephony/validation/ml-turkish-*.parquet
  - split: test
    path: telephony/test/ml-turkish-*.parquet
- config_name: all_dialogue
  data_files:
  - split: train
    path: dialogue/train/*.parquet
  - split: validation
    path: dialogue/validation/*.parquet
  - split: test
    path: dialogue/test/*.parquet
- config_name: aixblock_ja
  data_files:
  - split: train
    path:
    - dialogue/train/aixblock_ja-*.parquet
  - split: validation
    path: dialogue/validation/aixblock_ja-*.parquet
  - split: test
    path: dialogue/test/aixblock_ja-*.parquet
- config_name: apptek
  data_files:
  - split: train
    path:
    - dialogue/train/apptek-*.parquet
  - split: validation
    path: dialogue/validation/apptek-*.parquet
  - split: test
    path: dialogue/test/apptek-*.parquet
- config_name: callhome_de
  data_files:
  - split: train
    path:
    - dialogue/train/callhome_de-*.parquet
  - split: validation
    path: dialogue/validation/callhome_de-*.parquet
  - split: test
    path: dialogue/test/callhome_de-*.parquet
- config_name: callhome_en
  data_files:
  - split: train
    path:
    - dialogue/train/callhome_en-*.parquet
  - split: validation
    path: dialogue/validation/callhome_en-*.parquet
  - split: test
    path: dialogue/test/callhome_en-*.parquet
- config_name: callhome_es
  data_files:
  - split: train
    path:
    - dialogue/train/callhome_es-*.parquet
  - split: validation
    path: dialogue/validation/callhome_es-*.parquet
  - split: test
    path: dialogue/test/callhome_es-*.parquet
- config_name: callhome_ja
  data_files:
  - split: train
    path:
    - dialogue/train/callhome_ja-*.parquet
  - split: validation
    path: dialogue/validation/callhome_ja-*.parquet
  - split: test
    path: dialogue/test/callhome_ja-*.parquet
- config_name: callhome_zh
  data_files:
  - split: train
    path:
    - dialogue/train/callhome_zh-*.parquet
  - split: validation
    path: dialogue/validation/callhome_zh-*.parquet
  - split: test
    path: dialogue/test/callhome_zh-*.parquet
- config_name: imda_part3
  data_files:
  - split: train
    path:
    - dialogue/train/imda_part3-*.parquet
  - split: validation
    path: dialogue/validation/imda_part3-*.parquet
  - split: test
    path: dialogue/test/imda_part3-*.parquet
- config_name: imda_part4
  data_files:
  - split: train
    path:
    - dialogue/train/imda_part4-*.parquet
  - split: validation
    path: dialogue/validation/imda_part4-*.parquet
  - split: test
    path: dialogue/test/imda_part4-*.parquet
- config_name: imda_part5
  data_files:
  - split: train
    path:
    - dialogue/train/imda_part5-*.parquet
  - split: validation
    path: dialogue/validation/imda_part5-*.parquet
  - split: test
    path: dialogue/test/imda_part5-*.parquet
- config_name: imda_part6
  data_files:
  - split: train
    path:
    - dialogue/train/imda_part6-*.parquet
  - split: validation
    path: dialogue/validation/imda_part6-*.parquet
  - split: test
    path: dialogue/test/imda_part6-*.parquet
- config_name: magicdata_ja
  data_files:
  - split: train
    path:
    - dialogue/train/magicdata_ja-*.parquet
- config_name: magicdata_zh
  data_files:
  - split: train
    path:
    - dialogue/train/magicdata_zh-*.parquet
  - split: test
    path: dialogue/test/magicdata_zh-*.parquet
- config_name: magichub_ko
  data_files:
  - split: train
    path:
    - dialogue/train/magichub_ko-*.parquet
  - split: validation
    path: dialogue/validation/magichub_ko-*.parquet
  - split: test
    path: dialogue/test/magichub_ko-*.parquet
- config_name: seamless_improvised
  data_files:
  - split: train
    path:
    - dialogue/train/seamless_improvised-*.parquet
  - split: validation
    path: dialogue/validation/seamless_improvised-*.parquet
  - split: test
    path: dialogue/test/seamless_improvised-*.parquet
- config_name: seamless_naturalistic
  data_files:
  - split: train
    path:
    - dialogue/train/seamless_naturalistic-*.parquet
  - split: validation
    path: dialogue/validation/seamless_naturalistic-*.parquet
  - split: test
    path: dialogue/test/seamless_naturalistic-*.parquet
- config_name: spokenwoz
  data_files:
  - split: train
    path:
    - dialogue/train/spokenwoz-*.parquet
  - split: validation
    path: dialogue/validation/spokenwoz-*.parquet
  - split: test
    path: dialogue/test/spokenwoz-*.parquet
- config_name: synthetic_calls
  data_files:
  - split: train
    path:
    - dialogue/train/synthetic_calls-*.parquet
  - split: validation
    path: dialogue/validation/synthetic_calls-*.parquet
  - split: test
    path: dialogue/test/synthetic_calls-*.parquet
- config_name: aixblock_ja_telephony
  data_files:
  - split: train
    path:
    - dialogue_telephony/train/aixblock_ja-*.parquet
  - split: validation
    path: dialogue_telephony/validation/aixblock_ja-*.parquet
  - split: test
    path: dialogue_telephony/test/aixblock_ja-*.parquet
- config_name: apptek_telephony
  data_files:
  - split: train
    path:
    - dialogue_telephony/train/apptek-*.parquet
  - split: validation
    path: dialogue_telephony/validation/apptek-*.parquet
  - split: test
    path: dialogue_telephony/test/apptek-*.parquet
- config_name: imda_part4_telephony
  data_files:
  - split: train
    path:
    - dialogue_telephony/train/imda_part4-*.parquet
  - split: validation
    path: dialogue_telephony/validation/imda_part4-*.parquet
  - split: test
    path: dialogue_telephony/test/imda_part4-*.parquet
- config_name: imda_part5_telephony
  data_files:
  - split: train
    path:
    - dialogue_telephony/train/imda_part5-*.parquet
  - split: validation
    path: dialogue_telephony/validation/imda_part5-*.parquet
  - split: test
    path: dialogue_telephony/test/imda_part5-*.parquet
- config_name: imda_part6_telephony
  data_files:
  - split: train
    path:
    - dialogue_telephony/train/imda_part6-*.parquet
  - split: validation
    path: dialogue_telephony/validation/imda_part6-*.parquet
  - split: test
    path: dialogue_telephony/test/imda_part6-*.parquet
- config_name: magicdata_ja_telephony
  data_files:
  - split: train
    path:
    - dialogue_telephony/train/magicdata_ja-*.parquet
- config_name: magicdata_zh_telephony
  data_files:
  - split: train
    path:
    - dialogue_telephony/train/magicdata_zh-*.parquet
  - split: test
    path: dialogue_telephony/test/magicdata_zh-*.parquet
- config_name: magichub_ko_telephony
  data_files:
  - split: train
    path:
    - dialogue_telephony/train/magichub_ko-*.parquet
  - split: validation
    path: dialogue_telephony/validation/magichub_ko-*.parquet
  - split: test
    path: dialogue_telephony/test/magichub_ko-*.parquet
- config_name: seamless_improvised_telephony
  data_files:
  - split: train
    path:
    - dialogue_telephony/train/seamless_improvised-*.parquet
  - split: validation
    path: dialogue_telephony/validation/seamless_improvised-*.parquet
  - split: test
    path: dialogue_telephony/test/seamless_improvised-*.parquet
- config_name: seamless_naturalistic_telephony
  data_files:
  - split: train
    path:
    - dialogue_telephony/train/seamless_naturalistic-*.parquet
  - split: validation
    path: dialogue_telephony/validation/seamless_naturalistic-*.parquet
  - split: test
    path: dialogue_telephony/test/seamless_naturalistic-*.parquet
- config_name: synthetic_calls_telephony
  data_files:
  - split: train
    path:
    - dialogue_telephony/train/synthetic_calls-*.parquet
  - split: validation
    path: dialogue_telephony/validation/synthetic_calls-*.parquet
  - split: test
    path: dialogue_telephony/test/synthetic_calls-*.parquet
---

# Semantic-VAD EOT

End-of-turn (semantic VAD) turns, schema-compatible with
[`livekit/eot-bench-data`](https://huggingface.co/datasets/livekit/eot-bench-data).

Each row is one user **turn**: an audio clip (16 kHz **mp3**), its `words`, and ordered `silence_spans`. Per the
eot-bench convention the **last** silence span is the true end-of-turn (`eot`); earlier spans are mid-turn `hold`
pauses (labels positional, not stored). Timestamps are on the decoded-audio axis, and the decoded length equals
`duration`.

```python
from datasets import load_dataset, Audio
ds = load_dataset("Scicom-intl/semantic-vad-eot", "en")                  # ('train', 'validation', 'test')
tel = load_dataset("Scicom-intl/semantic-vad-eot", "en_telephony")       # the same turns through a phone channel
ds = ds.cast_column("audio", Audio(decode=False))
```

## Configs

- **`all`** (default) and 13 languages `en, fr, de, it, ja, ko, zh, pl, pt, ru, es, th, tr`: read speech from
  `AAdonis/multilingual_audio_alignments`, one sentence per turn, **end-of-turn silence de-padded** (below).
- **`all_telephony`** and **`<lang>_telephony`**: the same de-padded turns through a call-centre channel simulation
  (below), for training telephony detectors without telephony transcripts.
- **`all_dialogue`** and one config per conversational source (below): turns with observed ends.

Read-speech splits: for every language, all shards except the last form **train**; the last shard was split 50/25/25
into extra **train** / **validation** / **test** (shuffled, seed 42). De-padding and the telephony copy keep every row
in its split.

## De-padded end-of-turn silence (2026-10-01)

Until revision `80fc04131a1122b67859f9549203ac0e4b87a09d` the end-of-turn silence of these configs was recorded audio for its first part and exact
digital zeros after it: padding added at build time to reach 0.5 s when the source clip ended sooner, while holds are
never padded. A model can learn "exact zeros mean the turn is over". Every row now keeps only the silence that was
recorded: the last span (and `duration`) ends at the first run of at least 50 ms of exact zeros after its start,
rounded down to a whole mp3 frame (36 ms), and the mp3 stream is cut after that frame, so the audio up to the cut is
bit-identical to before (checked on every row) and nothing was re-encoded. Rows left with less than 0.2 s of recorded
end-of-turn silence (the shortest end of turn LiveKit's benchmark scores) were dropped. Holds, words and messages are
unchanged. Italian and Mandarin had almost no recorded trailing silence, so little of them is left.

| config | rows before | after de-padding | after the speech filter | kept | eot silence p50 before, s | after, s |
|---|---:|---:|---:|---:|---:|---:|
| `en` | 500,000 | 280,688 | 273,806 | 54.8 % | 0.525 | 0.525 |
| `fr` | 500,000 | 93,247 | 77,952 | 15.6 % | 0.521 | 0.511 |
| `de` | 500,000 | 20,780 | 12,639 | 2.5 % | 0.521 | 0.347 |
| `it` | 500,000 | 572 | 30 | 0.0 % | 0.52 | 1.015 |
| `ja` | 500,000 | 164,154 | 100,068 | 20.0 % | 0.525 | 0.517 |
| `ko` | 500,000 | 184,199 | 170,952 | 34.2 % | 0.525 | 0.631 |
| `zh` | 500,000 | 5,048 | 4,826 | 1.0 % | 0.521 | 0.699 |
| `pl` | 276,461 | 52,973 | 45,627 | 16.5 % | 0.521 | 0.293 |
| `pt` | 91,384 | 64,594 | 58,275 | 63.8 % | 0.527 | 0.525 |
| `ru` | 500,000 | 61,046 | 41,827 | 8.4 % | 0.521 | 0.419 |
| `es` | 500,000 | 160,353 | 131,452 | 26.3 % | 0.521 | 0.329 |
| `th` | 500,000 | 213,833 | 208,639 | 41.7 % | 0.529 | 0.701 |
| `tr` | 500,000 | 25,670 | 17,215 | 3.4 % | 0.521 | 0.441 |
| **total** | **5,867,845** | **1,327,157** | **1,143,308** | **19.5 %** | | |

The telephony configs hold the same turns (1,143,308 rows).

Per-language build manifests: `meta/depad/<lang>.json`. The padded version stays available:
`load_dataset("Scicom-intl/semantic-vad-eot", "en", revision="80fc04131a1122b67859f9549203ac0e4b87a09d")`.

## Conversational subsets (`dialogue/`, 2026-10)

Turns from real two-party conversations where every speaker is on their own channel (stereo telephone calls, or one
recording per participant), so the end of a turn is **observed**: the other party took the floor. The rule is the one
that built `Scicom-intl/semantic-vad-eot-emgs`: Silero VAD per channel; a pause is a `hold` unless the other party
spoke for at least 0.5 s inside it (shorter = backchannel), lasted 3 s or more, or the call ended; a speaker
interrupted (the other already talking more than 0.5 s before they stopped) is dropped; the clip is the speaker's own
channel with 0.3 s lead-in and up to 0.5 s of trailing silence, which must be quiet all the way. Echo is measured per
channel (its energy while it is silent, correlated with the other channel's at -100 to +600 ms): where the other voice is
audible, speech segments that are mostly the other party are dropped and the labelled silence stops 50 ms before the
reply; a channel with strong echo is not used. `words` and `messages` are empty; the speaker,
call, how the turn ended and the clip's offset in the call are in `meta/dialogue/<config>.parquet`, and the build
statistics in `meta/dialogue/<config>.json`. Splits are 90 / 5 / 5 by call. Wideband sources also have a
`<config>_telephony` copy through the channel simulation described below. Licences are the sources' own; check them
before any commercial use.

`synthetic_calls` is the exception to "observed": synthetic Malaysian call-centre dialogues whose agent and customer
turns were voiced by voice conversion and mixed into one clip with room-tone gaps. Its turn ends are known from the
renderer's timeline (refined by VAD inside each turn's slot), every turn of either role is used, the labelled silence
always stops before the next voice, and its gaps are synthetic rather than human timing.

| config | lang | source | licence | calls | hours | turns | with a hold, % | telephony copy |
|---|---|---|---|---:|---:|---:|---:|---|
| `aixblock_ja` | ja | AIxBlock/Human-to-machine-Japanese-audio-call-center-conversations | CC BY-NC 4.0 | 1746 | 31.3 | 7,312 | 31.0 | yes |
| `apptek` | en | apptek-com/apptek_callcenter_dialogues | CC BY-SA 4.0 | 873 | 151.1 | 49,296 | 53.4 | yes |
| `callhome_de` | de | BrunoHays/callhome-telephony (TalkBank CallHome) | CC BY-NC-SA 4.0 | 120 | 58.1 | 33,796 | 29.8 | - |
| `callhome_en` | en | BrunoHays/callhome-telephony (TalkBank CallHome) | CC BY-NC-SA 4.0 | 176 | 76.5 | 44,156 | 31.2 | - |
| `callhome_es` | es | BrunoHays/callhome-telephony (TalkBank CallHome) | CC BY-NC-SA 4.0 | 131 | 41.9 | 28,448 | 29.9 | - |
| `callhome_ja` | ja | BrunoHays/callhome-telephony (TalkBank CallHome) | CC BY-NC-SA 4.0 | 120 | 49.3 | 38,496 | 29.3 | - |
| `callhome_zh` | zh | BrunoHays/callhome-telephony (TalkBank CallHome) | CC BY-NC-SA 4.0 | 140 | 39.5 | 29,488 | 32.8 | - |
| `imda_part3` | en | recursal/reprocessed_singapore_national_speech_corpus (IMDA NSC) | Singapore Open Data Licence | 209 | 503.7 | 199,561 | 40.8 | - |
| `imda_part4` | zh+ta+ms+en | recursal/reprocessed_singapore_national_speech_corpus (IMDA NSC) | Singapore Open Data Licence | 484 | 514.6 | 217,452 | 41.2 | yes |
| `imda_part5` | en | recursal/reprocessed_singapore_national_speech_corpus (IMDA NSC) | Singapore Open Data Licence | 2751 | 937.8 | 148,072 | 42.7 | yes |
| `imda_part6` | en | recursal/reprocessed_singapore_national_speech_corpus (IMDA NSC) | Singapore Open Data Licence | 5965 | 1068.9 | 341,930 | 53.4 | yes |
| `magicdata_ja` | ja | MagicDataTech/Japanese_Spontaneous_Conversation_Training_Dataset | non-commercial (card) | 10 | 5.2 | 3,742 | 34.0 | yes |
| `magicdata_zh` | zh | MagicDataTech/multi-stream-spontaneous-conversation-training-datasets_chinese | academic use only (card) | 27 | 5.2 | 763 | 68.1 | yes |
| `magichub_ko` | ko | MagicHub/korean-conversational-speech-corpus | non-commercial (README) | 11 | 2.7 | 1,296 | 38.8 | yes |
| `seamless_improvised` | en | pengyizhou/seamless-interaction-audio-only (Meta Seamless Interaction) | CC BY-NC 4.0 | 17809 | 1343.8 | 526,281 | 38.5 | yes |
| `seamless_naturalistic` | en | pengyizhou/seamless-interaction-audio-only (Meta Seamless Interaction) | CC BY-NC 4.0 | 46810 | 2689.7 | 840,887 | 41.0 | yes |
| `spokenwoz` | en | ssz1111/SpokenWOZ-* | CC BY-NC 4.0 | 5700 | 251.5 | 193,214 | 31.0 | - |
| `synthetic_calls` | ms+en+ta+zh | Scicom-intl/Synthetic-User-Turn-TTS (dialogue_calls, hard_entities_calls, hard_entities_tamil_calls) | synthetic text; voices keep upstream licences | 9039 | 65.5 | 24,150 | 65.8 | yes |

## End-of-turn speech filter (2026-10-01)

A last pass dropped every turn whose labelled end-of-turn silence contains speech by **both** Silero VAD (threshold 0.5,
no padding) and WebRTC VAD (mode 3): at least 0.1 s each after the silence's first 100 ms, the part a smoothed detector's
decay after the last syllable cannot reach. On the telephony set `Scicom-intl/semantic-vad-eot-emgs`, whose turn ends
are observed, the rule fires on 0.5 % of turns; on the conversational configs here on 0.07-1.1 %, except the four IMDA
parts (2-5 %); on the de-padded read speech on far more (Japanese 39 %, German 39 %, French 16 %, English 2.5 %). There it is the alignment: in 93-99.5 % of
the flagged read turns the speech begins before the labelled end and runs on for a median 0.3-0.7 s, a last word the
aligner ended early or missed. The decision is made on the clean audio; telephony twins and sidecars lose the same ids.
Each manifest records `eot_speech_filter` (judged, dropped).

### Label check

![Share of labelled end-of-turn silences in which Silero VAD and WebRTC VAD hear speech after their first 100 ms, per config, before the speech filter](https://huggingface.co/datasets/Scicom-intl/semantic-vad-eot/resolve/main/assets/label_check.png)

Before the filter, the share of each config's labelled end-of-turn silences in which each detector hears at least 0.1 s
of speech after the first 100 ms: 600 sampled turns per config, or an unfiltered rebuild of 300 random calls for the
configs the builder already filters. Every conversational config except the four IMDA parts is at or below 2.7 % by
Silero. The IMDA recordings (each speaker on their own phone or telephone line) sit only 20-25 dB above their noise
floor, and what the detectors hear there is the last syllable fading into it (a median 0.12 s, starting at the label,
equally common whether the other party answers at once or not at all), not a missed turn end or the other party. The
de-padded read speech is the weak set: there the aligner ended the last word early or missed one.

## Telephony copies

Each turn through `semvad.callcentre.CallCentreDegrader` (a port of the call-centre profile of the Sidon
speech-restoration project), one draw per turn seeded from its `id`:

1. 10 %: only the measured response of a real phone line (about 3.4 kHz brick wall, gentle tilt, steep high-pass);
2. otherwise, caller-side background noise from
   [`Scicom-intl/callcentre-noise48k`](https://huggingface.co/datasets/Scicom-intl/callcentre-noise48k) with 85 %
   probability (office, street, ambient, keyboard, cafe, car, domestic, hvac, phone; no babble or hold music), then the
   channel: high-pass 200-350 Hz, an 8 / 11.025 / 12 / 16 kHz bottleneck with a low-pass at the band ceiling, GSM 06.10
   (8 kHz) / G.711 mu-law / no codec, 16-40 kbps mp3, line noise, and the measured line response with 50 % probability;
3. speech level set to -36 to -18 dBFS.

Noise SNR and the band ceiling are drawn from quantile tables calibrated so the chain reproduces the SNR and bandwidth
measured on real call-centre call legs (median band ceiling about 3.2 kHz, SNR mostly 33-55 dB); 15 % of turns use
harsher hand-set ranges instead. Left out on purpose because they would change what the labels say: cross-talk and
babble (a voice inside a labelled pause), packet loss (digital zeros inside pauses) and jitter or frame repeats (they
move timestamps). Every stage keeps the time axis, and the re-encode drops the first 1105 samples (LAME's decoder
delay) so the decoded audio lines up exactly: each telephony row has the same `id`, `words`, `silence_spans` and
`duration` as its twin, and the first 69 ms of lead-in are codec start-up. The parameters drawn for each turn are in
`meta/telephony/<same file name>` and `meta/dialogue/<config>_telephony.parquet` (`id`, `trace`).

## Known caveat: read-speech holds

Read-speech holds are gaps between aligned words; in several languages a third or more of them contain speech by Silero
or WebRTC VAD (breaths, short words the aligner merged). End-of-turn silences are the cleaner label.

## Removed: the five Malaysian subsets (2026-10-01)

`ms_dialects`, `ms_imda`, `ms_malaysian`, `ms_parliament` and `ms_science_english` were removed from `main`. Their turns
were cut from long monologue recordings at sentence-final punctuation, so their "end of turn" is a sentence boundary
where the same speaker usually keeps talking, and a label study on decoded audio (energy at the label edge, Silero VAD
and WebRTC VAD) found speech inside 18-65 % of their end-of-turn silences past the first 100 ms. They are still
available at revision `379a8935e3ef4e323b5468fc44510b3f9f1fb02a`.
