---
license: cc-by-4.0
---

# DataHive — Multilingual Conversations

Unscripted conversations in English, Hindi, Indonesian, Russian, and Vietnamese between people who already know each other. Shared to support research on natural conversational voice models.

## Why These Conversations

Natural conversation depends on timing: when to respond, brief acknowledgments, pauses, and interruptions. People who know each other share context, leave thoughts unfinished, and switch turns without explicit signals. These recordings capture that.

## How We Collect

Participants call people they know through the Hive Calls app ([Google Play](https://play.google.com/store/apps/details?id=app.dhive.calls), [App Store](https://apps.apple.com/app/hive-calls/id6812719492)). They choose their own topics, with no scripts. Conversations take place in everyday environments and are recorded through LiveKit, with each participant saved separately using Opus compression in an Ogg container. These recordings are converted to stereo MP3 for this sample. Participants are paid for qualifying conversation time.

## What’s in This Sample

| Detail | Value |
| --- | --- |
| Languages | English (`en`), Hindi (`hi`), Indonesian (`id`), Russian (`ru`), Vietnamese (`vi`) |
| Conversations | 61 |
| Speakers | 80 unique speakers |
| Total duration | 19h 31m 47s |
| Conversation length | 2m 34s–58m 57s |
| Audio | Stereo MP3 · 48 kHz · 128 kbps |
| Channels | Caller left, callee right, on a shared timeline |
| Metadata | JSONL, one record per conversation |
| License | CC BY 4.0 |

Metadata per conversation:

- topic
- `language`: conversation language code (`en`, `hi`, `id`, `ru`, or `vi`)
- `languages`: detected language codes from the original export
- duration
- caller and callee country (from phone number)
- turns per minute
- median gap between turns
- speech balance
- overlap

Transcripts are not included.

## Files

```text
README.md
preview/metadata.jsonl
preview/en_000001.mp3
preview/hi_000001.mp3
preview/id_000001.mp3
preview/ru_000001.mp3
preview/vi_000001.mp3
...
```

Install the Hugging Face audio dependencies, then load the dataset from the extracted folder:

```sh
python -m pip install "datasets[audio]"
```

```python
from datasets import load_dataset

ds = load_dataset("audiofolder", data_dir="preview")
```

## Quality and Selection

Every conversation passes QA review and automated checks for:

- **Audio:** noise, signal-to-noise ratio, clipping, silence.
- **Dynamics:** both speakers participate, turn-taking, overlap, interruptions, gaps.
- **Content:** the same language on both tracks, topic, explicit content.

Conversations are at least two minutes long. Repeat appearances by the same speaker are limited. Labels and measurements are automatic.

## Consent and Privacy

Participants explicitly consent to recording, sharing with customers, and AI model training. Conversations flagged for personal information are excluded. Metadata contains no names, phone numbers, or account IDs. Audio and transcript redaction is available on request.

## Custom Collections

We collect conversations for voice-model labs by language, accent, and requirements. Deliverables can include separate speaker tracks, transcripts, and quality metadata. On request, we can arrange collection of audio captured on-device before lossy compression, delivered as PCM WAV or FLAC. Contact [contact@datahive.ai](mailto:contact@datahive.ai).

## License

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Free for research and commercial use, including model training, with attribution to DataHive.
