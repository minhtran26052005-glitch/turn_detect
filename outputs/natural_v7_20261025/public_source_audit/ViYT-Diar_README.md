---
dataset_info:
  features:
  - name: audio
    dtype: audio
  - name: audio_id
    dtype: string
  - name: speaker
    list: string
  - name: timestamps_start
    list: float32
  - name: timestamps_end
    list: float32
  splits:
  - name: test
    num_bytes: 307676963
    num_examples: 100
  download_size: 307689902
  dataset_size: 307676963
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
license: apache-2.0
task_categories:
- voice-activity-detection
language:
- vi
tags:
- speaker-diarization
- Vietnamese
- Youtube
pretty_name: Vietnamese YouTube Speaker Diarization Dataset
size_categories:
- n<1K
---
# ViYT-Diar: Vietnamese YouTube Diarization Dataset

**ViYT-Diar** is a specialized, manually annotated audio dataset extracted from Vietnamese YouTube videos. It is exclusively designed to serve as a **high-quality test benchmark** for evaluating Speaker Diarization models on in-the-wild data.

## 🚀 How to Use

You can easily load this dataset and access the audio arrays along with their diarization timestamps using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Load the test split
dataset = load_dataset("tuanduy1612/ViYT-Diar", split="test")

# Inspect the first video's data
sample = dataset[0]
print(f"Video ID: {sample['audio_id']}")
print(f"Number of segments: {len(sample['speaker'])}")
print(f"First segment: Speaker {sample['speaker'][0]} from {sample['timestamps_start'][0]}s to {sample['timestamps_end'][0]}s")
print(f"Waveform: {sample['audio']['array']}")
print(f"Sampling rate: {sample['audio'][sampling_rate']}")
```
## 📊 Dataset Structure
**Data Instances**
Each `row` in the dataset represents one complete audio file (video) and contains lists of segments (timestamps and speakers) representing the diarization timeline.

- `audio`: The audio feature (contains the decoded waveform array, sampling rate, and local path). You can play this directly on the Hugging Face Dataset Viewer.

- `audio_id` (string): The original YouTube video ID or filename for reference and traceability.

- `speaker` (list of strings): A sequence of speaker identifiers for each segment (e.g., ["SPEAKER_00", "SPEAKER_01", ...]).

- `timestamps_start` (list of floats): A sequence of start times for each speech segment (in seconds).

- `timestamps_end` (list of floats): A sequence of end times for each speech segment (in seconds).

**Data Splits**
This dataset is designed specifically for evaluation purposes.

- *Test*: Contains exactly 100 audio files.

## 🌍 Dataset Creation
**Source Data**
The audio data was collected from public YouTube videos, focusing on a diverse range of Vietnamese content (such as vlogs, podcasts, and interviews) to represent real-world conversational scenarios.

**Annotation Process**
To guarantee precise and reliable ground-truth labels for model evaluation, all 100 audio files were manually annotated from scratch using Label Studio. The raw annotations were originally exported in the standard RTTM format before being processed into this dataset.

**Audio Preprocessing**
For optimal compatibility with speech-processing AI models, all audio files have been standardized to:

- Format: `.wav`

- Sampling Rate: `16 kHz`

- Channels: `Mono (1 channel)`

## Citation
```
@inproceedings{thang2025adapting,
  title={Adapting WavLM for Vietnamese Speaker Diarization in Real-world Conversations},
  author={Thang, Tuan-Duy and Nguyen, Van-Huy and Do, Tri-Nhan and Nguyen, Quoc-Khanh and Phan, Trung-Kien and Mac, Dang-Khoa},
  booktitle={2025 International Conference on Multimedia Analysis and Pattern Recognition (MAPR)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}
```