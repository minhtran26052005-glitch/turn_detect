"""Small unseen conversational EN/ZH timing pilot with strictly causal ASR.

Uses only official test audio; never feeds these rows to training or dev.
Labels come from the publisher's observed-floor construction, not punctuation.
"""
import hashlib
import io
import json
from pathlib import Path
import time

from faster_whisper import WhisperModel
import numpy as np
import pyarrow.parquet as pq
import requests
import soundfile as sf
from scipy.signal import resample_poly

REPO = 'Scicom-intl/semantic-vad-eot'
REV = '98d1445eab71e4084ccb838dbd87a205f6f2e293'
ASR_REV = '536b0662742c02347bc0e980a01041f333bce120'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    output = Path('data_v4/en_zh_natural_pause_20261025')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cases, sources, audio_cache = [], [], {}
    for lang in ['en', 'zh']:
        remote = f'dialogue/test/callhome_{lang}-00000.parquet'
        local = output / f'callhome_{lang}_test.parquet'
        response = requests.get(f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{remote}', timeout=90, stream=True)
        response.raise_for_status()
        with local.open('wb') as stream:
            for block in response.iter_content(1024 * 1024):
                stream.write(block)
        rows = pq.read_table(local, columns=['id', 'language', 'duration', 'silence_spans', 'audio']).to_pylist()
        eligible = [r for r in rows if r['language'] == lang and r['duration'] <= 90
            and len(r['silence_spans']) > 1 and r['silence_spans'][-1]['end']-r['silence_spans'][-1]['start'] >= .2
            and any(s['end']-s['start'] >= .2 for s in r['silence_spans'][:-1])]
        assert len(eligible) >= 10
        selected = eligible[:10]
        sources.append({'language': lang, 'config': 'callhome_'+lang, 'official_split': 'test',
            'parquet': remote, 'bytes': local.stat().st_size, 'sha256': sha(local), 'eligible_turns': len(eligible),
            'selected_turn_ids': [r['id'] for r in selected],
            'url': f'https://huggingface.co/datasets/{REPO}/blob/{REV}/{remote}',
            'license': 'CC BY-NC-SA 4.0 (CallHome source); evaluation research only; not added to training'})
        for i, row in enumerate(selected):
            data = row['audio']['bytes']
            assert data
            path = output / f'{lang}_{i:03d}.mp3'
            path.write_bytes(data)
            audio, rate = sf.read(io.BytesIO(data), dtype='float32', always_2d=True)
            assert audio.shape[1] == 1, 'Publisher turn clips must be isolated mono channels'
            if rate != 16000:
                audio = resample_poly(audio, 16000, rate, axis=0).astype(np.float32)
            audio_cache[str(path)] = audio[:, 0]
            spans = row['silence_spans']
            holds = [j for j, s in enumerate(spans[:-1]) if s['end']-s['start'] >= .2]
            # One hold and one final per turn, fixed before ASR or model scoring.
            for j in [holds[len(holds)//2], len(spans)-1]:
                timestamp = float(spans[j]['start']) + .2
                assert timestamp <= len(audio)/16000 + .02
                cases.append({'language': lang, 'label': 'EOT' if j == len(spans)-1 else 'NEOT',
                    'source_id': row['id'], 'audio_path': str(path), 'audio_sha256': sha(path),
                    'span_index': j, 'score_timestamp': timestamp, 'audio_prefix_start': max(0., timestamp-15.),
                    'source_revision': REV, 'source': 'callhome_'+lang+'_official_test', 'is_human_gold': False,
                    'label_source': 'PUBLISHER_STEREO_VAD_OBSERVED_FLOOR_POSITIONAL_LABEL',
                    'context': '', 'case_id': row['id']+'__span_'+str(j)})
        save(output / 'selection_before_asr.json', {'cases': cases, 'selection': 'first10 eligible <=90s turns perlanguage, one midpoint hold and final each',
            'training_use': False, 'chosen_without_asr_or_model_predictions': True})
        save(output / 'progress.json', {'stage': 'SELECTED_BEFORE_ASR', 'languages': len(sources), 'cases': len(cases)})
        print(json.dumps({'stage': 'selected', 'language': lang, 'cases': sum(r['language']==lang for r in cases)}), flush=True)
    model_path = Path.home()/'.cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots'/ASR_REV
    assert (model_path/'model.bin').is_file()
    asr = WhisperModel(str(model_path), device='cuda', compute_type='float16', cpu_threads=2, num_workers=1)
    final = []
    for i, row in enumerate(cases):
        audio = audio_cache[row['audio_path']]
        observed = audio[int(row['audio_prefix_start']*16000):int(row['score_timestamp']*16000)]
        segments, _ = asr.transcribe(observed, language=row['language'], beam_size=1,
            condition_on_previous_text=False, vad_filter=False)
        final.append({**row, 'text': ' '.join(s.text.strip() for s in segments).strip(),
            'asr_model': 'Systran/faster-whisper-small', 'asr_revision': ASR_REV,
            'asr_scope': 'Only own-channel last15seconds through decision timestamp; no future audio/source transcript'})
        (output/'cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in final),encoding='utf-8')
        save(output/'progress.json', {'stage':'CAUSAL_ASR','cases':i+1,'total':len(cases),'elapsed_seconds':time.monotonic()-started})
        print(json.dumps({'stage':'CAUSAL_ASR','cases':i+1,'total':len(cases)}),flush=True)
    save(output/'sources.json', {'dataset':REPO,'revision':REV,'sources':sources,'cases':len(final),
        'asr_revision':ASR_REV,'is_human_gold':False,'training_use':False,
        'label_scope':'Publisher observed floor proxies from stereo VAD; not human semantic gold',
        'sampling_limits':'Small deterministically selected official test turns with at least one hold; not a representative population estimate',
        'elapsed_seconds':time.monotonic()-started})


if __name__ == '__main__':
    main()
