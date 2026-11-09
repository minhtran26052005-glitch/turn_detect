"""Fresh public Vietnamese stereo pilot: causal ASR, observed floor proxies.

Labels are automatic VAD/other-channel proxies, NEVER advertised as human gold.
Model candidates have not trained on these conversations. All candidates kept
except deterministic compute-bound sampling, selected before model inference.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from faster_whisper.vad import get_speech_timestamps, VadOptions


REPO='datahiveai/multilingual-conversations-sample'
REV='1120cd7533b16ba8dd7a7c07e6d316b74b60f72a'


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    out=Path('data_v4/vi_natural_pause_20261025');out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();sources=[];items=[];audio_cache={}
    for name in ['vi_000001.mp3','vi_000002.mp3']:
        path=out/name
        response=requests.get(f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/preview/{name}',timeout=90,stream=True)
        response.raise_for_status()
        with path.open('wb') as stream:
            for block in response.iter_content(1024*1024):stream.write(block)
        with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        audio,sr=sf.read(path,dtype='float32',always_2d=True,stop=300*48000)
        assert audio.shape[1]==2
        audio=resample_poly(audio,16000,sr,axis=0).astype(np.float32);rate=16000
        # Isolated channel VAD only; no transcript or model score used in labels.
        speech=[get_speech_timestamps(audio[:,channel],vad_options=VadOptions(threshold=.5,min_speech_duration_ms=200,min_silence_duration_ms=150,speech_pad_ms=0)) for channel in [0,1]]
        sources.append({'name':name,'url':f'https://huggingface.co/datasets/{REPO}/blob/{REV}/preview/{name}','bytes':path.stat().st_size,'sha256':sha,
            'seconds_inspected':len(audio)/rate,'sample_rate':rate,'channel_speech_segments':[len(x) for x in speech]})
        audio_cache[name]=audio
        candidates=[]
        for channel in [0,1]:
            segments=speech[channel];other=speech[1-channel]
            for index in range(len(segments)-1):
                end=segments[index]['end']/rate;resume=segments[index+1]['start']/rate
                gap=resume-end
                if not .2<=gap<=4.:continue
                # A long overlapping contribution before this speaker stopped is
                # an interruption, not an uncomplicated EOT decision.
                if any(s['start']/rate<end-.2 and s['end']/rate>end for s in other):continue
                others=sum(max(0.,min(s['end']/rate,resume)-max(s['start']/rate,end)) for s in other)
                if others>=.5:label='EOT'
                elif others<.2 and gap<=2.:label='NEOT'
                else:continue
                candidates.append({'source_id':name,'language':'vi','channel':channel,'label':label,'score_timestamp':end+.2,
                    'gap_seconds':gap,'other_channel_speech_during_gap':others,'label_source':'AUTOMATIC_STEREO_VAD_OBSERVED_FLOOR_PROXY',
                    'is_human_gold':False,'audio_prefix_start':max(0.,end+.2-15.),'source_revision':REV})
        # Fix pilot sample before any ASR/classifier score, every label per call.
        selected=[]
        for label in ['EOT','NEOT']:
            pool=sorted([x for x in candidates if x['label']==label],key=lambda x:x['score_timestamp'])
            if pool:
                indices=np.unique(np.linspace(0,len(pool)-1,min(8,len(pool)),dtype=int))
                selected.extend(pool[i] for i in indices)
        items.extend(sorted(selected,key=lambda x:x['score_timestamp']))
        save(out/'selection_before_asr.json',{'cases':items,'label_method':'Automatic channel floor proxies; no human adjudication','cases_used_for_training':False})
        save(out/'progress.json',{'stage':'VAD_AND_SELECTION','calls_completed':len(sources),'selected_cases':len(items),'elapsed_seconds':time.monotonic()-started})
        print(json.dumps({'stage':'selected_before_asr','call':name,'counts':dict(Counter(x['label'] for x in selected))}),flush=True)
    assert items
    asr=WhisperModel('small',device='cpu',compute_type='int8',cpu_threads=4,num_workers=1)
    rows=[]
    for i,item in enumerate(items):
        audio=audio_cache[item['source_id']]
        left=int(item['audio_prefix_start']*16000);right=int(item['score_timestamp']*16000)
        observed=audio[left:right,item['channel']]
        segments,_=asr.transcribe(observed,language='vi',beam_size=1,condition_on_previous_text=False,vad_filter=False)
        text=' '.join(s.text.strip() for s in segments).strip()
        rows.append({**item,'text':text,'context':'','asr_model':'Systran/faster-whisper-small','asr_scope':'Causal own-channel last15seconds only; no future audio or source transcript'})
        (out/'cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        save(out/'progress.json',{'stage':'CAUSAL_ASR','cases':i+1,'total':len(items),'elapsed_seconds':time.monotonic()-started})
        print(json.dumps({'stage':'CAUSAL_ASR','cases':i+1,'total':len(items)}),flush=True)
    save(out/'sources.json',{'dataset':REPO,'revision':REV,'license':'CC BY4.0; attribution DataHive','sources':sources,'cases':len(rows),
        'is_human_gold':False,'training_use':False,'scope':'Small independent natural Vietnamese conversation pilot with automatic VAD floor labels and causal ASR; not representative gold benchmark',
        'sampling':'First300seconds of first2 released VI calls; up to8 perlabel percall selected uniformly by time before ASR/scoring',
        'limitations':['VAD and ASR can be wrong','Observed floor proxy labels need human verification','Only2 conversations','15second text window may omit semantic history'],
        'elapsed_seconds':time.monotonic()-started})


if __name__=='__main__':main()
