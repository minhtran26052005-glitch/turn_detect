"""Recover genuine handoffs hidden by the old max4s own-speaker gap filter.

The partner may speak for much longer than4s before the owner resumes. Detect
the observed partner response in the first3s; never feed that future to ASR.
Existing V4 pilot records remain unchanged. This is new V5 data construction.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

from .natural_v5_data import (DATAHIVE,DATAHIVE_REV,download,WhisperProcessor,WhisperForConditionalGeneration,
    get_speech_timestamps,VadOptions)
from .common import file_hash,read_jsonl,write_json,write_jsonl


def run(args):
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    source=Path('data_v5/natural_20261025')
    manifest=json.loads((source/'manifest.json').read_text())
    original=[r for split in ['train','dev','sealed'] for r in read_jsonl(source/(split+'.jsonl'))]
    prior={r['case_id']:r for r in original if r['language']=='vi'}
    rows=[r for r in original if r['language']!='vi'];fresh=[];audio_cache={};receipts=[];reused=0
    for number in range(3,9):
        path=download(DATAHIVE,DATAHIVE_REV,f'preview/vi_{number:06d}.mp3',receipts)
        with sf.SoundFile(path) as stream:
            rate=stream.samplerate;audio=stream.read(min(len(stream),900*rate),dtype='float32',always_2d=True)
        audio=resample_poly(audio,16000,rate,axis=0).astype(np.float32)
        speech=[get_speech_timestamps(audio[:,channel],vad_options=VadOptions(threshold=.5,min_speech_duration_ms=200,min_silence_duration_ms=150,speech_pad_ms=0)) for channel in [0,1]]
        split='train' if number<=6 else 'dev' if number==7 else 'sealed';group=f'datahive_vi_{number:06d}'
        for channel in [0,1]:
            segments=speech[channel];other=speech[1-channel]
            for j,segment in enumerate(segments):
                end=segment['end']/16000
                resume=segments[j+1]['start']/16000 if j+1<len(segments) else len(audio)/16000
                gap=resume-end
                if gap<.2:continue
                if any(s['start']/16000<end-.2 and s['end']/16000>end for s in other):continue
                partner_first3=sum(max(0.,min(s['end']/16000,resume,end+3.)-max(s['start']/16000,end)) for s in other)
                partner_total=sum(max(0.,min(s['end']/16000,resume)-max(s['start']/16000,end)) for s in other)
                if partner_first3>=.5:label='EOT'
                elif j+1<len(segments) and partner_total<.2 and gap<=2.:label='NEOT'
                else:continue
                stamp=end+.2;left=max(0.,stamp-15.);clip=audio[int(left*16000):int(stamp*16000),channel].copy()
                identifier=group+f'__ch_{channel}__pause_{j:04d}';sha=hashlib.sha256(clip.tobytes()).hexdigest()
                row={'case_id':identifier,'source_id':f'preview/vi_{number:06d}.mp3','session_group':group,
                    'source':'datahive_vi','language':'vi','split':split,'label':label,'channel':channel,'score_timestamp':stamp,
                    'audio_prefix_start':left,'source_revision':DATAHIVE_REV,'gap_seconds':gap,'other_speech_first3seconds':partner_first3,
                    'label_source':'OBSERVED_PARTNER_RESPONSE_GE500MS_IN_FIRST3S; NEOT_SELF_RESUMES_WITHIN2S_WITHOUT_PARTNER',
                    'is_human_gold':False,'context':'','causal_audio_sha256':sha}
                if identifier in prior and prior[identifier]['causal_audio_sha256']==sha:
                    rows.append({**row,**{key:prior[identifier][key] for key in ['text','asr_model','asr_revision','asr_scope']}});reused+=1
                else:fresh.append(row);audio_cache[identifier]=clip
        print(json.dumps({'stage':'EXPANDED_VI_SELECTION','call':number,'new_cases':len(fresh),'reused':reused}),flush=True)
    write_json(args.output/'selection_before_asr.json',{'cases':rows+fresh,'new_asr_cases':len(fresh),'reused_asr_cases':reused,
        'rule':'Observed >=500ms partner speech within first3s; no upper bound on owner-resume gap for EOT; NEOT <=2s with no partner',
        'classifier_scores_used':False,'old_v4_pilot_modified':False})
    revision=manifest['asr_revision'];processor=WhisperProcessor.from_pretrained('openai/whisper-small',revision=revision,token=False)
    model=WhisperForConditionalGeneration.from_pretrained('openai/whisper-small',revision=revision,token=False,dtype=torch.float16).cuda().eval()
    torch.set_num_threads(2)
    for start in range(0,len(fresh),8):
        batch=fresh[start:start+8]
        features=processor([audio_cache[r['case_id']] for r in batch],sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
        with torch.inference_mode():
            tokens=model.generate(input_features=features.input_features.cuda().half(),attention_mask=features.attention_mask.cuda(),language='vi',
                task='transcribe',num_beams=1,do_sample=False,max_new_tokens=256,return_timestamps=False)
        text=processor.batch_decode(tokens,skip_special_tokens=True)
        rows.extend({**r,'text':s.strip(),'asr_model':'openai/whisper-small','asr_revision':revision,
            'asr_scope':'Only causal own-channel last15seconds; no partner future or future source transcript'} for r,s in zip(batch,text,strict=True))
        write_jsonl(args.output/'asr_partial.jsonl',rows)
        write_json(args.output/'progress.json',{'stage':'EXPANDED_VI_ASR','completed':min(start+8,len(fresh)),'total':len(fresh),'elapsed_seconds':time.monotonic()-started})
    files={}
    for split in ['train','dev','sealed']:
        values=sorted([r for r in rows if r['split']==split],key=lambda r:r['case_id'])
        write_jsonl(args.output/(split+'.jsonl'),values)
        files[split]={'cases':len(values),'sha256':file_hash(args.output/(split+'.jsonl')),'counts':dict(Counter(r['language']+':'+r['label'] for r in values))}
    write_json(args.output/'manifest.json',{'status':'EXPANDED_ASR_POOL_CANONICALIZATION_REQUIRED','files':files,'source_files':manifest['source_files'],
        'asr_revision':revision,'source_manifest_sha256':file_hash(source/'manifest.json'),'source_sha256':file_hash(Path(__file__)),
        'new_vi_asr':len(fresh),'reused_vi_asr':reused,'old_v4_pilot_modified':False,'human_gold':False,'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({'stage':'EXPANDED_COMPLETE','files':files,'new_vi_asr':len(fresh)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('data_v5/natural_vi_expanded_20261025'))
    run(parser.parse_args())
