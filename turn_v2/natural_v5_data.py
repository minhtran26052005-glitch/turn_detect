"""Public natural-conversation training data, grouped by original call.

Only decoded past audio enters ASR; classifier inputs remain text-only. Public
observed-floor labels are automatic proxies, not human semantic gold.
"""
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly
import torch
import transformers.utils as hf_utils
import transformers.utils.import_utils as hf_imports
hf_utils.is_torchvision_available=hf_imports.is_torchvision_available=lambda:False
hf_utils.is_librosa_available=hf_imports.is_librosa_available=lambda:False
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import hf_hub_download, model_info
from faster_whisper.vad import get_speech_timestamps, VadOptions

from .common import digest, file_hash, write_json, write_jsonl

SCICOM='Scicom-intl/semantic-vad-eot'
SCICOM_REV='98d1445eab71e4084ccb838dbd87a205f6f2e293'
DATAHIVE='datahiveai/multilingual-conversations-sample'
DATAHIVE_REV='1120cd7533b16ba8dd7a7c07e6d316b74b60f72a'


def call_group(identifier):
    return identifier.split('__turn_')[0]


def download(repo,revision,name,receipts):
    path=Path(hf_hub_download(repo_id=repo,filename=name,revision=revision,repo_type='dataset',token=False))
    receipts.append({'repo':repo,'revision':revision,'filename':name,'bytes':path.stat().st_size,'sha256':file_hash(path)})
    print(json.dumps({'stage':'DOWNLOADED','filename':name,'bytes':path.stat().st_size}),flush=True)
    return path


def sampled_turns(rows,limit):
    # Hash order and a per-call cap prevent the first sorted call dominating.
    ordered=sorted(rows,key=lambda row:digest(row['id']))
    selected=[];counts=Counter()
    for row in ordered:
        group=call_group(row['id'])
        if counts[group]>=8:continue
        selected.append(row);counts[group]+=1
        if len(selected)>=limit:break
    return selected


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();receipts=[];cases=[];observed={};split_groups=defaultdict(set)
    write_json(args.output/'protocol.json',{'architecture':'TEXT_ONLY','user_selected':'more public real conversation training data',
        'vi_calls':{'train':[3,4,5,6],'dev':[7],'sealed':[8],'regression_excluded':[1,2]},
        'en_zh':'official train grouped by call hash into train/dev; official validation kept sealed; earlier official-test pilot excluded',
        'train_turn_limit_per_language':args.train_turns,'dev_turn_limit_per_language':args.dev_turns,'sealed_turn_limit_per_language':args.sealed_turns,
        'maximum_turns_per_call':8,'vi_seconds_per_call':args.vi_seconds,'source_sha256':file_hash(Path(__file__)),
        'labels':'Automatic observed floor/VAD; not human gold','selection_uses_model_predictions':False,
        'input':'ASR own-channel past15seconds at silence.start+200ms; no future audio; no source full transcript'})
    for lang in ['en','zh']:
        metadata=[];files={}
        for official,numbers in [('train',[0,1]),('validation',[0])]:
            for shard in numbers:
                name=f'dialogue/{official}/callhome_{lang}-{shard:05d}.parquet'
                path=download(SCICOM,SCICOM_REV,name,receipts);files[name]=path
                table=pq.read_table(path,columns=['id','language','duration','silence_spans'])
                for row_index,row in enumerate(table.to_pylist()):
                    if row['language']!=lang or row['duration']>90 or not row['silence_spans']:continue
                    last=row['silence_spans'][-1]
                    if float(last['end'])-float(last['start'])<.2:continue
                    split='sealed' if official=='validation' else ('dev' if int(digest(call_group(row['id']))[:8],16)%100<15 else 'train')
                    metadata.append({**row,'file':name,'row_index':row_index,'split':split})
        selected=[]
        for split,limit in [('train',args.train_turns),('dev',args.dev_turns),('sealed',args.sealed_turns)]:
            subset=sampled_turns([r for r in metadata if r['split']==split],limit)
            assert subset,(lang,split)
            selected.extend(subset)
            split_groups[(lang,split)].update(call_group(r['id']) for r in subset)
        by_file=defaultdict(list)
        for row in selected:by_file[row['file']].append(row)
        for name,subset in by_file.items():
            audio_table=pq.read_table(files[name],columns=['audio']).take([r['row_index'] for r in subset])
            for row,payload in zip(subset,audio_table.to_pylist(),strict=True):
                raw=payload['audio']['bytes'];assert raw
                audio,rate=sf.read(io.BytesIO(raw),dtype='float32',always_2d=True);assert audio.shape[1]==1
                if rate!=16000:audio=resample_poly(audio,16000,rate,axis=0).astype(np.float32)
                spans=row['silence_spans'];holds=[j for j,s in enumerate(spans[:-1]) if float(s['end'])-float(s['start'])>=.2]
                indices=([holds[i] for i in np.unique(np.linspace(0,len(holds)-1,min(2,len(holds)),dtype=int))] if holds else [])+[len(spans)-1]
                for index in indices:
                    timestamp=float(spans[index]['start'])+.2
                    left=max(0.,timestamp-15.)
                    clip=audio[int(left*16000):int(timestamp*16000),0].copy()
                    identifier=row['id']+'__span_'+str(index)
                    record={'case_id':identifier,'source_id':row['id'],'session_group':call_group(row['id']),
                        'source':'callhome_'+lang,'language':lang,'split':row['split'],'label':'EOT' if index==len(spans)-1 else 'NEOT',
                        'score_timestamp':timestamp,'audio_prefix_start':left,'span_index':index,'source_revision':SCICOM_REV,
                        'source_file':name,'source_row':row['row_index'],'label_source':'PUBLISHER_STEREO_VAD_OBSERVED_FLOOR',
                        'is_human_gold':False,'context':'','causal_audio_sha256':hashlib.sha256(clip.tobytes()).hexdigest()}
                    cases.append(record);observed[identifier]=clip
            del audio_table;gc.collect()
    # Complete original VI calls stay in one split. Previously viewed calls1/2
    # never enter this campaign, including unviewed portions of those calls.
    for number in range(3,9):
        name=f'preview/vi_{number:06d}.mp3'
        path=download(DATAHIVE,DATAHIVE_REV,name,receipts)
        with sf.SoundFile(path) as stream:
            rate=stream.samplerate;audio=stream.read(min(len(stream),args.vi_seconds*rate),dtype='float32',always_2d=True)
        assert audio.shape[1]==2
        audio=resample_poly(audio,16000,rate,axis=0).astype(np.float32)
        speech=[get_speech_timestamps(audio[:,channel],vad_options=VadOptions(threshold=.5,min_speech_duration_ms=200,min_silence_duration_ms=150,speech_pad_ms=0)) for channel in [0,1]]
        split='train' if number<=6 else 'dev' if number==7 else 'sealed'
        group=f'datahive_vi_{number:06d}';split_groups[('vi',split)].add(group)
        for channel in [0,1]:
            segments=speech[channel];other=speech[1-channel]
            for j in range(len(segments)-1):
                end=segments[j]['end']/16000;resume=segments[j+1]['start']/16000;gap=resume-end
                if not .2<=gap<=4.:continue
                if any(s['start']/16000<end-.2 and s['end']/16000>end for s in other):continue
                duration=sum(max(0.,min(s['end']/16000,resume)-max(s['start']/16000,end)) for s in other)
                if duration>=.5:label='EOT'
                elif duration<.2 and gap<=2.:label='NEOT'
                else:continue
                timestamp=end+.2;left=max(0.,timestamp-15.)
                clip=audio[int(left*16000):int(timestamp*16000),channel].copy()
                identifier=group+f'__ch_{channel}__pause_{j:04d}'
                cases.append({'case_id':identifier,'source_id':name,'session_group':group,'source':'datahive_vi',
                    'language':'vi','split':split,'label':label,'channel':channel,'score_timestamp':timestamp,'audio_prefix_start':left,
                    'source_revision':DATAHIVE_REV,'gap_seconds':gap,'other_speech_seconds':duration,
                    'label_source':'AUTOMATIC_STEREO_VAD_OBSERVED_FLOOR','is_human_gold':False,'context':'',
                    'causal_audio_sha256':hashlib.sha256(clip.tobytes()).hexdigest()})
                observed[identifier]=clip
        print(json.dumps({'stage':'VI_SELECTED','call':number,'cases':sum(r['session_group']==group for r in cases)}),flush=True)
    for lang in ['vi','en','zh']:
        for left,right in [('train','dev'),('train','sealed'),('dev','sealed')]:
            assert not split_groups[(lang,left)].intersection(split_groups[(lang,right)])
    write_json(args.output/'selection_before_asr.json',{'cases':cases,'source_files':receipts,
        'call_groups':{lang+':'+split:sorted(groups) for (lang,split),groups in split_groups.items()},'chosen_without_classifier_scores':True})
    # Snapshot the actual model revision before any transcript is generated.
    asr_revision=model_info('openai/whisper-small',token=False).sha
    processor=WhisperProcessor.from_pretrained('openai/whisper-small',revision=asr_revision,token=False)
    model=WhisperForConditionalGeneration.from_pretrained('openai/whisper-small',revision=asr_revision,token=False,dtype=torch.float16).cuda().eval()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    final=[]
    for lang in ['vi','en','zh']:
        subset=[r for r in cases if r['language']==lang]
        for start in range(0,len(subset),args.asr_batch):
            batch=subset[start:start+args.asr_batch]
            features=processor([observed[r['case_id']] for r in batch],sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
            with torch.inference_mode():
                tokens=model.generate(input_features=features.input_features.cuda().half(),attention_mask=features.attention_mask.cuda(),
                    language=lang,task='transcribe',num_beams=1,do_sample=False,max_new_tokens=256,return_timestamps=False)
            transcripts=processor.batch_decode(tokens,skip_special_tokens=True)
            final.extend({**r,'text':text.strip(),'asr_model':'openai/whisper-small','asr_revision':asr_revision,
                'asr_scope':'Causal own-channel last15seconds; no future audio or complete source transcript'} for r,text in zip(batch,transcripts,strict=True))
            write_jsonl(args.output/'asr_partial.jsonl',final)
            write_json(args.output/'progress.json',{'stage':'CAUSAL_ASR','completed':len(final),'total':len(cases),
                'language':lang,'elapsed_seconds':time.monotonic()-started})
            if start%(args.asr_batch*10)==0:print(json.dumps({'stage':'ASR','completed':len(final),'total':len(cases),'language':lang}),flush=True)
    files={}
    for split in ['train','dev','sealed']:
        rows=sorted([r for r in final if r['split']==split],key=lambda r:r['case_id'])
        write_jsonl(args.output/(split+'.jsonl'),rows)
        files[split]={'cases':len(rows),'sha256':file_hash(args.output/(split+'.jsonl')),
            'counts':dict(Counter(r['language']+':'+r['label'] for r in rows)),
            'empty_asr_cases':sum(not r['text'] for r in rows)}
    write_json(args.output/'manifest.json',{'status':'PUBLIC_NATURAL_TEXT_READY_NOT_HUMAN_GOLD','files':files,
        'source_files':receipts,'selection_sha256':file_hash(args.output/'selection_before_asr.json'),
        'asr_revision':asr_revision,'call_groups':{lang+':'+split:sorted(groups) for (lang,split),groups in split_groups.items()},
        'source_sha256':file_hash(Path(__file__)),'old_v4_tests_used_in_training':False,
        'licenses':{'datahive_vi':'CC BY4.0 attribution DataHive','callhome_en_zh':'CC BY-NC-SA4.0 source; research experiment'},
        'scope':'Text-only classifier data from causal ASR and automatic observed-floor labels; not independent human adjudication',
        'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({'stage':'COMPLETE','files':files,'elapsed_seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-turns',type=int,default=300)
    parser.add_argument('--dev-turns',type=int,default=75)
    parser.add_argument('--sealed-turns',type=int,default=75)
    parser.add_argument('--vi-seconds',type=int,default=900)
    parser.add_argument('--asr-batch',type=int,default=8)
    run(parser.parse_args())
