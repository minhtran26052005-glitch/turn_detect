"""Rebuild the same cases using the current turn and immediate past partner.

Case IDs, labels and call splits stay fixed. All acoustic windows end at or
before the decision; partner context ends before the current speaker starts.
"""
import argparse
from collections import Counter,defaultdict
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

from .common import file_hash,read_jsonl,write_json,write_jsonl
from .natural_v5_data import (SCICOM,SCICOM_REV,DATAHIVE,DATAHIVE_REV,download,
    WhisperProcessor,WhisperForConditionalGeneration,get_speech_timestamps,VadOptions)


def speech_overlap(segments,left,right,rate=16000):
    return sum(max(0.,min(s['end']/rate,right)-max(s['start']/rate,left)) for s in segments)


def vi_turn_start(own,partner,index,rate=16000):
    turn_start=own[0]['start']/rate;previous_owner_end=0.
    for j in range(1,index+1):
        left=own[j-1]['end']/rate;right=own[j]['start']/rate
        if speech_overlap(partner,left,right,rate)>=.5:
            turn_start=right;previous_owner_end=left
    return turn_start,previous_owner_end


def choose_peer(current,metadata,turn_start):
    candidates=[r for r in metadata if r['call_id']==current['call_id'] and r['speaker']!=current['speaker']
        and r['call_start']+r['duration']-r['eot_len']<=turn_start+1e-6]
    return max(candidates,key=lambda r:r['call_start']+r['duration']-r['eot_len']) if candidates else None


def run(args):
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();source=Path('data_v5/canonical_20261025')
    cases=[dict(r) for split in ['train','dev','sealed'] for r in read_jsonl(source/(split+'.jsonl'))]
    prior=json.loads((source/'manifest.json').read_text());receipts=[];windows={};transcripts={}
    pool=Path('data_v5/natural_vi_expanded_20261025');pool_manifest=json.loads((pool/'manifest.json').read_text());revision=pool_manifest['asr_revision']
    for split in ['train','dev','sealed']:
        for row in read_jsonl(pool/(split+'.jsonl')):
            if row['asr_revision']==revision:transcripts[row['language']+':'+row['causal_audio_sha256']]=row['text']
    def register(audio,lang):
        audio=np.asarray(audio,dtype=np.float32).copy()
        assert len(audio)>0
        key=lang+':'+hashlib.sha256(audio.tobytes()).hexdigest()
        if key not in transcripts:windows[key]=audio
        return key
    # Provider turn clips already start at the prior handoff. Retrieve the
    # nearest other-speaker clip from ALL metadata, not the sampled cases.
    for lang in ['en','zh']:
        meta_path=download(SCICOM,SCICOM_REV,f'meta/dialogue/callhome_{lang}.parquet',receipts)
        metadata=pq.read_table(meta_path,columns=['id','speaker','call_id','call_start','duration','eot_len','split']).to_pylist()
        by_id={row['id']:row for row in metadata};by_call=defaultdict(list)
        for row in metadata:by_call[row['call_id']].append(row)
        files={};lookup={}
        for official,numbers in [('train',[0,1]),('validation',[0])]:
            for shard in numbers:
                filename=f'dialogue/{official}/callhome_{lang}-{shard:05d}.parquet'
                path=download(SCICOM,SCICOM_REV,filename,receipts);files[filename]=path
                identifiers=pq.read_table(path,columns=['id']).column('id').to_pylist()
                lookup.update({identifier:(filename,i) for i,identifier in enumerate(identifiers)})
        subset=[r for r in cases if r['language']==lang]
        # Candidate peer uses metadata times, with no access to EOT labels.
        needed={r['source_id'] for r in subset}
        for row in subset:
            meta=by_id[row['source_id']]
            # Allow a conservative .3s lead-in while collecting candidates;
            # after causal VAD we verify/select again at the exact speech start.
            peer=choose_peer(meta,[m for m in by_call[meta['call_id']] if m['id'] in lookup],meta['call_start']+.3)
            if peer:needed.add(peer['id'])
            # First speech may be earlier/later than .3s. Include the last3
            # possible partner clips so the exact causal cutoff can choose one.
            possible=[m for m in by_call[meta['call_id']] if m['id'] in lookup and m['speaker']!=meta['speaker']
                and m['call_start']+m['duration']-m['eot_len']<=meta['call_start']+min(row['score_timestamp'],1.)]
            possible.sort(key=lambda m:m['call_start']+m['duration']-m['eot_len'],reverse=True)
            needed.update(m['id'] for m in possible[:3])
        grouped=defaultdict(list)
        for identifier in needed:
            filename,index=lookup[identifier];grouped[filename].append((identifier,index))
        audio_by_id={}
        for filename,items in grouped.items():
            table=pq.read_table(files[filename],columns=['audio']).take([i for _,i in items])
            for (identifier,_),payload in zip(items,table.to_pylist(),strict=True):
                audio,rate=sf.read(io.BytesIO(payload['audio']['bytes']),dtype='float32',always_2d=True)
                if rate!=16000:audio=resample_poly(audio,16000,rate,axis=0).astype(np.float32)
                audio_by_id[identifier]=audio[:,0]
        for row in subset:
            meta=by_id[row['source_id']];audio=audio_by_id[row['source_id']];stamp=row['score_timestamp']
            observed=audio[:int(stamp*16000)]
            speech=get_speech_timestamps(observed,vad_options=VadOptions(threshold=.5,min_speech_duration_ms=200,min_silence_duration_ms=150,speech_pad_ms=0))
            first=speech[0]['start']/16000 if speech else 0.
            turn_start=meta['call_start']+first
            left=max(0.,stamp-30.)
            row['v6_current_key']=register(audio[int(left*16000):int(stamp*16000)],lang)
            row['v6_current_audio_start']=meta['call_start']+left
            row['v6_current_audio_end']=meta['call_start']+stamp
            row['v6_turn_start']=turn_start
            peer=choose_peer(meta,[m for m in by_call[meta['call_id']] if m['id'] in audio_by_id],turn_start)
            row['v6_partner_key']=None;row['v6_partner_audio_end']=None
            if peer:
                assert peer['split']==meta['split']
                end=min(peer['duration']-peer['eot_len']+.2,turn_start-peer['call_start']);begin=max(0.,end-30.)
                if end>begin:
                    row['v6_partner_key']=register(audio_by_id[peer['id']][int(begin*16000):int(end*16000)],lang)
                    row['v6_partner_audio_start']=peer['call_start']+begin;row['v6_partner_audio_end']=peer['call_start']+end
                    row['v6_partner_source_id']=peer['id']
            assert row['v6_partner_audio_end'] is None or row['v6_partner_audio_end']<=turn_start+1e-6
        del audio_by_id
        print(json.dumps({'stage':'V6_WINDOWS_SELECTED','language':lang,'cases':len(subset),'unique_new_windows':len(windows)}),flush=True)
    for number in range(3,9):
        filename=f'preview/vi_{number:06d}.mp3';path=download(DATAHIVE,DATAHIVE_REV,filename,receipts)
        with sf.SoundFile(path) as stream:
            rate=stream.samplerate;audio=stream.read(min(len(stream),900*rate),dtype='float32',always_2d=True)
        audio=resample_poly(audio,16000,rate,axis=0).astype(np.float32)
        speech=[get_speech_timestamps(audio[:,channel],vad_options=VadOptions(threshold=.5,min_speech_duration_ms=200,min_silence_duration_ms=150,speech_pad_ms=0)) for channel in [0,1]]
        for row in [r for r in cases if r['language']=='vi' and r['source_id']==filename]:
            channel=row['channel'];index=int(row['case_id'].rsplit('_',1)[1])
            turn_start,old_end=vi_turn_start(speech[channel],speech[1-channel],index)
            stamp=row['score_timestamp'];left=max(0.,turn_start-.2,stamp-30.)
            row['v6_current_key']=register(audio[int(left*16000):int(stamp*16000),channel],'vi')
            row['v6_current_audio_start']=left;row['v6_current_audio_end']=stamp;row['v6_turn_start']=turn_start
            row['v6_partner_key']=None;row['v6_partner_audio_end']=None
            past=[s for s in speech[1-channel] if s['start']/16000<turn_start and s['end']/16000>old_end]
            if past:
                end=min(turn_start,max(s['end']/16000 for s in past)+.2)
                begin=max(old_end,0.,end-30.)
                if end>begin:
                    row['v6_partner_key']=register(audio[int(begin*16000):int(end*16000),1-channel],'vi')
                    row['v6_partner_audio_start']=begin;row['v6_partner_audio_end']=end
                    row['v6_partner_source_id']=filename+':channel'+str(1-channel)
            assert row['v6_partner_audio_end'] is None or row['v6_partner_audio_end']<=turn_start+1e-6
        print(json.dumps({'stage':'V6_VI_WINDOWS_SELECTED','call':number,'unique_new_windows':len(windows)}),flush=True)
    required={r[key] for r in cases for key in ['v6_current_key','v6_partner_key'] if r[key]}
    write_json(args.output/'frozen_windows_before_asr.json',{'cases':cases,'source_manifest_sha256':file_hash(source/'manifest.json'),
        'required_windows':len(required),'new_asr_windows':len(windows),'reused_asr_windows':len(required.intersection(transcripts)),
        'labels_and_case_splits_unchanged':True,'classifier_scores_used':False,'source_sha256':file_hash(Path(__file__))})
    processor=WhisperProcessor.from_pretrained('openai/whisper-small',revision=revision,token=False)
    model=WhisperForConditionalGeneration.from_pretrained('openai/whisper-small',revision=revision,token=False,dtype=torch.float16).cuda().eval()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    completed=0
    for lang in ['vi','en','zh']:
        keys=[key for key in sorted(windows) if key.startswith(lang+':')]
        for start in range(0,len(keys),8):
            batch=keys[start:start+8]
            features=processor([windows[key] for key in batch],sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
            with torch.inference_mode():
                tokens=model.generate(input_features=features.input_features.cuda().half(),attention_mask=features.attention_mask.cuda(),language=lang,
                    task='transcribe',num_beams=1,do_sample=False,max_new_tokens=256,return_timestamps=False)
            text=processor.batch_decode(tokens,skip_special_tokens=True)
            transcripts.update({key:value.strip() for key,value in zip(batch,text,strict=True)});completed+=len(batch)
            write_json(args.output/'progress.json',{'stage':'CURRENT_TURN_AND_IMMEDIATE_PARTNER_ASR','completed':completed,'total':len(windows),
                'language':lang,'elapsed_seconds':time.monotonic()-started})
            if start%80==0:
                write_json(args.output/'transcript_cache.json',{key:transcripts[key] for key in required if key in transcripts})
                print(json.dumps({'stage':'V6_ASR','completed':completed,'total':len(windows),'language':lang}),flush=True)
    files={}
    for row in cases:
        row['v5_text']=row['text'];row['v5_context']=row['context']
        row['text']=transcripts[row['v6_current_key']]
        row['context']=transcripts[row['v6_partner_key']] if row['v6_partner_key'] else ''
        row['asr_revision']=revision;row['asr_scope']='Current turn last30s through decision; immediate partner last30s ending before current turn begins'
    for split in ['train','dev','sealed']:
        values=sorted([r for r in cases if r['split']==split],key=lambda r:r['case_id'])
        write_jsonl(args.output/(split+'.jsonl'),values)
        files[split]={'cases':len(values),'sha256':file_hash(args.output/(split+'.jsonl')),
            'counts':dict(Counter(r['language']+':'+r['label'] for r in values)),
            'with_immediate_partner':sum(bool(r['context']) for r in values),'empty_current_transcript':sum(not r['text'] for r in values)}
    write_json(args.output/'manifest.json',{'status':'PAST_PARTNER_AND_CURRENT_TURN_TEXT_READY_NOT_HUMAN_GOLD','files':files,
        'prior_canonical_manifest_sha256':file_hash(source/'manifest.json'),'source_files':receipts,'source_sha256':file_hash(Path(__file__)),
        'same_case_ids_labels_and_call_splits_as_v5':True,'held_classifier_scores_never_computed':True,
        'asr_revision':revision,'required_windows':len(required),'new_asr_windows':len(windows),'reused_asr_windows':len(required.intersection(set(transcripts)-set(windows))),
        'past_partner_selection_uses_target_label':False,'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({'stage':'V6_READY','files':files,'new_asr_windows':len(windows)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('data_v6/turn_context_20261025'))
    run(parser.parse_args())
