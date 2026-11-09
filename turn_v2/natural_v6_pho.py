"""Same causal VI windows, new pinned ASR; labels/cases/splits unchanged."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import transformers.utils as hf_utils
import transformers.utils.import_utils as hf_imports
hf_utils.is_torchvision_available=hf_imports.is_torchvision_available=lambda:False
hf_utils.is_librosa_available=hf_imports.is_librosa_available=lambda:False
from transformers import WhisperForConditionalGeneration,WhisperProcessor
from huggingface_hub import hf_hub_download

from .common import file_hash,read_jsonl,write_json,write_jsonl,normalize


def repetition_rate(text):
    words=normalize(text).split()
    if len(words)<12:return 0.
    triples=[tuple(words[i:i+3]) for i in range(len(words)-2)]
    return 1.-len(set(triples))/len(triples)


def run(args):
    output=args.output;output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    source=Path('data_v6/turn_context_20261025');rows=[dict(r) for split in ['train','dev','sealed'] for r in read_jsonl(source/(split+'.jsonl'))]
    receipt=json.loads(Path('outputs/natural_v6_20261025/pho_download.json').read_text());model_path=Path(receipt['local_path'])
    audio={};windows={};reused_keys={}
    for row in rows:
        if row['language']!='vi':continue
        filename=row['source_id']
        if filename not in audio:
            path=hf_hub_download('datahiveai/multilingual-conversations-sample',filename,revision=row['source_revision'],repo_type='dataset',token=False)
            with sf.SoundFile(path) as stream:
                rate=stream.samplerate;value=stream.read(min(len(stream),900*rate),dtype='float32',always_2d=True)
            audio[filename]=resample_poly(value,16000,rate,axis=0).astype(np.float32)
        value=audio[filename]
        for role,channel in [('current',row['channel']),('partner',1-row['channel'])]:
            end=row.get('v6_'+role+'_audio_end')
            if end is None:continue
            begin=row['v6_'+role+'_audio_start'];observed=value[int(begin*16000):int(end*16000),channel].copy()
            # Time slicing and float resampling must match previous causal work.
            key='vi:'+hashlib.sha256(observed.tobytes()).hexdigest()
            assert key==row['v6_'+role+'_key'],(row['case_id'],role)
            windows[key]=observed
    write_json(output/'frozen_before_asr.json',{'source_manifest_sha256':file_hash(source/'manifest.json'),'asr':receipt,
        'windows':len(windows),'decoding':{'beam':3,'do_sample':False,'max_new_tokens':256},
        'cases_or_labels_filtered':False,'model_predictions_used':False,'source_sha256':file_hash(Path(__file__))})
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    processor=WhisperProcessor.from_pretrained(model_path,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(model_path,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    keys=sorted(windows,key=lambda k:len(windows[k]));texts={}
    for start in range(0,len(keys),4):
        batch=keys[start:start+4]
        inputs=processor([windows[k] for k in batch],sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
        with torch.inference_mode():
            tokens=model.generate(input_features=inputs.input_features.cuda().half(),attention_mask=inputs.attention_mask.cuda(),language='vi',task='transcribe',
                num_beams=3,do_sample=False,max_new_tokens=256,return_timestamps=False)
        transcripts=processor.batch_decode(tokens,skip_special_tokens=True)
        texts.update({k:t.strip() for k,t in zip(batch,transcripts,strict=True)})
        write_json(output/'progress.json',{'stage':'PHOWHISPER_SAME_CAUSAL_WINDOWS','completed':min(start+4,len(keys)),'total':len(keys),
            'elapsed_seconds':time.monotonic()-started,'VRAM_allocated':torch.cuda.memory_allocated()})
        if start%40==0:
            write_json(output/'transcripts_partial.json',texts)
            print(json.dumps({'stage':'PHO_ASR','completed':len(texts),'total':len(keys),'elapsed_seconds':time.monotonic()-started}),flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
    for row in rows:
        if row['language']!='vi':continue
        row['whisper_small_text']=row['text'];row['whisper_small_context']=row['context']
        row['text']=texts[row['v6_current_key']];row['context']=texts[row['v6_partner_key']] if row['v6_partner_key'] else ''
        row['asr_model']=receipt['repo'];row['asr_revision']=receipt['revision']
    files={}
    for split in ['train','dev','sealed']:
        selected=sorted([r for r in rows if r['split']==split],key=lambda r:r['case_id'])
        write_jsonl(output/(split+'.jsonl'),selected)
        files[split]={'cases':len(selected),'sha256':file_hash(output/(split+'.jsonl')),'counts':dict(Counter(r['language']+':'+r['label'] for r in selected))}
    # This measures repetition and transcript changes; it is not WER/ASR quality
    # without a human transcript. No bad-ASR samples are silently removed.
    diagnostics={}
    for split in ['train','dev']:
        subset=[r for r in rows if r['split']==split and r['language']=='vi']
        diagnostics[split]={'cases':len(subset),'old_high_repetition':sum(repetition_rate(r['whisper_small_text'])>.3 for r in subset),
            'new_high_repetition':sum(repetition_rate(r['text'])>.3 for r in subset),
            'changed_texts':sum(normalize(r['text'])!=normalize(r['whisper_small_text']) for r in subset),
            'empty_new_texts_kept':sum(not r['text'] for r in subset),'human_transcript_available':False}
    write_json(output/'manifest.json',{'status':'CAUSAL_SAME_CASES_PHOWHISPER_VI_READY_NOT_HUMAN_GOLD','files':files,
        'parent_manifest_sha256':file_hash(source/'manifest.json'),'asr':receipt,'diagnostics':diagnostics,
        'label_case_split_changed':False,'empty_or_hard_cases_filtered':False,'held_classifier_scores_used':False,
        'source_sha256':file_hash(Path(__file__)),'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({'status':'PHO_DATA_READY','files':files,'diagnostics':diagnostics,'elapsed_seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('data_v6/pho_text_20261025'));run(parser.parse_args())
