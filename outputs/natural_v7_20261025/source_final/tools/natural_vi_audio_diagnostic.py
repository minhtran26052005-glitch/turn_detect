"""CPU-only stereo dominance/correlation diagnostic on training/development.

Flags aid listening; they never remove/relabel a case or score a classifier.
"""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import correlate,resample_poly
import soundfile as sf
from huggingface_hub import hf_hub_download


def main():
    output=Path('outputs/natural_v7_20261025/audio_diagnostic');output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for split in ['train','dev']:
        rows.extend(json.loads(line) for line in Path('data_v6/turn_context_20261025',split+'.jsonl').read_text(encoding='utf-8').splitlines())
    rows=[r for r in rows if r['language']=='vi'];sources=[];results=[]
    for filename in sorted({r['source_id'] for r in rows}):
        cases=[r for r in rows if r['source_id']==filename];revision=cases[0]['source_revision']
        path=Path(hf_hub_download('datahiveai/multilingual-conversations-sample',filename,revision=revision,repo_type='dataset',token=False))
        with sf.SoundFile(path) as stream:
            rate=stream.samplerate;audio=stream.read(min(len(stream),900*rate),dtype='float32',always_2d=True)
        audio=resample_poly(audio,16000,rate,axis=0).astype(np.float32)
        for row in cases:
            end=max(0,int((row['score_timestamp']-.2)*16000));start=max(0,end-16000)
            current=audio[start:end,row['channel']].astype(np.float64);other=audio[start:end,1-row['channel']].astype(np.float64)
            current-=current.mean();other-=other.mean()
            energy_current=float(np.mean(current**2));energy_other=float(np.mean(other**2))
            ratio=10*np.log10((energy_current+1e-12)/(energy_other+1e-12))
            denominator=np.linalg.norm(current)*np.linalg.norm(other)
            if denominator>0:
                values=correlate(current,other,mode='full',method='fft');middle=len(current)-1;width=min(9600,len(current)-1)
                peak=float(np.max(np.abs(values[middle-width:middle+width+1]))/denominator)
            else:peak=0.
            results.append({'case_id':row['case_id'],'split':row['split'],'source_id':filename,'label':row['label'],
                'own_minus_other_energy_dB':float(ratio),'peak_abs_correlation_within_600ms':peak,
                'own_channel_below_other_by_6dB':bool(ratio < -6),'high_cross_channel_correlation':bool(peak>.6)})
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        sources.append({'source_id':filename,'sha256':digest,'revision':revision})
    summaries={}
    for split in ['train','dev']:
        values=[r for r in results if r['split']==split]
        summaries[split]={'cases':len(values),'high_cross_channel_correlation':sum(r['high_cross_channel_correlation'] for r in values),
            'own_below_other_6dB':sum(r['own_channel_below_other_by_6dB'] for r in values),
            'median_own_minus_other_dB':float(np.median([r['own_minus_other_energy_dB'] for r in values]))}
    report={'status':'AUDIO_DIAGNOSTIC_ONLY','summaries':summaries,'sources':sources,'cases':results,
        'method':'Own and other channels last1second ending at pause start; RMS energy and peak normalized crosscorrelation +/-600ms',
        'limitations':'Energy and correlation do not prove incorrect labels or transcript. Listening/human adjudication required.',
        'cases_removed_or_relabeled':False,'held_data_used':False,'classifier_scores_used':False}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'summaries':summaries}),flush=True)


if __name__=='__main__':main()
