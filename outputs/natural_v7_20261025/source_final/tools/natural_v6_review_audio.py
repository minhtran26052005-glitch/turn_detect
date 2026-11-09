"""Export deterministic train/dev VI listening examples of matched ASR routes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from huggingface_hub import hf_hub_download


def main():
    root=Path('outputs/natural_v7_20261025');output=root/'asr_review';output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for split in ['train','dev']:
        values=[json.loads(line) for line in Path('data_v6/pho_text_20261025',split+'.jsonl').read_text(encoding='utf-8').splitlines()]
        values=[r for r in values if r['language']=='vi']
        calls=sorted({r['source_id'] for r in values})
        for call in calls:
            cases=sorted([r for r in values if r['source_id']==call],key=lambda r:r['score_timestamp'])
            # Fixed time-spaced examples; selection uses no model score/label.
            indices=np.unique(np.linspace(0,len(cases)-1,4,dtype=int))
            rows.extend(cases[i] for i in indices)
    audio_cache={};items=[]
    for i,row in enumerate(rows):
        name=row['source_id']
        if name not in audio_cache:
            source=hf_hub_download('datahiveai/multilingual-conversations-sample',name,revision=row['source_revision'],repo_type='dataset',token=False)
            with sf.SoundFile(source) as stream:
                rate=stream.samplerate;audio=stream.read(min(len(stream),900*rate),dtype='float32',always_2d=True)
            audio_cache[name]=resample_poly(audio,16000,rate,axis=0).astype(np.float32)
        values=audio_cache[name];left=int(row['v6_current_audio_start']*16000);right=int(row['v6_current_audio_end']*16000)
        path=output/f'case_{i:03d}.wav';sf.write(path,values[left:right,row['channel']],16000,subtype='PCM_16')
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        items.append({'case_id':row['case_id'],'split':row['split'],'label':row['label'],'is_human_gold':False,
            'text_old':row['whisper_small_text'],'text_new':row['text'],'context_old':row['whisper_small_context'],'context_new':row['context'],
            'source_id':name,'audio':path.name,'audio_sha256':digest,'bytes':path.stat().st_size,
            'audio_end':row['v6_current_audio_end'],'current_turn_start':row['v6_turn_start']})
    payload=json.dumps(items,ensure_ascii=False).replace('<','\\u003c')
    page=r'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Đối chiếu ASR Việt V7</title><style>body{font:17px system-ui;color:#18313e;background:#f3f7fb}main{max-width:1100px;margin:auto;padding:24px}section{background:white;padding:24px;border:1px solid #c9dae4;border-radius:12px;margin:16px 0}button{font:inherit;padding:10px;margin:5px}p{line-height:1.7;white-space:pre-wrap}audio{width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style><main><h1>20 ví dụ ASR Việt trên cùng audio đã quan sát</h1><p>Chọn đều theo thời gian từ4cuộc gọi train và1cuộc gọi dev của DataHive, CC BY4.0. Không có tập giữ riêng ở đây. Transcript mới chưa được người chấm xác nhận; ít lặp hơn không tự chứng minh WER thấp hơn.</p><section><button id="prev">← Trước</button><button id="next">Sau →</button><span id="count"></span><p id="meta"></p><audio id="audio" controls preload="none"></audio><div class="pair"><div><h3>Whisper-small</h3><p id="old"></p><details><summary>Lời đối diện trước đó</summary><p id="oldContext"></p></details></div><div><h3>PhoWhisper-large</h3><p id="new"></p><details><summary>Lời đối diện trước đó</summary><p id="newContext"></p></details></div></div><p id="label"></p></section><p>Audio chỉ đến thời điểm quyết định; model turn detection tiếp tục chỉ nhận chữ. Nhãn khoảng dừng tự động chưa phải human gold.</p></main><script id="data" type="application/json">__DATA__</script><script>const R=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);let i=0;function show(){const r=R[i];$('count').textContent=(i+1)+'/'+R.length;$('meta').textContent=r.case_id+' · '+r.split;$('audio').src=r.audio;$('old').textContent=r.text_old;$('new').textContent=r.text_new;$('oldContext').textContent=r.context_old;$('newContext').textContent=r.context_new;$('label').textContent='Nhãn tự động: '+r.label+' · chưa có người xác nhận'}$('prev').onclick=()=>{i=Math.max(0,i-1);show()};$('next').onclick=()=>{i=Math.min(R.length-1,i+1);show()};show();</script></html>'''.replace('__DATA__',payload)
    (output/'review.html').write_text(page,encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps({'attribution':'DataHive CC BY4.0','selection':'4 uniformly time-spaced cases per train/dev call, no labels/scores used',
        'held_cases_included':False,'items':items},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'cases':len(items),'path':str(output/'review.html')}),flush=True)


if __name__=='__main__':main()
