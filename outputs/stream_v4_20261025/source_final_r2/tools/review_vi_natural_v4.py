"""Listening/review pack; automatic floor labels are hidden until a vote."""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def main():
    source = Path('data_v4/vi_natural_pause_20261025')
    output = Path('outputs/stream_v4_20261025')
    clips = output / 'vi_review_audio'
    clips.mkdir(exist_ok=True)
    rows = [json.loads(line) for line in (source / 'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    audio_cache = {}
    reviewed = []
    files = []
    for i, row in enumerate(rows):
        if row['source_id'] not in audio_cache:
            audio, sr = sf.read(source / row['source_id'], dtype='float32', always_2d=True)
            audio_cache[row['source_id']] = resample_poly(audio, 16000, sr, axis=0).astype(np.float32)
        audio = audio_cache[row['source_id']]
        left = int(row['audio_prefix_start'] * 16000)
        decision = int(row['score_timestamp'] * 16000)
        right = min(len(audio), decision + 3 * 16000)
        for kind, samples in [('causal', audio[left:decision, row['channel']]), ('audit', audio[left:right])]:
            path = clips / f'{i:03d}_{kind}.wav'
            sf.write(path, samples, 16000, subtype='PCM_16')
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            files.append({'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size})
        reviewed.append({**row, 'case_index': i, 'decision_clip_seconds': (decision-left)/16000,
            'causal_audio': f'vi_review_audio/{i:03d}_causal.wav', 'audit_audio': f'vi_review_audio/{i:03d}_audit.wav'})
    data = json.dumps(reviewed, ensure_ascii=False).replace('<', '\\u003c')
    html = r'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Nghe kiểm tra 30 điểm dừng tiếng Việt</title>
<style>body{font:17px system-ui;color:#18313e;background:#f3f7fb}main{max-width:1000px;margin:auto;padding:24px}section{background:white;border:1px solid #ccdce5;padding:24px;border-radius:12px;margin:18px 0}button{font:inherit;padding:10px;margin:4px;border:1px solid #aac2cf;border-radius:6px;background:white;cursor:pointer}p{line-height:1.65}audio{display:block;width:100%;margin:12px 0}.chosen{background:#cdeadd}#text{white-space:pre-wrap;font-size:21px}textarea{width:98%;font:inherit;min-height:60px}small{color:#46616f}</style>
<main><h1>Nghe kiểm tra 30 điểm dừng hội thoại Việt</h1>
<p>Hai cuộc gọi công khai từ <a href="https://huggingface.co/datasets/datahiveai/multilingual-conversations-sample/tree/1120cd7533b16ba8dd7a7c07e6d316b74b60f72a">DataHive</a>, CC BY 4.0. Nhãn hiện có được suy ra bằng VAD hai kênh, chưa có người xác nhận. ASR có thể sai. Hãy nghe và chọn EOT/NEOT/AMBIGUOUS trước khi mở nhãn tự động.</p>
<section><button id="prev">← Trước</button><button id="next">Sau →</button><span id="count"></span><p id="meta"></p>
<h3>Đúng âm thanh model được phép quan sát</h3><audio id="causal" controls preload="none"></audio><div id="text"></div>
<h3>Âm thanh đối chiếu nhãn</h3><p id="point"></p><audio id="audit" controls preload="none"></audio><small>Clip đối chiếu có thêm 3 giây tương lai, chỉ để kiểm tra hành vi nhường lượt. Phần này không được đưa vào ASR hoặc model.</small>
<p><button data-label="EOT">EOT — đã nhường lượt</button><button data-label="NEOT">NEOT — còn giữ lượt</button><button data-label="AMBIGUOUS">AMBIGUOUS — chưa xác định</button></p><textarea id="reason" placeholder="Lý do hoặc lỗi ASR/VAD quan sát được"></textarea>
<p><button id="reveal">Mở nhãn tự động sau khi chọn</button><span id="automatic"></span></p></section>
<section><button id="export">Tải nhận xét JSONL</button><span id="status"></span><p>Nhãn được giữ trong bộ nhớ tab. Tải file trước khi đóng trang. Một lượt xem chưa tự tạo thành bộ gold: cần người chấm độc lập và giải quyết bất đồng.</p></section></main>
<script type="application/json" id="data">__DATA__</script><script>
const rows=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);let index=0;
function save(){rows[index].review_reason=$('reason').value}
function show(){const r=rows[index];$('count').textContent=(index+1)+'/'+rows.length;$('meta').textContent=r.source_id+' · kênh '+r.channel+' · thời điểm '+r.score_timestamp.toFixed(2)+'s';$('text').textContent=r.text;$('causal').src=r.causal_audio;$('audit').src=r.audit_audio;$('point').textContent='Thời điểm quyết định: '+r.decision_clip_seconds.toFixed(2)+'s trong clip; nghe diễn biến sau đó để kiểm tra nhãn.';$('reason').value=r.review_reason||'';$('automatic').textContent='';document.querySelectorAll('[data-label]').forEach(b=>b.classList.toggle('chosen',r.review_label===b.dataset.label));$('status').textContent=rows.filter(r=>r.review_label).length+'/'+rows.length+' đã chấm'}
$('prev').onclick=()=>{save();index=Math.max(0,index-1);show()};$('next').onclick=()=>{save();index=Math.min(rows.length-1,index+1);show()};document.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>{save();rows[index].review_label=b.dataset.label;show()});$('reveal').onclick=()=>{$('automatic').textContent=rows[index].review_label?'VAD tự động: '+rows[index].label:'Chọn nhận xét trước để tránh bị nhãn gợi ý.'};
$('export').onclick=()=>{save();const data=rows.filter(r=>r.review_label).map(r=>JSON.stringify({...r,automatic_label:r.label,label:r.review_label,is_human_gold:false,label_status:'REVIEW_EXPORT_PENDING_IDENTITY_AGREEMENT'})).join('\n')+'\n';const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([data],{type:'application/x-ndjson;charset=utf-8'}));link.download='vi_natural_review_20261025.jsonl';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)};show();
</script></html>'''.replace('__DATA__', data)
    (output / 'vi_natural_review.html').write_text(html, encoding='utf-8')
    (output / 'vi_review_audio_manifest.json').write_text(json.dumps({'sources': str(source / 'sources.json'), 'audio_clips': files,
        'model_receives_future_audio': False, 'human_gold_created': False}, indent=2), encoding='utf-8')
    print(json.dumps({'page': str(output / 'vi_natural_review.html'), 'cases': len(rows), 'audio_clips': len(files)}))


if __name__ == '__main__':
    main()
