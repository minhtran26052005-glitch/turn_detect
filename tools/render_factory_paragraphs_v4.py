"""Create a small local review of synthetic paragraphs, with honest labels."""
import collections
import html
import json
from pathlib import Path

ROOT=Path('outputs/factory_dialogue_pilot_20261030/qwen_v4_paragraphs_review')


def load(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    dialogues=load(ROOT/'dialogues.jsonl');groups=collections.defaultdict(list)
    for lang in ('vi','en','zh'):
        for row in load(ROOT/('examples_'+lang+'_sample.jsonl')):groups[row['dialogue_id']].append(row)
    supplemental=ROOT.parent/'qwen_v5_zh_review'
    if supplemental.exists():
        dialogues+=load(supplemental/'dialogues.jsonl')
        for row in load(supplemental/'examples_zh_sample.jsonl'):groups[row['dialogue_id']].append(row)
    views=[]
    for d in dialogues:
        turns=''.join('<div class="turn"><b>'+html.escape(t['speaker'])+'</b><p>'+html.escape(' '.join(s['display'] for s in t['sentences']))+'</p></div>' for t in d['turns'])
        choices={}
        for x in groups[d['dialogue_id']]:
            if x['lookahead_words']==2 and x['label'] not in choices:choices[x['label']]=x
        rows=''.join('<tr><td>'+html.escape(label)+'</td><td>'+html.escape(x['context'])+'</td><td>'+html.escape(x['text'])+'</td></tr>' for label,x in choices.items())
        table='<table><tr><th>Nhãn</th><th>Chữ trước điểm xét</th><th>Hai từ sau</th></tr>'+rows+'</table>' if rows else '<p class="muted">Mẫu điểm xét của đoạn này không nằm trong bản xem nhỏ.</p>'
        views.append('<article data-lang="'+html.escape(d['language'])+'"><h2>'+html.escape(d['dialogue_id'])+'</h2>'+turns+table+'</article>')
    page='''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Hội thoại tư vấn nhà máy</title>
<style>body{font:16px system-ui;line-height:1.65;max-width:1100px;margin:32px auto;padding:0 20px;background:#f4f6fa;color:#172239}header,article{background:white;border:1px solid #dce3ed;border-radius:14px;padding:20px;margin:16px 0}h1{margin:0}h2{font-size:1.15rem}select{padding:8px}.turn{border-left:3px solid #8da7e8;padding-left:14px;margin:14px 0}.turn p{margin:4px 0}table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #e5eaf0;padding:8px}.muted,small{color:#59677a}</style>
<header><h1>Đoạn hội thoại tư vấn nhà máy</h1><p>Qwen sinh các đoạn nhiều câu, theo từng ngôn ngữ. Dấu chấm tạo EOT; dấu phẩy hoặc chấm phẩy bên trong câu tạo NEOT. Nhãn được tạo tự động và chưa được con người kiểm tra; không phải test hội thoại thực.</p>
<label>Ngôn ngữ <select id="language"><option value="">Tất cả</option><option value="vi">Tiếng Việt</option><option value="zh">Tiếng Trung</option><option value="en">Tiếng Anh</option></select></label></header>'''+''.join(views)+'''<script>const s=document.querySelector('#language');s.onchange=()=>document.querySelectorAll('article').forEach(e=>e.hidden=s.value&&e.dataset.lang!==s.value);</script></html>'''
    (ROOT/'review.html').write_text(page,encoding='utf-8')
    print(json.dumps({'dialogues':len(dialogues),'counts':dict(collections.Counter(d['language'] for d in dialogues)),'manifest':manifest},ensure_ascii=False))

if __name__=='__main__':main()
