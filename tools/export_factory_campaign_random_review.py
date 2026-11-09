"""Export only a bounded dialogue review from the current Colab runtime."""
import argparse
import base64
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import secrets
import shutil

from colab_runtime import assigned_servers, ColabRuntime

REPO = Path(__file__).resolve().parents[1]
LIMIT = 5 * 1024 * 1024


def runtime():
    matches = []
    for index in range(len(assigned_servers())):
        rt = ColabRuntime(server_index=index)
        for session in rt.request('GET', 'api/sessions').json():
            name = Path(session.get('path', '')).name
            if name == 'a.ipynb' or name.startswith('a-jvsc-'):
                matches.append((index, session))
    if len(matches) != 1 or matches[0][1]['kernel'].get('execution_state') != 'idle':
        raise RuntimeError('Need one idle current a.ipynb session; never interrupt or reset it')
    index, session = matches[0]
    rt = ColabRuntime(server_index=index, session_path=session['path'])
    rt.connect()
    return rt


def execute(rt, code):
    result = rt.execute(code, timeout=60, echo=False)
    if result['status'] != 'ok':
        raise RuntimeError(str([x for x in result['outputs'] if x['output_type'] == 'error']))
    return json.loads(''.join(x.get('text', '') for x in result['outputs'] if x['output_type'] == 'stream'))


def paragraph(turn):
    return ' '.join(s['display'] for s in turn['sentences'])


def render(dest):
    data = json.loads((dest / 'samples.json').read_text(encoding='utf-8'))
    manifest = json.loads((dest / 'manifest.json').read_text(encoding='utf-8'))
    samples = data['samples']
    assert Counter(s['language'] for s in samples) == {'vi': 100, 'en': 100}
    assert len({s['dialogue']['dialogue_id'] for s in samples}) == 200
    escape = lambda s: html.escape(str(s), quote=True)
    cards = []
    texts = {}
    for lang in ('vi', 'en'):
        lines = []
        for sample in (s for s in samples if s['language'] == lang):
            d = sample['dialogue']; turn = sample['selected_turn']; task = d.get('generation_task', {})
            turns = ''.join('<section class="context-turn' + (' chosen' if i == sample['selected_turn_index'] else '')
                            + '"><b>' + escape(t['speaker']) + '</b><p>' + escape(paragraph(t)) + '</p></section>'
                            for i, t in enumerate(d['turns']))
            meta = ' · '.join(str(task[k]) for k in ('industry', 'equipment', 'topic') if task.get(k))
            cards.append('<article data-lang="' + lang + '"><div class="eyebrow">'
                         + ('TIẾNG VIỆT' if lang == 'vi' else 'ENGLISH') + ' · MẪU '
                         + str(sample['number']).zfill(3) + '/100</div><p class="topic">' + escape(meta)
                         + '</p><h2>' + escape(turn['speaker']) + '</h2><p class="selected">'
                         + escape(paragraph(turn)) + '</p><details><summary>Xem toàn bộ hội thoại ('
                         + str(len(d['turns'])) + ' lượt)</summary>' + turns
                         + '<p class="source">ID: ' + escape(d['dialogue_id']) + ' · Commit '
                         + str(sample['source_commit_index']) + '</p></details></article>')
            lines.extend([f"{sample['number']:03d}. {turn['speaker']}", paragraph(turn),
                          f"Hội thoại: {d['dialogue_id']}", ''])
        texts[f'{lang}_100_luot_loi.txt'] = '\n'.join(lines).encode('utf-8')
    page = '''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>200 lượt lời Việt & Anh — mẫu ngẫu nhiên</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#1b293b;font:17px/1.75 system-ui,sans-serif}main{max-width:940px;margin:36px auto;padding:0 20px}h1{font-size:30px;line-height:1.25;margin:8px 0 16px}header p{margin:8px 0}.sub,.topic,.source{color:#566779;font-size:14px}.toolbar{position:sticky;top:0;background:#f3f5f8f5;z-index:2;padding:15px 0;display:flex;gap:10px;flex-wrap:wrap;align-items:center}select,input,button{font:inherit;border:1px solid #c7d0dc;border-radius:9px;padding:8px 12px;background:white;color:#203149}input{flex:1;min-width:180px}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}article{background:white;border:1px solid #dce3ec;border-radius:15px;margin:0 0 18px;padding:26px;box-shadow:0 3px 12px #17305005}.eyebrow{font-size:12px;letter-spacing:.1em;font-weight:750;color:#315fc4}.topic{margin:7px 0 18px}h2{font-size:17px;margin:0 0 6px}.selected{font-size:19px;white-space:pre-wrap;margin:0 0 20px}summary{cursor:pointer;color:#315fc4;font-weight:600}details{border-top:1px solid #e6ebf1;padding-top:14px}.context-turn{border-left:3px solid #dbe2ec;padding:5px 0 5px 15px;margin-top:18px}.context-turn p{margin:4px 0}.chosen{border-left-color:#315fc4;background:#f3f7ff}.pager{display:flex;gap:14px;align-items:center;justify-content:center;padding:12px 0 35px}#count{font-size:14px;color:#526276}footer{font-size:13px;color:#68778b}a{color:#315fc4}article[hidden]{display:none}@media(max-width:550px){main{margin:20px auto;padding:0 12px}article{padding:19px}h1{font-size:26px}.selected{font-size:18px}}
</style></head><body><main><header><div class="eyebrow">HỘI THOẠI NHÀ MÁY · QWEN</div><h1>100 lượt lời Việt<br>100 lượt lời Anh</h1><p>Mỗi mẫu là một lượt nói nguyên đoạn, lấy từ một hội thoại khác nhau. Mở bên dưới mỗi mẫu để đọc đầy đủ ngữ cảnh.</p><p class="sub">Chọn ngẫu nhiên từ dữ liệu đã được hệ thống chấp nhận; giữ nguyên chữ và dấu câu. Dữ liệu do Qwen sinh, chưa duyệt nội dung và chưa được đưa vào train.</p></header>
<div class="toolbar"><select id="language" aria-label="Ngôn ngữ"><option value="vi">Tiếng Việt · 100</option><option value="en">Tiếng Anh · 100</option><option value="">Cả hai · 200</option></select><input id="search" type="search" placeholder="Tìm trong nội dung, vai nói, chủ đề…" aria-label="Tìm nội dung"><span id="count" aria-live="polite"></span></div><div id="cards">'''
    page += ''.join(cards) + '''</div><div class="pager"><button id="previous">← Trước</button><span id="page"></span><button id="next">Sau →</button></div><footer><p>Bản chữ: <a href="vi_100_luot_loi.txt">Việt</a> · <a href="en_100_luot_loi.txt">Anh</a>. Mỗi tiếng có 100 hội thoại nguồn khác nhau; một lượt được chọn ngẫu nhiên trong mỗi hội thoại. Đây là bản đọc, chưa phải bộ nhãn do người duyệt.</p><details><summary>Thông tin lần lấy mẫu</summary><p>Seed: '''
    page += escape(manifest['seed']) + ' · Chốt nguồn tại commit ' + str(manifest['frozen_commit_index'])
    page += '''. Nguồn và SHA nằm trong manifest.json.</p></details></footer></main><script>
const cards=[...document.querySelectorAll('article')], lang=document.querySelector('#language'), search=document.querySelector('#search');let page=0;const size=10;
function show(){const q=search.value.toLocaleLowerCase();const found=cards.filter(c=>(!lang.value||c.dataset.lang===lang.value)&&(!q||c.textContent.toLocaleLowerCase().includes(q)));const pages=Math.max(1,Math.ceil(found.length/size));page=Math.min(page,pages-1);cards.forEach(c=>c.hidden=true);found.slice(page*size,(page+1)*size).forEach(c=>c.hidden=false);document.querySelector('#count').textContent=found.length+' mẫu';document.querySelector('#page').textContent='Trang '+(page+1)+' / '+pages;document.querySelector('#previous').disabled=page===0;document.querySelector('#next').disabled=page===pages-1;}
lang.onchange=()=>{page=0;show()};search.oninput=()=>{page=0;show()};document.querySelector('#previous').onclick=()=>{page--;show();window.scrollTo(0,0)};document.querySelector('#next').onclick=()=>{page++;show();window.scrollTo(0,0)};show();
</script></body></html>'''
    blobs = {'review.html': page.encode('utf-8'), **texts}
    assert sum(p.stat().st_size for p in dest.iterdir() if p.is_file()) + sum(map(len, blobs.values())) < LIMIT
    for name, data in blobs.items():
        (dest / name).write_bytes(data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['start', 'status', 'collect'])
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(); dest = a.output_dir.resolve()
    assert shutil.disk_usage(dest.anchor).free >= 10 * 2**30 + LIMIT, 'Local reserve prevents sample transfer'
    if a.action == 'start':
        dest.mkdir(parents=True, exist_ok=False)
        seed = secrets.randbits(64)
        remote = f'/content/factory_random_review_{seed}'
        source = (REPO / 'turn_v2/factory_dialogue_random_review_v1.py').read_bytes()
        code = f'''import pathlib,base64,hashlib,subprocess,sys,json,time
r=pathlib.Path({remote!r});r.mkdir(exist_ok=False)
data=base64.b64decode({base64.b64encode(source).decode()!r});assert hashlib.sha256(data).hexdigest()=={hashlib.sha256(source).hexdigest()!r}
(r/'sample.py').write_bytes(data)
with (r/'worker.log').open('a') as f:proc=subprocess.Popen([sys.executable,'-u',str(r/'sample.py'),'--output',str(r/'output'),'--seed',{str(seed)!r},'--count','100'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(.5)
print(json.dumps({{'remote_root':str(r),'seed':{str(seed)!r},'pid':proc.pid,'source_sha256':hashlib.sha256(data).hexdigest()}}))'''
    else:
        job = json.loads((dest / 'job.json').read_text(encoding='utf-8')); remote = job['remote_root']
        code = f'''import pathlib,json,base64,hashlib
r=pathlib.Path({remote!r})/'output'
s=json.loads((r/'status.json').read_text())
v={{'status':s}}
if s['stage']=='ERROR':v['error']=json.loads((r/'error.json').read_text())
if {a.action!r}=='collect':
 assert s['stage']=='COMPLETE'
 files={{name:(r/name).read_bytes() for name in ['samples.json','manifest.json']}}
 assert sum(map(len,files.values()))<2*1024*1024
 v['files']={{name:{{'base64':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}} for name,data in files.items()}}
print(json.dumps(v,ensure_ascii=False))'''
    rt = runtime()
    try:
        value = execute(rt, code)
    finally:
        rt.close()
    if a.action == 'start':
        (dest / 'job.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
        print(json.dumps(value)); return
    if a.action == 'collect':
        receipt = {}
        for name, entry in value.pop('files').items():
            data = base64.b64decode(entry['base64']); assert hashlib.sha256(data).hexdigest() == entry['sha256']
            (dest / name).write_bytes(data); assert hashlib.sha256((dest / name).read_bytes()).hexdigest() == entry['sha256']
            receipt[name] = {k: entry[k] for k in ('bytes', 'sha256')}
        (dest / 'transfer_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        render(dest)
        value.update(review=str(dest / 'review.html'), transfer_receipt=receipt,
                     total_local_bytes=sum(p.stat().st_size for p in dest.iterdir() if p.is_file()))
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
