"""Fetch a small verified diagnostic report and fixed20-pair review from SSH."""
import base64
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import subprocess

REMOTE = '/home/edabk/hoangbpm/turn_detect/outputs/multilingual_stream_test_20261102_v1'
LOCAL = Path('outputs/multilingual_stream_test_20261102_v1')


def pct(x):
    return '—' if x is None else f'{100*x:.2f}%'


def render(report, cases):
    rows = []
    for k in ('2', '1'):
        for mode, title in [('monolingual', 'Riêng từng tiếng'), ('mixed', 'Ghép Việt → Anh')]:
            for lang in ('vi', 'en'):
                m = report['metrics_every_eligible_word_gap'][mode][k][lang]
                rows.append(f'<tr><td>{k}</td><td>{title}</td><td>{lang.upper()}</td><td>{m["samples"]}</td>'
                            f'<td>{m["fp"]}/{m["tp"]+m["fp"]} · {pct(m["false_share_of_cuts"])}</td>'
                            f'<td>{pct(m["eot_precision"])}</td><td>{m["fn"]}/{m["tp"]+m["fn"]}</td><td>{pct(m["eot_f1"])}</td></tr>')
    cards = []
    for c in cases:
        panels = []
        for k in ('2', '1'):
            v = c['lookahead'][k]; segments = []
            for s in v['segments']:
                flush = s['reason'] != 'MODEL_EOT'
                crossed = s['start'] < c['switch_word_index'] < s['end']
                title = 'Xả cuối luồng · không tính model tự ngắt' if flush else f'EOT · p={s["p_eot"]:.3f}'
                if crossed: title += ' · gộp qua chỗ ghép VI → EN'
                segments.append('<li class="'+('flush' if flush else 'bad' if crossed else '')+'"><small>'+escape(title)+'</small><p>'+escape(s['text'])+'</p></li>')
            errors = []
            for p in v['problem_points']:
                actual = p['reference']['label'] if p['reference'] else 'Không chấm'
                label = 'Cắt nhầm' if actual == 'NEOT' and p['predicted']=='EOT' else 'Bỏ sót' if actual=='EOT' and p['predicted']=='NEOT' else 'Chỗ ghép'
                errors.append('<tr><td>'+label+'</td><td>'+escape(' '.join(p['context'].split()[-24:]))+' <b>|</b> '+escape(p['text'])+'</td><td>'+actual+'</td><td>'+p['predicted']+'</td><td>'+f'{p["p_eot"]:.3f}'+'</td></tr>')
            switch = v['switch']; m = v['metrics']
            panels.append('<section class="mode" data-k="'+k+'"><p><b>Tại chỗ ghép: '+switch['predicted']+'</b> · p='+f'{switch["p_eot"]:.3f}'+
                          f' · cắt nhầm {m["fp"]}, bỏ sót {m["fn"]}</p><ol>'+''.join(segments)+'</ol><details><summary>Xem các lỗi và đầu vào tại chỗ ghép</summary>'+
                          '<div class="scroll"><table><thead><tr><th>Loại</th><th>Buffer trước | nhìn trước</th><th>Nhãn dấu câu</th><th>Model</th><th>p EOT</th></tr></thead><tbody>'+''.join(errors)+'</tbody></table></div></details></section>')
        cards.append('<article data-number="'+str(c['number'])+'"><h2>Cặp '+str(c['number'])+'</h2><div class="original"><section><b>Đoạn Việt gốc</b><p>'+escape(c['vi_original'])+'</p></section><section><b>Đoạn Anh gốc</b><p>'+escape(c['en_original'])+'</p></section></div>'+''.join(panels)+'</article>')
    old = []
    for k, c in report.get('same_first20_old_probe_comparison', {}).items():
        a, b = c['old'], c['new']
        old.append(f'<tr><td>{k}</td><td>{a["fp"]}</td><td>{b["fp"]}</td><td>{pct(a["eot_precision"])}</td><td>{pct(b["eot_precision"])}</td><td>{a["fn"]}</td><td>{b["fn"]}</td></tr>')
    checkpoint = report['checkpoint']; switches = report['language_splice_observations']
    style = '*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:#172939;font:16px/1.6 system-ui}main{max-width:1200px;margin:32px auto;padding:0 18px}h1{line-height:1.3;font-size:30px}h2{font-size:22px}article,.summary{background:#fff;border:1px solid #d8e1eb;border-radius:14px;padding:22px;margin:18px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:left;vertical-align:top;border-bottom:1px solid #dce3ec}.original{display:grid;grid-template-columns:1fr 1fr;gap:16px}.original section{background:#f5f7fa;padding:14px;border-radius:9px}p{margin:8px 0}small,.muted{color:#607181;font-size:13px}li{padding:8px 12px;margin:8px 0;border-left:3px solid #a8bdcf}.bad{background:#fff0ed}.flush{background:#f0f2f5}summary{cursor:pointer;font-weight:600;color:#285daf}.toolbar{position:sticky;top:0;background:#edf2f7f5;padding:12px;z-index:2;display:flex;gap:20px;flex-wrap:wrap}select{font:inherit;padding:5px}.mode[hidden],article[hidden]{display:none}@media(max-width:650px){.original{grid-template-columns:1fr}article,.summary{padding:14px}}'
    page = '<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Test streaming bản mới · 200 đoạn Việt/Anh</title><style>'+style+'</style></head><body><main>'
    page += '<h1>Test streaming bản mới: 100 đoạn Việt + 100 đoạn Anh</h1><p>Chấm riêng 200 đoạn và ghép thành 100 luồng Việt → Anh. Bản xem giữ đúng 20 cặp đầu đã khóa trước khi chấm; số liệu tổng hợp dùng đủ 200 đoạn.</p>'
    page += f'<section class="summary"><p>Checkpoint <b>{checkpoint["step"]}</b> · bước nhánh mới <b>{checkpoint["branch_step"]}</b> · ngưỡng chung <b>{report["threshold_from_same_checkpoint_dev"]:.2f}</b>, chọn trên DEV của chính checkpoint trước khi test.</p>'
    page += '<p>Model chỉ thấy buffer chữ thường không dấu câu cùng 1 hoặc 2 từ nhìn trước. Chỉ xả buffer theo EOT dự đoán; không dùng nhãn đúng, mã ngôn ngữ hay vai nói. Xả cuối luồng không được tính là dự đoán EOT.</p><p><b>Phạm vi:</b> dữ liệu Qwen tổng hợp đã từng xem, loại khỏi TRAIN/DEV mới theo ID hội thoại; có thể cùng nhóm chủ đề với TRAIN. Nhãn dấu câu, không phải human/audio gold hay test hội thoại độc lập.</p>'
    page += '<h2>Cắt nhầm và bỏ sót trên mọi khe từ đủ nhìn trước</h2><p>Cắt nhầm / tổng lần cắt là tỷ lệ đáng xem khi đưa đoạn sang dịch. Các điểm kết thúc bằng ?/! được để ngoài điểm số.</p><div class="scroll"><table><thead><tr><th>Nhìn trước</th><th>Chế độ</th><th>Tiếng</th><th>Điểm xét</th><th>Cắt nhầm / tổng cắt</th><th>Precision</th><th>Bỏ sót / EOT thật</th><th>F1 EOT</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
    page += '<p>Chỗ ghép có EOT dự đoán: '+', '.join(f'{k} từ: {v["predicted_eot"]}/{v["total"]}' for k, v in switches.items())+'. Đây là điểm ghép nhân tạo, không phải accuracy nhận diện ngôn ngữ.</p>'
    if old:
        page += '<h2>Cùng 20 cặp của bài thử cũ</h2><p>Dùng lại trace cũ, không chạy lại model cũ. Trọng số và ngưỡng DEV của hai checkpoint khác nhau; không dùng test để chọn ngưỡng/model.</p><div class="scroll"><table><thead><tr><th>Nhìn trước</th><th>Cắt nhầm cũ</th><th>Cắt nhầm mới</th><th>Precision cũ</th><th>Precision mới</th><th>Bỏ sót cũ</th><th>Bỏ sót mới</th></tr></thead><tbody>'+''.join(old)+'</tbody></table></div>'
    page += f'<p>Đã audit {report["causal_audit"]["all_inferences_audited"]:,} lượt suy luận: đúng lượng chữ nhìn trước, reset chỉ theo EOT dự đoán và giữ nguyên toàn bộ từ. Training chính tiếp tục độc lập. Frozen optimizer/RNG và toàn bộ trace giữ trên SSH.</p></section>'
    page += '<div class="toolbar"><label>Nhìn trước <select id="k"><option value="2">2 từ</option><option value="1">1 từ</option></select></label><label>Cặp <select id="which"><option value="">20 cặp đầu</option>'+''.join(f'<option value="{c["number"]}">{c["number"]}</option>' for c in cases)+'</select></label></div>'+''.join(cards)
    page += '<p class="muted">Report SHA256 '+escape(report['_report_sha256'])+'</p></main><script>const k=document.querySelector("#k"),w=document.querySelector("#which");function show(){document.querySelectorAll(".mode").forEach(e=>e.hidden=e.dataset.k!==k.value);document.querySelectorAll("article").forEach(e=>e.hidden=!!w.value&&e.dataset.number!==w.value)}k.onchange=w.onchange=show;show();</script></body></html>'
    return page.encode('utf-8')


def main():
    assert shutil.disk_usage(Path.cwd()).free > 10*2**30 + 5*2**20
    code = f'''import json,base64,hashlib
from pathlib import Path
p=Path({REMOTE!r});assert json.loads((p/'status.json').read_text())['stage']=='COMPLETE'
names=['report.json','review_cases.json','frozen_dev.json','checkpoint_receipt.json','input_manifest.json','input_audit.json','status.json','launcher.json']
files={{n:(p/n).read_bytes() for n in names}};assert sum(map(len,files.values()))<4*2**20
print(json.dumps({{n:{{'base64':base64.b64encode(b).decode(),'sha256':hashlib.sha256(b).hexdigest()}} for n,b in files.items()}}))'''
    proc = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','edabk@100.91.209.124','python3 -'], input=code, capture_output=True, text=True, encoding='utf-8', timeout=45, check=True)
    values = json.loads(proc.stdout); files = {}
    for name, v in values.items():
        b = base64.b64decode(v['base64']); assert hashlib.sha256(b).hexdigest()==v['sha256']; files[name]=b
    report = json.loads(files['report.json']); cases = json.loads(files['review_cases.json'])
    assert hashlib.sha256(files['review_cases.json']).hexdigest()==report['review_cases_sha256']
    report['_report_sha256'] = values['report.json']['sha256']; files['review.html'] = render(report, cases)
    assert sum(map(len,files.values())) < 5*2**20
    LOCAL.mkdir(parents=True,exist_ok=True)
    receipt = {}
    for name, b in files.items():
        p=LOCAL/name;p.write_bytes(b);assert p.read_bytes()==b
        receipt[name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    (LOCAL/'transfer_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'review':str((LOCAL/'review.html').resolve()),'bytes':sum(map(len,files.values())), 'report_sha256':values['report.json']['sha256'], 'checkpoint':report['checkpoint'], 'metrics_lookahead2':{mode:m['2'] for mode,m in report['metrics_every_eligible_word_gap'].items()},'old_first20':report['same_first20_old_probe_comparison'].get('2')},ensure_ascii=False))


if __name__=='__main__':main()
