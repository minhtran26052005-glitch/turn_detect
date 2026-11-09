"""Fetch compact VI->EN probe reports; checkpoints and full traces stay on SSH."""
import argparse
import base64
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import subprocess

REMOTE = '/mnt/hdd500/turn_detect_training/multilingual_vi_en_stream_probe_20261102_v1'
LOCAL = Path('outputs/multilingual_vi_en_stream_probe_20261102_v1')


def pct(value):
    return '—' if value is None else f'{100 * value:.2f}%'


def render(dest, report, cases):
    threshold = report['threshold_from_same_checkpoint_dev']
    cards = []; plain = []; cuts = {'1': 0, '2': 0}
    for case in cases:
        parts = []
        for k in ('1', '2'):
            result = case['lookahead'][k]; switch = result['switch']
            refs = {r['boundary_word_index'] for r in result['anchors'] if r['reference'] and r['reference']['label'] == 'EOT'}
            cuts[k] += sum(s['reason'] == 'MODEL_EOT' and s['end'] not in refs for s in result['segments'])
            segments = []
            for s in result['segments']:
                flush = s['reason'] != 'MODEL_EOT'
                crosses = s['start'] < case['switch_word_index'] < s['end']
                label = 'Xả cuối luồng · không phải dự đoán model' if flush else f"EOT · p={s['p_eot']:.3f}"
                if crosses:
                    label += ' · Gộp qua chỗ ghép Việt → Anh'
                segments.append('<li class="' + ('flush' if flush else 'cross' if crosses else '') + '"><small>'
                                + escape(label) + '</small><p>' + escape(s['text']) + '</p></li>')
            rows = []
            for row in result['anchors']:
                ref = row['reference']['label'] if row['reference'] else None
                predicted = row['predicted']; correct = ref is not None and ref == predicted
                location = 'VI → EN' if row['boundary_word_index'] == case['switch_word_index'] else 'Việt' if row['boundary_word_index'] < case['switch_word_index'] else 'Anh'
                preview = ' '.join(row['context'].split()[-16:])
                rows.append('<tr class="' + ('ok' if correct else 'bad' if ref else '') + '"><td>'
                            + location + '</td><td>' + escape(preview) + ' <b>|</b> '
                            + escape(row['text']) + '</td><td>' + (ref or '—') + '</td><td>' + predicted
                            + '</td><td>' + f"{row['p_eot']:.3f}" + '</td></tr>')
            parts.append('<section class="mode" data-k="' + k + '"><div class="switch ' + ('ok' if switch['predicted'] == 'EOT' else 'bad')
                         + '"><b>Tại chỗ ghép Việt → Anh: ' + switch['predicted'] + '</b> · p(EOT)='
                         + f"{switch['p_eot']:.3f}" + '</div><h3>Các đoạn model phát ra</h3><ol>'
                         + ''.join(segments) + '</ol><details><summary>Xem từng điểm tham chiếu và đầu vào nhìn trước</summary>'
                         + '<p>Chỉ các dấu chấm cuối câu và dấu phẩy/chấm phẩy được chấm nhãn. Các điểm khác không có nhãn tham chiếu.</p>'
                         + '<div class="table"><table><tr><th>Vị trí</th><th>Chữ trước | nhìn trước</th><th>Nhãn từ dấu câu</th><th>Dự đoán</th><th>p(EOT)</th></tr>'
                         + ''.join(rows) + '</table></div></details></section>')
        missed = {k: case['lookahead'][k]['switch']['predicted'] != 'EOT' for k in ('1', '2')}
        cards.append('<article data-case="' + str(case['number']) + '" data-miss-1="' + str(int(missed['1']))
                     + '" data-miss-2="' + str(int(missed['2'])) + '"><h2>Cặp ' + str(case['number']).zfill(2)
                     + ' · Việt → Anh</h2><div class="original"><section><b>Đoạn Việt gốc</b><p>'
                     + escape(case['vi_original']) + '</p></section><section><b>Đoạn Anh gốc</b><p>'
                     + escape(case['en_original']) + '</p></section></div><details><summary>Luồng chữ đầu vào — cấp dần, không đưa cả tương lai vào model</summary><p class="raw">'
                     + escape(case['stream_input']) + '</p></details>' + ''.join(parts)
                     + '<p class="muted">Nguồn: ' + escape(case['vi_source']) + ' + ' + escape(case['en_source']) + '</p></article>')
        plain.extend([f"CẶP {case['number']:02d}", 'VI: ' + case['vi_original'], 'EN: ' + case['en_original'],
                      'INPUT: ' + case['stream_input'], 'LOOKAHEAD 2 / SWITCH: ' + case['lookahead']['2']['switch']['predicted'],
                      *[f"[{s['reason']}] {s['text']}" for s in case['lookahead']['2']['segments']], ''])
    metrics = []
    for k in ('1', '2'):
        for lang in ('vi', 'en'):
            solo = report['metrics_on_matched_fullstop_comma_anchors']['monolingual'][k][lang]
            mixed = report['metrics_on_matched_fullstop_comma_anchors']['mixed_matched'][k][lang]
            assert solo['samples'] == mixed['samples']
            metrics.append(f"<tr><td>{k}</td><td>{lang.upper()}</td><td>{solo['samples']}</td><td>{pct(solo['accuracy'])} / {pct(solo['eot_f1'])}</td><td>{pct(mixed['accuracy'])} / {pct(mixed['eot_f1'])}</td></tr>")
    switches = report['language_splice_observations']
    heading = ''.join('<div class="stat"><b>Nhìn trước ' + k + ' từ</b><strong>' + str(switches[k]['predicted_eot']) + '/'
                      + str(switches[k]['total']) + '</strong><span>chỗ ghép có dự đoán EOT</span><small>Trong đó '
                      + str(switches[k]['full_stop_predicted_eot']) + '/' + str(switches[k]['full_stop_reference_total'])
                      + ' chỗ kết thúc bằng dấu chấm.</small></div>' for k in ('1', '2'))
    page = '''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Student distill · thử luồng Việt → Anh</title><style>
*{box-sizing:border-box}body{font:16px/1.65 system-ui;margin:0;background:#f3f5f8;color:#1b293b}main{max-width:1120px;margin:32px auto;padding:0 20px}h1{font-size:29px;line-height:1.3}h2{font-size:21px}h3{font-size:17px}article,.summary{background:white;border:1px solid #dce3ec;border-radius:14px;padding:24px;margin:18px 0}.stats,.original{display:grid;grid-template-columns:1fr 1fr;gap:18px}.stat{background:#eef3ff;border-radius:12px;padding:18px;display:flex;flex-direction:column}.stat strong{font-size:32px}.muted,small{font-size:13px;color:#647488}.toolbar{position:sticky;top:0;background:#f3f5f8f5;padding:12px 0;display:flex;gap:16px;align-items:center;flex-wrap:wrap;z-index:2}select{font:inherit;padding:8px;border:1px solid #becbdc;border-radius:8px}.original section{background:#f7f9fc;border-radius:9px;padding:14px}.original p{margin:7px 0}.switch{padding:12px;border-radius:8px;margin-top:18px}.ok{background:#edf9f1}.bad,.cross{background:#fff0ee}.flush{background:#f0f2f5}li{padding:8px 12px;margin:9px 0;border-left:3px solid #b9c8dc}li p{margin:2px 0}.raw{font-family:ui-monospace,monospace;white-space:pre-wrap}summary{cursor:pointer;color:#315fc4;font-weight:600}details{margin:14px 0}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #dce3ec;vertical-align:top}.mode[hidden],article[hidden]{display:none}@media(max-width:650px){.stats,.original{grid-template-columns:1fr}main{padding:0 10px}article,.summary{padding:16px}}
</style></head><body><main><h1>Student distill: luồng Việt → Anh</h1><p>Ghép 20 cặp từ các đoạn Qwen đã sinh: một đoạn Việt rồi một đoạn Anh. Đây là các nguồn độc lập, không phải cặp bản dịch hay hội thoại song ngữ tự nhiên.</p><div class="stats">'''
    page += heading + '</div><section class="summary"><p>Checkpoint <b>' + str(report['checkpoint']['step'])
    page += '</b> · Ngưỡng chung <b>' + f'{threshold:.2f}' + '</b>, chọn trên DEV của chính checkpoint trước bài thử.</p>'
    policy = 'Bộ đệm được xả sau EOT do model dự đoán, không xả theo nhãn đúng.' if report.get('predicted_eot_buffer_reset') else 'Lịch sử chữ trượt theo luồng, không reset theo nhãn đúng.'
    page += '<p>Model chỉ xuất EOT/NEOT, không xuất nhãn ngôn ngữ. Đầu vào chữ thường, không dấu câu, không mã tiếng hoặc vai nói; chỉ nhìn trước 1/2 từ. ' + policy + '</p><p>Chỗ ghép cố ý dùng nhìn trước sang ngôn ngữ/đoạn nguồn khác, khác quy tắc nhìn trước cùng người của dữ liệu train. Nhãn từ dấu câu Qwen, chưa phải human/audio gold. Đoạn cuối xả khi hết luồng không được tính là model tự ngắt.</p>'
    if report.get('comparison_to_rolling_prefix'):
        comparisons = []
        for k, value in report['comparison_to_rolling_prefix'].items():
            a = value['rolling_prefix_splice']; b = value['predicted_reset_splice']
            comparisons.append(f"<tr><td>{k}</td><td>{a['predicted_eot']}/{a['total']}</td><td>{b['predicted_eot']}/{b['total']}</td><td>{value['rolling_prefix_cuts_outside_fullstop']}</td><td>{value['predicted_reset_cuts_outside_fullstop']}</td></tr>")
        page += '<h3>Đối chiếu hai cách chạy bộ đệm</h3><p>Cùng checkpoint và ngưỡng DEV; không đổi model hoặc chọn lại ngưỡng bằng bài thử này.</p><div class="table"><table><tr><th>Nhìn trước</th><th>Giữ lịch sử: EOT ở chỗ ghép</th><th>Xả theo EOT dự đoán: EOT ở chỗ ghép</th><th>Giữ lịch sử: cắt ngoài dấu chấm</th><th>Xả buffer: cắt ngoài dấu chấm</th></tr>' + ''.join(comparisons) + '</table></div>'
    page += '<details><summary>Đối chiếu cùng điểm xét: đơn ngữ và luồng ghép</summary><p>Accuracy / F1 EOT chỉ trên các điểm dấu chấm và phẩy/chấm phẩy có ở cả hai chế độ; không phải điểm toàn bộ các khoảng giữa từ. Cột đơn ngữ là đối chứng giữ lịch sử của bản thử đầu.</p><table><tr><th>Nhìn trước</th><th>Tiếng</th><th>Số điểm</th><th>Đơn ngữ: Acc / F1</th><th>Luồng ghép: Acc / F1</th></tr>' + ''.join(metrics) + '</table><p>Các nhát cắt không trùng điểm EOT theo dấu chấm: nhìn trước 1 từ ' + str(cuts['1']) + ', nhìn trước 2 từ ' + str(cuts['2']) + '. Đây là thống kê theo đích dấu câu, không phải phán quyết về độ đủ nghĩa.</p></details></section>'
    page += '<div class="toolbar"><label>Nhìn trước <select id="k"><option value="2">2 từ</option><option value="1">1 từ</option></select></label><label>Cặp <select id="case"><option value="">Tất cả</option>'
    page += ''.join(f'<option value="{case["number"]}">{case["number"]:02d}</option>' for case in cases)
    page += '</select></label><label><input type="checkbox" id="miss"> Chỉ cặp chưa ngắt tại chỗ ghép</label><span id="count"></span></div>'
    page += ''.join(cards) + '''<footer><p class="muted">Bài thử không cập nhật trọng số, không chọn ngưỡng/checkpoint bằng dữ liệu này. Distill chính chạy độc lập.</p></footer></main><script>
const k=document.querySelector('#k'),which=document.querySelector('#case'),miss=document.querySelector('#miss');function show(){let n=0;document.querySelectorAll('.mode').forEach(e=>e.hidden=e.dataset.k!==k.value);document.querySelectorAll('article').forEach(e=>{e.hidden=(which.value&&e.dataset.case!==which.value)||(miss.checked&&e.getAttribute('data-miss-'+k.value)!=='1');if(!e.hidden)n++});document.querySelector('#count').textContent=n+' cặp'};k.onchange=which.onchange=miss.onchange=show;show();
</script></body></html>'''
    return {'review.html': page.encode('utf-8'), 'paired_streams.txt': '\n'.join(plain).encode('utf-8')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--remote-output', default=REMOTE)
    p.add_argument('--output-dir', type=Path, default=LOCAL)
    args = p.parse_args(); destination = args.output_dir
    assert shutil.disk_usage(Path.cwd()).free >= 10 * 2**30 + 5 * 1024**2
    code = f'''import pathlib,json,base64,hashlib
p=pathlib.Path({args.remote_output!r});assert json.loads((p/'status.json').read_text())['stage']=='COMPLETE'
names=['report.json','cases.json','frozen_dev.json','checkpoint_receipt.json','input_manifest.json','status.json','launcher.json']
files={{n:(p/n).read_bytes() for n in names}};assert sum(map(len,files.values()))<4*1024**2
assert json.loads(files['report.json'])['complete']
print(json.dumps({{n:{{'base64':base64.b64encode(v).decode(),'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)}} for n,v in files.items()}}))'''
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'edabk@100.91.209.124', 'python3 -'],
                            input=code, capture_output=True, text=True, encoding='utf-8', timeout=45, check=True)
    files = json.loads(result.stdout); blobs = {}
    for name, item in files.items():
        data = base64.b64decode(item['base64']); assert hashlib.sha256(data).hexdigest() == item['sha256']
        blobs[name] = data
    report = json.loads(blobs['report.json']); cases = json.loads(blobs['cases.json'])
    assert hashlib.sha256(blobs['cases.json']).hexdigest() == report['cases_sha256']
    blobs.update(render(destination, report, cases))
    assert sum(map(len, blobs.values())) < 5 * 1024**2
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in blobs.items():
        (destination / name).write_bytes(data); assert (destination / name).read_bytes() == data
    receipt = {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for name, data in blobs.items()}
    (destination / 'transfer_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'checkpoint_step': report['checkpoint']['step'], 'switches': report['language_splice_observations'],
                      'report_sha256': receipt['report.json']['sha256'], 'local_bytes': sum(map(len, blobs.values())),
                      'review': str((destination / 'review.html').resolve())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
