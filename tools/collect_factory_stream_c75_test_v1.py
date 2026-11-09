"""Collect small live/final C75 diagnostics; models and traces remain on SSH."""
import base64
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import subprocess

REMOTE = '/home/edabk/hoangbpm/turn_detect/outputs/multilingual_stream_c75_test_20261102_v1'
LOCAL = Path('outputs/multilingual_stream_c75_test_20261102_v1')
LIMIT = 5*2**20


def render(bundle):
    report = bundle['files']['report.json']
    cells = []
    comparisons = []
    lookahead = []
    examples = []
    for lang, result in report['languages'].items():
        m = result['metrics_after_audit']
        fmt = lambda x: '—' if x is None else f'{100*x:.2f}%'
        row = [lang.upper(), str(m['samples']), fmt(m['accuracy']), fmt(m['eot_f1']), fmt(m['eot_precision']),
               fmt(m['eot_recall']), str(m['fp']), str(m['fn']), fmt(m['false_share_of_cuts'])]
        cells.append('<tr>'+''.join('<td>'+x+'</td>' for x in row)+'</tr>')
        comp = report['comparison_to_previous_c75']['by_language'][lang]
        comparisons.append('<tr>'+''.join('<td>'+x+'</td>' for x in [lang.upper(), fmt(comp['old']['accuracy']),
            fmt(comp['old']['eot_f1']), fmt(m['accuracy']), fmt(m['eot_f1']), f"{comp['eot_f1_change_pp']:+.2f}"])+'</tr>')
        for k, values in result['by_lookahead_words'].items():
            lookahead.append('<tr>'+''.join('<td>'+x+'</td>' for x in [lang.upper(), k, str(values['samples']),
                fmt(values['eot_f1']), fmt(values['eot_precision']), fmt(values['eot_recall'])])+'</tr>')
        examples.append('<h2>Ví dụ '+lang.upper()+'</h2>')
        for example in bundle['files'][f'review_{lang}.json']:
            examples.append('<article><p><b>'+('Đúng' if example['correct'] else 'Sai')+'</b> · Nhãn '+escape(example['label'])+
                ' · Dự đoán '+escape(example['predicted'])+f" · P(EOT) {example['p_eot']:.4f} · nhìn trước {example['lookahead_words']} từ</p>"+
                '<p><b>Context:</b> '+escape(example['context'])+'</p><p><b>Text:</b> '+escape(example['text'])+'</p></article>')
    checkpoint = report['checkpoint']
    text = f'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C75 · checkpoint streaming mới</title><style>body{{font:16px/1.6 system-ui;max-width:1250px;margin:35px auto;padding:0 22px;background:#f4f7fb;color:#18283b}}table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}th,td{{padding:10px;border:1px solid #d8e2ee;text-align:left}}article,.note{{padding:16px 22px;background:white;margin:16px 0;border-radius:10px;overflow-wrap:anywhere}}.note{{border-left:4px solid #cb9c2c}}code{{font-size:13px;overflow-wrap:anywhere}}</style>
<h1>Test chẩn đoán C75 · model streaming đang train</h1><p>Checkpoint global <b>{checkpoint['step']:,}</b> · nhánh <b>{checkpoint['branch_step']:,}</b> · ngưỡng chung <b>{report['threshold_from_same_checkpoint_dev']:.2f}</b> chọn trên DEV cùng checkpoint trước khi chấm.</p>
<div class="note">C75 là bộ văn bản nhãn theo dấu câu đã từng xem, dùng context cố định tại từng điểm; không phải luồng xả bộ đệm theo EOT dự đoán, test hội thoại/audio độc lập hoặc human gold. Kết quả này không chọn model/ngưỡng và không được trộn train. Model này chưa học spoken v9 mới; trainer chính tiếp tục chạy. Đây là checkpoint Transformer mới, không phải gói INT8 branch28000.</div>
<h2>Kết quả sau audit</h2><table><tr><th>Tiếng</th><th>Mẫu</th><th>Accuracy</th><th>F1 EOT</th><th>Precision</th><th>Recall</th><th>FP EOT</th><th>FN EOT</th><th>EOT dự đoán sai / mọi EOT dự đoán</th></tr>{''.join(cells)}</table>
<p>FP/FN tính trên các điểm chấm C75; không chuyển thành tỷ lệ cắt nhầm trong luồng streaming hay so trực tiếp với tỷ lệ của bài test 200 đoạn.</p>
<h2>Theo số từ nhìn trước</h2><table><tr><th>Tiếng</th><th>Lookahead, từ</th><th>Mẫu</th><th>F1 EOT</th><th>Precision</th><th>Recall</th></tr>{''.join(lookahead)}</table>
<h2>Đối chiếu C75 trước đó · cùng các mẫu còn lại</h2><p>Checkpoint cũ {report['comparison_to_previous_c75']['checkpoint_step']:,} dùng ngưỡng DEV riêng {report['comparison_to_previous_c75']['old_threshold']:.2f}; cách hiệu chuẩn và miền dữ liệu huấn luyện đã thay đổi. Dùng lại trace cũ để tính đối chiếu, không chạy lại model cũ.</p>
<table><tr><th>Tiếng</th><th>Accuracy cũ</th><th>F1 cũ</th><th>Accuracy mới</th><th>F1 mới</th><th>Δ F1, điểm %</th></tr>{''.join(comparisons)}</table>
<h2>Kiểm tra trùng dữ liệu</h2><p>Đã đối chiếu lại SHA bộ C75, trace cũ và fingerprint cycle84 để dùng lại quyết định audit cycle84 trước đó; không tuyên bố quét lại toàn bộ SQLite. Đã kiểm tra thêm context+future chuẩn hóa xuất hiện nguyên văn trong mọi đoạn TRAIN/DEV của corpus nhà máy source722; có {report['factory_overlap_audit']['matching_examples']} mẫu trùng tiềm năng và đã loại thận trọng.</p>
<p>Chưa kiểm hết va chạm chỉ ở tokenizer, lịch sử khác nhưng trùng phần bị cắt, các phép nối hai nguồn tổng hợp hoặc paraphrase. Nên không gọi đây là test độc lập.</p>
<p><a href="report.json">Report JSON</a> · <a href="frozen_dev.json">DEV khóa ngưỡng</a> · <a href="checkpoint_receipt.json">Biên nhận checkpoint</a> · <a href="factory_overlap_audit.json">Audit corpus mới</a> · <a href="completion_verification.json">Kiểm tra toàn bộ trace</a></p>
{''.join(examples)}<p>Report SHA-256: <code>{bundle['report_sha256']}</code></p></html>'''
    (LOCAL/'review.html').write_text(text, encoding='utf-8')


def main():
    free = shutil.disk_usage(Path.cwd()).free/2**30
    assert free >= 10+5/1024, 'Keep Windows reserve; no report bundle until sufficient space'
    code = f'''import pathlib,json,base64,hashlib,os,subprocess
r=pathlib.Path({REMOTE!r});v={{'files':{{}}}}
for n in ['status.json','error.json','launcher.json','checkpoint_receipt.json','frozen_dev.json','input_manifest.json','factory_overlap_audit.json','partial_report.json','report.json','review_vi.json','review_en.json','review_zh.json','completion_verification.json']:
 if (r/n).exists():v['files'][n]=json.loads((r/n).read_text())
if (r/'report.json').exists():
 raw=(r/'report.json').read_bytes();v['report_sha256']=hashlib.sha256(raw).hexdigest();v['report_b64']=base64.b64encode(raw).decode()
if (r/'launcher.json').exists():
 expected=v['files']['launcher.json'];p=pathlib.Path('/proc')/str(expected['pid'])
 if p.exists():
  raw=(p/'stat').read_text();ticks=raw[raw.rfind(')')+2:].split()[19];v['process']={{'present':True,'start_ticks':ticks,'expected_ticks_match':ticks==expected['start_ticks'],'cmdline':(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()}}
 else:v['process']={{'present':False}}
main=r.parent/'multilingual_stream_v2_20261102';v['main_status']=json.loads((main/'status.json').read_text())
p=pathlib.Path('/proc/2827094')
if p.exists():
 raw=(p/'stat').read_text();v['main_process']={{'start_ticks':raw[raw.rfind(')')+2:].split()[19],'cmdline':(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()}}
v['gpu_processes']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
for path in [str(r),'/mnt/hdd500']:
 s=os.statvfs(path);v.setdefault('disk_free_gib',{{}})[path]=s.f_bavail*s.f_frsize/2**30
raw=json.dumps(v,ensure_ascii=False);assert len(raw.encode())<={LIMIT};print(raw)
'''
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'edabk@100.91.209.124', 'python3 -'],
                            input=code, capture_output=True, text=True, encoding='utf-8', check=True, timeout=45)
    assert len(result.stdout.encode()) <= LIMIT
    bundle = json.loads(result.stdout)
    bundle.update(checked_utc=datetime.now(timezone.utc).isoformat(), windows_free_gib=free)
    LOCAL.mkdir(exist_ok=True, parents=True)
    for name, value in bundle['files'].items():
        (LOCAL/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    if 'report_b64' in bundle:
        raw = base64.b64decode(bundle.pop('report_b64'))
        assert hashlib.sha256(raw).hexdigest() == bundle['report_sha256']
        (LOCAL/'report.json').write_bytes(raw)
    (LOCAL/'heartbeat_verification.json').write_text(json.dumps(bundle, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    if bundle['files'].get('report.json', {}).get('complete') and bundle['files']['status.json']['stage'] == 'COMPLETE':
        render(bundle)
        receipt = {'report_sha256': bundle['report_sha256'], 'transfer_bytes': len(result.stdout.encode()),
                   'models_or_full_traces_downloaded': False,
                   'local_files': {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                   for p in LOCAL.iterdir() if p.is_file() and p.name != 'transfer_receipt.json'}}
        (LOCAL/'transfer_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    summary = {k: bundle['files'].get(k) for k in ['status.json', 'error.json', 'checkpoint_receipt.json']}
    summary.update(process=bundle.get('process'), main_status=bundle['main_status'], disk=bundle['disk_free_gib'])
    if bundle['files'].get('report.json', {}).get('complete'):
        summary.update(report_sha256=bundle['report_sha256'], languages={l: v['metrics_after_audit'] for l, v in bundle['files']['report.json']['languages'].items()},
                       review=str((LOCAL/'review.html').resolve()))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
