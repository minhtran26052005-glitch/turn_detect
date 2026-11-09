"""Small inspectable teacher dashboard and verified read-only notebook cell."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
from pathlib import Path
import shutil

from build_notebook import cell
from teacher_first_colab import campaign_runtime

ROOT = Path('outputs/teacher_first_v14_20261026')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--notebook', action='store_true'); args = parser.parse_args()
    value = json.loads((ROOT / 'status_receipt.json').read_text(encoding='utf-8'))
    status = value.get('status.json', {}); rows = []
    for lang in ['vi', 'en', 'zh']:
        report = value.get('champions', {}).get(lang)
        identity = value.get('teachers', {}).get(lang, {}).get('identity.json', {})
        progress = value.get('teachers', {}).get(lang, {}).get('progress.json', {})
        row = {'language': lang, 'parameters': identity.get('parameters'), 'progress': progress, 'report': report}
        rows.append(row)
    summary = {'queried_at_utc': value['queried_at_utc'], 'worker': status,
               'parent_verified': value.get('launcher.json_process_verified'), 'child_verified': value.get('child.json_process_verified'),
               'GPU': value.get('GPU'), 'remote_disk_free_GiB': round(value.get('remote_disk_free_bytes', 0) / 2**30, 2),
               'teachers': rows, 'data_counts': value.get('pool_counts'), 'gate': value.get('distillation_gate.json'),
               'snapshot': value.get('recovery/latest_snapshot.json'), 'quality_target_certified': False,
               'teacher_weights_backed_up_externally': False, 'serving_models_replaced': False}
    punctuation_reports = sorted(ROOT.glob('terminal_punctuation_eval_*/report.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if punctuation_reports:
        latest = punctuation_reports[0]
        terminal = json.loads(latest.read_text(encoding='utf-8'))
        summary['terminal_punctuation_evaluation'] = {
            'protocol': terminal.get('protocol'), 'scope': terminal.get('scope'),
            'report': latest.relative_to(ROOT).as_posix(),
            'review': latest.with_name('review.html').relative_to(ROOT).as_posix(),
            'languages': terminal.get('languages'),
            'quality_target_certified': terminal.get('quality_target_certified', False),
        }
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    def esc(v): return html.escape(json.dumps(v, ensure_ascii=False, indent=2))
    def scores(metrics): return f"{100*metrics['accuracy']:.2f}% / {100*metrics['eot_f1']:.2f}% ({metrics['samples']} mẫu)"
    table = ''
    for row in rows:
        report = row['report']; progress = row['progress']
        table += '<tr><td>' + row['language'] + '</td><td>' + (f"{row['parameters']:,}" if row['parameters'] else 'Đang chờ nạp') + '</td><td>'
        table += str(progress.get('total_steps', 0)) + '</td><td>' + (scores(report['semantic_dev']) if report else 'Chưa đánh giá') + '</td><td>'
        table += (scores(report['natural_floor_dev']) if report else 'Chưa đánh giá') + '</td></tr>'
    queried = datetime.fromisoformat(value['queried_at_utc']).astimezone(timezone(timedelta(hours=7))).isoformat()
    page = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60"><title>Teacher lớn trước — VI EN ZH</title><style>body{font:17px system-ui;line-height:1.65;max-width:1200px;margin:24px auto;padding:24px;background:#f1f5fa;color:#19354a}section{background:white;border-radius:12px;padding:22px;margin:20px 0}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}.scroll{overflow:auto}</style>
<h1>Train teacher lớn trước rồi mới distill</h1><p>Ba model riêng VI / EN / ZH từ <a href="https://huggingface.co/microsoft/mdeberta-v3-base">microsoft/mdeberta-v3-base</a>: 12 lớp, hidden 768, khoảng 278 triệu params/model. Fine-tune toàn bộ; chưa nén trong pha này.</p>
<section><h2>Trạng thái lần kiểm tra</h2><p>Thời điểm: ''' + html.escape(queried) + '</p><pre>' + esc({k: summary[k] for k in ['parent_verified', 'child_verified', 'GPU', 'remote_disk_free_GiB']}) + '</pre><pre>' + esc(status) + '''</pre><p>Đây là snapshot lần kiểm tra; lịch theo dõi cập nhật mỗi 15 phút khi app có thể chạy. Ô cuối a.ipynb đọc kernel hiện tại. Tải lại trang không giữ VM Colab sống.</p></section>
<section><h2>Teacher được giữ tốt nhất theo silver dev</h2><div class="scroll"><table><tr><th>Tiếng</th><th>Params</th><th>Bước train gần nhất</th><th>Accuracy / F1 semantic silver dev</th><th>Accuracy / F1 observed-floor dev</th></tr>''' + table + '''</table></div><p>Checkpoint baseline dùng encoder pretrained với head EOT mới khởi tạo. Điểm được chọn trên silver dev nhỏ (VI31 / EN52 / ZH22), không phải bài kiểm tra độc lập. Cột floor dùng ngưỡng của semantic dev và đo việc người nói thực sự nhường lượt; hai mục tiêu khác nhau.</p><p><a href="natural_floor_review/review.html">Mở natural-floor: input và dự đoán từng mẫu</a> · <a href="natural_floor_review/vi_natural_floor_predictions.jsonl">VI JSONL</a> · <a href="natural_floor_review/en_natural_floor_predictions.jsonl">EN JSONL</a> · <a href="natural_floor_review/zh_natural_floor_predictions.jsonl">ZH JSONL</a></p></section>
<section><h2>Điều kiện trước khi distill</h2><p>Teacher đạt cả accuracy và F1 EOT ≥90%, có kiểm chứng độc lập đúng nhiệm vụ trước khi distill. Chưa có human/publisher semantic EOT gold độc lập cho cả ba tiếng; silver90 không tự mở khóa distillation. Hiện chưa thay các gói INT4.</p><pre>''' + esc(summary['gate']) + '''</pre></section>
<section><h2>Dữ liệu tích lũy</h2><pre>''' + esc(summary['data_counts']) + '''</pre><p>Qwen chỉ nhìn text đã quan sát và ngữ cảnh quá khứ; nhãn cắt prefix cũ được chấm lại. Không lấy dấu câu hay phần tương lai làm đáp án.</p></section>
<section><h2>Lưu và khôi phục</h2><p>Teacher best và optimizer/RNG được lưu trên Colab. Metadata nhỏ tải về máy không chứa teacher weights và không đủ khôi phục training khi mất VM. Chưa xác nhận backup toàn bộ teacher ra ngoài VM; không tải model lớn xuống Windows.</p><pre>''' + esc(summary['snapshot']) + '</pre><p><a href="summary.json">JSON</a> · <a href="../../README_TEACHER_FIRST.md">Cách chạy</a> · <a href="../continuous_v13_20261026/review.html">Đối chứng GRU v13 đã dừng để nhường GPU</a></p></section></html>'
    if punctuation_reports:
        review_href = punctuation_reports[0].with_name('review.html').relative_to(ROOT).as_posix()
        page = page.replace('<p><a href="natural_floor_review/review.html">',
                            f'<p><a href="{html.escape(review_href, quote=True)}">Mở kiểm tra mới: NEOT trước dấu chấm, EOT tại dấu chấm</a> · <a href="natural_floor_review/review.html">',
                            1)
    (ROOT / 'review.html').write_text(page, encoding='utf-8')
    if args.notebook:
        path = Path('a.ipynb'); nb = json.loads(path.read_text(encoding='utf-8')); tag = 'teacher_first_v14_status'
        source = Path('tools/templates/teacher_first_status_cell.py').read_text(encoding='utf-8')
        found = [c for c in nb['cells'] if tag in c.get('metadata', {}).get('tags', []) and c['cell_type'] == 'code']
        if not found:
            shutil.copy2(path, ROOT / 'a_before_teacher_first.ipynb')
            nb['cells'] += [cell('markdown', '## Teacher lớn trước — VI / EN / ZH\n\nFine-tune ba teacher pretrained lớn độc lập, đánh giá trước khi distill. [Dashboard](outputs/teacher_first_v14_20261026/review.html).', tag), cell('code', source, tag)]
            found = [nb['cells'][-1]]
        current = found[0]; assert ''.join(current['source']) == source
        runtime = campaign_runtime(); runtime.connect(); result = runtime.execute(source, timeout=60, echo=False); runtime.close()
        assert result['status'] == 'ok'
        proof = ROOT / ('notebook_receipt_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '.json')
        proof.write_text(json.dumps({'source_sha256': hashlib.sha256(source.encode()).hexdigest(), **result}, ensure_ascii=False, indent=2), encoding='utf-8')
        current['outputs'] = result['outputs']; current['metadata']['verification_receipt'] = str(proof)
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'status': status.get('stage'), 'teachers_loaded': sum(r['parameters'] is not None for r in rows),
                      'notebook_refreshed': args.notebook, 'quality_target_certified': False}))


if __name__ == '__main__': main()
