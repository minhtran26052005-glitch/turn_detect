"""Collect the fresh 200-paragraph diagnostic, without downloading weights."""
from pathlib import Path
import collect_factory_stream_test_v1 as original

original.REMOTE = '/home/edabk/hoangbpm/turn_detect/outputs/multilingual_stream_test_20261103_v1'
original.LOCAL = Path('outputs/multilingual_stream_test_20261103_v1')
base_render = original.render


def render(report, cases):
    page = base_render(report, cases).decode('utf-8')
    prior = report['previous_stream_test_comparison']
    rows = []
    for mode, label in [('monolingual', 'Riêng từng tiếng'), ('mixed', 'Ghép Việt → Anh')]:
        for language in ('vi', 'en'):
            a = prior['metrics'][mode]['2'][language]
            b = report['metrics_every_eligible_word_gap'][mode]['2'][language]
            rows.append(f'<tr><td>{label}</td><td>{language.upper()}</td>'
                        f'<td>{a["fp"]} → {b["fp"]}</td><td>{a["fn"]} → {b["fn"]}</td>'
                        f'<td>{original.pct(a["eot_f1"])} → {original.pct(b["eot_f1"])}</td></tr>')
    extra = '<section class="summary"><h2>So với lần test streaming trước trên cùng 200 đoạn</h2>'
    extra += f'<p>Checkpoint cũ {prior["checkpoint"]["step"]:,}, ngưỡng DEV riêng {prior["threshold"]:.2f}. Dùng báo cáo cũ đã kiểm SHA; không chạy lại model cũ hoặc chọn ngưỡng từ test.</p>'
    extra += '<table><tr><th>Chế độ · nhìn trước 2 từ</th><th>Tiếng</th><th>Cắt nhầm cũ → mới</th><th>Bỏ sót cũ → mới</th><th>F1 cũ → mới</th></tr>'+''.join(rows)+'</table>'
    extra += '<p>Checkpoint này chưa học dữ liệu spoken v9 mới. Bộ 200 đoạn đã từng xem, nhãn dấu câu; không phải test độc lập trên lời nói thật. Toàn bộ trace đã được kiểm tra lại, không chỉ các ví dụ trong trang.</p></section>'
    return page.replace('<div class="toolbar">', extra+'<div class="toolbar">', 1).encode('utf-8')


original.render = render
if __name__ == '__main__':
    original.main()
