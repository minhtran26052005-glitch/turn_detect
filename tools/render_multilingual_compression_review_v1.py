"""Render compact compression evidence; never retrieve model weights."""
from datetime import datetime, timezone
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT/'outputs/multilingual_tiny_compression_20261102_v1'
CONTROL = ROOT/'outputs/multilingual_transformer_compression_20261102_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def main():
    tiny = read(LOCAL/'final_report.json')
    control = read(CONTROL/'final_report.json')
    assert tiny and tiny['complete'], 'Tiny experiment must finish before rendering final evidence'
    if not tiny['accepted']:
        assert control and control['complete'], 'Compression control has not finished'
    selected = tiny if tiny['accepted'] else control if control.get('accepted') else None
    teacher = tiny.get('teacher') or read(LOCAL/'teacher_dev.json')
    receipt = read(LOCAL/'teacher_receipt.json')
    candidates = tiny['candidates'] + (control.get('candidates', []) if control else [])
    rows = []
    def metric_cell(m):
        return f"{100*m['eot_f1']:.2f}% / {100*m['eot_precision']:.2f}% / {100*m['eot_recall']:.2f}%"
    def table_row(name, size, results, passed):
        columns = [html.escape(name), f'{size/2**20:.3f} MiB',
                   *[metric_cell(results[lang]) for lang in ['vi', 'en', 'zh']], passed]
        return '<tr>'+''.join('<td>'+x+'</td>' for x in columns)+'</tr>'
    rows.append(table_row('Frozen teacher: shared multilingual DistilBERT', receipt['weight_bytes'], teacher['by_language'], 'Mốc đối chiếu'))
    for candidate in candidates:
        name = candidate.get('candidate') or Path(candidate['path']).name
        rows.append(table_row(name, candidate['model_bytes'], candidate['dev']['by_language'],
                              'ĐẠT mức giữ DEV' if candidate['retention_passed'] else 'KHÔNG ĐẠT mức giữ DEV'))
    title = 'Đã có bản nén giữ chất lượng DEV' if selected else 'Chưa có bản nén qua mức giữ chất lượng DEV'
    artifact = ''
    benchmarks = ''
    if selected:
        winner = selected['winner']
        artifact = ('<p>Gói giữ trên SSH: <code>'+html.escape(selected['package'])+'</code></p>'
                    '<p>ZIP SHA-256: <code>'+selected['archive_sha256']+'</code></p>'
                    '<p>Model SHA-256: <code>'+winner['model_sha256']+'</code></p>')
        benchmark_rows = []
        for name, result in selected['cpu_benchmark'].items():
            benchmark_rows.append('<tr>'+''.join('<td>'+x+'</td>' for x in [html.escape(name),
                                   f"{result['median_ms']:.3f}", f"{result['p95_ms']:.3f}",
                                   f"{result['peak_process_rss_bytes']/2**20:.1f}"])+'</tr>')
        benchmarks = ('<h2>CPU thực đo</h2><p>Batch một, hai thread, gồm chuẩn hóa và tokenizer; máy dùng chung. '
                      'RSS là đỉnh của toàn tiến trình, không chỉ tensor. Thư viện runtime không nằm trong dung lượng gói model.</p>'
                      '<table><tr><th>Backend/model</th><th>Median ms</th><th>P95 ms</th><th>Peak RSS MiB</th></tr>'
                      +''.join(benchmark_rows)+'</table>')
    summary = {'complete': True, 'accepted': selected is not None,
               'tiny_distilled_model_accepted': tiny['accepted'],
               'selected_family': 'tiny_GRU_KD_QAT' if tiny['accepted'] else 'original_architecture_PTQ_control' if selected else None,
               'teacher_receipt': receipt, 'selected': selected,
               'source_spoken_v9_used': False, 'independent_test_evaluated': False,
               'scope': 'VI/EN predicted-reset synthetic punctuation DEV; ZH legacy punctuation DEV anchors',
               'created_utc': datetime.now(timezone.utc).isoformat()}
    (LOCAL/'compression_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    page = f'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nén model dùng chung Việt–Anh–Trung</title><style>
body{{font:16px/1.55 system-ui,sans-serif;max-width:1280px;margin:40px auto;padding:0 24px;color:#182432;background:#f7f9fc}}
table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{padding:12px;border:1px solid #dce2e9;text-align:left;vertical-align:top}}
th{{background:#eaf0f7}}code{{overflow-wrap:anywhere;font-size:13px}}.note{{padding:16px;background:#fff5d9;border-left:4px solid #dca72c}}h2{{margin-top:36px}}
</style><h1>{title}</h1><p>Model chung Việt–Anh–Trung, teacher đóng băng branch {receipt['branch_step']}, global {receipt['step']}.</p>
<p class="note">Kết quả dưới đây là DEV nhãn dấu câu, không phải test hội thoại/audio độc lập. Nguồn học là source722 và ZH cũ, chưa dùng dữ liệu spoken v9 mới. INT4/INT8 là định dạng lưu trọng số; tính toán FP32. Chưa thay model đang chạy và chưa backup gói SSH lên Drive.</p>
<p>Đã distill GRU trên 300.000 mẫu TRAIN, 100.000 mỗi tiếng; tokenizer chỉ học TRAIN. Các gói GRU không đạt được giữ làm bằng chứng thử nghiệm. Phương án Transformer giữ nguyên kiến trúc sáu lớp và chỉ lượng tử hóa lưu trọng số, không gọi là tiny GRU/QAT.</p>
<h2>Đối chiếu chất lượng và dung lượng</h2><p>Mỗi ô ngôn ngữ ghi <strong>F1 / precision / recall EOT</strong>. Đạt khi từng tiếng giảm tối đa 2 điểm phần trăm F1, 2 điểm precision, 3 điểm recall so với teacher. Mỗi model dùng ngưỡng chung chọn trên DEV, không dùng test.</p>
<table><tr><th>Model</th><th>Tệp trọng số</th><th>Việt</th><th>Anh</th><th>Trung</th><th>Kết luận giữ DEV</th></tr>{''.join(rows)}</table>
{benchmarks}<h2>Artifact và bằng chứng</h2>{artifact}<p>Không tải trọng số xuống Windows. Report, optimizer, RNG, cache, mô hình gốc và các ứng viên giữ trên SSH.</p>
<p><a href="compression_summary.json">Report tổng hợp JSON</a> · <a href="final_report.json">Thử nghiệm GRU</a></p></html>'''
    target = LOCAL/'review.html'
    target.write_text(page, encoding='utf-8')
    assert target.stat().st_size+(LOCAL/'compression_summary.json').stat().st_size < 5*2**20
    print(json.dumps({'review': str(target), 'accepted': selected is not None,
                      'tiny_accepted': tiny['accepted'], 'family': summary['selected_family']}))


if __name__ == '__main__':
    main()
