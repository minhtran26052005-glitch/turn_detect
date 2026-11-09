"""Read-only result review builder; requires COMPLETE report and proof."""
from pathlib import Path
import gzip
import hashlib
import html
import json
import re
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/context_window_ab_20261108_v1'
ARMS = ('predicted_reset', 'source_window')


def pct(value):
    return f'{100*(value or 0):.2f}'


def render_markdown(text):
    """Small dependency-free renderer for this generated report's headings/table."""
    def inline(value):
        return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html.escape(value))
    output = []; table = False
    for line in text.splitlines():
        if line.startswith('|'):
            if not table:
                output.append('<table>'); table = True
            if re.match(r'^\|[-:| ]+\|$', line):
                continue
            output.append('<tr>'+''.join('<td>'+inline(x.strip())+'</td>' for x in line.strip('|').split('|'))+'</tr>')
            continue
        if table:
            output.append('</table>'); table = False
        match = re.match(r'^(#{1,3}) (.*)$', line)
        if match:
            level = len(match[1]); output.append(f'<h{level}>'+inline(match[2])+f'</h{level}>')
        elif line:
            output.append('<p>'+inline(line)+'</p>')
    if table:
        output.append('</table>')
    return '\n'.join(output)


def build():
    report = json.loads((OUT/'report.json').read_text())
    proof = json.loads((OUT/'completion_verification.json').read_text())
    assert report['complete'] and proof['complete']
    assert hashlib.sha256((OUT/'report.json').read_bytes()).hexdigest() == proof['report_sha256']
    selections = {}
    for arm in ARMS:
        selection = json.loads((OUT/arm/'selection_receipt.json').read_text())
        best = max(selection['calibration'], key=lambda x: (min((m['eot_f1'] or 0) for m in x['by_language'].values()),
                                                           sum((m['eot_f1'] or 0) for m in x['by_language'].values())))
        assert best['threshold'] == report['arms'][arm]['selected']['threshold'] == selection['threshold']
        selections[arm] = dict(threshold=selection['threshold'], unique_calibration_model_forwards=selection['unique_calibration_model_forwards'])
    monitor = json.load(gzip.open(OUT/'dev_monitor.json.gz', 'rt', encoding='utf-8'))
    positions = {}
    with gzip.open(OUT/ARMS[0]/'monitor.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line); positions[row['stream'], row['boundary']] = row
    examples = []; counts = Counter(); transitions = Counter()
    with gzip.open(OUT/ARMS[1]/'monitor.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            b = json.loads(line); a = positions.pop((b['stream'], b['boundary']))
            assert all(a[k] == b[k] for k in ('stream', 'boundary', 'label', 'language', 'scored'))
            if not a['scored']:
                continue
            transitions[a['language']+':'+str(a['label'])+':'+str(int(a['predicted_eot']))+'->'+str(int(b['predicted_eot']))] += 1
            if a['predicted_eot'] == b['predicted_eot']:
                continue
            kind = ('TP_gain' if a['label'] else 'FP_added') if b['predicted_eot'] else ('TP_lost' if a['label'] else 'FP_removed')
            s = monitor[b['stream']]; bucket = (b['language'], kind)
            if counts[bucket] >= 2:
                continue
            counts[bucket] += 1; boundary = b['boundary']
            examples.append(dict(language=b['language'], domain=s['domain'], family=s['family'],
                stream=b['stream'], boundary=boundary, silver_eot_label=b['label'], change=kind,
                source_left=' '.join(s['words'][max(0,boundary-35):boundary]),
                future=' '.join(s['words'][boundary:boundary+2]),
                a_context_tail=' '.join(s['words'][max(a['context_start'],boundary-35):boundary]),
                b_context_tail=' '.join(s['words'][max(b['context_start'],boundary-35):boundary]),
                a_start=a['context_start'], b_start=b['context_start'],
                a_p_eot=a['p_eot'], b_p_eot=b['p_eot'], a_cut=a['predicted_eot'], b_cut=b['predicted_eot']))
    assert not positions
    (OUT/'paired_dev_cases.json').write_text(json.dumps(dict(complete=True,examples=examples,transitions=dict(transitions),
        scope='Deterministic first two disagreements per language/change category; silver punctuation agreement, not semantic annotation'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    headers = ['## Material Passport', '', '- Origin Skill: academic-research-suite / experiment-agent',
               '- Origin Mode: run and descriptive validation', '- Origin Date: 2026-11-08',
               '- Verification Status: VERIFIED for declared technical checks; CAUTION for generalization',
               '- Version Label: context_window_ab_20261108_v1', '', '# Đối chứng ngữ cảnh A/B', '',
               'Hai nhánh mới cùng xuất phát từ checkpoint v7 3.731.173, giữ nguyên model 11.941.122 tham số, model/27 trạng thái AdamW/RNG/thứ tự nguồn ban đầu. Mỗi nhánh học đủ 333.315 điểm ASR tự nhiên trong một lượt; effective batch 16 mẫu được chấm, 20.833 lần cập nhật. LR 4e-6, loss NEOT1/EOT2 và future 1–2 từ cùng lượt nguồn giữ nguyên.', '',
               '- **A:** encoder nhìn từ lần predicted EOT gần nhất.',
               '- **B:** encoder nhìn tối đa 256 từ nguồn gần nhất, độc lập với quyết định cắt; cả hai được encode tối đa 256 token tổng. Quyết định cắt cho decoder vẫn được theo dõi.',
               '- Chỉ thay chính sách ngữ cảnh trong TRAIN và DEV. Hai nhánh fit xong trước khi chấm monitor. Không chạy TEST/translator hoặc thay model mặc định.', '',
               '## Kết quả DEV', '', '| Nguồn/ngôn ngữ | A: F1 / P / R (%) | B: F1 / P / R (%) | ΔF1 (điểm %) |',
               '|---|---:|---:|---:|']
    a, b = (report['arms'][arm]['selected'] for arm in ARMS)
    for lang in ('vi','en'):
        x,y = a['by_language'][lang],b['by_language'][lang]
        headers.append(f"| Toàn DEV {lang.upper()} | {' / '.join(pct(x[k]) for k in ('eot_f1','eot_precision','eot_recall'))} | {' / '.join(pct(y[k]) for k in ('eot_f1','eot_precision','eot_recall'))} | {100*((y['eot_f1'] or 0)-(x['eot_f1'] or 0)):+.2f} |")
    da = {(x['domain'],x['language']):x for x in a['by_domain']}
    for y in b['by_domain']:
        x = da[y['domain'],y['language']]
        headers.append(f"| {y['domain']} {y['language'].upper()} | {' / '.join(pct(x[k]) for k in ('eot_f1','eot_precision','eot_recall'))} | {' / '.join(pct(y[k]) for k in ('eot_f1','eot_precision','eot_recall'))} | {100*((y['eot_f1'] or 0)-(x['eot_f1'] or 0)):+.2f} |")
    u = report['uncertainty']; lo,hi = u['primary_percentile95_ci']
    headers += ['', f"Ngưỡng calibration chọn riêng: A={a['threshold']}, B={b['threshold']}. Đây là so sánh hai pipeline đã calibration, không phải weights-only ở cùng ngưỡng.", '',
        f"Chênh lệch chỉ số chính (F1 ngôn ngữ yếu hơn trên toàn DEV): **{100*u['primary_difference']:+.2f} điểm %**, CI95% bootstrap theo {u['families']} family: **[{100*lo:+.2f}; {100*hi:+.2f}]**, 2.000 lượt lấy mẫu ghép cặp, seed 20261108.", '',
        f"Điều kiện thăm dò khóa trước: **{'ĐẠT' if u['exploratory_promising'] else 'CHƯA ĐẠT'}**. Đây không phải gate chấp nhận sản phẩm. Không tự nhận/adopt hay mở thêm epoch.", '',
        '## Kiểm chứng và phạm vi', '',
        '- Fixture mới của cả hai chính sách: ngắt giữa gradient window rồi resume cho kết quả bitwise bằng chạy liền.',
        '- Audit 100 cập nhật đầu mỗi nhánh: 1.600 mẫu được chấm; kiểm source, nhãn, mask, future, reset và encoded-token hash; model hữu hạn và đã thay đổi.',
        '- Kiểm đủ một lượt 333.315 điểm/420 task mỗi nhánh; 27 AdamW, toàn bộ RNG/order/cursor/lanes và trạng thái accumulator; weights khớp full resume.',
        f"- Audit {proof['audited_trace_records']:,} bản ghi calibration/monitor; {proof['canonical_cases']} ca đơn đối chiếu, sai lệch xác suất lớn nhất {proof['max_probability_delta']:.3g}. Không audit lại mọi training forward.",
        '- B dùng lại xác suất calibration chỉ khi source-anchored input giống hệt. Số bản ghi audit không phải số forward GPU mới; số forward riêng nằm trong selection_receipt.json.',
        '- Checkpoint cha v7 giữ nguyên SHA, vẫn PAUSED và STOP_REQUESTED. Không phục hồi checkpoint fail đã xóa.', '',
        'Padding, lượng tính toán và lịch sử reset/dropout có thể khác sau khi đổi context. Cùng RNG ban đầu không có nghĩa mọi dropout mask sau đó giống nhau. Kết quả không tách riêng tác động context lúc train khỏi lúc suy luận.', '',
        '## Diễn giải thống kê — 11/11 loại sai lệch đã xem xét', '',
        '1. Simpson: báo cả toàn bộ và từng nguồn/ngôn ngữ; không suy từ điểm gộp rằng mọi nguồn tốt hơn.',
        '2. Ecological: CI theo family, không suy thành độ đúng ngữ nghĩa từng câu.',
        '3. Selection/Berkson: đây là ASR TRAIN đã lọc và DEV cố định; kết quả có điều kiện trên các nguồn này.',
        '4. Collider: không lọc mẫu theo dự đoán/model-score; cùng denominator ở hai nhánh.',
        '5. Base rate: báo TP/FP/FN và số EOT trong report; không dùng accuracy tổng làm thành công.',
        '6. Regression to mean: có control cùng điểm khởi đầu, không chỉ so với một điểm thấp trước đây.',
        '7. Survivorship: cả hai hoàn tất ngân sách; báo cả kết quả âm, không giữ riêng nhánh tốt.',
        '8. Look-elsewhere: chỉ một chỉ số chính khóa trước; chỉ số nguồn/ngôn ngữ là mô tả/guard, không gán significance hàng loạt.',
        '9. Forking paths: protocol được hash trước fit; DEV đã xem nhiều lần nên vẫn là thăm dò, không phải đăng ký trước trên tập kín mới.',
        '10. Causation: một seed, can thiệp chính sách context cả TRAIN+DEV; không tách riêng train-context, eval-context hay chứng minh nguyên nhân ngữ nghĩa duy nhất.',
        '11. Reverse causality: chính sách khóa trước kết quả; không sửa ngưỡng bằng monitor/TEST.', '',
        'CI percentile là độ bất định lấy mẫu có điều kiện trên outputs hiện tại và giả định family là đơn vị độc lập. Không bao gồm phương sai seed training, lỗi nhãn punctuation-silver, chồng lặp lineage chưa biết hoặc lệch domain. Không diễn giải CI như xác suất tham số nằm trong khoảng; không suy từ CI chứa 0 rằng hai phương pháp tương đương.', '',
        'Không có human/audio gold, latency đầu cuối hoặc điểm dịch mới. Giới hạn nguồn nhãn/future và population được giữ nguyên. Xem locked_protocol.json, report.json, completion_verification.json và paired_dev_cases.json để kiểm tra chi tiết.', '',
        '## Các điều kiện thăm dò khóa trước', '']
    headers += [f"- {'PASS' if value else 'FAIL'}: {key}" for key,value in u['checks'].items()]
    markdown_text = '\n'.join(headers)+'\n'
    (OUT/'README.md').write_text(markdown_text,encoding='utf-8')
    cases_html = ''
    for c in examples:
        cases_html += '<details><summary>'+html.escape(f"{c['language'].upper()} · {c['domain']} · {c['change']}")+'</summary>'
        cases_html += '<p>'+html.escape(c['source_left'])+' <b>│ '+html.escape(c['future'])+'</b></p>'
        cases_html += '<p>'+html.escape(f"Nhãn silver={c['silver_eot_label']}; A p={c['a_p_eot']:.5f}, cut={c['a_cut']}; B p={c['b_p_eot']:.5f}, cut={c['b_cut']}; context start A/B={c['a_start']}/{c['b_start']}")+'</p></details>'
    body = render_markdown(markdown_text)
    page = '<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Đối chứng ngữ cảnh A/B</title><style>body{font:17px/1.6 system-ui;background:#f1f4f8;color:#172235;margin:0}main{max-width:1100px;margin:24px auto;padding:30px;background:white}table{border-collapse:collapse;width:100%;display:block;overflow:auto;font-size:15px}td,th{border:1px solid #ccd5e2;padding:8px}th{background:#edf2f9}h1{font-size:30px}details{padding:12px;border:1px solid #d5deea;margin:10px 0}summary{cursor:pointer}@media(max-width:600px){main{margin:0;padding:18px}}</style><main>'+body+'<h2>Ca DEV ghép cặp</h2><p>Ví dụ bất đồng với nhãn dấu câu; không phải đánh giá ngữ nghĩa bởi người.</p>'+cases_html+'</main></html>'
    (OUT/'review.html').write_text(page,encoding='utf-8')
    summary = dict(complete=True, selections=selections, overall={arm:report['arms'][arm]['selected']['by_language'] for arm in ARMS},
                   domains={arm:report['arms'][arm]['selected']['by_domain'] for arm in ARMS},
                   uncertainty=u, adopted=False, parent_paused=True, new_test=False)
    (OUT/'user_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    checks = dict(complete=True, report_sha256=proof['report_sha256'], calibration_selection_recomputed=True,
                  monitor_populations_paired_exact=True, paired_cases=len(examples), statistical_fallacy_checks=11,
                  files={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ('README.md','review.html','user_summary.json','paired_dev_cases.json')})
    (OUT/'review_verification.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(checks))


if __name__ == '__main__':
    build()
