"""Render bounded verified DEV calibration results, without choosing a model."""
from pathlib import Path
import json,html,hashlib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/malbert_calibration_audit_20261107_v1'

def pct(v):return 'N/A' if v is None else f'{100*v:.2f}'

def main():
    r=json.loads((OUT/'report.json').read_text());p=json.loads((OUT/'completion_verification.json').read_text())
    assert r['complete'] and p['complete'] and hashlib.sha256((OUT/'report.json').read_bytes()).hexdigest()==p['report_sha256']
    rows=[]
    for lang in ['vi','en']:
        a=r['baseline']['by_language'][lang];b=r['selected']['by_language'][lang]
        rows.append(dict(domain='ALL',language=lang,before=a,after=b))
    rows+=r['comparison']
    header='| Nguồn | Tiếng | F1 trước → sau | Precision trước → sau | Recall trước → sau | FP trước → sau | FN trước → sau |'
    table=[header,'|---|---|---|---|---|---|---|']
    cells=[]
    for x in rows:
        a,b=x['before'],x['after'];values=[x['domain'],x['language']]+[f'{pct(a[k])} → {pct(b[k])}' for k in ['eot_f1','eot_precision','eot_recall']]+[f'{a[k]} → {b[k]}' for k in ['fp','fn']]
        table.append('| '+' | '.join(values)+' |');cells.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
    text=f'''# Calibration DEV-only trên model v7 cố định

Hoàn thành kiểm chứng. Checkpoint {r['checkpoint_step']}, 11.941.122 tham số FP32. Không train, không thay weights, không TEST, không áp ngưỡng/default mới. Parent vẫn tạm dừng.

Ngưỡng cũ {r['old_threshold']:g}; ngưỡng chọn bằng calibration {r['new_threshold']:g}. Tiêu chí giữ nguyên: tối đa F1 của tiếng thấp hơn, rồi tổng F1; khi bằng nhau lấy ngưỡng đầu theo grid tăng dần. Chỉ một ngưỡng mới được chấm trên monitor; baseline monitor lấy đúng report trước của cùng weights.

Kết luận của lần kiểm này: operating point cũ góp phần gây bỏ sót. Trên ASR VI, F1 tăng10,43→22,88 và recall6,32→20,07 chỉ bằng ngưỡng; nhưng precision29,82→26,60, FP40→149 và FN vẫn215/269. Vì vậy chưa thể coi đây là sửa xong hoặc chứng minh model đủ dùng. Phép kiểm không áp dụng cho weights của nhánh ASR-adapted đã bị từ chối; không suy ra nó chữa được sự sụt điểm của nhánh đó.

Các ngưỡng mới từ 0,00001 tới 0,1 chạy actual predicted-reset riêng từng mức, không threshold lại trace cũ. Grid cũ >0,1 dùng lại calibration summaries cùng checkpoint. Mốc 0,1 chạy lại trên calibration và khớp confusion/metrics gốc. Không tối ưu theo monitor.

{chr(10).join(table)}

Tỷ lệ ở bảng là phần trăm. Nguồn ASR riêng nhỏ, không được coi là kiểm chứng độc lập đủ rộng. Nhãn punctuation silver; không human/audio gold, không đo độ trễ endpoint hay chất lượng dịch. Đây là chẩn đoán operating point trên một checkpoint, không chứng minh mô hình mới hoặc cách train nào tốt hơn. Không khẳng định ngưỡng tối ưu ngoài grid hữu hạn.

Kiểm chứng {p['calibration_inferences_audited']:,} inference calibration mới và {p['monitor_inferences_audited']:,} inference monitor; {p['canonical_cases']} canonical batch đơn, maxdelta {p['max_probability_delta']:.9g}. Toàn bộ saved new predictions đã audit population/causalinput/future/reset/tokenhash/labels/confusion. Summary calibration cũ >0,1 được dùng lại, không có tuyên bố audit mới toàn bộ trace của các mức cũ. Parent full checkpoint và weights SHA không đổi. Report SHA {p['report_sha256']}.

Full weights và traces ở SSH, không tải Windows/backup Drive. Lượt này không mở nhánh training mới.
'''
    (OUT/'README.md').write_text(text,encoding='utf-8')
    doc='''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Calibration v7</title><style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 20px;color:#152536;background:#f5f7fa}table{border-collapse:collapse;background:white;width:100%}td,th{padding:12px;border:1px solid #d4dce7;text-align:left}p{line-height:1.6}th{background:#dce9f4}.note{padding:16px;background:#fff2d8}h1{font-size:27px}</style>'''
    doc+=f'<h1>Calibration trên cùng model v7</h1><p>Checkpoint {r["checkpoint_step"]} · 11,94M · ngưỡng {r["old_threshold"]:g} → {r["new_threshold"]:g}</p><p class="note">Không train hoặc chấm TEST. Training vẫn tạm dừng; chưa áp ngưỡng vào model mặc định. F1/P/R là dấu chấm silver, không phải chứng nhận chất lượng dịch.</p>'
    doc+='<table><thead><tr>'+''.join('<th>'+html.escape(v.strip())+'</th>' for v in header.strip('|').split('|'))+'</tr></thead><tbody>'+''.join(cells)+'</tbody></table>'
    doc+=f'<p>Chọn trên calibration riêng trước khi xem monitor. Audit {p["calibration_inferences_audited"]:,} calibration và {p["monitor_inferences_audited"]:,} monitor decisions; {p["canonical_cases"]} canonical cases.</p><p>Report SHA: {p["report_sha256"]}</p></html>'
    (OUT/'review.html').write_text(doc,encoding='utf-8')
    summary=dict(complete=True,checkpoint_step=r['checkpoint_step'],old_threshold=r['old_threshold'],new_threshold=r['new_threshold'],rows=rows,
        report_sha256=p['report_sha256'],no_training=True,no_test=True,parent_paused=True,adopted=False,
        assessment='Threshold contributes to misses; lower threshold improves recall with more false cuts, not a deployable quality fix. ASR-adapted weights were not evaluated.')
    (OUT/'user_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(threshold=r['new_threshold'],all=[x for x in rows if x['domain']=='ALL']),ensure_ascii=False))

if __name__=='__main__':main()
