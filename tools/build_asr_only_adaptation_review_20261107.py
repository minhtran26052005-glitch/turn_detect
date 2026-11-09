"""Build a bounded before/after DEV report for the ASR-only trial."""
from datetime import datetime,timezone
import hashlib
import html
import json
from pathlib import Path

P=Path(__file__).resolve().parents[1]/'outputs/asr_only_adaptation_20261107_v1'


def main():
    r=json.loads((P/'report.json').read_text(encoding='utf-8'));proof=json.loads((P/'completion_verification.json').read_text(encoding='utf-8'))
    assert r['complete'] and proof['complete'] and hashlib.sha256((P/'report.json').read_bytes()).hexdigest()==proof['report_sha256']
    m=json.loads((P/'dataset_manifest.json').read_text(encoding='utf-8'))
    rows=[]
    for x in r['comparison']:
        before,after=x['before'],x['after']
        rows.append(dict(mode=x['mode'],domain=x['domain'],language=x['language'],
            before_f1_percent=100*(before['eot_f1'] or 0),after_f1_percent=100*(after['eot_f1'] or 0),f1_delta_pp=x['f1_delta_pp'],
            before_precision_percent=100*(before['eot_precision'] or 0),after_precision_percent=100*(after['eot_precision'] or 0),
            before_recall_percent=100*(before['eot_recall'] or 0),after_recall_percent=100*(after['eot_recall'] or 0),
            before_tp=before['tp'],after_tp=after['tp'],before_fp=before['fp'],after_fp=after['fp'],before_fn=before['fn'],after_fn=after['fn']))
    summary=dict(complete=True,at_utc=datetime.now(timezone.utc).isoformat(),baseline_step=r['baseline_step'],
        final_step=r['training']['global_step'],training_updates=r['training']['updates'],full_asr_epochs=1,
        scored_train_points=m['eligible_points_per_epoch'],streams=m['streams'],families=m['canonical_video_families'],words=m['counts']['words'],
        retained_source_rows=m['counts']['source_rows'],parameters=11941122,learning_rate=4e-6,loss_weights={'NEOT':1,'EOT':2},
        baseline_threshold=.1,selected_threshold=r['dev']['selected']['threshold'],
        all_canonical_decisions_identical=proof['all_canonical_decisions_identical'],canonical_cases=proof['canonical_predictor_cases'],
        all_dev_inferences_audited=proof['all_dev_inferences_audited'],comparison=rows,
        report_sha256=proof['report_sha256'],parent_v7_remains_paused=True,no_model_default_replacement=True,no_test_evaluation=True,
        limitations='One ASR-only epoch, same retained original DEV, punctuation silver not audio/human gold. ASR focus and natural source ordering both differ from v7; no matched-budget general-data arm. Improvements on this DEV do not certify translation quality or deployment readiness.')
    summary['assessment']='Current ASR-only setup severely reduces Vietnamese DEV performance; original model retained, no deployment adoption'
    for name in ['posthoc_asr_vi_score_distribution.json','posthoc_training_language_schedule.json']:
        if (P/name).exists():summary[name.removesuffix('.json')]=json.loads((P/name).read_text(encoding='utf-8'))
    (P/'user_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    esc=html.escape
    page='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Thử học riêng ASR — DEV trước/sau</title><style>body{font:17px/1.6 system-ui;max-width:1180px;margin:30px auto;padding:0 16px;background:#f7f9fc;color:#1f2937}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d1d5db;padding:10px;text-align:left}h2{margin-top:32px}.up{color:#15803d}.down{color:#b91c1c}code{overflow-wrap:anywhere}details{background:white;padding:14px;margin:14px 0}</style><h1>Học riêng ASR: trước và sau một vòng</h1>'''
    page+='<p>Xuất phát từ checkpoint '+str(summary['baseline_step'])+', kết thúc '+str(summary['final_step'])+'. Model mALBERT 11.941.122 tham số. Học hết '+format(summary['scored_train_points'],',')+' vị trí hợp lệ từ 210 luồng / 81 nhóm video / 169.011 từ / '+str(summary['retained_source_rows'])+' dòng nguồn đã giữ. Đây là phần TRAIN hợp lệ đã lọc, không phải toàn bộ 12.415 dòng ASR thô. Không ghép xáo trộn câu trong thử nghiệm này.</p>'
    page+='<p>Giữ nguyên AdamW và toàn bộ RNG từ parent; learning rate 4e-6, weighted CE NEOT=1/EOT=2, batch16, 16 luồng nhân quả. Nhánh v7 cũ vẫn tạm dừng; model mặc định chưa đổi. TEST không được chấm hay dùng chọn tham số trong thử nghiệm.</p>'
    page+='<p><strong>Kết quả: cấu hình ASR-only lần này làm ASR Việt suy giảm mạnh; bản gốc được giữ nguyên. Anh tăng F1 nhẹ, nhưng recall không tăng và các nguồn khác giảm.</strong></p>'
    modes=[('fixed_baseline_threshold','Cùng ngưỡng 0,1 — đối chiếu thay đổi trọng số')]
    if summary['selected_threshold']!=.1:modes.append(('selected','Ngưỡng chọn lại từ DEV calibration của model sau học'))
    for mode,label in modes:
        page+='<h2>'+label+'</h2><p>Ngưỡng sau học được chọn theo cùng chính sách worst-language F1 trên 119 luồng calibration; báo cáo trên 394 luồng monitor riêng. Ngưỡng chọn: '+str(summary['selected_threshold'])+'. ASR monitor gồm 7 luồng Việt / 7 luồng Anh; dấu câu là nhãn silver.</p><table><tr><th>Nguồn</th><th>Tiếng</th><th>F1 trước</th><th>F1 sau</th><th>Δ điểm %</th><th>Precision trước → sau</th><th>Recall trước → sau</th></tr>'
        for x in rows:
            if x['mode']!=mode:continue
            color='up' if x['f1_delta_pp']>=0 else 'down'
            values=[esc(x['domain']),esc(x['language']),format(x['before_f1_percent'],'.2f'),format(x['after_f1_percent'],'.2f'),'<span class="'+color+'">'+format(x['f1_delta_pp'],'+.2f')+'</span>',format(x['before_precision_percent'],'.2f')+' → '+format(x['after_precision_percent'],'.2f'),format(x['before_recall_percent'],'.2f')+' → '+format(x['after_recall_percent'],'.2f')]
            page+='<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>'
        page+='</table>'
    page+='<h2>Phạm vi kết luận</h2><p>Thử nghiệm thay đổi cả tỷ trọng nguồn ASR và thứ tự câu tự nhiên so với v7 ghép câu; chưa tách hiệu ứng từng yếu tố bằng nhánh đối chứng cùng ngân sách. Baseline DEV được tái dùng từ đúng checkpoint parent, không chạy lại model cũ. Một vòng và bộ DEV này chưa chứng minh chất lượng dịch hoặc endpoint audio thật.</p>'
    page+='<details><summary>Chẩn đoán sau chạy, chưa thay đổi cấu hình</summary><p>Điểm p(EOT) trung vị tại các EOT ASR Việt giảm từ khoảng 0,00507 xuống 0,000449. AUROC trên cùng quần thể điểm giảm từ 0,744 xuống 0,727; các context reset khác nhau nên không coi đây là phép tách riêng hiệu ứng calibration. Không chọn hoặc áp dụng ngưỡng mới từ monitor.</p><p>1.000 update cuối chỉ có 3.316 điểm hợp lệ, gồm Việt 2.470/Anh 846; batch hiệu dụng nhỏ dần ở cuối các luồng dài. Đây là dữ kiện lịch mẫu, chưa chứng minh nguyên nhân suy giảm. Cần đối chứng trước khi sửa cách chia batch/loss.</p></details>'
    page+='<details><summary>Kiểm chứng và định danh</summary><p>Đã kiểm đủ '+str(summary['all_dev_inferences_audited'])+' input monitor DEV nhân quả/encoding/future/reset/confusion, '+str(summary['canonical_cases'])+' trường hợp parity số học, đủ epoch và checkpoint model/27 AdamW/RNG/order/cursor/lanes. Curve calibration được kiểm lại cách chọn ngưỡng; raw calibration predictions không được lưu riêng. Parent không đổi. Report SHA256: <code>'+summary['report_sha256']+'</code></p></details></html>'
    (P/'review.html').write_text(page,encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
