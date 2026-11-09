"""Produce a small review after the verified bounded effective16 experiment."""
import json,hashlib,html
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'outputs/asr_effective_batch16_20261107_v1'
def pc(v):return 'N/A' if v is None else f'{100*v:.2f}'
def main():
 r=json.loads((P/'report.json').read_text());proof=json.loads((P/'completion_verification.json').read_text())
 assert r['complete'] and proof['complete'] and hashlib.sha256((P/'report.json').read_bytes()).hexdigest()==proof['report_sha256']
 a=r['arms']['control']['selected'];b=r['arms']['effective16']['selected'];rows=[]
 for l in ['vi','en']:rows.append(dict(domain='ALL',language=l,before=a['by_language'][l],after=b['by_language'][l]))
 old={(v['domain'],v['language']):v for v in a['by_domain']}
 rows += [dict(domain=v['domain'],language=v['language'],before=old[v['domain'],v['language']],after=v) for v in b['by_domain']]
 head=['Nguồn','Tiếng','F1 cũ → mới','Precision cũ → mới','Recall cũ → mới','FP cũ → mới','FN cũ → mới'];lines=['| '+' | '.join(head)+' |','|---|---|---|---|---|---|---|'];tr=[]
 for v in rows:
  x,y=v['before'],v['after'];cells=[v['domain'],v['language']]+[pc(x[k])+' → '+pc(y[k]) for k in ['eot_f1','eot_precision','eot_recall']]+[str(x[k])+' → '+str(y[k]) for k in ['fp','fn']]
  lines.append('| '+' | '.join(cells)+' |');tr.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>')
 note=f'''Same source/order/initial model+27AdamW+allRNG from3731173. Same333315eligiblepoints. Control22147updates; treatment20833 (20832windows16 + final3). WeightedCE1/2, LR4e-6 unchanged. Effective-window grouping changes update count, dropout grouping and potentially predicted-reset contexts; not normalization-only causal proof. Each arm uses the same locked20-threshold calibration policy; selected thresholds {a['threshold']:g}/{b['threshold']:g}. Thresholds may differ, so selected-monitor comparison measures the calibrated training pipeline, not weights alone at a universal fixed threshold. No TEST/default replacement. Silver punctuation DEV, not human/audio gold or translation-quality/latency acceptance. Parentv7 remains paused; no autoextend.'''
 parent_path=P.parent/'malbert_calibration_audit_20261107_v1/report.json';parent=json.loads(parent_path.read_text())
 assert parent['checkpoint_step']==3731173 and parent['selected']['by_language']['vi']['samples']==a['by_language']['vi']['samples']
 refs=[x for x in parent['selected']['by_domain'] if x['domain']=='asr']
 parentnote='Parent v7 trước ASR, cùng DEV, ngưỡng0.03 chọn theo cùng mục tiêu calibration, dùng lại báo cáo đã lưu không chấm lại: '+', '.join(x['language']+' ASR F1 '+pc(x['eot_f1'])+' / P '+pc(x['eot_precision'])+' / R '+pc(x['eot_recall']) for x in refs)+'. Cần đọc mốc này cùng hai nhánh thử nghiệm; tốt hơn control ASR cũ không tự động có nghĩa tốt hơn parent.'
 assessment='Kết luận: gom đủ16 chỉ cho thay đổi nhỏ trong phép thử này, chưa giải quyết chất lượng. ASRVI F127,47→31,20/P30,09→33,77/R25,28→29,00; ASREN F153,58→52,65/P47,83→47,88/R60,90→58,48. VI vẫn FN191/269 và FP153. OverallF1VI30,74→30,88/EN47,87→48,29 gần như không đổi. Hai ngưỡng chọn khác nhau(.003/.03), vì vậy đây là so sánh pipeline đãcalibration chứ không phải weights-only ở cùng một ngưỡng. So với parent chưaASRđãcalibration, ASR tăng nhưng overallVI34,88→30,88, dailyVI46,20→34,03 và factoryVI48,00→34,25: không có cải thiện toàn diện. Không áp model/default hoặc tự train thêm. Không chứng minh batch nhỏ là nguyên nhân duy nhất hoặc ASR specialization nói chung không thể hiệu quả.'
 text='# ASR effective16: bounded controlled experiment\n\n'+assessment+'\n\n'+note+'\n\n'+parentnote+'\n\n'+'\n'.join(lines)+f"\n\nAll new calibration/monitor traces audited: {proof['calibration_and_monitor_inferences_audited']:,}; canonical cases {proof['canonical_cases']}; maxdelta {proof['max_probability_delta']:.9g}. ReportSHA {proof['report_sha256']}. All weights/optimizerRNG/traces stay SSH, no Drive backup.\n"
 (P/'README.md').write_text(text,encoding='utf-8')
 doc='<html lang="vi"><meta charset="utf-8"><title>ASR effective16</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px}td,th{padding:10px;border:1px solid #ccd}table{border-collapse:collapse}p{line-height:1.6}</style><h1>ASR: cập nhật đủ 16 mẫu</h1><p>'+html.escape(assessment)+'</p><p>'+html.escape(note)+'</p><p>'+html.escape(parentnote)+'</p><table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in head)+'</tr></thead><tbody>'+''.join(tr)+'</tbody></table><p>Report SHA '+proof['report_sha256']+'</p></html>'
 (P/'review.html').write_text(doc,encoding='utf-8')
 summary=dict(complete=True,rows=rows,assessment=assessment,control_threshold=a['threshold'],effective16_threshold=b['threshold'],report_sha256=proof['report_sha256'],parent_pre_asr_reference=refs,parent_reference_report_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),parent_reference_reused_not_rescored=True,no_default_replacement=True,parent_paused=True)
 (P/'user_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
