"""Collect only small status/reports for the user-requested external ZIP test."""
from datetime import datetime,timezone
from html import escape
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

LOCAL=Path('outputs/malbert_eval_zip_20261104_v2')
REMOTE='/home/edabk/hoangbpm/turn_detect/outputs/malbert_eval_zip_20261104_v2'


def percent(value):return '—' if value is None else f'{value*100:.2f}%'


def add_current_model_comparison(html,report,comparison,coverage):
    assert comparison['same_source_streams_and_eligible_point_population']
    assert coverage['fingerprint']['train_sha256']==report['main_dataset_sha256']
    rows=[]
    for item in sorted((x for x in comparison['rows'] if x['domain']=='ALL'),key=lambda x:(x['split'],x['language']!='vi')):
        old,new=item['previous'],item['current']
        values=[item['split'],item['language'].upper(),percent(old['eot_f1']),percent(new['eot_f1']),
                str(new['fp'])+'/'+str(new['tp']+new['fp'])+' ('+percent(new['false_share_of_cuts'])+')',
                str(new['fn'])+'/'+str(new['actual_eot'])]
        rows.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in values)+'</tr>')
    block='<h2>Model mới so với lần chấm trước</h2><p>Cùng nguồn và các điểm đủ điều kiện, lookahead 2, dấu chấm là EOT. Checkpoint '+str(comparison['old_checkpoint'])+' dùng ngưỡng '+str(comparison['old_threshold'])+'; checkpoint '+str(comparison['new_checkpoint'])+' dùng '+str(comparison['new_threshold'])+'. Mỗi ngưỡng lấy từ DEV gốc của chính checkpoint trước khi chấm ZIP. So sánh phản ánh cả trọng số và hiệu chuẩn DEV; model cũ dùng lại trace/report đã lưu.</p><div class=table><table><tr><th>Split</th><th>Tiếng</th><th>F1 cũ</th><th>F1 mới</th><th>Cắt sai mới / lượt cắt đã chấm</th><th>Bỏ sót mới / EOT</th></tr>'+''.join(rows)+'</table></div>'
    block+='<h2>Dữ liệu nào thực sự đã học</h2><div class=intro><p><b>Snapshot 399.960 mẫu: 200.000 replay + 199.960 điểm cắt bổ sung đã chọn.</b> Model đã học lặp lại toàn bộ snapshot. Bản được chấm ở epoch '+str(report['checkpoint']['epoch']+1)+', có '+format(report['checkpoint']['additional_row_training_exposures'],',')+' lượt tiếp xúc mẫu bổ sung; đây là các lần lặp, không phải từng ấy mẫu mới.</p><p>Hai JSONL đã được dùng qua các mẫu TRAIN chọn lọc. factory/gen500 và daily_conversation/full trong ZIP trùng các nguồn raw đó. asr_crawl/run2b, factory/gen4000 và dữ liệu spoken mới của Colab chưa vào fingerprint đang train. Không phải mọi dòng raw hay mọi khe từ ứng viên đều đã học.</p></div><h2>Chi tiết model mới theo nguồn</h2>'
    return html.replace('<div class=table>',block+'<div class=table>',1)


def render(report,cases):
    rows=[];diagnostic=[]
    for r in report['metric_rows']:
        if r['cohort']!='all_requested' or r['lookahead_words']!=2:continue
        rendered='<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in [r['domain'],r['split'],r['language'].upper(),r['samples'],r['actual_eot'],percent(r['eot_f1']),percent(r['eot_precision']),percent(r['eot_recall']),str(r['fp'])+'/'+str(r['tp']+r['fp']),percent(r['false_share_of_cuts']),r['fn']])+'</tr>'
        (rows if r['label_policy']==report['primary_label_policy'] else diagnostic).append(rendered)
    cards=[]
    for r in cases:
        if r['lookahead_words']!=2 or r['case'] not in ('fp','fn'):continue
        cards.append('<article><h3>'+escape(r['domain']+' · '+r['split']+' · '+r['language'].upper()+' · '+r['case'].upper())+'</h3><p>'+escape(r['source_display'])+'</p><p class=point>'+escape(r['visible_context_tail'])+' <b>│</b> <mark>'+escape(r['future'])+'</mark></p><small>'+escape(r['dialogue_id'])+' · p(EOT)='+str(round(r['p_eot'],4))+(' · nguồn đã có trong train' if r['known_train_source_overlap'] else '')+'</small></article>')
    ck=report['checkpoint']
    header='<tr><th>Nguồn</th><th>Split</th><th>Tiếng</th><th>Điểm</th><th>EOT</th><th>F1</th><th>Precision</th><th>Recall</th><th>FP/cuts</th><th>Tỷ lệ cắt sai</th><th>FN</th></tr>'
    return '''<!doctype html><html lang=vi><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Chấm EOT trên eval.zip</title><style>body{font:15px/1.6 system-ui;color:#203345;background:#f5f7fb;margin:28px auto;max-width:1250px;padding:0 16px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #d9e1eb;text-align:left}article,.intro{background:white;border:1px solid #d9e1eb;border-radius:10px;padding:18px;margin:18px 0}h1{font-size:28px}.point{background:#eef3fa;padding:12px}small{color:#62768a}mark{background:#d7f6e4}.table{overflow:auto}summary{cursor:pointer;padding:16px;font-weight:bold}</style><h1>Đánh giá dấu câu trên eval.zip</h1><div class=intro><p><b>mALBERT 11,94M FP32 · checkpoint '''+str(ck['step'])+''' · ngưỡng '''+str(ck['threshold'])+'''</b>. Checkpoint/ngưỡng khóa từ DEV gốc trước khi chấm ZIP. Train tiếp tục độc lập.</p><p>Đã chấm nguồn source_raw có dấu câu ở mọi nhóm DEV/TEST, bao gồm keep=false và 54 hội thoại factory đã có mẫu trong train. Kết quả chính là toàn bộ mẫu theo yêu cầu; báo cáo JSON có lát cắt không gồm nhóm trùng đã biết. Đây là nhãn dấu câu, không phải human/audio gold.</p><p>Bảng chính dùng dấu chấm EOT như train; ?/! và dấu kết thúc mơ hồ không tính vào nhãn chính. Bảng mở rộng bên dưới có thêm góc nhìn .?! cùng dự đoán/ngưỡng, không chấm hay hiệu chỉnh model lần nữa. F1 trống nghĩa là phần đó không có EOT đủ điều kiện.</p><p>Lookahead 2: chỉ hai từ tiếp theo trong cùng lượt nguồn. Buffer chỉ xả sau EOT dự đoán, không reset theo nhãn hay người nói. Các tác vụ vốn là câu/mã đơn lẻ được chấm từng item. Thiếu future ở cuối lượt và flush cuối luồng không được tính thành EOT do model dự đoán.</p><p><b>FP/cuts</b> = số cắt sai trên tổng lần model cắt trong các điểm có nhãn đủ điều kiện. FN = bỏ sót dấu chấm đủ điều kiện. Không dùng accuracy cao trên số đông NEOT làm bằng chứng model tốt.</p></div><div class=table><table>'''+header+''.join(rows)+'''</table></div><details><summary>Chẩn đoán khi tính thêm dấu hỏi / dấu than là EOT</summary><div class=table><table>'''+header+''.join(diagnostic)+'''</table></div></details><h2>Ví dụ cắt nhầm / bỏ sót</h2>'''+''.join(cards)+'''<p><small>Toàn bộ trace, inputs, frozen weights và optimizer/RNG giữ trên SSH. Windows chỉ nhận báo cáo nhỏ. Không chọn model, ngưỡng hay chỉnh training theo kết quả này.</small></p></html>'''


def main():
    assert shutil.disk_usage(Path.cwd()).free>=10*2**30+5*2**20
    code=f'''import pathlib,json,subprocess,hashlib,time
r=pathlib.Path({REMOTE!r});v={{'raw_files':{{}},'process':{{'present':False}}}}
v['remote_sha256']={{}}
for n in ['launcher.json','status.json','error.json','checkpoint_receipt.json','locked_protocol.json','input_manifest.json','input_source_order_audit.json','report.json','completion_verification.json','independent_completion_verification.json','review_cases.json','comparison_with_827000.json','training_coverage_receipt.json','training_source_usage_receipt.json']:
 p=r/n
 if p.exists():
  assert p.stat().st_size<3*2**20
  data=p.read_bytes();v['raw_files'][n]=data.decode('utf-8');v['remote_sha256'][n]=hashlib.sha256(data).hexdigest()
v['artifact_progress']={{n:{{'bytes':(r/n).stat().st_size,'age_s':round(time.time()-(r/n).stat().st_mtime,2)}} for n in ['predictions.jsonl.gz','prepared_streams.json.gz'] if (r/n).exists()}}
main=r.parent/'multilingual_malbert_vi_en_checked_20261103_v3'
v['main_status']=json.loads((main/'status.json').read_text())
v['main_error']=json.loads((main/'error.json').read_text()) if (main/'error.json').exists() else None
l=json.loads(v['raw_files']['launcher.json']) if 'launcher.json' in v['raw_files'] else None
if l:
 p=pathlib.Path('/proc')/str(l['pid']);v['process']['present']=p.exists()
 if p.exists():
  raw=(p/'stat').read_text();ticks=raw[raw.rfind(')')+2:].split()[19]
  v['process'].update(pid=l['pid'],start_ticks=ticks,expected_ticks_match=ticks==l['start_ticks'],cmdline=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode())
v['gpu_processes']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
print(json.dumps(v))'''
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','edabk@100.91.209.124','python3 -'],input=code,text=True,capture_output=True,encoding='utf-8',timeout=45,check=True)
    v=json.loads(p.stdout);raw_files=v.pop('raw_files');v['files']={n:json.loads(value) for n,value in raw_files.items()}
    v['checked_utc']=datetime.now(timezone.utc).isoformat();v['windows_free_gib']=shutil.disk_usage(Path.cwd()).free/2**30
    LOCAL.mkdir(exist_ok=True);data=json.dumps(v,ensure_ascii=False,indent=2).encode();assert len(data)<5*2**20
    (LOCAL/'heartbeat_verification.json').write_bytes(data)
    for name,value in v['files'].items():
        data=raw_files[name].encode('utf-8')
        assert hashlib.sha256(data).hexdigest()==v['remote_sha256'][name],name
        (LOCAL/name).write_bytes(data)
    state_path=LOCAL/'task_state.json';state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    state.update(phase=v['files'].get('status.json',{}).get('stage','UNKNOWN'),process=v['process'],checked_utc=v['checked_utc'])
    if v['files'].get('report.json',{}).get('complete') and v['files'].get('completion_verification.json',{}).get('complete'):
        report=v['files']['report.json'];state['complete']=True
        proof=v['files']['completion_verification.json']
        assert proof['complete'] and proof['report_sha256']==v['remote_sha256']['report.json']
        assert proof['all_inferences_audited']==report['inferences']
        assert v['files']['input_source_order_audit.json']['complete']
        assert report['comparison_with_827000_sha256']==v['remote_sha256']['comparison_with_827000.json']
        html=render(report,v['files'].get('review_cases.json',[]))
        html=add_current_model_comparison(html,report,v['files']['comparison_with_827000.json'],v['files']['training_coverage_receipt.json'])
        assert len(html.encode())<5*2**20;(LOCAL/'review.html').write_text(html,encoding='utf-8')
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary={'checked_utc':v['checked_utc'],'process':v['process'],'status':v['files'].get('status.json'),'error':v['files'].get('error.json'),'gpu_processes':v['gpu_processes'],'artifact_progress':v['artifact_progress'],'main_status':v['main_status'],'main_error':v['main_error']}
    if v['files'].get('report.json'):
        report=v['files']['report.json'];summary['all_requested_test_k2']=[r for r in report['metric_rows'] if r['domain']=='ALL' and r['split']=='test' and r['lookahead_words']==2 and r['cohort']=='all_requested' and r['label_policy']==report['primary_label_policy']]
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
