"""Collect bounded experiment reports and progress while preserving remote artifacts."""
from pathlib import Path
from datetime import datetime, timezone
from html import escape
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/malbert_learning_diagnostic_20261106_v1'


def main():
    assert shutil.disk_usage(ROOT).free > 10*2**30+5*2**20
    code = r'''
from pathlib import Path
import json, subprocess, shutil, hashlib
root=Path('/home/edabk/hoangbpm/turn_detect')
out=root/'outputs/malbert_learning_diagnostic_20261106_v1'
names=['launcher.json','status.json','error.json','checkpoint_receipt.json','fixed_sample_receipt.json',
       'sanity/progress.json','sanity/report.json','lr4e6/training_complete.json','lr1e5/training_complete.json',
       'baseline/dev_report.json','lr4e6/dev_report.json','lr1e5/dev_report.json','report.json',
       'completion_verification.json','verification_status.json']
files={};hashes={}
for name in names:
 p=out/name
 if p.exists():
  assert p.stat().st_size<2**20
  data=p.read_bytes();files[name]=json.loads(data);hashes[name]=hashlib.sha256(data).hexdigest()
l=files['launcher.json'];proc=Path('/proc')/str(l['pid']);process={'present':proc.exists()}
if proc.exists():
 raw=(proc/'stat').read_text();f=raw[raw.rfind(')')+2:].split();cmd=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
 process.update(pid=l['pid'],start_ticks=f[19],cmdline=cmd,identity_match=f[19]==l['start_ticks'] and l['module'] in cmd)
main=root/'outputs/multilingual_malbert_vi_en_conflict_weighted_20261106_v5'
print(json.dumps(dict(files=files,sha256=hashes,process=process,
 main_status=json.loads((main/'status.json').read_text()),
 main_error=json.loads((main/'error.json').read_text()) if (main/'error.json').exists() else None,
 gpu_processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),
 disks_gib=dict(ssd=shutil.disk_usage(root).free/2**30,hdd=shutil.disk_usage('/mnt/hdd500').free/2**30))))
'''
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','edabk@100.91.209.124','python3 -'],input=code,text=True,capture_output=True,encoding='utf-8',check=True,timeout=60)
    value=json.loads(result.stdout)
    value['checked_utc']=datetime.now(timezone.utc).isoformat()
    value['windows_free_gib']=shutil.disk_usage(ROOT).free/2**30
    blob=json.dumps(value,ensure_ascii=False,indent=2).encode('utf-8')
    assert len(blob)<5*2**20
    OUT.mkdir(exist_ok=True)
    (OUT/'heartbeat_verification.json').write_bytes(blob)
    for name, content in value['files'].items():
        path=OUT/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=value['files'].get('report.json')
    proof=value['files'].get('completion_verification.json')
    if report and proof and proof.get('complete'):
        rows=[]
        for arm in report['reports']:
            for language,metrics in arm['monitor']['by_language'].items():
                cells=[arm['arm'],language,f"{arm['selected_threshold']:.1f}",arm['selection_status']]
                cells.extend(f"{100*(metrics[key] or 0):.2f}%" for key in ['eot_f1','eot_precision','eot_recall'])
                cells.extend(str(metrics[key]) for key in ['tp','fp','fn'])
                rows.append('<tr>'+''.join('<td>'+escape(cell)+'</td>' for cell in cells)+'</tr>')
        headers=['Nhánh','Tiếng','Ngưỡng','Điều kiện calibration','F1 monitor','Precision monitor','Recall monitor','TP','FP','FN']
        html='''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Chẩn đoán học mALBERT</title>
<style>body{font:16px system-ui;max-width:1250px;margin:36px auto;padding:0 20px;color:#17212b}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ccd4dd;text-align:left}th{background:#edf3fa}code{overflow-wrap:anywhere}</style>
<h1>Chẩn đoán học mALBERT 11,94M</h1>'''
        html+=f"<p>Checkpoint gốc {report['parent_step']}. Mỗi nhánh LR chạy {report['paired_updates']} cập nhật. Phép thử học thuộc đạt 100% trên 1.000 mẫu sau {report['sanity']['steps']} bước.</p>"
        html+='<p>Mẫu học thuộc được phân tầng chủ động với 125 EOT và 375 NEOT mỗi tiếng. Phép thử dùng AdamW mới, LR 3e-5, batch hiệu dụng 32, microbatch 4 sau khi phục hồi lỗi giới hạn VRAM.</p>'
        html+='<p>Hai nhánh LR kế thừa cùng model, optimizer, RNG và thứ tự nguồn. Dự đoán thay đổi có thể làm ngữ cảnh reset về sau khác nhau. Ngưỡng được chọn riêng trên calibration với mục tiêu precision tối thiểu 50% mỗi tiếng. Precision thực tế trên monitor được thể hiện trong bảng.</p>'
        html+='<table><thead><tr>'+''.join('<th>'+escape(header)+'</th>' for header in headers)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
        html+='<p>Đã tái kiểm toàn bộ rollout monitor, input nhân quả, nhãn, reset dự đoán và checkpoint. Nhãn DEV dựa trên dấu câu. Đây là thí nghiệm ngắn trên DEV và chưa chứng nhận chất lượng hội thoại thực.</p>'
        html+='<p>SHA256 báo cáo <code>'+escape(proof['report_sha256'])+'</code></p></html>'
        (OUT/'review.html').write_text(html,encoding='utf-8')
    state_path=OUT/'task_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(reported=False)
    state.update(phase=value['files'].get('status.json',{}).get('stage','UNKNOWN'),process=value['process'],checked_utc=value['checked_utc'])
    if value['files'].get('report.json',{}).get('complete'):
        state['complete']=True
    state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    summary=dict(status=value['files'].get('status.json'),error=value['files'].get('error.json'),process=value['process'],
                 gpu=value['gpu_processes'],main_error=value['main_error'])
    for name in ['sanity/progress.json','sanity/report.json']:
        if name in value['files']:
            record=value['files'][name]
            summary[name]={k:v for k,v in record.items() if k!='history'}
    if 'report.json' in value['files']:
        summary['report']=value['files']['report.json']
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
