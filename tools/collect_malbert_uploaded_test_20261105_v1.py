"""Collect bounded reports, never weights or full TEST traces, and render a review."""
from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NAME = 'malbert_uploaded_test_20261105_v1'


def main():
    assert shutil.disk_usage(ROOT).free > 10*2**30+5*2**20
    code = f'''import pathlib,json,hashlib,subprocess,time,shutil
r=pathlib.Path('/home/edabk/hoangbpm/turn_detect');out=r/'outputs'/{NAME!r};files={{}};hashes={{}}
for n in ['launcher.json','status.json','error.json','checkpoint_receipt.json','locked_protocol.json','input_manifest.json','input_source_order_audit.json','report.json','completion_verification.json','independent_completion_verification.json','comparison_with_previous.json','review_cases.json']:
 p=out/n
 if p.exists():
  assert p.stat().st_size<2**20
  data=p.read_bytes();files[n]=data.decode('utf-8');hashes[n]=hashlib.sha256(data).hexdigest()
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip();proc={{'present':False}}
l=json.loads(files['launcher.json'])
p=pathlib.Path('/proc')/str(l['pid'])
if p.exists():
 raw=(p/'stat').read_text();ticks=raw[raw.rfind(')')+2:].split()[19];cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 proc=dict(present=True,pid=l['pid'],start_ticks=ticks,cmdline=cmd,boot_id=boot,identity_match=ticks==l['start_ticks'] and boot==l['boot_id'] and l['module'] in cmd)
main=r/'outputs/multilingual_malbert_vi_en_uploaded_20261104_v4'
v=dict(raw_files=files,remote_sha256=hashes,process=proc,main_status=json.loads((main/'status.json').read_text()),main_error=json.loads((main/'error.json').read_text()) if (main/'error.json').exists() else None,
 artifact_progress={{n:dict(bytes=(out/n).stat().st_size,age_s=time.time()-(out/n).stat().st_mtime) for n in ['predictions.jsonl.gz','prepared_streams.json.gz'] if (out/n).exists()}},
 disk_free_gib=dict(ssd=shutil.disk_usage(r).free/2**30,hdd=shutil.disk_usage('/mnt/hdd500').free/2**30),
 gpu_processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip())
print(json.dumps(v))
'''
    p = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','edabk@100.91.209.124','python3 -'],
                       input=code,text=True,capture_output=True,encoding='utf-8',check=True,timeout=60)
    v=json.loads(p.stdout);raw=v.pop('raw_files');v['files']={n:json.loads(s) for n,s in raw.items()}
    v['checked_utc']=datetime.now(timezone.utc).isoformat();v['windows_free_gib']=shutil.disk_usage(ROOT).free/2**30
    local=ROOT/'outputs'/NAME;local.mkdir(exist_ok=True)
    blob=json.dumps(v,ensure_ascii=False,indent=2).encode('utf-8');assert len(blob)<5*2**20
    (local/'heartbeat_verification.json').write_bytes(blob)
    for n,s in raw.items():
        data=s.encode('utf-8');assert hashlib.sha256(data).hexdigest()==v['remote_sha256'][n]
        (local/n).write_bytes(data)
    state_path=local/'task_state.json';state=json.loads(state_path.read_text()) if state_path.exists() else {}
    state.update(phase=v['files'].get('status.json',{}).get('stage','UNKNOWN'),process=v['process'],checked_utc=v['checked_utc'])
    report=v['files'].get('report.json');proof=v['files'].get('completion_verification.json');numeric=v['files'].get('independent_completion_verification.json')
    rows=[]
    if report and proof and numeric:
        assert report['complete'] and proof['complete'] and numeric['complete']
        assert proof['report_sha256']==numeric['report_sha256']==v['remote_sha256']['report.json']
        assert proof['all_inferences_audited']==report['inferences']
        state['complete']=True
        rows=[r for r in report['metric_rows'] if r['domain']=='ALL' and r['lookahead_words']==2 and r['cohort']=='all_requested' and r['label_policy']==report['primary_label_policy']]
        fmt=lambda x:'—' if x is None else f'{x*100:.2f}%'
        table=''.join('<tr><td>'+r['language']+'</td><td>'+fmt(r['eot_f1'])+'</td><td>'+fmt(r['eot_precision'])+'</td><td>'+fmt(r['eot_recall'])+'</td><td>'+str(r['fp'])+'/'+str(r['tp']+r['fp'])+'</td><td>'+str(r['fn'])+'/'+str(r['actual_eot'])+'</td></tr>' for r in rows)
        ck=report['checkpoint'];cases=v['files'].get('review_cases.json',[])
        cards=''.join('<article><b>'+escape(r['language']+' · '+r['domain']+' · '+r['case'])+'</b><p>'+escape(r['visible_context_tail'])+' <mark>'+escape(r['future'])+'</mark></p></article>' for r in cases if r['lookahead_words']==2)
        html='<!doctype html><html lang=vi><meta charset=utf-8><title>TEST model dữ liệu tải lên</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:0 18px;background:#f5f7fb}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #ccd;padding:10px}article{background:white;padding:16px;margin:16px 0}</style><h1>TEST Việt / Anh · mALBERT 11,94M FP32</h1><p>Checkpoint '+str(ck['step'])+' / branch '+str(ck['branch_step'])+', ngưỡng '+str(ck['threshold'])+' khóa từ DEV mới của chính checkpoint trước khi chấm TEST. Train tiếp tục độc lập.</p><p>Nhãn dấu câu silver; không phải human/audio gold. Bao gồm các nguồn TEST có dấu và cả keep=false như yêu cầu trước. Có nhóm factory từng được parent học; JSON có cohort loại nhóm trùng đã biết. Chưa exhaust mọi dạng trùng.</p><table><tr><th>Tiếng</th><th>F1</th><th>Precision</th><th>Recall</th><th>Cắt sai / lượt cắt</th><th>Bỏ sót / EOT</th></tr>'+table+'</table><p>Lookahead hai từ cùng lượt nguồn, chỉ reset theo EOT dự đoán. Dấu chấm là EOT, ?/! và dấu mơ hồ không chấm chính; không tính flush cuối thành EOT model. Không chọn model/ngưỡng/train theo TEST này.</p>'+cards+'</html>'
        assert len(html.encode('utf-8'))<5*2**20;(local/'review.html').write_text(html,encoding='utf-8')
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(process=v['process'],status=v['files'].get('status.json'),error=v['files'].get('error.json'),artifact_progress=v['artifact_progress'],main_status=v['main_status'],main_error=v['main_error'],test_k2=rows),ensure_ascii=False))


if __name__=='__main__': main()
