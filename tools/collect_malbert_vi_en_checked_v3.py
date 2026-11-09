"""Small reports only: pretrained mALBERT training on SSH."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

REMOTE='/home/edabk/hoangbpm/turn_detect/outputs/multilingual_malbert_vi_en_checked_20261103_v3'
LOCAL=Path('outputs/multilingual_malbert_vi_en_checked_20261103_v3')


def main():
    assert shutil.disk_usage(Path.cwd()).free>10*2**30+5*2**20
    code=f'''import pathlib,json,os,subprocess
r=pathlib.Path({REMOTE!r});v={{'files':{{}},'process':{{'present':False}}}}
names=['launcher.json','status.json','error.json','cutover_v2_launcher.json','cutover_v2_status.json','cutover_v2_error.json','preparation_status.json','preparation_error.json','dataset_manifest.json','additional_data_audit.json','additional_data_review.json','independent_training_verification.json','migration_receipt.json','initialization_receipt.json','recipe.json','train_probe/final.json','cache/progress.json','cache/receipt.json','technical_fixture/verification.json','technical_fixture/int4_verification.json','first_checkpoint_verification.json','latest_dev.json','candidate_results.json','final_report.json']
for n in names:
 p=r/n
 if p.exists():
  assert p.stat().st_size<2**20
  v['files'][n]=json.loads(p.read_text())
l=v['files'].get('launcher.json')
if l:
 p=pathlib.Path('/proc')/str(l['pid']);v['process']['present']=p.exists()
 if p.exists():
  s=(p/'stat').read_text();t=s[s.rfind(')')+2:].split()[19]
  v['process'].update(pid=l['pid'],start_ticks=t,expected_ticks_match=t==l['start_ticks'],cmdline=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode())
v['cutover_process']={{'present':False}}
c=v['files'].get('cutover_v2_launcher.json')
if c:
 p=pathlib.Path('/proc')/str(c['pid']);v['cutover_process']['present']=p.exists()
 if p.exists():
  raw=(p/'stat').read_text();t=raw[raw.rfind(')')+2:].split()[19]
  v['cutover_process'].update(pid=c['pid'],start_ticks=t,expected_ticks_match=t==c['start_ticks'],cmdline=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode())
v['parent_process_present']=pathlib.Path('/proc/914852').exists()
v['runs']={{}}
for p in (r/'runs').glob('*'):
 v['runs'][p.name]={{}}
 for n in ['resume_receipt.json','best_report.json','training_report.json']:
  if (p/n).exists():v['runs'][p.name][n]=json.loads((p/n).read_text())
v['old_large_trainer_absent']=not pathlib.Path('/proc/2827094').exists()
v['old_compact_trainer_absent']=not pathlib.Path('/proc/427431').exists()
v['old_trilingual_malbert_absent']=not pathlib.Path('/proc/889316').exists()
v['gpu_processes']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
v['disks']={{str(p):os.statvfs(p).f_bavail*os.statvfs(p).f_frsize/2**30 for p in [r,pathlib.Path('/mnt/hdd500')]}}
v['mount']=subprocess.run(['findmnt','-n','-o','SOURCE,TARGET,OPTIONS','/mnt/hdd500'],capture_output=True,text=True).stdout.strip()
print(json.dumps(v))'''
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','edabk@100.91.209.124','python3 -'],input=code,capture_output=True,text=True,encoding='utf-8',timeout=45,check=True)
    d=json.loads(p.stdout);d['checked_utc']=datetime.now(timezone.utc).isoformat()
    d['windows_free_gib']=shutil.disk_usage(Path.cwd()).free/2**30
    data=(json.dumps(d,ensure_ascii=False,indent=2)+'\n').encode('utf-8');assert len(data)<5*2**20
    LOCAL.mkdir(parents=True,exist_ok=True);(LOCAL/'heartbeat_verification.json').write_bytes(data)
    for n,value in d['files'].items():
        path=LOCAL/n;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    path=LOCAL/'task_state.json';state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    state.update(checked_utc=d['checked_utc'],phase=d['files'].get('status.json',{}).get('stage',d['files'].get('cutover_v2_status.json',{}).get('stage','UNKNOWN')),
          process=d['process'],cutover_process=d['cutover_process'],latest_status=d['files'].get('status.json'),source_spoken_v9_used=False,
          active_large_trainer_must_not_resume=True,model_parameters=11941122,first_checkpoint_verified=d['files'].get('first_checkpoint_verification.json'))
    final=d['files'].get('final_report.json',{})
    if final.get('complete'):state.update(complete=True,accepted=final['accepted'])
    path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    review=d['files'].get('additional_data_review.json')
    if review:
        from html import escape
        audit=review['audit'];cards=[]
        for row in review['cases']:
            cards.append('<article><h3>'+escape(row['language'].upper()+' · '+row['label'])+'</h3><p>'+escape(row['raw_display'])+'</p><p class=point>'+escape(row['context'])+' <b>│</b> <mark>'+escape(row['text'])+'</mark></p><small>'+escape(row['origin']+' · '+row['dialogue_id']+' · turn '+str(row['turn_idx']))+'</small></article>')
        html='<!doctype html><html lang=vi><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Dữ liệu bổ sung Việt Anh</title><style>body{font:16px/1.65 system-ui;background:#f5f7fa;color:#203345;max-width:1050px;margin:32px auto;padding:0 18px}article,.intro{background:white;border:1px solid #dce3eb;padding:20px;margin:18px 0;border-radius:12px}.point{background:#f0f4f8;padding:12px}small{color:#647587}mark{background:#d8f5e6}</style><h1>Dữ liệu bổ sung chỉ Việt/Anh</h1><div class=intro><p><b>'+str(audit['additional_rows'])+' điểm cắt mới · '+str(audit['total_train_rows'])+' mẫu train tổng cộng.</b></p><p>Chỉ dùng phần TRAIN hợp lệ. DEV/TEST gốc được giữ ngoài train; ghép các chunk liên tiếp cùng người nói trước khi tạo nhãn. Không dùng target dịch hoặc rest_of_sentence làm đầu vào.</p><p>Nhãn EOT/NEOT suy từ dấu câu, không phải human/audio gold. Đã kiểm tra toàn bộ mẫu mới và khử trùng theo token đầu vào mô hình với tập được bảo vệ. Không chấm test để chọn model/ngưỡng. Đây là mẫu TRAIN để kiểm tra dữ liệu, không phải kết quả chất lượng mô hình.</p><p>Trong ví dụ: bên trái vạch là lịch sử đã thấy, nền xanh là một/hai từ nhìn trước. Dấu câu chỉ xuất hiện trong văn bản nguồn để kiểm nhãn.</p></div>'+''.join(cards)+'</html>'
        assert len(html.encode('utf-8'))<5*2**20
        (LOCAL/'review.html').write_text(html,encoding='utf-8')
    print(json.dumps({k:d[k] for k in ['checked_utc','process','disks','gpu_processes','old_large_trainer_absent']},ensure_ascii=False))
    print(json.dumps({n:d['files'].get(n) for n in ['status.json','error.json','preparation_status.json','cutover_v2_status.json','cutover_v2_error.json','preparation_error.json','first_checkpoint_verification.json','latest_dev.json','final_report.json']},ensure_ascii=False))


if __name__=='__main__':main()
