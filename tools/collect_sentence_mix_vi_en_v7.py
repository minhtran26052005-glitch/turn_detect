"""Collect bounded curriculum reports, with current process and storage evidence."""
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import shutil
import subprocess

LOCAL = Path(__file__).resolve().parents[1]/'outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7'


def main():
    assert shutil.disk_usage(LOCAL.parent).free > 10*2**30+5*2**20
    code = r'''
from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/home/edabk/hoangbpm/turn_detect')
out=root/'outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7'
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
names=['launcher.json','status.json','error.json','preparation_launcher.json','preparation_status.json',
 'sentence_pack_audit.json','sentence_pack_examples.json','manifest_clarification.json','dataset_manifest.json','curriculum_plan.json','independent_preparation_verification.json',
 'technical_fixture_verification.json','independent_pack_verification.json','migration_receipt.json','independent_training_verification.json',
 'independent_actual_curriculum_training_verification.json','latest_dev.json','teacher_new_dev.json',
 'handover_liveness_verification.json',
 'cutover_launcher.json','cutover_status.json','cutover_error.json','cutover_complete.json',
 'candidate_results.json','final_report.json']
files={};hashes={}
for name in names:
 p=out/name
 if p.exists():
  assert p.stat().st_size<(2**21 if name=='curriculum_plan.json' else 2**20)
  data=p.read_bytes();files[name]=json.loads(data);hashes[name]=hashlib.sha256(data).hexdigest()
def process(launch):
 if not launch:return dict(present=False)
 p=Path('/proc')/str(launch['pid'])
 if not p.exists():return dict(present=False,pid=launch['pid'])
 raw=(p/'stat').read_text();f=raw[raw.rfind(')')+2:].split();cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
 return dict(present=True,pid=launch['pid'],start_ticks=f[19],cmdline=cmd,cpu_ticks=int(f[11])+int(f[12]),
  identity_match=f[19]==launch['start_ticks'] and launch['boot_id']==boot and launch['module'] in cmd)
runs={}
for folder in (out/'runs').glob('*'):
 runs[folder.name]={}
 for p in list(folder.glob('*receipt.json'))+list(folder.glob('*coverage.json'))+list(folder.glob('curriculum_stage_*_complete.json')):
  assert p.stat().st_size<2**20
  runs[folder.name][p.name]=json.loads(p.read_text())
mounted=Path('/mnt/hdd500').is_mount()
mount=subprocess.run(['findmnt','-n','-o','SOURCE,TARGET,OPTIONS','/mnt/hdd500'],capture_output=True,text=True).stdout.strip()
parent=root/'outputs/multilingual_malbert_vi_en_curriculum_20261106_v6'
print(json.dumps(dict(files=files,remote_sha256=hashes,runs=runs,boot_id=boot,
 process=process(files.get('launcher.json')),preparation_process=process(files.get('preparation_launcher.json')),
 cutover_process=process(files.get('cutover_launcher.json')),
 parent_process=process(json.loads((parent/'launcher.json').read_text())),
 parent_status=json.loads((parent/'status.json').read_text()),
 ssd_free_gib=shutil.disk_usage(root).free/2**30,hdd_mounted=mounted,hdd_free_gib=shutil.disk_usage('/mnt/hdd500').free/2**30 if mounted else None,mount=mount,
 gpu_processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip())))
'''
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','edabk@100.91.209.124','python3 -'],
                            input=code,text=True,capture_output=True,encoding='utf-8',check=True,timeout=60)
    value=json.loads(result.stdout)
    value.update(checked_utc=datetime.now(timezone.utc).isoformat(), windows_free_gib=shutil.disk_usage(LOCAL.parent).free/2**30)
    blob=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    assert len(blob)<5*2**20
    LOCAL.mkdir(exist_ok=True)
    (LOCAL/'heartbeat_verification.json').write_bytes(blob)
    for name,record in value['files'].items():
        (LOCAL/name).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    state_path=LOCAL/'task_state.json'
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else dict(last_notified_dev_branch_step=0,reported_cutover=False)
    status=value['files'].get('status.json',value['files'].get('preparation_status.json',{}))
    resume=value['runs'].get('finetune',{}).get('resume_receipt.json',{})
    phase=status.get('stage','PREPARING_CURRICULUM')
    if value['files'].get('error.json') or value['files'].get('cutover_error.json'):
        phase='ERROR'
    elif value['files'].get('launcher.json') and not value['process'].get('identity_match'):
        phase='REMOTE_TRAINER_IDENTITY_UNKNOWN'
    state.update(phase=phase,desired_state='ACTIVE',checked_utc=value['checked_utc'],process=value['process'],
                 active_languages=['vi','en'],parameters=11941122,current_training_version=7,
                 loss_policy={'NEOT':1.,'EOT':2.},drive_data_excluded=True,latest_status=status,
                 curriculum_stage=status.get('curriculum_stage',resume.get('curriculum_stage',state.get('curriculum_stage'))),
                 curriculum_stage_seen=status.get('curriculum_stage_seen',resume.get('curriculum_stage_seen',state.get('curriculum_stage_seen'))),
                 full_filtered_epochs_complete=status.get('full_filtered_epochs_complete',resume.get('full_filtered_epochs_complete',state.get('full_filtered_epochs_complete',0))))
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    stages=value['files'].get('curriculum_plan.json',{}).get('stages',[])
    pack_audit=value['files'].get('sentence_pack_audit.json',{})
    histogram=pack_audit.get('histogram',{})
    rows=[]
    total=pack_audit.get('pack_count',0)
    for number in range(1,11):
        vi=histogram.get('vi:'+str(number),0);en=histogram.get('en:'+str(number),0)
        cells=[str(number),str(vi),str(en),str(vi+en),f"{100*(vi+en)/total:.2f}%" if total else 'Đang chuẩn bị']
        rows.append('<tr>'+''.join('<td>'+escape(cell)+'</td>' for cell in cells)+'</tr>')
    review='''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Sentence mix Việt Anh</title><style>body{font:16px system-ui;max-width:1100px;margin:36px auto;padding:0 20px;color:#17212b}td,th{padding:12px;border:1px solid #ccd4dd;text-align:left}table{border-collapse:collapse;width:100%}pre{white-space:pre-wrap;background:#edf3fa;padding:18px}</style><h1>mALBERT 11,94M · Đoạn TRAIN ghép 1–10 câu</h1><p>Tỷ lệ theo số đoạn, riêng từng tiếng. Năm nhóm 1–5 câu mỗi nhóm 15%, năm nhóm 6–10 mỗi nhóm 5%. Các câu cùng tiếng và cùng nguồn được ghép thành đoạn TRAIN tổng hợp. Provenance và dấu chấm nguồn được kiểm lại. Các điểm nối có thể qua lượt nói/người nói gốc. Phần 100% của lịch mới dùng toàn bộ đoạn đã chuẩn bị.</p><table><tr><th>Câu/đoạn</th><th>Việt</th><th>Anh</th><th>Tổng đoạn</th><th>Tỷ lệ</th></tr>'''+''.join(rows)+'</table><h2>Trạng thái mới kiểm tra</h2><pre>'+escape(json.dumps(status,ensure_ascii=False,indent=2))+'</pre><p>DEV cố định dùng các turn nguồn thực và future cùng turn. Nhãn TRAIN/DEV là dấu câu silver. Full checkpoint, nguồn và provenance lưu trên SSH.</p></html>'
    examples=value['files'].get('sentence_pack_examples.json',[])
    cards=''.join('<article><h3>'+escape(example['language']+' · '+str(example['sentence_count'])+' câu · '+example['origin'])+'</h3><p>'+escape(example['normalized_text_with_source_fullstops'])+'</p></article>' for example in examples)
    review=review.replace('</html>','<h2>Ví dụ đoạn TRAIN tổng hợp</h2>'+cards+'</html>')
    (LOCAL/'review.html').write_text(review,encoding='utf-8')
    print(json.dumps(dict(phase=phase,status=status,process=value['process'],preparation_process=value['preparation_process'],
        parent_process=value['parent_process'],error=value['files'].get('error.json'),cutover_error=value['files'].get('cutover_error.json'),
        stages=[{key:stage[key] for key in ['index','family_count','eligible_points_per_pass','update_budget']} for stage in stages],
        disks_gib=dict(ssd=value['ssd_free_gib'],hdd=value['hdd_free_gib'],windows=value['windows_free_gib'])),ensure_ascii=False))


if __name__=='__main__':
    main()
