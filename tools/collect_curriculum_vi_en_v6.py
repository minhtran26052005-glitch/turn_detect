"""Collect bounded curriculum reports, with current process and storage evidence."""
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import shutil
import subprocess

LOCAL = Path(__file__).resolve().parents[1]/'outputs/multilingual_malbert_vi_en_curriculum_20261106_v6'


def main():
    assert shutil.disk_usage(LOCAL.parent).free > 10*2**30+5*2**20
    code = r'''
from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/home/edabk/hoangbpm/turn_detect')
out=root/'outputs/multilingual_malbert_vi_en_curriculum_20261106_v6'
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
names=['launcher.json','status.json','error.json','preparation_launcher.json','preparation_status.json',
 'dataset_manifest.json','curriculum_plan.json','independent_preparation_verification.json',
 'technical_fixture_verification.json','migration_receipt.json','independent_training_verification.json',
 'independent_actual_curriculum_training_verification.json','latest_dev.json','teacher_new_dev.json',
 'handover_liveness_verification.json',
 'cutover_launcher.json','cutover_status.json','cutover_error.json','cutover_complete.json',
 'candidate_results.json','final_report.json']
files={};hashes={}
for name in names:
 p=out/name
 if p.exists():
  assert p.stat().st_size<2**20
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
parent=root/'outputs/multilingual_malbert_vi_en_conflict_weighted_20261106_v5'
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
                 active_languages=['vi','en'],parameters=11941122,current_training_version=6,
                 loss_policy={'NEOT':1.,'EOT':2.},drive_data_excluded=True,latest_status=status,
                 curriculum_stage=status.get('curriculum_stage',resume.get('curriculum_stage',state.get('curriculum_stage'))),
                 curriculum_stage_seen=status.get('curriculum_stage_seen',resume.get('curriculum_stage_seen',state.get('curriculum_stage_seen'))),
                 full_filtered_epochs_complete=status.get('full_filtered_epochs_complete',resume.get('full_filtered_epochs_complete',state.get('full_filtered_epochs_complete',0))))
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    stages=value['files'].get('curriculum_plan.json',{}).get('stages',[])
    rows=[]
    for stage in stages:
        cells=[str(stage['index']+1),f"{100*stage['fraction_of_families']:.0f}%",str(stage['family_count']),
               str(len(stage['stream_indices'])),str(stage['eligible_points_per_pass']),
               str(stage['update_budget']) if stage['update_budget'] else 'Toàn bộ TRAIN, tối thiểu 6 epoch đầy đủ']
        rows.append('<tr>'+''.join('<td>'+escape(cell)+'</td>' for cell in cells)+'</tr>')
    review='''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Curriculum Việt Anh</title><style>body{font:16px system-ui;max-width:1100px;margin:36px auto;padding:0 20px;color:#17212b}td,th{padding:12px;border:1px solid #ccd4dd;text-align:left}table{border-collapse:collapse;width:100%}pre{white-space:pre-wrap;background:#edf3fa;padding:18px}</style><h1>mALBERT 11,94M · Curriculum Việt Anh</h1><p>Các pool tăng dần theo cả family hội thoại và giữ replay tự nhiên từ các mức trước. Tỷ lệ chỉ số family. Các giai đoạn đầu giới hạn số cập nhật và có thể kết thúc giữa một lượt qua pool.</p><table><tr><th>Giai đoạn</th><th>Family mục tiêu</th><th>Số family</th><th>Streams</th><th>Điểm hợp lệ mỗi lượt</th><th>Ngân sách cập nhật</th></tr>'''+''.join(rows)+'</table><h2>Trạng thái mới kiểm tra</h2><pre>'+escape(json.dumps(status,ensure_ascii=False,indent=2))+'</pre><p>DEV calibration và monitor giữ bộ nhóm nguồn đã khóa. Nhãn là dấu câu silver. TRAIN dùng context nhân quả, reset theo dự đoán, future cùng turn nguồn. Full checkpoint và dữ liệu lưu trên SSH.</p></html>'
    (LOCAL/'review.html').write_text(review,encoding='utf-8')
    print(json.dumps(dict(phase=phase,status=status,process=value['process'],preparation_process=value['preparation_process'],
        parent_process=value['parent_process'],error=value['files'].get('error.json'),cutover_error=value['files'].get('cutover_error.json'),
        stages=[{key:stage[key] for key in ['index','family_count','eligible_points_per_pass','update_budget']} for stage in stages],
        disks_gib=dict(ssd=value['ssd_free_gib'],hdd=value['hdd_free_gib'],windows=value['windows_free_gib'])),ensure_ascii=False))


if __name__=='__main__':
    main()
