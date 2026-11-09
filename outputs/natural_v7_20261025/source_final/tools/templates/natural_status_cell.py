import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from IPython.display import display

NC_ROOT=Path('/content/turn_detect_v2') if Path('/content/turn_detect_v2').is_dir() else Path.cwd()
def nc_read(relative):
    roots=[NC_ROOT]+[NC_ROOT/'outputs'/name for name in ['natural_v5_20261025','natural_v6_20261025','natural_v7_20261025','natural_v8_20261025']]
    for base in roots:
        path=base/relative
        if path.exists():
            try:return json.loads(path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:return None
    return None

nc_jobs=[]
for name in ['natural_v5_train_20261025','natural_v6_finetune_20261025','natural_v6_pho_repair_20261025',
    'natural_v7_text_20261025','natural_v8_lora_20261025','natural_v8_lora_20261025_r2','natural_v8_lora_20261025_r3','natural_final_eval_20261025']:
    status=nc_read('jobs/'+name+'/status.json') or {}
    nc_jobs.append({'job':name,'status':status.get('status','chưa có receipt'),'exit_code':status.get('exit_code'),
        'elapsed_seconds':status.get('elapsed_seconds')})
print('Đọc trạng thái lúc UTC:',datetime.now(timezone.utc).isoformat())
display(pd.DataFrame(nc_jobs))
nc_progress=nc_read('runs/natural_v8_lora_20261025_r3/progress.json') or nc_read('runs/natural_v7_text_20261025/progress.json')
if nc_progress:print('Tiến độ train đã lưu:',json.dumps(nc_progress,ensure_ascii=False))

nc_frozen=nc_read('outputs/natural_v7_20261025/selected_before_held.json')
nc_selection=nc_read('runs/natural_v8_lora_20261025_r3/selection.json') or nc_read('runs/natural_v7_text_20261025/selection.json')
nc_dev=nc_frozen['selected']['dev'] if nc_frozen else nc_selection['best_dev'] if nc_selection else None
if nc_dev:
    print('Development — không phải điểm test giữ riêng:')
    display(pd.DataFrame([{'language':lang,'samples':m['samples'],'accuracy_pct':round(100*m['accuracy'],2),
        'F1_EOT_pct':round(100*m['eot_f1'],2),'false_EOT_pct_of_holds':round(100*m['false_eot_rate'],2)}
        for lang,m in nc_dev['natural_languages'].items()]))
nc_final=nc_read('outputs/natural_v7_20261025/final_eval/report.json')
if nc_final:
    print('Kiểm tra cuối — nhãn observed-floor/VAD tự động, chưa có human gold:')
    display(pd.DataFrame([{'language':lang,'samples':m['samples'],'accuracy_pct':round(100*m['accuracy'],2),
        'F1_EOT_pct':round(100*m['eot_f1'],2),'false_EOT_pct_of_holds':round(100*m['false_eot_rate'],2)}
        for lang,m in nc_final['metrics']['selected'].items()]))
    print('Numeric target:',nc_final['automatic_held_numeric_target_met'],'| Số cuộc gọi:',nc_final['held_calls'])
else:
    print('384 mẫu giữ riêng: chưa có kết quả classifier đã hoàn tất.')
print('Đầu vào model: TEXT_ONLY. Ô này chỉ đọc trạng thái; không khởi chạy/ngắt train. Crawl204prefix chưa có gold nên accuracy/F1 chưa xác định.')
