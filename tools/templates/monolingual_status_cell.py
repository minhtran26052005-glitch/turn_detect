from pathlib import Path
import json
import pandas as pd
from IPython.display import display

M_ROOT=Path('/content/turn_detect_v2') if Path('/content/turn_detect_v2').is_dir() else Path.cwd()
def m_read(relative):
    for base in [M_ROOT,M_ROOT/'outputs/monolingual_20261025']:
        path=base/relative
        if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    return {}

print('Ba model độc lập: VI dùng bản QAT đã có; EN/ZH train teacher → distill → QAT riêng.')
print('VI cũ:93,55% accuracy /90,19% F1 trên dev tiền tố cũ; không phải điểm hội thoại mới.')
records=[]
for lang in ['en','zh']:
    for phase in ['teacher','student']:
        name='mono_'+lang+'_'+phase+'_20261025'
        status=m_read('jobs/'+name+'/status.json');progress=m_read('runs/'+name+'/progress.json')
        records.append({'language':lang,'phase':phase,'status':status.get('status','chưa bắt đầu'),
            'step':progress.get('step'),'stage':progress.get('stage',progress.get('phase')),'exit_code':status.get('exit_code')})
display(pd.DataFrame(records))
metrics=[]
for lang in ['en','zh']:
    for phase in ['teacher','student']:
        name='mono_'+lang+'_'+phase+'_20261025';report=m_read('runs/'+name+('/training_report.json' if phase=='teacher' else '/monolingual_report.json'))
        for scope,value in report.get('dev_slices',{}).items():
            metrics.append({'language':lang,'phase':phase,'scope':'DEV '+scope,'samples':value['samples'],
                'accuracy_pct':round(value['accuracy']*100,2),'F1_EOT_pct':round(value['eot_f1']*100,2)})
if metrics:
    print('Development dùng chọn model/ngưỡng — chưa là kiểm chứng độc lập:')
    display(pd.DataFrame(metrics))
final=m_read('outputs/monolingual_20261025/report.json')
if final:
    print('Regression đã xem; nhãn tự động, chưa có human semantic gold:')
    display(pd.DataFrame([{'language':lang,'samples':v['samples'],'accuracy_pct':round(v['accuracy']*100,2),
        'F1_EOT_pct':round(v['eot_f1']*100,2)} for lang,v in final['regression_metrics'].items()]))
print('Ô này chỉ đọc kết quả/trạng thái, không khởi chạy thêm job.204prefix crawl chưa có nhãn nên chưa tính accuracy/F1.')
