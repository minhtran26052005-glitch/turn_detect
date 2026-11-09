import json
from pathlib import Path
import pandas as pd
from IPython.display import display
V10_ROOT=Path('/content/turn_detect_v2') if Path('/content/turn_detect_v2').is_dir() else Path.cwd()
def v10_read(relative):
    path=V10_ROOT/relative
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    return {}
print('Ba model riêng. V10 thêm data và thử reward một quyết định; các điểm dưới là dev, không phải test mới.')
jobs=[]
for name in ['reward_v10_campaign_20261025','reward_v10_qwen_20261025','reward_v10_recovery_20261025']+['reward_v10_'+lang+'_20261025_r2' for lang in ['vi','en','zh']]:
    state=v10_read('jobs/'+name+'/status.json');jobs.append({'job':name,'status':state.get('status','chưa bắt đầu'),'exit_code':state.get('exit_code'),'seconds':state.get('elapsed_seconds')})
display(pd.DataFrame(jobs))
progress=[];metrics=[]
for lang in ['vi','en','zh']:
    p=v10_read('outputs/reward_v10_20261025/'+lang+'/progress.json')
    if p:progress.append({'language':lang,'phase':p.get('phase'),'step':p.get('step'),'max_steps':p.get('steps_max'),'batch':p.get('batch_size')})
    report=v10_read('outputs/reward_v10_20261025/'+lang+'/report.json')
    if report:
        for phase,record in [('baseline',report['baseline'])]+[(r['phase'],r['dev']) for r in report['phases']]:
            m=record['slices']['natural'];metrics.append({'language':lang,'phase':phase,'accuracy_pct':round(100*m['accuracy'],2),'F1_EOT_pct':round(100*m['eot_f1'],2),'false_EOT_pct_of_holds':round(100*m['false_eot_rate'],2),'min_dev_slice_score_pct':round(100*record['selection_score'],2)})
if progress:display(pd.DataFrame(progress))
if metrics:display(pd.DataFrame(metrics))
qwen=v10_read('outputs/reward_v10_20261025/qwen/report.json')
if qwen:print('Qwen two-view silver:',qwen['by_language'],'| Chưa hiệu chỉnh confidence, không phải human gold.')
print('Reward: đúng+1/ngắt sớm−2/chờ trênEOT−0,5. Không có measured latency. Crawl silver không nhận policy reward. Ô này không khởi chạy/ngắt train.')
