import json
from pathlib import Path
import pandas as pd
from IPython.display import display
AR_ROOT=Path('/content/turn_detect_v2') if Path('/content/turn_detect_v2').is_dir() else Path.cwd()
def ar_read(relative):
    path=AR_ROOT/relative
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    return {}
print('Automatic crawl → model action → Qwen judge reward → one-step RL → dev gate; three independent policies.')
state=ar_read('jobs/auto_rl_v12_20261025/status.json');report=ar_read('outputs/auto_rl_v12_20261025/report.json')
print('Job:',state.get('status'),'| GPU memory peak GiB:',round(report.get('GPU_peak_bytes',0)/2**30,2))
print('Judge Qwen3-8B:',report.get('probe_passed'),'| reward is NOT human gold or measured turn timing.')
if report.get('rounds'):display(pd.DataFrame(report['rounds']))
rows=[]
for rnd in [1,2]:
    for lang in ['vi','en','zh']:
        r=ar_read('outputs/auto_rl_v12_20261025/round_%02d/'%rnd+lang+'/report.json')
        if r.get('dev_after'):
            before=r['dev_before']['slices']['natural'];after=r['dev_after']['slices']['natural']
            rows.append({'round':rnd,'language':lang,'accuracy_before_pct':round(100*before['accuracy'],2),'F1_before_pct':round(100*before['eot_f1'],2),
                'accuracy_after_pct':round(100*after['accuracy'],2),'F1_after_pct':round(100*after['eot_f1'],2),'kept_by_dev_gate':r['candidate_retained']})
if rows:display(pd.DataFrame(rows))
print('Development only. No old384regression tuning. No90%real certificate. This cell reads receipts; it does not start or stop training.')
