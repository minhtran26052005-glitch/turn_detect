import json
from pathlib import Path
import sqlite3
import subprocess
from datetime import datetime,timezone
import pandas as pd
from IPython.display import display

C13_ROOT=Path('/content/turn_detect_v2') if Path('/content/turn_detect_v2').is_dir() else Path.cwd()
C13_RUN=C13_ROOT/'outputs/continuous_v13_20261026'
print('Đọc kernel lúc UTC:',datetime.now(timezone.utc).isoformat())
print('Chiến dịch liên tục: không giới hạn số vòng, không dừng vì dev plateau hoặc silver90%.')
if (C13_RUN/'status.json').exists():
    s=json.loads((C13_RUN/'status.json').read_text(encoding='utf-8'))
    print(json.dumps({k:s.get(k) for k in ['status','stage','cycle','language','branch','step','steps','heartbeat_utc']},ensure_ascii=False))
if (C13_RUN/'state.sqlite').exists():
    c=sqlite3.connect('file:'+str(C13_RUN/'state.sqlite')+'?mode=ro',uri=True)
    counts=[dict(zip(['language','split','label','samples'],r)) for r in c.execute('SELECT language,split,label,COUNT(*) FROM samples GROUP BY language,split,label')]
    display(pd.DataFrame(counts))
    champions=[(key.split(':')[1],json.loads(value)) for key,value in c.execute("SELECT key,value FROM meta WHERE key LIKE 'champion:%'")]
    rows=[]
    for lang,r in champions:
        semantic=r['semantic_dev']['metrics'];floor=r['natural_floor_dev']
        rows.append({'language':lang,'semantic_silver_dev_samples':semantic['samples'],'semantic_accuracy_pct':round(100*semantic['accuracy'],2),'semantic_F1_pct':round(100*semantic['eot_f1'],2),
            'natural_floor_accuracy_pct':round(100*floor['accuracy'],2),'natural_floor_F1_pct':round(100*floor['eot_f1'],2),'precision':r.get('precision','FP32_LATENT')})
    if rows:display(pd.DataFrame(rows))
    c.close()
if Path('/content').exists():
    print(subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip())
if (C13_RUN/'recovery/latest_snapshot.json').exists():print('Snapshot:',json.loads((C13_RUN/'recovery/latest_snapshot.json').read_text(encoding='utf-8')))
print('Semantic silver do Qwen chấm không phải human gold. Natural-floor là mục tiêu khác. Chưa chứng nhận90%hội thoại thực.')
print('Các candidate hiện học FP32 latent; chưa thay bản INT4 serving. Ô này chỉ đọc, không tạo worker mới.')
