import json
from pathlib import Path
import sqlite3
import subprocess
from datetime import datetime, timezone, timedelta
import pandas as pd
from IPython.display import display

TF_ROOT = Path('/content/turn_detect_v2') if Path('/content/turn_detect_v2').exists() else Path.cwd()
TF_RUN = TF_ROOT / 'outputs/teacher_first_v14_20261026'
print('Kiểm tra lúc:', datetime.now(timezone(timedelta(hours=7))).isoformat())
print('Teacher lớn trước, ba checkpoint riêng VI/EN/ZH; chưa distill hoặc thay model INT4.')
if (TF_RUN / 'status.json').exists():
    tf_status = json.loads((TF_RUN / 'status.json').read_text(encoding='utf-8'))
    print(json.dumps({k: tf_status.get(k) for k in ['status', 'stage', 'cycle', 'language', 'step', 'steps', 'total_steps', 'loss', 'heartbeat_utc']}, ensure_ascii=False))
if (TF_RUN / 'state.sqlite').exists():
    tf_db = sqlite3.connect('file:' + str(TF_RUN / 'state.sqlite') + '?mode=ro', uri=True)
    tf_rows = []
    for tf_key, tf_value in tf_db.execute("SELECT key,value FROM meta WHERE key LIKE 'large_teacher:%'"):
        tf_report = json.loads(tf_value); tf_dev = tf_report['semantic_dev']; tf_floor = tf_report['natural_floor_dev']
        tf_rows.append({'tiếng': tf_key.split(':')[1], 'params': tf_report['identity']['parameters'],
                       'bước': tf_report['total_steps'], 'mẫu silver dev': tf_dev['samples'],
                       'silver accuracy %': round(100 * tf_dev['accuracy'], 2), 'silver F1 %': round(100 * tf_dev['eot_f1'], 2),
                       'floor accuracy %': round(100 * tf_floor['accuracy'], 2), 'floor F1 %': round(100 * tf_floor['eot_f1'], 2)})
    if tf_rows: display(pd.DataFrame(tf_rows))
    tf_db.close()
if Path('/content').exists():
    print(subprocess.check_output(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader'], text=True).strip())
print('Silver Qwen không phải human gold; chưa chứng nhận 90% trên hội thoại thật.')
print('Teacher và optimizer đầy đủ hiện lưu trên Colab; metadata snapshot không chứa trọng số.')
print('Ô này chỉ đọc trạng thái, không tạo worker hoặc chạy lại training.')
