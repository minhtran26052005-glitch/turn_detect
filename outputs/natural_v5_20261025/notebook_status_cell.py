import json
from pathlib import Path
N5_ROOT = Path('/content/turn_detect_v2') if Path('/content/turn_detect_v2').is_dir() else Path.cwd()
N5_NAME = 'natural_v5_train_20261025'
def n5_read(relative):
    for base in [N5_ROOT, N5_ROOT/'outputs/natural_v5_20261025']:
        path = base/relative
        if path.exists(): return json.loads(path.read_text(encoding='utf-8'))
    return None
n5_status = n5_read('jobs/'+N5_NAME+'/status.json')
n5_progress = n5_read('runs/'+N5_NAME+'/progress.json')
n5_selection = n5_read('runs/'+N5_NAME+'/selection.json')
n5_manifest = n5_read('data_v5/canonical_20261025/manifest.json')
print(json.dumps({'job_status':n5_status, 'progress':n5_progress,
    'best_dev': n5_selection['best_dev'] if n5_selection else None,
    'natural_dataset_counts': n5_manifest['files'] if n5_manifest else None,
    'model_input':'TEXT_ONLY', 'sealed_test_scored':bool(n5_read('data_v5/canonical_20261025/sealed_consumed.json')),
    'note':'Dev is not held-test quality. Rerun this cell to read saved job state; it never launches/restarts training.'},ensure_ascii=False,indent=2))
