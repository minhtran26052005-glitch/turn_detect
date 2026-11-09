import hashlib, json
from pathlib import Path
V4_ROOT = Path('/content/turn_detect_v2') if Path('/content/turn_detect_v2').is_dir() else Path.cwd()
v4_summary = json.loads((V4_ROOT/'outputs/stream_v4_20261025/summary.json').read_text(encoding='utf-8'))
v4_weights = V4_ROOT/'outputs/stream_v4_20261025'/v4_summary['selected_checkpoint']/'model.safetensors'
if not v4_weights.exists():
    v4_weights = V4_ROOT/v4_summary['selected_checkpoint']/'model.safetensors'
with v4_weights.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == v4_summary['weights_sha256']
assert not v4_summary['user_target_achieved']
print(json.dumps({'hash_verified':True, 'status':v4_summary['status'],
    'fresh_natural_AUTO_labels':v4_summary['fresh_natural_auto_label_metrics'],
    'fresh_crawl_gold_labels':v4_summary['fresh_crawl_gold_labels'],
    'fresh_crawl_accuracy':None, 'serving_default_changed':False},ensure_ascii=False,indent=2))
