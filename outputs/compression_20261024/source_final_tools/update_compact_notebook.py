"""Append the verified compact result without replacing existing notebook cells."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil

from build_notebook import source_bundle


def main():
    path = Path('a.ipynb')
    root = Path('outputs/compression_20261024')
    manifest = json.loads(Path('models/turn_v2_compact_20261024/model_manifest.json').read_text())
    assert manifest['status'] == 'SIZE_AND_FULL_DEV_TARGET_MET'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    backup = root/'a_before_compression.ipynb'
    if not backup.exists():
        shutil.copy2(path,backup)
    package, checksum = source_bundle()
    # Only extend the model preference in the existing crawl cell.
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'bootstrap' in cell.get('metadata',{}).get('tags',[]):
            source = ''.join(cell['source'])
            source, replacements = re.subn(r"^package = base64\.b64decode\(.*\)$",f"package = base64.b64decode({package!r})",source,flags=re.MULTILINE)
            assert replacements == 1
            source, replacements = re.subn(r"^assert hashlib\.sha256\(package\)\.hexdigest\(\) == .*?$",f"assert hashlib.sha256(package).hexdigest() == {checksum!r}",source,flags=re.MULTILINE)
            assert replacements == 1
            cell['source'] = source.splitlines(keepends=True)
            cell['execution_count'], cell['outputs'] = None, []
            cell['metadata'].pop('colab_verified_at',None)
        if cell['cell_type'] == 'code' and 'crawl_predict' in cell.get('metadata',{}).get('tags',[]):
            source = ''.join(cell['source'])
            old = "PREDICT_MODEL_DIR = TRIAL_ONNX if TRIAL_ONNX.is_dir() else None"
            new = "COMPACT_ONNX = ROOT/'models/turn_v2_compact_20261024/onnx'\nPREDICT_MODEL_DIR = COMPACT_ONNX if COMPACT_ONNX.is_dir() else (TRIAL_ONNX if TRIAL_ONNX.is_dir() else None)"
            if old in source:
                cell['source'] = source.replace(old,new).splitlines(keepends=True)
                cell['execution_count'], cell['outputs'] = None, []
                cell['metadata'].pop('colab_verified_at',None)
    metrics = manifest['dev_metrics']
    markdown = f"""# Bản nén ngày 24/10/2026

Đã đạt ngân sách **≤1/10 dung lượng ONNX gốc**: ONNX **{manifest['files']['model.onnx']['bytes']/1e6:.3f} MB**, toàn bộ gói **{manifest['deployment_bytes']/1e6:.3f} MB**, so với 1.116,285 MB ban đầu.

Trên **17.833 mẫu dev**, accuracy **{metrics['accuracy']*100:.3f}%**, F1 EOT **{metrics['eot_f1']*100:.3f}%**. Đây là kết quả chọn bản nén trên dev; **chưa chấm holdout độc lập mới**. Holdout lần train trước đã dùng và không được mở lại để chọn bản nén.

Thu gọn vocab từ train và lượng tử hóa INT8 trọng số; giữ 12 lớp, activation và phép tính FP32. Không train lại, không distillation/RL. Mục tiêu 1/10 là file lưu trữ, không phải RAM, số tham số hay tốc độ.

Ô dưới chạy độc lập, kiểm tra SHA256 và suy luận với bản nén trên Windows hoặc Colab đã đồng bộ. [Review chi tiết](outputs/compression_20261024/review.html) · [Experiment record](EXPERIMENT_COMPRESSION_20261024.md)
"""
    code = '''import hashlib, json, sys
from pathlib import Path
COMPACT_ROOT = Path('/content/turn_detect_v2') if Path('/content/turn_detect_v2').is_dir() else Path.cwd()
sys.path.insert(0,str(COMPACT_ROOT))
from turn_v2.predict import TurnDetectorV2
COMPACT_MANIFEST = json.loads((COMPACT_ROOT/'models/turn_v2_compact_20261024/model_manifest.json').read_text(encoding='utf-8'))
PREDICT_MODEL_DIR = COMPACT_ROOT/'models/turn_v2_compact_20261024/onnx'
assert COMPACT_MANIFEST['status']=='SIZE_AND_FULL_DEV_TARGET_MET'
for filename, identity in COMPACT_MANIFEST['files'].items():
    file_path = PREDICT_MODEL_DIR/filename
    assert file_path.stat().st_size==identity['bytes']
    assert hashlib.file_digest(file_path.open('rb'),'sha256').hexdigest()==identity['sha256']
compact_detector = TurnDetectorV2(PREDICT_MODEL_DIR)
demo_texts = ['tôi muốn đặt vé đi hà nội', 'tôi muốn đặt vé đi', 'cảm ơn bạn']
compact_demos = [{'text':text,**compact_detector.predict(text)} for text in demo_texts]
print(json.dumps({'deployment_MB':COMPACT_MANIFEST['deployment_bytes']/1e6,
    'dev':COMPACT_MANIFEST['dev_metrics'],'scope':COMPACT_MANIFEST['quality_scope'],
    'model':str(PREDICT_MODEL_DIR),'hashes_verified':True,'demos':compact_demos},ensure_ascii=False,indent=2))
'''
    if any('compression_result' in cell.get('metadata',{}).get('tags',[]) for cell in notebook['cells']):
        raise ValueError('Compression cells already exist; preserve their recorded outputs')
    notebook['cells'].extend([
        {'cell_type':'markdown','id':'compact-result-note','metadata':{'tags':['compression_result']},'source':markdown.splitlines(keepends=True)},
        {'cell_type':'code','id':'compact-result-check','metadata':{'tags':['compression_result']},'execution_count':None,'outputs':[], 'source':code.splitlines(keepends=True)}])
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(json.dumps({'cells':len(notebook['cells']),'backup':str(backup),'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()}))


if __name__ == '__main__':
    main()
