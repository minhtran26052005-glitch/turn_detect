"""Keep existing notebook work and append the executed tiny-student result."""
from __future__ import annotations
import json
from pathlib import Path
import re
import shutil
from build_notebook import source_bundle


def main():
    path=Path('a.ipynb');root=Path('outputs/distillation_20261024')
    manifest=json.loads(Path('models/turn_v2_distilled_min_20261024/model_manifest.json').read_text())
    notebook=json.loads(path.read_text(encoding='utf-8'))
    if any('distillation_result' in c.get('metadata',{}).get('tags',[]) for c in notebook['cells']):raise ValueError('Distillation result already exists')
    shutil.copy2(path,root/'a_before_distillation.ipynb')
    package,checksum=source_bundle()
    for cell in notebook['cells']:
        tags=cell.get('metadata',{}).get('tags',[])
        if cell['cell_type']=='code' and 'bootstrap' in tags:
            text=''.join(cell['source'])
            text,n=re.subn(r'^package = base64\.b64decode\(.*\)$',f'package = base64.b64decode({package!r})',text,flags=re.MULTILINE);assert n==1
            text,n=re.subn(r'^assert hashlib\.sha256\(package\)\.hexdigest\(\) == .*?$',f'assert hashlib.sha256(package).hexdigest() == {checksum!r}',text,flags=re.MULTILINE);assert n==1
            cell['source']=text.splitlines(keepends=True);cell['outputs']=[];cell['execution_count']=None;cell['metadata'].pop('colab_verified_at',None)
        if cell['cell_type']=='code' and 'crawl_predict' in tags:
            text=''.join(cell['source']);old='PREDICT_MODEL_DIR = COMPACT_ONNX if COMPACT_ONNX.is_dir() else (TRIAL_ONNX if TRIAL_ONNX.is_dir() else None)'
            new="DISTILLED_ONNX = ROOT/'models/turn_v2_distilled_min_20261024/onnx'\nPREDICT_MODEL_DIR = DISTILLED_ONNX if DISTILLED_ONNX.is_dir() else (COMPACT_ONNX if COMPACT_ONNX.is_dir() else (TRIAL_ONNX if TRIAL_ONNX.is_dir() else None))"
            assert old in text;cell['source']=text.replace(old,new).splitlines(keepends=True);cell['outputs']=[];cell['execution_count']=None;cell['metadata'].pop('colab_verified_at',None)
    metrics=manifest['dev_metrics']
    markdown=f'''# Distillation: student nhỏ nhất đã thử đạt mục tiêu

**GRU 96, embedding 64, 1 lớp**, {manifest['parameters_deployment']:,} tham số triển khai. ONNX **{manifest['onnx_bytes']/1000:.3f} KB**, cả thư mục model/tokenizer/runner/requirements **{manifest['deployment_bytes']/1000:.3f} KB** (không tính thư viện runtime đã cài), nhỏ hơn **{manifest['deployment_reduction_factor']:.1f} lần** gói teacher 109,6 MB.

Trên **17.833 mẫu dev**: accuracy **{metrics['accuracy']*100:.3f}%**, F1 EOT **{metrics['eot_f1']*100:.3f}%**, ngưỡng **{metrics['threshold']:.3f}**. Đã thử 6 lượt train/refinement và 18 biến thể lưu trọng số. Đây là bản nhỏ nhất **trong các bản đã thử** đạt cả hai chỉ số, không chứng minh mức nhỏ nhất tuyệt đối.

Student học từ nhãn train, logits và đặc trưng teacher cố định trên GPU L4. **Dev được dùng chọn bản/ngưỡng; holdout đã dùng không chấm lại.** F1 chỉ nhỉnh hơn 90% một ít, chưa bảo đảm trên hội thoại thực/câu ngắn.

Ô dưới kiểm tra hashes và suy luận bằng runner độc lập chỉ cần NumPy, Tokenizers và ONNX Runtime; không cần PyTorch/Transformers cho suy luận. [Model và hướng dẫn](models/turn_v2_distilled_min_20261024/README.md) · [Review](outputs/distillation_20261024/review.html) · [Hồ sơ](EXPERIMENT_DISTILLATION_20261024.md)
'''
    code='''import hashlib, importlib.util, json
from pathlib import Path
DISTILL_ROOT = Path('/content/turn_detect_v2') if Path('/content/turn_detect_v2').is_dir() else Path.cwd()
DISTILL_DIR = DISTILL_ROOT/'models/turn_v2_distilled_min_20261024'
distill_manifest = json.loads((DISTILL_DIR/'model_manifest.json').read_text(encoding='utf-8'))
PREDICT_MODEL_DIR = DISTILL_DIR/'onnx'
for filename, identity in distill_manifest['files'].items():
    file_path=PREDICT_MODEL_DIR/filename
    assert file_path.stat().st_size==identity['bytes']
    assert hashlib.file_digest(file_path.open('rb'),'sha256').hexdigest()==identity['sha256']
runtime_spec=importlib.util.spec_from_file_location('tiny_portable_runtime',PREDICT_MODEL_DIR/'run_model.py')
tiny_runtime=importlib.util.module_from_spec(runtime_spec)
runtime_spec.loader.exec_module(tiny_runtime)
tiny_detector=tiny_runtime.TinyOnnxDetector(PREDICT_MODEL_DIR)
tiny_texts=['hà nội là thủ đô và là một trong những trung tâm văn hóa lớn của việt nam','tôi muốn đặt vé đi','cảm ơn bạn']
print(json.dumps({'onnx_KB':distill_manifest['onnx_bytes']/1000,'deployment_KB':distill_manifest['deployment_bytes']/1000,
    'dev':distill_manifest['dev_metrics'],'scope':distill_manifest['quality_scope'],
    'hashes_verified':True,'model':str(PREDICT_MODEL_DIR),
    'demos':[{'text':text,**tiny_detector.predict(text)} for text in tiny_texts]},ensure_ascii=False,indent=2))
'''
    notebook['cells'].extend([{'cell_type':'markdown','id':'tiny-distill-result','metadata':{'tags':['distillation_result']},'source':markdown.splitlines(keepends=True)},
        {'cell_type':'code','id':'tiny-distill-verify','metadata':{'tags':['distillation_result']},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}])
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(json.dumps({'cells':len(notebook['cells']),'last_tag':'distillation_result'}))


if __name__=='__main__':main()
