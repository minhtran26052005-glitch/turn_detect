"""Append the INT4 result to the existing notebook, preserving all old cells."""
import json
from pathlib import Path
import re
import shutil

from build_notebook import cell, source_bundle


def main():
    notebook_path = Path("a.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    if any("int4_result" in c.get("metadata", {}).get("tags", []) for c in notebook["cells"]):
        raise ValueError("INT4 cells already present")
    shutil.copy2(notebook_path, Path("outputs/quant4_20261024/a_before_int4.ipynb"))
    package, checksum = source_bundle()
    for entry in notebook["cells"]:
        tags = entry.get("metadata", {}).get("tags", [])
        text = "".join(entry["source"])
        if entry["cell_type"] == "code" and "bootstrap" in tags:
            text, count = re.subn(r"^package = base64\.b64decode\(.*\)$", f"package = base64.b64decode({package!r})", text, flags=re.MULTILINE)
            assert count == 1
            text, count = re.subn(r"^assert hashlib\.sha256\(package\)\.hexdigest\(\) == .*?$", f"assert hashlib.sha256(package).hexdigest() == {checksum!r}", text, flags=re.MULTILINE)
            assert count == 1
            entry["source"] = text.splitlines(keepends=True)
            entry["execution_count"] = None
            entry["outputs"] = []
            entry["metadata"].pop("colab_verified_at", None)
        if entry["cell_type"] == "code" and "crawl_predict" in tags:
            old = "PREDICT_MODEL_DIR = DISTILLED_ONNX if DISTILLED_ONNX.is_dir() else (COMPACT_ONNX if COMPACT_ONNX.is_dir() else (TRIAL_ONNX if TRIAL_ONNX.is_dir() else None))"
            assert old in text
            new = "INT4_ONNX = ROOT/'models/turn_v2_distilled_int4_20261024/onnx'\nPREDICT_MODEL_DIR = INT4_ONNX if INT4_ONNX.is_dir() else (DISTILLED_ONNX if DISTILLED_ONNX.is_dir() else (COMPACT_ONNX if COMPACT_ONNX.is_dir() else (TRIAL_ONNX if TRIAL_ONNX.is_dir() else None)))"
            entry["source"] = text.replace(old, new).splitlines(keepends=True)
            entry["execution_count"] = None
            entry["outputs"] = []
            entry["metadata"].pop("colab_verified_at", None)
    markdown = """
    ## Student INT4 đã kiểm tra

    Trọng số ma trận **INT4**, nhóm32, scale FP16; bias/norm và phép tính FP32. **ONNX 185,933 KB**, cả model/tokenizer/runner **412,682 KB** (thư viện đã cài không tính). INT8 đối chứng tương ứng 328,761 / 555,479 KB.

    Đủ **17.833 dev**: accuracy **93,4335%**, F1 EOT **90,0620%**, ngưỡng hiệu chỉnh **0,430**. Nhóm32 nhỏ nhất đạt trong bốn cấu hình đã chấm. Dev dùng chọn cấu hình và ngưỡng; không phải holdout độc lập. Giữ ngưỡng INT8 0,545 thì F1 INT4 chỉ89,8236%. False EOT tăng744→864, missed EOT giảm404→307.

    Ô dưới dùng runner độc lập, kiểm tra hashes và chạy ba ví dụ cố định; dự đoán sai trên câu ngắn vẫn được giữ để xem giới hạn. Bản INT8 cũ còn nguyên.

    [Model/hướng dẫn](models/turn_v2_distilled_int4_20261024/README.md) · [ZIP](models/turn_v2_distilled_int4_20261024.zip) · [Hồ sơ](EXPERIMENT_INT4_20261024.md)
    """
    code = """
    import hashlib, importlib.util, json, sys
    from pathlib import Path
    INT4_ROOT = Path('/content/turn_detect_v2') if Path('/content/turn_detect_v2').is_dir() else Path.cwd()
    INT4_DIR = INT4_ROOT/'models/turn_v2_distilled_int4_20261024'
    int4_manifest = json.loads((INT4_DIR/'model_manifest.json').read_text(encoding='utf-8'))
    PREDICT_MODEL_DIR = INT4_DIR/'onnx'
    for filename, identity in int4_manifest['files'].items():
        file_path = PREDICT_MODEL_DIR/filename
        assert file_path.stat().st_size == identity['bytes']
        with file_path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == identity['sha256']
    int4_spec = importlib.util.spec_from_file_location('int4_portable_runtime', PREDICT_MODEL_DIR/'run_model.py')
    int4_runtime = importlib.util.module_from_spec(int4_spec)
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try: int4_spec.loader.exec_module(int4_runtime)
    finally: sys.dont_write_bytecode = previous_bytecode
    int4_detector = int4_runtime.TinyOnnxDetector(PREDICT_MODEL_DIR)
    int4_texts = ['hà nội là thủ đô và là một trong những trung tâm văn hóa lớn của việt nam', 'tôi muốn đặt vé đi', 'cảm ơn bạn']
    print(json.dumps({'hashes_verified': True, 'onnx_KB': int4_manifest['onnx_bytes']/1000,
        'deployment_KB': int4_manifest['deployment_bytes']/1000, 'dev': int4_manifest['dev_metrics'],
        'scope': int4_manifest['quality_scope'], 'model': str(PREDICT_MODEL_DIR),
        'demos': [{'text': text, **int4_detector.predict(text)} for text in int4_texts]}, ensure_ascii=False, indent=2))
    """
    notebook["cells"].extend([cell("markdown", markdown, "int4_result"), cell("code", code, "int4_result")])
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    paragraph = "Đã kiểm tra thêm [student INT4](models/turn_v2_distilled_int4_20261024/README.md): **ONNX 186 KB, cả gói 413 KB**, dev accuracy **93,43%**, F1 EOT **90,06%**, ngưỡng mới **0,430**. Lưu trọng số 4-bit, phép tính FP32; dev dùng chọn bản/ngưỡng, chưa có holdout độc lập mới. [ZIP](models/turn_v2_distilled_int4_20261024.zip) · ô `int4_result` cuối `a.ipynb`.\n\n"
    for filename in ["README_VI.md", "README_TRAIN_V2.md"]:
        path = Path(filename)
        text = path.read_text(encoding="utf-8")
        first, rest = text.split("\n\n", 1)
        path.write_text(first + "\n\n" + paragraph + rest, encoding="utf-8")
    print(json.dumps({"cells": len(notebook["cells"]), "tag": "int4_result", "source_bundle_sha256": checksum}))


if __name__ == "__main__":
    main()
