"""Append QAT results to a.ipynb while retaining earlier experiment outputs."""
import json
from pathlib import Path
import re
import shutil

from build_notebook import cell, source_bundle


def main():
    path = Path("a.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert not any("qat_result" in c.get("metadata", {}).get("tags", []) for c in notebook["cells"])
    evidence = Path("outputs/qat_20261024")
    shutil.copy2(path, evidence / "a_before_qat.ipynb")
    manifest = json.loads(Path("models/turn_v2_distilled_qat4_20261024/model_manifest.json").read_text(encoding="utf-8"))
    package, checksum = source_bundle()
    for entry in notebook["cells"]:
        tags = entry.get("metadata", {}).get("tags", [])
        text = "".join(entry["source"])
        if entry["cell_type"] == "code" and "bootstrap" in tags:
            text, n = re.subn(r"^package = base64\.b64decode\(.*\)$", f"package = base64.b64decode({package!r})", text, flags=re.MULTILINE)
            assert n == 1
            text, n = re.subn(r"^assert hashlib\.sha256\(package\)\.hexdigest\(\) == .*?$", f"assert hashlib.sha256(package).hexdigest() == {checksum!r}", text, flags=re.MULTILINE)
            assert n == 1
            entry["source"] = text.splitlines(keepends=True)
            entry["outputs"] = []
            entry["execution_count"] = None
            entry["metadata"].pop("colab_verified_at", None)
        if entry["cell_type"] == "code" and "crawl_predict" in tags:
            old = "PREDICT_MODEL_DIR = INT4_ONNX if INT4_ONNX.is_dir() else (DISTILLED_ONNX if DISTILLED_ONNX.is_dir() else (COMPACT_ONNX if COMPACT_ONNX.is_dir() else (TRIAL_ONNX if TRIAL_ONNX.is_dir() else None)))"
            assert old in text
            new = "QAT_ONNX = ROOT/'models/turn_v2_distilled_qat4_20261024/onnx'\nPREDICT_MODEL_DIR = QAT_ONNX if QAT_ONNX.is_dir() else (INT4_ONNX if INT4_ONNX.is_dir() else (DISTILLED_ONNX if DISTILLED_ONNX.is_dir() else (COMPACT_ONNX if COMPACT_ONNX.is_dir() else (TRIAL_ONNX if TRIAL_ONNX.is_dir() else None))))"
            entry["source"] = text.replace(old, new).splitlines(keepends=True)
            entry["outputs"] = []
            entry["execution_count"] = None
            entry["metadata"].pop("colab_verified_at", None)
    metrics = manifest["dev_metrics"]
    markdown = f"""
    ## Student QAT INT4

    Đã chạy QAT thật trên **315.174 train**, GPU L4: giả lập4-bit ở mỗi forward, cập nhật trọng số bằng clipped STE và loss distillation. Scale FP16 giữ nguyên từ train sang export, không fit lại khi nén. Hai lượt QAT nhóm64/32; chọn nhóm**{manifest['quantization_group_size']}**, best epoch**{manifest['best_epoch']}**.

    ONNX **{manifest['onnx_bytes']/1000:.3f} KB**, cả gói **{manifest['deployment_bytes']/1000:.3f} KB**. Chính ONNX packed INT4 đạt accuracy **{metrics['accuracy']*100:.4f}%**, F1 EOT **{metrics['eot_f1']*100:.4f}%**, ngưỡng **{metrics['threshold']:.3f}** trên đủ17.833 dev. Dev dùng chọn checkpoint/ngưỡng, chưa có holdout độc lập mới. Trọng số4-bit, activations/phép tính FP32.

    [Model/hướng dẫn](models/turn_v2_distilled_qat4_20261024/README.md) · [ZIP](models/turn_v2_distilled_qat4_20261024.zip) · [Hồ sơ](EXPERIMENT_QAT_20261024.md). Bản INT8 và PTQ giữ nguyên.
    """
    code = """
    import hashlib, importlib.util, json, sys
    from pathlib import Path
    QAT_ROOT = Path('/content/turn_detect_v2') if Path('/content/turn_detect_v2').is_dir() else Path.cwd()
    QAT_DIR = QAT_ROOT/'models/turn_v2_distilled_qat4_20261024'
    qat_manifest = json.loads((QAT_DIR/'model_manifest.json').read_text(encoding='utf-8'))
    PREDICT_MODEL_DIR = QAT_DIR/'onnx'
    for filename, identity in qat_manifest['files'].items():
        file_path = PREDICT_MODEL_DIR/filename
        assert file_path.stat().st_size == identity['bytes']
        with file_path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == identity['sha256']
    qat_spec = importlib.util.spec_from_file_location('qat_portable_runtime', PREDICT_MODEL_DIR/'run_model.py')
    qat_runtime = importlib.util.module_from_spec(qat_spec)
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try: qat_spec.loader.exec_module(qat_runtime)
    finally: sys.dont_write_bytecode = previous_bytecode
    qat_detector = qat_runtime.TinyOnnxDetector(PREDICT_MODEL_DIR)
    qat_texts = ['hà nội là thủ đô và là một trong những trung tâm văn hóa lớn của việt nam', 'tôi muốn đặt vé đi', 'cảm ơn bạn']
    print(json.dumps({'hashes_verified': True, 'training_method': qat_manifest['training_method'],
        'onnx_KB': qat_manifest['onnx_bytes']/1000, 'deployment_KB': qat_manifest['deployment_bytes']/1000,
        'dev': qat_manifest['dev_metrics'], 'scope': qat_manifest['quality_scope'], 'model': str(PREDICT_MODEL_DIR),
        'demos': [{'text': text, **qat_detector.predict(text)} for text in qat_texts]}, ensure_ascii=False, indent=2))
    """
    notebook["cells"].extend([cell("markdown", markdown, "qat_result"), cell("code", code, "qat_result")])
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    paragraph = f"Đã chạy [QAT INT4](models/turn_v2_distilled_qat4_20261024/README.md) trên Colab: ONNX **{manifest['onnx_bytes']/1000:.1f} KB**, cả gói **{manifest['deployment_bytes']/1000:.1f} KB**, dev accuracy **{metrics['accuracy']*100:.2f}%**, F1 EOT **{metrics['eot_f1']*100:.2f}%**, ngưỡng **{metrics['threshold']:.3f}**. Trọng số4-bit và activations/phép tính FP32; chưa có holdout mới. [ZIP](models/turn_v2_distilled_qat4_20261024.zip) · ô `qat_result` cuối `a.ipynb`.\n\n"
    for filename in ["README_VI.md", "README_TRAIN_V2.md"]:
        target = Path(filename)
        title, body = target.read_text(encoding="utf-8").split("\n\n", 1)
        target.write_text(title + "\n\n" + paragraph + body, encoding="utf-8")
    print(json.dumps({"cells": len(notebook["cells"]), "source_bundle_sha256": checksum}))


if __name__ == "__main__":
    main()
