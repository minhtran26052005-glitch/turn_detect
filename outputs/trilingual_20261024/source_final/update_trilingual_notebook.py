"""Add an explicit pilot result; keep the accepted Vietnamese serving default."""
import json
from pathlib import Path
import re
import shutil

from build_notebook import cell, source_bundle


def main():
    path = Path("a.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert not any("trilingual_result" in c.get("metadata", {}).get("tags", []) for c in notebook["cells"])
    root = Path("outputs/trilingual_20261024")
    shutil.copy2(path, root / "a_before_trilingual.ipynb")
    package, checksum = source_bundle()
    for entry in notebook["cells"]:
        if entry["cell_type"] == "code" and "bootstrap" in entry.get("metadata", {}).get("tags", []):
            text = "".join(entry["source"])
            text, n = re.subn(r"^package = base64\.b64decode\(.*\)$", f"package = base64.b64decode({package!r})", text, flags=re.MULTILINE); assert n == 1
            text, n = re.subn(r"^assert hashlib\.sha256\(package\)\.hexdigest\(\) == .*?$", f"assert hashlib.sha256(package).hexdigest() == {checksum!r}", text, flags=re.MULTILINE); assert n == 1
            entry["source"] = text.splitlines(keepends=True); entry["outputs"] = []; entry["execution_count"] = None
            entry["metadata"].pop("colab_verified_at", None)
    markdown = """
    ## Pilot Việt–Anh–Trung: đã train, chưa đạt chất lượng để thay bản Việt

    Teacher12lớp đã fine-tune800bước; student GRU128 distill11epoch rồi QAT6epoch. Train71.329 mẫu, dev7.190 mẫu từ Việt frozen-train + MultiWOZ2.2 + CrossWOZ. ONNX729KB, cả gói1,889MB, vocabulary18.711. Trọng số INT4, phép tính FP32.

    **Không thay `PREDICT_MODEL_DIR` đang dùng bản Việt.** Student có regression Việt; English/Chinese chưa đạt90%. Nhãn cắt prefix là proxy, không phải semantic gold. Ngoài test nhãn suy ra, đã kiểm tra744turn thật ở LiveKit,2.057điểm dừng đủ200ms; chỉ dùng transcript/timing, không tải audio, không tune trên benchmark.

    [Báo cáo tổng](outputs/trilingual_20261024/review.html) · [Player từng prefix và dấu phẩy/chấm](outputs/trilingual_stream_20261024/review.html) · [Hướng dẫn/dữ liệu/model](README_TRILINGUAL_V3.md).

    Ô dưới kiểm tra hashes và chạy3ngôn ngữ trên đúng file INT4. Đầu vào dài bị cắt sẽ được đánh dấu; dấu câu tự nó không phải tín hiệu EOT trong model này.
    """
    code = """
    import hashlib, importlib.util, json, sys
    from pathlib import Path
    TRI_ROOT = Path('/content/turn_detect_v2') if Path('/content/turn_detect_v2').is_dir() else Path.cwd()
    TRI_DIR = TRI_ROOT/'models/turn_v3_trilingual_pilot_20261024'
    tri_manifest = json.loads((TRI_DIR/'model_manifest.json').read_text(encoding='utf-8'))
    for filename, identity in tri_manifest['files'].items():
        file_path = TRI_DIR/'onnx'/filename
        assert file_path.stat().st_size == identity['bytes']
        with file_path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == identity['sha256']
    tri_spec = importlib.util.spec_from_file_location('trilingual_portable', TRI_DIR/'onnx/run_model.py')
    tri_runtime = importlib.util.module_from_spec(tri_spec)
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try: tri_spec.loader.exec_module(tri_runtime)
    finally: sys.dont_write_bytecode = previous_bytecode
    tri_detector = tri_runtime.TinyOnnxDetector(TRI_DIR/'onnx')
    tri_cases = [('vi', 'tôi muốn đặt một vé tàu đi Hà Nội vào sáng mai'), ('vi', 'tôi muốn đặt một vé tàu đi'),
                 ('en', 'I would like to book a flight to London tomorrow'), ('en', 'I would like to book a flight to'),
                 ('zh', '我想订一张明天从北京到上海的火车票'), ('zh', '我想订一张从北京到')]
    print(json.dumps({'hashes_verified': True, 'status': tri_manifest['status'],
        'onnx_KB': tri_manifest['onnx_bytes']/1000, 'deployment_KB': tri_manifest['deployment_bytes']/1000,
        'proxy_test': tri_manifest['derived_official_test'], 'real_pause_text_only': tri_manifest['external_real_pause_text_only'],
        'vi_full_dev': tri_manifest['vi_full_dev_regression'], 'serving_default_changed': False,
        'demos': [{'text': text, **tri_detector.predict(text, language=lang)} for lang, text in tri_cases]}, ensure_ascii=False, indent=2))
    """
    notebook["cells"].extend([cell("markdown", markdown, "trilingual_result"), cell("code", code, "trilingual_result")])
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    paragraph = "Đã chạy [pilot Việt–Anh–Trung](README_TRILINGUAL_V3.md) theo teacher→distill→QAT, **chưa đạt90% cả ba tiếng và có regression Việt**; giữ bản Việt làm mặc định. [Kết quả](outputs/trilingual_20261024/review.html) · [Player streaming từng prefix](outputs/trilingual_stream_20261024/review.html). Ô `trilingual_result` cuối notebook dùng bản thử riêng.\n\n"
    for filename in ["README_VI.md", "README_TRAIN_V2.md"]:
        target = Path(filename); title, body = target.read_text(encoding="utf-8").split("\n\n", 1)
        target.write_text(title + "\n\n" + paragraph + body, encoding="utf-8")
    print(json.dumps({"cells": len(notebook["cells"]), "tag": "trilingual_result", "default_model_changed": False}))


if __name__ == "__main__": main()
