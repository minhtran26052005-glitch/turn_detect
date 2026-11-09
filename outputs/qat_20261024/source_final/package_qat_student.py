"""Package the smallest tested QAT student meeting the frozen-dev targets."""
import json
from pathlib import Path
import shutil
import zipfile

from turn_v2.common import file_hash, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    evidence = Path("outputs/qat_20261024")
    protocol = read(evidence / "protocol.json")
    assert all(file_hash(Path(p)) == sha for p, sha in protocol["controls"].items())
    selection = read(evidence / "selection.json")
    assert selection["complete"] and selection["selected"]
    winner = selection["selected"]
    assert winner["quality_passed"]
    student = evidence / "runs" / winner["name"]
    training = read(student / "training_report.json")
    export = read(student / "qat_export_report.json")
    assert export["technical_parity_passed"] and export["frozen_scales_used_without_refit"]
    package = Path("models/turn_v2_distilled_qat4_20261024")
    model = package / "onnx"
    model.mkdir(parents=True, exist_ok=False)
    for path in (student / "onnx_int4").iterdir():
        if path.is_file():
            shutil.copy2(path, model / path.name)
    for name in ["run_model.py", "requirements.txt"]:
        shutil.copy2(Path("models/turn_v2_distilled_int4_20261024/onnx") / name, model / name)
    contract = read(model / "turn_detector_config.json")
    contract.update(compression_dev_report=winner["report"], quality_scope="Full frozen dev selection; no fresh holdout or real-dialogue guarantee")
    write_json(model / "turn_detector_config.json", contract)
    files = {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in sorted(model.iterdir()) if p.is_file()}
    controls = {"INT8": read("models/turn_v2_distilled_min_20261024/model_manifest.json"),
                "PTQ_INT4": read("models/turn_v2_distilled_int4_20261024/model_manifest.json")}
    manifest = {"status": "SMALLEST_PASSING_TESTED_QAT_INT4", "files": files,
        "onnx_bytes": files["model.onnx"]["bytes"], "deployment_bytes": sum(f["bytes"] for f in files.values()),
        "deployment_includes": "ONNX, tokenizer/configuration, native runner, requirements; installed libraries excluded",
        "training_method": "INT4_QAT_WITH_DISTILLATION", "storage_precision": "INT4 matrices, frozen FP16 group scales; FP32 biases/norm",
        "execution_precision": "FP32; weight dequantization at session load", "quantization_group_size": winner["group_size"],
        "parameters_deployment": 293090, "dev_metrics": winner["dev_metrics"], "selected_report": winner["report"],
        "source_run": winner["name"], "train_samples": training["train_samples"], "optimizer_steps": training["optimizer_steps"],
        "best_epoch": training["best_epoch"], "epochs_completed": len(read(student / "history.json")),
        "latent_student_sha256": training["student_sha256"], "quant_params_sha256": training["quant_params_sha256"],
        "training_fake_quant_vs_packed_max_probability_delta": winner["gpu_fake_quant_vs_cpu_packed_max_probability_delta"],
        "gradient_checks": read(student / "gradient_checks.json"), "candidates_trained": 2,
        "controls": {name: {"model_sha256": c["files"]["model.onnx"]["sha256"], "onnx_bytes": c["onnx_bytes"], "deployment_bytes": c["deployment_bytes"], "dev_metrics": c["dev_metrics"]} for name, c in controls.items()},
        "quality_scope": "Full frozen dev selection; no fresh holdout or real-dialogue guarantee", "holdout_evaluated": False}
    write_json(package / "model_manifest.json", manifest)
    metrics = winner["dev_metrics"]
    table = "\n".join(f"| QAT g{r['group_size']} | {r['model_bytes']/1000:.3f} | {r['dev_metrics']['accuracy']*100:.4f}% | {r['dev_metrics']['eot_f1']*100:.4f}% | {r['dev_metrics']['threshold']:.3f} |" for r in selection["results"])
    fixed64 = read("outputs/quant4_20261024/int4_g64_quality/report.json")["dev_selected_threshold"]
    comparison = "\n".join(f"| {name} | {c['onnx_bytes']/1000:.3f} | {c['dev_metrics']['accuracy']*100:.4f}% | {c['dev_metrics']['eot_f1']*100:.4f}% | {c['dev_metrics']['threshold']:.3f} |" for name, c in controls.items())
    description = f"""Đã fine-tune **QAT trọng số INT4** trên Colab L4 với toàn bộ315.174 mẫu train: giả lập lượng tử ở mỗi forward, gradient clipped STE đến cả năm ma trận. Nhóm32 và64, tối đa8epoch, LR0,0001, seed43, batch256. Loss gồm nhãn train và distillation logits/đặc trưng teacher cố định. Scale FP16 khởi tạo bằng MSE trọng số rồi giữ nguyên cả lúc train và export; không tái fit scale sau QAT.

Chọn **nhóm{winner['group_size']}**, {training['optimizer_steps']:,} bước, best epoch{training['best_epoch']}. **ONNX{manifest['onnx_bytes']:,}byte**, cả model/tokenizer/runner/requirements **{manifest['deployment_bytes']:,}byte**, không tính thư viện đã cài. Accuracy **{metrics['accuracy']*100:.4f}%**, F1 EOT **{metrics['eot_f1']*100:.4f}%**, ngưỡng **{metrics['threshold']:.3f}** trên đủ17.833 frozen dev bằng chính ONNX INT4 đã đóng gói. GPU fake-quant so với CPU packed lệch xác suất tối đa{winner['gpu_fake_quant_vs_cpu_packed_max_probability_delta']:.3g}.

| Bản | ONNX KB | Accuracy dev | F1 EOT dev | Ngưỡng |
| --- | ---: | ---: | ---: | ---: |
{comparison}
| PTQ_INT4 g64 | 176.999 | {fixed64['accuracy']*100:.4f}% | {fixed64['eot_f1']*100:.4f}% | {fixed64['threshold']:.3f} |
{table}

Trọng số ma trận lưu4-bit thực sự, hai mã mỗi byte. Bias/norm, activations và phép tính GRU vẫn FP32; QAT không tự tạo kernel INT4. Mục tiêu lượt này là dung lượng và điểm dev. Dev dùng chọn checkpoint/ngưỡng, không phải đánh giá độc lập; không chấm lại consumed holdout. Nhiều NEOT là tiền tố EOT và các câu ngắn như “cảm ơn bạn” có thể sai. Không quy toàn bộ thay đổi điểm cho QAT vì chưa có đối chứng fine-tune FP32 với cùng ngân sách.
"""
    readme = "# Student GRU QAT INT4\n\n" + description + """
## Chạy

```powershell
python -m pip install -r models/turn_v2_distilled_qat4_20261024/onnx/requirements.txt
python models/turn_v2_distilled_qat4_20261024/onnx/run_model.py --text "tôi muốn đặt vé đi"
```

Runner không cần Torch/Transformers. Ô `qat_result` cuối `a.ipynb` kiểm tra hash và chạy trên Colab. Bản INT8/PTQ cũ còn nguyên.

[Đối chiếu đầy đủ](../../outputs/qat_20261024/selection.json) · [Hồ sơ](../../EXPERIMENT_QAT_20261024.md) · [PyTorch QAT](https://pytorch.org/blog/quantization-aware-training/)
"""
    (package / "README.md").write_text(readme, encoding="utf-8")
    experiment = "# QAT INT4 student — 2026-10-24\n\nMaterial Passport: **TRAINED_AND_PACKED_FULL_DEV_EVALUATED; CROSS_PLATFORM_VERIFICATION_PENDING**.\n\n" + description
    experiment += "\nLần khởi động nhóm64 đầu lỗi cuDNN vì stride từ nhóm cuối chưa đầy trước bước optimizer nào; sửa output fakequant `.contiguous()`, thêm test CUDA forward/backward lặp lại, rồi chạy `r2` riêng. Giữ log lỗi và mã nguồn trước sửa. Không tính lần khởi động lỗi là mô hình QAT hoàn tất. Source/receipts/controls trong `outputs/qat_20261024/`.\n"
    Path("EXPERIMENT_QAT_20261024.md").write_text(experiment, encoding="utf-8")
    archive = package.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                out.write(path, path.relative_to(package.parent))
    write_json(evidence / "package_receipt.json", {"zip": str(archive), "bytes": archive.stat().st_size, "sha256": file_hash(archive),
        "manifest_sha256": file_hash(package / "model_manifest.json"), "onnx_bytes": manifest["onnx_bytes"], "deployment_bytes": manifest["deployment_bytes"]})
    print(json.dumps({"model": str(package), "onnx_bytes": manifest["onnx_bytes"], "deployment_bytes": manifest["deployment_bytes"], "dev": metrics}))


if __name__ == "__main__":
    main()
