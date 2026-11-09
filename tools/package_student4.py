"""Publish a separate INT4 student package; preserve the original INT8."""
import json
from pathlib import Path
import shutil
import zipfile

from turn_v2.common import file_hash, write_json


def main():
    evidence = Path("outputs/quant4_20261024")
    selection = json.loads((evidence / "selection.json").read_text(encoding="utf-8"))
    assert selection["complete"] and selection["controls_unchanged"]
    winner = selection["selected"]
    assert winner and winner["quality_passed"]
    destination = Path("models/turn_v2_distilled_int4_20261024")
    model = destination / "onnx"
    model.mkdir(parents=True, exist_ok=False)
    for path in (evidence / winner["name"]).iterdir():
        if path.is_file():
            shutil.copy2(path, model / path.name)
    for name in ["run_model.py", "requirements.txt"]:
        shutil.copy2(Path("models/turn_v2_distilled_min_20261024/onnx") / name, model / name)
    metrics = winner["metrics"]
    contract_path = model / "turn_detector_config.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["compression_dev_report"] = winner["report"].replace("\\", "/")
    contract["quality_scope"] = "Full frozen dev selection; no fresh holdout or real-dialogue guarantee"
    write_json(contract_path, contract)
    baseline = json.loads(Path("models/turn_v2_distilled_min_20261024/model_manifest.json").read_text(encoding="utf-8"))
    files = {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in sorted(model.iterdir()) if p.is_file()}
    manifest = {
        "status": "SMALLEST_PASSING_OF_FOUR_INT4_CANDIDATES", "files": files,
        "onnx_bytes": files["model.onnx"]["bytes"], "deployment_bytes": sum(p["bytes"] for p in files.values()),
        "deployment_includes": "ONNX, tokenizer/configs, standalone runner, requirements; installed libraries excluded",
        "storage_precision": "INT4 (292288 matrix weights), FP16 group scales, FP32 biases/norm",
        "execution_precision": "FP32; dequantize constant weights at session load; no native INT4 GRU kernel",
        "quantization_group_size": contract["quantization_group_size"], "parameters_deployment": 293090,
        "source_fp32_sha256": json.loads((evidence / (winner["name"] + "_export.json")).read_text(encoding="utf-8"))["source_sha256"],
        "baseline_int8_sha256": baseline["files"]["model.onnx"]["sha256"],
        "baseline_model_bytes": baseline["onnx_bytes"], "baseline_deployment_bytes": baseline["deployment_bytes"],
        "dev_metrics": metrics, "baseline_dev_metrics": baseline["dev_metrics"],
        "selected_report": winner["report"].replace("\\", "/"), "candidates_tested": 4,
        "threshold_recalibrated_on_dev": True, "holdout_evaluated": False,
        "quality_scope": "Full frozen dev selection; no fresh holdout or real-dialogue guarantee",
    }
    write_json(destination / "model_manifest.json", manifest)
    table = "\n".join(f"| {r['name']} | {r['model_bytes'] / 1000:.3f} | {r['metrics']['accuracy'] * 100:.4f}% | {r['metrics']['eot_f1'] * 100:.4f}% | {'Đạt' if r['quality_passed'] else 'Không đạt'} |" for r in selection["results"])
    readme = f"""# Student GRU INT4

Đã nén trọng số student GRU96/embedding64 xuống **INT4 thực sự: hai trọng số mỗi byte**. File ONNX **{manifest['onnx_bytes']:,} byte**, cả thư mục `onnx/` **{manifest['deployment_bytes']:,} byte**, chưa tính thư viện runtime đã cài. INT8 đối chứng: 328.761 / 555.479 byte. Tokenizer giữ nguyên 219.195 byte nên cả gói không giảm một nửa.

Trên đủ **17.833 mẫu frozen dev**: accuracy **{metrics['accuracy'] * 100:.4f}%**, F1 EOT **{metrics['eot_f1'] * 100:.4f}%**, ngưỡng **{metrics['threshold']:.3f}**. Chọn nhóm 32 nhỏ nhất đạt trong bốn cấu hình đã thử, không cần train lại. Dev đã dùng chọn cấu hình/ngưỡng; không phải holdout độc lập. Holdout đã dùng không được chấm lại.

## Dùng ngay

```powershell
python -m pip install -r models/turn_v2_distilled_int4_20261024/onnx/requirements.txt
python models/turn_v2_distilled_int4_20261024/onnx/run_model.py --text "tôi muốn đặt vé đi"
```

Runner chỉ cần NumPy, Tokenizers và ONNX Runtime 1.23.2; không cần Torch/Transformers. Ô `int4_result` cuối `a.ipynb` kiểm tra hashes và chạy model trên kernel Colab đang kết nối.

## Kiểu nén và giới hạn

292.288 trọng số ma trận (embedding, hai ma trận GRU, hai lớp Linear) lưu INT4, nhóm 32 với scale FP16. Bias/LayerNorm giữ FP32. ONNX opset21 `DequantizeLinear` rồi Cast giải nén trọng số về FP32 lúc nạp session; phép tính và activations vẫn FP32. Dung lượng file giảm, không suy ra RAM hay thời gian chạy giảm tương ứng. [Đặc tả ONNX](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html#dequantizelinear-21) và [phạm vi INT4 của ORT](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#quantize-to-int4uint4).

Ngưỡng đổi từ 0,545 (INT8) sang 0,430 (INT4). Giữ ngưỡng INT8 thì F1 chỉ **89,8236%**. Ngưỡng mới cho false EOT **864**, missed EOT **307**; đối chứng INT8 lần lượt **744 / 404**. Ưu tiên cả accuracy/F1 không có nghĩa là mọi loại lỗi đều tốt hơn. F1 chỉ nhỉnh hơn 90% khoảng 0,062 điểm phần trăm.

Data chưa có benchmark hội thoại người gán nhãn độc lập; nhiều NEOT là tiền tố từ EOT. Các ví dụ ngắn như “cảm ơn bạn” vẫn có thể bị dự đoán NEOT. Bản INT8 gốc được giữ nguyên ở `../turn_v2_distilled_min_20261024/`.

| Cấu hình | ONNX KB | Accuracy dev | F1 EOT dev | Cả hai ≥90% |
| --- | ---: | ---: | ---: | --- |
{table}

[Báo cáo dev](../../{manifest['selected_report']}) · [Tất cả ứng viên](../../outputs/quant4_20261024/selection.json) · [Hồ sơ thực nghiệm](../../EXPERIMENT_INT4_20261024.md)
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    protocol = json.loads((evidence / "protocol.json").read_text(encoding="utf-8"))
    experiment = f"""# INT4 student experiment — 2026-10-24

Material Passport: **EVALUATED_ON_FULL_DEV; PACKAGING_VERIFICATION_PENDING**.

Mục tiêu: trọng số 4-bit và model nhỏ hơn INT8, accuracy và F1 EOT ≥90% trên frozen dev; không mở lại holdout đã dùng. Nguồn FP32 là checkpoint GRU96 distill trước đó, SHA256 `{manifest['source_fp32_sha256']}`. INT8 đối chứng SHA256 `{manifest['baseline_int8_sha256']}`.

Phương pháp: ONNX version converter 18→21; năm ma trận INT4 signed [-8,7], hai mã mỗi byte. Scale FP16 cho mỗi nhóm theo trục cuối; chọn trong hệ số [0,8;0,85;0,9;0,95;1,0] bằng MSE trọng số, không dùng activation dev để định lượng. Giữ bias/norm FP32. DequantizeLinear block và Cast đưa về FP32 lúc load; không có kernel GRU INT4. Seed42 cho kiểm tra kỹ thuật batch1/3/2, sequence8/24/256, thêm8padding; delta0.

Đã chấm đủ 17.833 dev cho bốn nhóm [64,32,16,8], cùng tokenizer và cùng lưới ngưỡng 181 điểm 0,05..0,95 như đối chứng. Chọn file ONNX nhỏ nhất đạt cả hai, nhóm32: **{manifest['onnx_bytes']:,} byte**, cả gói **{manifest['deployment_bytes']:,} byte**. Không có train/QAT/RL/crawl mới trong lượt này.

| Cấu hình | ONNX KB | Accuracy dev | F1 EOT dev | Cả hai ≥90% |
| --- | ---: | ---: | ---: | --- |
{table}

F1 tại ngưỡng INT8 0,545 là89,8236%; sau hiệu chỉnh dev0,430 đạt90,06195%. False EOT tăng744→864, missed EOT giảm404→307. Chỉ chênh0,06195 điểm phần trăm so với yêu cầu90, nên không khẳng định chắc chắn trên dữ liệu mới. Data/holdout limitations kế thừa nguyên bản distillation. Không coi cờ size_target_1/10 của evaluate_compressed là điều kiện lần này; budget lần này là nhỏ hơn328.761byte/555.479byte INT8.

Nguồn: [ONNX DQ21](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html#dequantizelinear-21), [ORT INT4](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#quantize-to-int4uint4). Receipts subprocess ghi exit_code thật; nguồn và frozen identities nằm trong `outputs/quant4_20261024/`. Mọi đánh giá là dev selection, không phải chứng nhận độc lập.
"""
    Path("EXPERIMENT_INT4_20261024.md").write_text(experiment, encoding="utf-8")
    source_copy = evidence / "source"
    source_copy.mkdir(exist_ok=False)
    for p in [Path("turn_v2/quantize_student4.py"), Path("tools/test_student4.py"), Path("tools/package_student4.py"), Path("tests/test_quantize_student4.py")]:
        shutil.copy2(p, source_copy / p.name)
    archive = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                out.write(path, path.relative_to(destination.parent))
    write_json(evidence / "package_receipt.json", {"zip": str(archive), "bytes": archive.stat().st_size,
        "sha256": file_hash(archive), "manifest_sha256": file_hash(destination / "model_manifest.json"),
        "deployment_bytes": manifest["deployment_bytes"]})
    print(json.dumps({"onnx_bytes": manifest["onnx_bytes"], "deployment_bytes": manifest["deployment_bytes"], "zip_bytes": archive.stat().st_size}))


if __name__ == "__main__":
    main()
