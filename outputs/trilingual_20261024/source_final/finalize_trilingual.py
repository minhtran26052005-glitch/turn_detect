"""Package and report a failed-target pilot honestly; retain the VI default."""
import html
import json
from pathlib import Path
import zipfile

from turn_v2.common import file_hash, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    root = Path("outputs/trilingual_20261024")
    package = Path("models/turn_v3_trilingual_pilot_20261024")
    model = package / "onnx"
    dev = read(root / "student_dev_report.json")
    test = read(root / "runs/tri_final_20261024/report.json")
    real = read(root / "runs/tri_real_pause_20261024/report.json")
    teacher = read(root / "runs/tri_teacher_20261024/training_report.json")
    student = read(root / "runs/tri_student_20261024/training_report.json")
    data = read("data_v3/pilot_trilingual_20261024/manifest.json")
    export = read(root / "runs/tri_student_20261024/qat_best/qat_export_report.json")
    files = {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in sorted(model.iterdir()) if p.is_file()}
    manifest = {"status": "EXPERIMENTAL_PILOT_NOT_READY; ALL_LANGUAGE_90_TARGET_NOT_MET; VI_REGRESSION",
        "files": files, "onnx_bytes": files["model.onnx"]["bytes"], "deployment_bytes": sum(v["bytes"] for v in files.values()),
        "deployment_includes": "ONNX, tokenizer/configuration, portable runner, requirements; installed libraries excluded",
        "architecture": student["config"], "deployment_parameters": student["deployment_parameters"], "vocab_size": student["config"]["vocab_size"],
        "languages": ["vi", "en", "zh"], "storage_precision": "INT4 matrices; FP16 frozen scales; FP32 bias/norm", "execution_precision": "FP32",
        "teacher_checkpoint_sha256": teacher["checkpoint_sha256"], "latent_student_sha256": export["latent_student_sha256"],
        "quant_params_sha256": export["quant_params_sha256"], "data_manifest_sha256": teacher["data_manifest_sha256"],
        "dev": dev["dev"], "vi_full_dev_regression": dev["vi_full_dev"], "derived_official_test": test["student_INT4"],
        "external_real_pause_text_only": real["models"]["student_INT4"], "thresholds_frozen_before_tests": True,
        "all_language_target_met": False, "promoted_over_vi_control": False,
        "scope": "Structural proxy pilot plus fixed 200ms external real-pause text-only check; no production quality claim"}
    write_json(package / "model_manifest.json", manifest)
    md = """# Bản thử Việt–Anh–Trung: teacher → distill → QAT INT4

**Chưa đạt mục tiêu 90% cả ba ngôn ngữ. Không thay bản Việt QAT đang có.** Bản này dùng kiểm chứng đường train và lỗi dữ liệu/streaming.

"""
    md += f"Teacher mDeBERTa12lớp/768: khôi phục encoder/head và embedding đã có của bản Việt, mở lại vocabulary đa ngôn ngữ pretrained rồi fine-tune cân bằng ba tiếng800bước trên Colab L4. Train **{student['train_samples']:,} mẫu**, dev **{student['dev_samples']:,} mẫu**. Student GRU128/embedding64/1lớp: distill {student['phases'][0]['epochs']}epoch, tiếp QAT {student['phases'][1]['epochs']}epoch. Vocabulary **{manifest['vocab_size']:,}**, chỉ fit train, giữ fallback Latin/CJK. ONNX **{manifest['onnx_bytes']:,}byte**, cả thư mục `onnx/` **{manifest['deployment_bytes']:,}byte**; không tính thư viện đã cài. Trọng số matrix4-bit, phép tính FP32.\n\n"
    md += "## Kết quả trên chính file INT4\n\n| Tập | Ngôn ngữ | Mẫu | Accuracy | F1 EOT |\n| --- | --- | ---: | ---: | ---: |\n"
    for lang, m in dev["dev"].items():md += f"| Dev pilot nhãn suy ra | {lang} | {m['samples']} | {m['accuracy']*100:.2f}% | {m['eot_f1']*100:.2f}% |\n"
    for lang, m in test["student_INT4"].items():md += f"| Official-test hội thoại, nhãn suy ra | {lang} | {m['samples']} | {m['accuracy']*100:.2f}% | {m['eot_f1']*100:.2f}% |\n"
    for lang, m in real["models"]["student_INT4"]["span_metrics"].items():md += f"| LiveKit: văn bản tại khoảng dừng thực | {lang} | {m['samples']} | {m['accuracy']*100:.2f}% | {m['eot_f1']*100:.2f}% |\n"
    vi = dev["vi_full_dev"]
    md += f"\nFull Vietnamese dev17.833: student accuracy **{vi['accuracy']*100:.2f}%**, F1 **{vi['eot_f1']*100:.2f}%**; bản Việt QAT riêng F1 **90,19%**. Teacher ba tiếng giữ F1Việt **{test['teacher_vi_full_dev_regression']['eot_f1']*100:.2f}%**. Student nhỏ bị regression tiếng Việt.\n\n"
    md += """## Dùng để kiểm tra

```powershell
python -m pip install -r models/turn_v3_trilingual_pilot_20261024/onnx/requirements.txt
python models/turn_v3_trilingual_pilot_20261024/onnx/run_model.py --language en --text "I would like to book a flight to London tomorrow"
python models/turn_v3_trilingual_pilot_20261024/onnx/run_model.py --language zh --text "我想订一张明天从北京到上海的火车票"
```

Python: `TinyOnnxDetector(path).predict(text, context, language='vi'|'en'|'zh')`. Ngôn ngữ cần truyền rõ; không có bộ nhận diện ngôn ngữ tự động. Ngưỡng từng tiếng lấy từ dev, đã khóa trước cả hai lượt kiểm tra test/external. Runner không cần Torch/Transformers.

## Cách đọc kết quả

- MultiWOZ2.2/CrossWOZ có lượt hội thoại, không có nhãn semantic EOT cho mọi prefix. Bản thử lấy cuối lượt làm EOT, cắt phần có lời tiếp làm NEOT proxy. Một prefix có thể đã đủ ý; không gọi các nhãn này là gold chuẩn semantic.
- LiveKit EOT Bench có transcript/timing và khoảng im lặng thực. Bài kiểm tra chỉ dùng văn bản quan sát tại `silence.start+0,2s`, không tải audio, không dùng tương lai, không quét threshold/policy. Đây không phải điểm leaderboard audio chính thức; transcript lag=0 là điều kiện lý tưởng hơn ASR thực.
- Crawl39đoạn Wikipedia, replay12đoạn/2.253prefix cho mỗi model. Dấu phẩy/dấu chấm không tự là EOT. Model bỏ dấu câu nên chỉ thêm dấu ở cuối cho delta0. Đầu vào quá253token hiện giữ phần đầu; runner gắn `input_truncated` nhưng chưa sửa chính sách này trong bản thử.
- Dữ liệu tiếngViệt cũ/holdout đã dùng giữ nguyên, không chấm lại holdout. Official-test EN/ZH và LiveKit chưa được đưa vào train hay dùng chọn ngưỡng.

## Nguồn và artefacts

- [MultiWOZ2.2](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2), revision `fe0c8e65cfcd8462bd33c86e35f21addc84ca82b`. Attribution dữ liệu CC BY4.0 theo Cambridge; code repo MIT.
- [CrossWOZ](https://github.com/thu-coai/CrossWOZ), revision `df82c9fdff91b9b130f2d6b89110d3870ba6260e`; upstream Apache2.0.
- [LiveKit EOT Bench](https://huggingface.co/datasets/livekit/eot-bench-data), revision `ca9d98a9686b920a2d8c9eb984224ba9be74e4dd`; Apache2.0 theo upstream.400turnAnh và344turnTrung, tổng2.057điểm dừng đủ200ms.
- Wikipedia: nguồn/attribution trong `outputs/trilingual_stream_20261024/crawl/sources.json` (CC BY-SA theo điều khoản Wikipedia).

[Báo cáo tổng](../../outputs/trilingual_20261024/review.html) · [Phát lại từng prefix](../../outputs/trilingual_stream_20261024/review.html) · [Hồ sơ thí nghiệm](../../EXPERIMENT_TRILINGUAL_20261024.md).
"""
    (package / "README.md").write_text(md, encoding="utf-8")
    Path("README_TRILINGUAL_V3.md").write_text(md.replace("../../", ""), encoding="utf-8")
    experiment = "# Trilingual pilot — 2026-10-24\n\nMaterial Passport: **TRAINED_AND_EVALUATED_NOT_QUALITY_QUALIFIED**.\n\n" + md
    experiment += "\nTechnical notes: initial CrossWOZ loader expected user instead of usr; invalid incomplete data folder was never trained. Crawl API returned429 on one batch; retained sources, used batched requests after cooldown. First GPU/ONNX full-dev tolerance check failed with max0,000581 due cuDNN TF32; all five decoded weights match exactly, strict TF32-disabled GPU reference versus ONNX max9,414e-6. Dev thresholds frozen before external cases, no retrain after test results. See receipts and source bundles for exact identities.\n"
    Path("EXPERIMENT_TRILINGUAL_20261024.md").write_text(experiment, encoding="utf-8")
    def table(title, values):
        content = f"<h2>{html.escape(title)}</h2><table><tr><th>Model</th><th>Ngôn ngữ</th><th>N</th><th>Accuracy</th><th>F1 EOT</th></tr>"
        for name, langs in values.items():
            for lang, m in langs.items():
                content += f"<tr><td>{html.escape(name)}</td><td>{lang}</td><td>{m['samples']}</td><td>{m['accuracy']*100:.2f}%</td><td class='{('pass' if m['joint_score']>=.9 else 'fail')}'>{m['eot_f1']*100:.2f}%</td></tr>"
        return content + "</table>"
    baseline = read(root / "vi_qat_baseline_dev.json")["dev"]
    page = """<!doctype html><html lang="vi"><meta charset="utf-8"><title>Việt–Anh–Trung: kết quả pilot</title><style>body{font:16px system-ui;background:#f4f7fa;color:#172c3b;margin:0}main{max-width:1100px;margin:30px auto;padding:24px}h1{font-size:28px}h2{font-size:21px;margin-top:28px}.box{background:white;padding:22px;border-radius:12px;margin:18px 0}.alert{background:#fff0e5;border-left:5px solid #d06b2d;padding:18px;line-height:1.6}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border-bottom:1px solid #dce6ef;text-align:left}.fail{color:#a23428;font-weight:700}.pass{color:#087260;font-weight:700}a{color:#155e9e}.note{line-height:1.7;color:#4c6271}li{margin:10px 0}</style><main><h1>Teacher → distill → QAT: bản thử ba ngôn ngữ</h1><div class="alert"><b>Chưa đạt yêu cầu90% cả ba tiếng. Không thay bản Việt đang dùng.</b><br>TiếngViệt riêng đạtdev90,19%F1, nhưng streaming chưa được xác nhận. Student ba tiếng có regression tiếngViệt. Điểm trên khoảng dừng hội thoại thực thấp hơn nhiều so với nhãn tiền tố tự tạo.</div>"""
    page += f"<p class='note'>Train{student['train_samples']:,}mẫu, dev{student['dev_samples']:,}mẫu. Teacher800bước; studentdistill{student['phases'][0]['epochs']}epoch +QAT{student['phases'][1]['epochs']}epoch. Vocabulary{manifest['vocab_size']:,}; ONNX{manifest['onnx_bytes']/1000:.1f}KB; cả gói{manifest['deployment_bytes']/1000:.1f}KB. CPU Windows/Colab khớp xác suất sai lệch tối đa2,22e-16.</p>"
    page += table("Dev pilot — cùng tập, nhãn EOT/cuối lượt và NEOT/tiền tố suy ra", {"Bản Việt QAT cũ": baseline, "Teacher12lớp": teacher["dev"], "StudentINT4": dev["dev"]})
    page += table("Official-test EN/ZH — ngưỡng khóa từ dev, nhãn suy ra", {"Teacher": test["teacher"], "StudentINT4": test["student_INT4"]})
    page += table("LiveKit: văn bản tại khoảng dừng hội thoại thực, fixed200ms", {name: value["span_metrics"] for name, value in real["models"].items()})
    page += "<p class='note'>Không so trực tiếp bảng này với leaderboard audio/policy-sweep. Không dùng LiveKit để train/chọn ngưỡng.400turnAnh +344turnTrung,2.057điểm dừng đủ200ms, transcript lag0. Nhãn dựa trên khoảng im lặng cuối lượt thực; các khoảng trước là hold.</p>"
    page += table("Regression tiếngViệt trên toàn17.833dev cũ", {"ViệtQATcũ": {"vi": read("models/turn_v2_distilled_qat4_20261024/model_manifest.json")["dev_metrics"]}, "Teacherba tiếng": {"vi": test["teacher_vi_full_dev_regression"]}, "Studentba tiếng": {"vi": dev["vi_full_dev"]}})
    page += """<section class="box"><h2>Xem trực tiếp lúc model phát EOT</h2><p><a href="../trilingual_stream_20261024/review.html">▶ Mở trình phát từng prefix, có slider/xác suất/ngưỡng và dấu phẩy/chấm</a></p><p class="note">Crawl39đoạn, replay12đoạn,2.253prefix/model. Văn xuôi không phải nhãn chuẩn hội thoại. Model bỏ dấu câu nên thêm dấu ở cuối không đổi xác suất. Các prefix quá253token bị giữ phần đầu; player đánh dấu input bị cắt.</p></section><section class="box"><h2>Dữ liệu và model</h2><ul><li><a href="../../models/turn_v3_trilingual_pilot_20261024.zip">ZIP student INT4 — thử nghiệm</a> · <a href="../../models/turn_v3_trilingual_pilot_20261024/README.md">Hướng dẫn/cảnh báo</a></li><li><a href="../../data_v3/pilot_trilingual_20261024/manifest.json">Manifest split/hash/nhãn suy ra</a> · <a href="../../data_v3/sources_20261024/sources.json">Nguồn MultiWOZ/CrossWOZ</a></li><li><a href="../../data_v3/livekit_eot_20261024/sources.json">Nguồn LiveKit và protocol</a> · <a href="runs/tri_real_pause_20261024/report.json">Số đo đầy đủ tại khoảng dừng thực</a></li><li><a href="../../EXPERIMENT_TRILINGUAL_20261024.md">Hồ sơ thí nghiệm</a> · <a href="completion_validation.json">Kiểm tra hoàn tất</a></li></ul><p class="note">Bước tiếp theo: nâng teacher bằng dữ liệu pause/hội thoại sát triển khai, kiểm tra nhãn prefix bị mơ hồ; dùng đầy đủ dữ liệu Việt để rehearsal/distill và tăng capacity nếu còn regression. Giữ các tập test đã xem làm regression, cần tập chưa xem mới cho tuyên bố chất lượng độc lập.</p></section></main></html>"""
    (root / "review.html").write_text(page, encoding="utf-8")
    archive = package.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
        for path in sorted(package.rglob("*")):
            if path.is_file():out.write(path, path.relative_to(package.parent))
    write_json(root / "package_receipt.json", {"zip": str(archive), "bytes": archive.stat().st_size, "sha256": file_hash(archive),
        "manifest_sha256": file_hash(package / "model_manifest.json"), "onnx_bytes": manifest["onnx_bytes"], "deployment_bytes": manifest["deployment_bytes"]})
    print(json.dumps({"status": manifest["status"], "onnx_bytes": manifest["onnx_bytes"], "deployment_bytes": manifest["deployment_bytes"], "zip_bytes": archive.stat().st_size}))


if __name__ == "__main__": main()
