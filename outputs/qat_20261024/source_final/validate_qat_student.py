"""Final evidence checks for QAT training, exact packing and deployment."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import onnx
from IPython.core.inputtransformer2 import TransformerManager

from turn_v2.common import file_hash, write_json
from turn_v2.metrics import classification_metrics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    root = Path("outputs/qat_20261024")
    package = Path("models/turn_v2_distilled_qat4_20261024")
    model = package / "onnx"
    manifest = read(package / "model_manifest.json")
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    for p, sha in read(root / "protocol.json")["controls"].items():
        check("frozen " + p, file_hash(Path(p)) == sha)
    for folder in [package, Path("models/turn_v2_distilled_int4_20261024"), Path("models/turn_v2_distilled_min_20261024")]:
        for name, expected in read(folder / "model_manifest.json")["files"].items():
            path = folder / "onnx" / name
            check(str(path) + " identity", file_hash(path) == expected["sha256"] and path.stat().st_size == expected["bytes"])
    for split, sha in read("data_v2/frozen/manifest.json")["split_sha256"].items():
        check("frozen bytes only " + split, file_hash(Path("data_v2/frozen") / (split + ".jsonl")) == sha)
    selection = read(root / "selection.json")
    check("two QAT trials complete", selection["complete"] and len(selection["results"]) == 2)
    passing = [r for r in selection["results"] if r["quality_passed"]]
    check("smallest passing QAT selected", min(passing, key=lambda r: r["model_bytes"])["name"] == selection["selected"]["name"])
    for result in selection["results"]:
        name = result["name"]
        student = root / "runs" / name
        status = read(root / "jobs" / name / "status.json")
        check(name + " completed true exit", status["status"] == "COMPLETED" and status["exit_code"] == 0)
        training = read(student / "training_report.json")
        identity = read(student / "run_identity.json")
        exported = read(student / "qat_export_report.json")
        report = read(student / "quality_int4/report.json")
        check(name + " full train/dev", training["train_samples"] == 315174 and training["dev_samples"] == 17833 and report["samples"] == 17833)
        check(name + " trained checkpoint", training["best_epoch"] >= 1 and training["optimizer_steps"] > 0)
        check(name + " frozen QAT scales", identity["quant_params_sha256"] == training["quant_params_sha256"] == exported["quant_params_sha256"] == file_hash(student / "quant_params.pt"))
        check(name + " latent checkpoint hash", training["student_sha256"] == exported["latent_student_sha256"] == file_hash(student / "student.pt"))
        check(name + " five matrix gradients", len(read(student / "gradient_checks.json")) == 5 and all(read(student / "gradient_checks.json").values()))
        check(name + " exact QAT weight decode", exported["baked_weight_decode_exact"] and exported["frozen_scales_used_without_refit"] and len(exported["tensors"]) == 5)
        check(name + " technical parity", exported["technical_parity_passed"] and exported["right_padding_probability_delta"] < 1e-5)
        check(name + " training packed inference parity", result["gpu_fake_quant_vs_cpu_packed_max_probability_delta"] < 1e-4)
        check(name + " full packed targets", result["quality_passed"] and report["dev_selected_threshold"] == result["dev_metrics"])
        check(name + " holdout untouched", not training["holdout_evaluated"] and training["holdout_marker_unchanged"] and not report["holdout_evaluated"] and report["holdout_marker_unchanged"])
        for step in [0, 1]:
            check(name + " export/evaluate subprocess " + str(step), read(student / ("evaluation_" + str(step) + "_receipt.json"))["exit_code"] == 0)
        for filename, sha in identity["source_sha256"].items():
            check(name + " source " + filename, file_hash(Path("turn_v2") / filename) == sha)
    reference_path = Path(manifest["selected_report"]).parent / "dev_predictions.npz"
    values = np.load(reference_path)
    contract = read(model / "turn_detector_config.json")
    metrics = classification_metrics(values["labels"], values["prob_EOT"], contract["threshold"])
    check("selected packed metrics recomputed", metrics == manifest["dev_metrics"])
    check("QAT method explicitly recorded", contract["training_method"] == manifest["training_method"] == "INT4_QAT_WITH_DISTILLATION")
    check("both targets at deployed threshold", metrics["accuracy"] >= .9 and metrics["eot_f1"] >= .9)
    graph = onnx.load(str(model / "model.onnx"))
    onnx.checker.check_model(graph, full_check=True)
    packed = [p for p in graph.graph.initializer if p.data_type == onnx.TensorProto.INT4]
    check("actual signed INT4 two codes per byte", len(packed) == 5 and all(len(p.raw_data) == (int(np.prod(p.dims)) + 1) // 2 for p in packed))
    check("292288 packed matrix weights", sum(int(np.prod(p.dims)) for p in packed) == 292288)
    check("no external weight data", not any(p.external_data for p in graph.graph.initializer))
    check("seven deployment files only", {p.name for p in model.iterdir()} == set(manifest["files"]))
    check("deployment exact byte count", sum(p.stat().st_size for p in model.iterdir()) == manifest["deployment_bytes"])
    check("smaller than both previous packages", all(manifest["deployment_bytes"] < c["deployment_bytes"] for c in manifest["controls"].values()))
    for filename in ["native_verification.json", "colab_native_verification.json"]:
        native = read(root / filename)
        check(filename + " full dev native runner", native["passed"] and native["samples"] == 17833 and not native["torch_imported"] and not native["transformers_imported"])
        check(filename + " matching metrics", all(native["dev_metrics"][k] == metrics[k] for k in ["accuracy", "eot_f1", "false_eot", "missed_eot"]))
    check("Windows Colab parity", read(root / "windows_colab_parity.json")["passed"])
    check("QAT notebook executed on Colab", read(root / "notebook_verification.json")["passed"])
    notebook = read("a.ipynb")
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    last = next(c for c in code_cells if "qat_result" in c.get("metadata", {}).get("tags", []))
    check("QAT notebook outputs saved", last["outputs"] and not any(o["output_type"] == "error" for o in last["outputs"]))
    transformer = TransformerManager()
    for cell in code_cells:
        compile(transformer.transform_cell("".join(cell["source"])), "a.ipynb", "exec")
    check("notebook cells compile with IPython magics", True)
    bootstrap = next(c for c in code_cells if "bootstrap" in c.get("metadata", {}).get("tags", []))
    line = next(t for t in "".join(bootstrap["source"]).splitlines() if t.startswith("package = base64.b64decode("))
    embedded = base64.b64decode(ast.literal_eval(line[len("package = base64.b64decode("):-1]))
    with zipfile.ZipFile(io.BytesIO(embedded)) as archive:
        for name in ["qat_student.py", "export_qat_student.py"]:
            check("embedded " + name, archive.read("turn_v2/" + name) == (Path("turn_v2") / name).read_bytes())
    tests = read(root / "unit_tests.json")
    check("39 unit tests including CUDA pass", tests["exit_code"] == 0 and "39 passed" in (root / "unit_tests.log").read_text(encoding="utf-8"))
    receipt = read(root / "package_receipt.json")
    zip_path = Path(receipt["zip"])
    check("delivery ZIP identity", zip_path.stat().st_size == receipt["bytes"] and file_hash(zip_path) == receipt["sha256"])
    with zipfile.ZipFile(zip_path) as archive:
        check("ZIP package files only", len(archive.namelist()) == len(manifest["files"]) + 2 and not any("__pycache__" in n for n in archive.namelist()))
        for name, identity in manifest["files"].items():
            content = archive.read(package.name + "/onnx/" + name)
            check("ZIP " + name, len(content) == identity["bytes"] and hashlib.sha256(content).hexdigest() == identity["sha256"])
    final_source = root / "source_final"
    final_source.mkdir(exist_ok=True)
    sources = [*Path("turn_v2").glob("*qat*.py"), *Path("tools").glob("*qat*.py"), Path("tools/verify_tiny_package.py"), Path("tests/test_qat_student.py")]
    for source in sources:
        target = final_source / source.name
        if target.exists():
            assert target.read_bytes() == source.read_bytes()
        else:
            shutil.copy2(source, target)
    write_json(final_source / "identities.json", {p.name: file_hash(p) for p in final_source.iterdir() if p.is_file() and p.name != "identities.json"})
    outcome = {"status": "PASS" if all(c["passed"] for c in checks) else "FAIL", "checks_passed": sum(c["passed"] for c in checks),
        "checks_total": len(checks), "checks": checks, "model_sha256": manifest["files"]["model.onnx"]["sha256"],
        "onnx_bytes": manifest["onnx_bytes"], "deployment_bytes": manifest["deployment_bytes"], "zip_bytes": receipt["bytes"],
        "dev_metrics": metrics, "holdout_evaluated": False, "scope": manifest["quality_scope"]}
    write_json(root / "completion_validation.json", outcome)
    print(json.dumps({k: v for k, v in outcome.items() if k != "checks"}))
    assert outcome["status"] == "PASS", str([c for c in checks if not c["passed"]])
    experiment = Path("EXPERIMENT_QAT_20261024.md")
    text = experiment.read_text(encoding="utf-8").replace("TRAINED_AND_PACKED_FULL_DEV_EVALUATED; CROSS_PLATFORM_VERIFICATION_PENDING", "VERIFIED: TRAINED, PACKED FULL DEV, WINDOWS, COLAB, ZIP")
    if "Hoàn tất kiểm tra" not in text:
        text += f"\nHoàn tất kiểm tra **{len(checks)}/{len(checks)}**: hai job QAT completed/exit0, fakequant có gradient qua năm ma trận, scale giữ nguyên từ lúc khởi tạo, packed decode đúng trọng số QAT, full dev hai nền tảng, identities frozen/bản cũ không đổi, notebook đã chạy Colab, **39 tests pass** gồm CUDA partial-group forward/backward. ZIP{receipt['bytes']:,}byte, SHA256 `{receipt['sha256']}`.\n"
        experiment.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
