"""Verify the completed INT4 experiment and its immutable controls."""
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import onnx

from turn_v2.common import file_hash, write_json
from turn_v2.metrics import classification_metrics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    root = Path("outputs/quant4_20261024")
    package = Path("models/turn_v2_distilled_int4_20261024")
    model = package / "onnx"
    manifest = read(package / "model_manifest.json")
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    for path, sha in read(root / "protocol.json")["controls"].items():
        check("frozen: " + path, file_hash(Path(path)) == sha)
    for folder in [package, Path("models/turn_v2_distilled_min_20261024")]:
        for name, identity in read(folder / "model_manifest.json")["files"].items():
            p = folder / "onnx" / name
            check(str(p) + " identity", p.stat().st_size == identity["bytes"] and file_hash(p) == identity["sha256"])
    frozen = read("data_v2/frozen/manifest.json")
    for split, sha in frozen["split_sha256"].items():
        check("frozen bytes only: " + split, file_hash(Path("data_v2/frozen") / (split + ".jsonl")) == sha)
    candidates = read(root / "selection.json")
    check("four candidates completed", candidates["complete"] and len(candidates["results"]) == 4)
    for result in candidates["results"]:
        for step in [0, 1]:
            receipt = read(root / (result["name"] + "_" + str(step) + "_receipt.json"))
            check(result["name"] + " subprocess " + str(step), receipt["exit_code"] == 0)
        report = read(result["report"])
        check(result["name"] + " full dev only", report["samples"] == 17833 and not report["holdout_evaluated"] and report["holdout_marker_unchanged"])
    passing = [r for r in candidates["results"] if r["quality_passed"]]
    check("smallest passing INT4 selected", min(passing, key=lambda r: r["model_bytes"])["name"] == candidates["selected"]["name"])
    values = np.load(root / "int4_g32_quality/dev_predictions.npz")
    recomputed = classification_metrics(values["labels"], values["prob_EOT"], manifest["dev_metrics"]["threshold"])
    check("full dev metrics recomputed", recomputed == manifest["dev_metrics"])
    check("quality targets", recomputed["accuracy"] >= .9 and recomputed["eot_f1"] >= .9)
    contract = read(model / "turn_detector_config.json")
    check("deployed threshold matches selected", contract["threshold"] == recomputed["threshold"])
    graph = onnx.load(str(model / "model.onnx"))
    onnx.checker.check_model(graph, full_check=True)
    check("opset21", any(p.domain == "" and p.version == 21 for p in graph.opset_import))
    int4 = [p for p in graph.graph.initializer if p.data_type == onnx.TensorProto.INT4]
    check("five matrices are actually packed INT4", len(int4) == 5 and all(len(p.raw_data) == (int(np.prod(p.dims)) + 1) // 2 for p in int4))
    check("packed weight count and bytes", sum(int(np.prod(p.dims)) for p in int4) == 292288 and sum(len(p.raw_data) for p in int4) == 146144)
    check("no external weights", not any(p.external_data for p in graph.graph.initializer))
    check("seven deployment files only", {p.name for p in model.iterdir()} == set(manifest["files"]))
    check("deployment byte count", sum(p.stat().st_size for p in model.iterdir()) == manifest["deployment_bytes"])
    check("model smaller than INT8", manifest["onnx_bytes"] < manifest["baseline_model_bytes"])
    check("package smaller than INT8", manifest["deployment_bytes"] < manifest["baseline_deployment_bytes"])
    for filename in ["native_verification.json", "colab_native_verification.json"]:
        native = read(root / filename)
        check(filename + " full dev native-only", native["passed"] and native["samples"] == 17833
              and not native["torch_imported"] and not native["transformers_imported"] and not native["holdout_evaluated"])
    check("Windows Colab parity", read(root / "windows_colab_parity.json")["passed"])
    check("notebook actually executed on Colab", read(root / "notebook_verification.json")["passed"])
    notebook = read("a.ipynb")
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    final = [c for c in code_cells if "int4_result" in c.get("metadata", {}).get("tags", [])]
    check("INT4 cell outputs saved", len(final) == 1 and final[0]["outputs"] and not any(o["output_type"] == "error" for o in final[0]["outputs"]))
    from IPython.core.inputtransformer2 import TransformerManager
    transformer = TransformerManager()
    for cell in code_cells:
        compile(transformer.transform_cell("".join(cell["source"])), "notebook_cell", "exec")
    check("all notebook cells compile", True)
    bootstrap = next(c for c in code_cells if "bootstrap" in c.get("metadata", {}).get("tags", []))
    line = next(t for t in "".join(bootstrap["source"]).splitlines() if t.startswith("package = base64.b64decode("))
    import ast
    embedded = base64.b64decode(ast.literal_eval(line[len("package = base64.b64decode("):-1]))
    with zipfile.ZipFile(io.BytesIO(embedded)) as archive:
        check("notebook embeds current INT4 exporter", archive.read("turn_v2/quantize_student4.py") == Path("turn_v2/quantize_student4.py").read_bytes())
    check("35 unit tests passed", read(root / "unit_tests.json")["exit_code"] == 0 and "35 passed" in (root / "unit_tests.log").read_text(encoding="utf-8"))
    archive_receipt = read(root / "package_receipt.json")
    archive_path = Path(archive_receipt["zip"])
    check("ZIP identity", file_hash(archive_path) == archive_receipt["sha256"] and archive_path.stat().st_size == archive_receipt["bytes"])
    with zipfile.ZipFile(archive_path) as archive:
        check("ZIP contains complete package only", len(archive.namelist()) == len(manifest["files"]) + 2 and all("__pycache__" not in n for n in archive.namelist()))
        for name, identity in manifest["files"].items():
            data = archive.read(package.name + "/onnx/" + name)
            check("ZIP " + name, len(data) == identity["bytes"] and hashlib.sha256(data).hexdigest() == identity["sha256"])
    for path in [Path("tools/verify_student4.py"), Path("tools/connect_student4.py"), Path("tools/update_student4_notebook.py"),
                 Path("tools/validate_student4.py"), Path("pytest.ini")]:
        shutil.copy2(path, root / "source" / path.name)
    write_json(root / "source/identities.json", {p.name: file_hash(p) for p in (root / "source").iterdir() if p.is_file() and p.name != "identities.json"})
    result = {"status": "PASS" if all(c["passed"] for c in checks) else "FAIL", "checks_passed": sum(c["passed"] for c in checks),
        "checks_total": len(checks), "checks": checks, "model_sha256": manifest["files"]["model.onnx"]["sha256"],
        "onnx_bytes": manifest["onnx_bytes"], "deployment_bytes": manifest["deployment_bytes"],
        "zip_bytes": archive_receipt["bytes"], "dev_metrics": recomputed, "holdout_evaluated": False,
        "scope": "Full dev selection; no independent unseen quality claim"}
    write_json(root / "completion_validation.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}))
    assert result["status"] == "PASS"
    experiment = Path("EXPERIMENT_INT4_20261024.md")
    text = experiment.read_text(encoding="utf-8").replace("EVALUATED_ON_FULL_DEV; PACKAGING_VERIFICATION_PENDING", "VERIFIED: FULL DEV, WINDOWS, COLAB, ZIP")
    if "Hoàn tất kiểm tra" not in text:
        text += f"\nHoàn tất kiểm tra **{len(checks)}/{len(checks)}**: model/ZIP hashes, hai nền tảng mỗi nền tảng đủ17.833 dev (Windows delta0, Colab2,22e-16), runner không Torch/HF, bốn cấu hình, marker/frozen bytes còn nguyên, notebook đã thực thi Colab, **35 unit tests pass**. Lần pytest đầu quét cả bản chụp mã nguồn trong outputs gây trùng tên module; `pytest.ini` giới hạn discovery vào `tests/` rồi chạy lại thành công. ZIP{archive_receipt['bytes']:,}byte, SHA256 `{archive_receipt['sha256']}`.\n"
        experiment.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
