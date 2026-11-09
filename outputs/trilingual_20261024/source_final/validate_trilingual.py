"""Verify data isolation, measured failures, exact artifacts and completed jobs."""
import ast
import base64
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import numpy as np
import onnx
from IPython.core.inputtransformer2 import TransformerManager

from turn_v2.common import file_hash, input_id, read_jsonl, write_json
from turn_v2.metrics import classification_metrics


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    root = Path("outputs/trilingual_20261024")
    model = Path("models/turn_v3_trilingual_pilot_20261024/onnx")
    manifest = read(model.parent / "model_manifest.json")
    checks = []
    def check(name, passed):checks.append({"name": name, "passed": bool(passed)})
    for path, sha in read(root / "protocol.json")["controls"].items():check("control " + path, file_hash(Path(path)) == sha)
    for package in [model.parent, Path("models/turn_v2_distilled_qat4_20261024"), Path("models/turn_v2_compact_20261024")]:
        for name, identity in read(package / "model_manifest.json")["files"].items():
            path = package / "onnx" / name
            check(str(path), path.stat().st_size == identity["bytes"] and file_hash(path) == identity["sha256"])
    data = Path("data_v3/pilot_trilingual_20261024")
    data_manifest = read(data / "manifest.json")
    rows = {split: list(read_jsonl(data / (split + ".jsonl"))) for split in ["train", "dev", "test"]}
    for split, sha in data_manifest["split_sha256"].items():check("pilot split " + split, file_hash(data / (split + ".jsonl")) == sha)
    check("complete trilingual train/dev", all({r["language"] for r in rows[split]} == {"vi", "en", "zh"} for split in ["train", "dev"]))
    groups = {split: {r["source_id"] for r in value if r["language"] != "vi"} for split, value in rows.items()}
    check("dialogue split isolation", not (groups["train"] & groups["dev"] or groups["train"] & groups["test"] or groups["dev"] & groups["test"]))
    inputs = {split: {r["language"] + input_id(r) for r in value} for split, value in rows.items()}
    check("normalized input split isolation", not (inputs["train"] & inputs["dev"] or inputs["train"] & inputs["test"] or inputs["dev"] & inputs["test"]))
    check("prefix proxies explicitly tagged", all(r["label_source"] == "observed_continuation_prefix_proxy" and "label_limit" in r for value in rows.values() for r in value if r["language"] != "vi" and r["label"] == "NEOT"))
    for split, sha in read("data_v2/frozen/manifest.json")["split_sha256"].items():check("legacy bytes " + split, file_hash(Path("data_v2/frozen") / (split + ".jsonl")) == sha)
    for phase in ["teacher", "student", "final", "real_pause"]:
        status = read(root / "jobs" / ("tri_" + phase + "_20261024") / "status.json")
        check(phase + " completed exit0", status["status"] == "COMPLETED" and status["exit_code"] == 0)
        upload = read(root / ("tri_" + phase + "_20261024_launch.json"))
        metadata = json.loads("".join(x["text"] for x in upload["outputs"] if x["output_type"] == "stream"))
        check(phase + " source/data bundle", file_hash(root / ("tri_" + phase + "_20261024_upload.zip")) == metadata["bundle_sha256"])
    teacher = read(root / "runs/tri_teacher_20261024/training_report.json")
    backup = read(root / "teacher_checkpoint_backup.json")
    check("exact full teacher checkpoint backup", file_hash(Path(backup["path"])) == backup["sha256"] == teacher["checkpoint_sha256"])
    check("teacher 800 actual steps", teacher["steps"] == 800 and teacher["status"] == "TEACHER_AND_TRAIN_CACHE_COMPLETE")
    initialization = read(root / "runs/tri_teacher_20261024/initialization.json")
    check("teacher restored VI weights parity", initialization["mapped_input_probability_delta"] < 1e-5 and initialization["restored_control_rows"] == 23078)
    student = root / "runs/tri_student_20261024"
    check("student vocabulary train-only segmentation", read(student / "vocabulary/train_segmentation_audit.json")["changed"] == 0 and not read(student / "vocabulary/report.json")["dev_or_test_vocab_fit"])
    check("QAT five matrix gradients", len(read(student / "qat_best/gradient_checks.json")) == 5 and all(read(student / "qat_best/gradient_checks.json").values()))
    export = read(student / "qat_best/qat_export_report.json")
    check("QAT frozen scales exact decode", export["technical_parity_passed"] and export["frozen_scales_used_without_refit"] and export["baked_weight_decode_exact"])
    graph = onnx.load(str(model / "model.onnx")); onnx.checker.check_model(graph, full_check=True)
    packed = [t for t in graph.graph.initializer if t.data_type == onnx.TensorProto.INT4]
    check("actual packed INT4 matrices", len(packed) == 5 and all(len(t.raw_data) == (int(np.prod(t.dims)) + 1) // 2 for t in packed))
    check("all seven deployment files, no cache", {p.name for p in model.iterdir()} == set(manifest["files"]))
    check("deployment bytes", sum(p.stat().st_size for p in model.iterdir()) == manifest["deployment_bytes"])
    dev = read(root / "student_dev_report.json")
    check("strict GPU CPU reference parity", dev["strict_gpu_reference_vs_packed_dev_max_delta"] < 1e-4)
    check("portable native imports only", not dev["torch_imported"] and not dev["transformers_imported"])
    values = np.load(root / "student_dev_predictions.npz")
    for lang in ["vi", "en", "zh"]:
        selected = values["language"] == lang
        computed = classification_metrics(values["labels"][selected], values["prob_EOT"][selected], dev["deployed_thresholds"][lang])
        check("dev metric recomputation " + lang, computed == manifest["dev"][lang])
    check("negative result labelled as pilot", not manifest["all_language_target_met"] and not manifest["promoted_over_vi_control"] and "NOT_READY" in manifest["status"])
    final = read(root / "runs/tri_final_20261024/report.json")
    check("final test frozen contract", file_hash(model / "turn_detector_config.json") == final["frozen_before_test"]["student_contract_sha256"])
    check("no old holdout/test retune", final["thresholds_unchanged"] and not final["old_vi_holdout_evaluated"] and not final["test_used_for_selection"])
    for name, key in [("student", "student_INT4"), ("teacher", "teacher")]:
        values = np.load(root / "runs/tri_final_20261024" / (name + "_test_predictions.npz"))
        for lang in ["en", "zh"]:
            selected = values["language"] == lang; expected = final[key][lang]
            check(name + " final metric " + lang, classification_metrics(values["labels"][selected], values["prob_EOT"][selected], expected["threshold"]) == expected)
    real = read(root / "runs/tri_real_pause_20261024/report.json")
    real_root = Path("data_v3/livekit_eot_20261024")
    sources = read(real_root / "sources.json")
    turns = {r["id"]: r for r in (json.loads(line) for line in (real_root / "turns.jsonl").read_text(encoding="utf-8").splitlines())}
    cases = list(read_jsonl(real_root / "cases.jsonl"))
    check("real-pause metadata 744 turns 2057 spans", len(turns) == 744 and len(cases) == 2057 and not sources["audio_downloaded"])
    check("real-pause inputs causal", all(r["text"] == ("" if r["language"] == "zh" else " ").join(w["word"] for w in turns[r["source_id"]]["words"] if w["end"] <= r["timestamp_seconds"] + 1e-8) for r in cases))
    check("real-pause labels true final-span", all((r["label"] == "EOT") == (r["span_index"] == len(turns[r["source_id"]]["silence_spans"]) - 1) for r in cases))
    check("real benchmark evaluation only", real["evaluation_only"] and not real["thresholds_or_weights_retuned"])
    for name, result in real["models"].items():
        values = np.load(root / "runs/tri_real_pause_20261024" / (name + "_predictions.npz"))
        for lang in ["en", "zh"]:
            expected = result["span_metrics"][lang]; selected = values["language"] == lang
            check(name + " real-pause metric " + lang, classification_metrics(values["labels"][selected], values["prob_EOT"][selected], expected["threshold"]) == expected)
    check("Windows Colab full dev parity", read(root / "windows_colab_parity.json")["passed"])
    for name in ["vi_qat_baseline", "trilingual_student"]:
        report = read(Path("outputs/trilingual_stream_20261024") / name / "report.json")
        check(name + " real prefix replay", report["prefix_events"] == 2253 and report["paragraphs"] == 12 and report["punctuation_only_max_probability_delta"] == 0.)
    notebook = read("a.ipynb"); before = read(root / "a_before_trilingual.ipynb")
    old_crawl = next(c for c in before["cells"] if "crawl_predict" in c.get("metadata", {}).get("tags", []))
    new_crawl = next(c for c in notebook["cells"] if "crawl_predict" in c.get("metadata", {}).get("tags", []))
    check("Vietnamese default retained", old_crawl["source"] == new_crawl["source"])
    check("notebook Colab run passed", read(root / "notebook_verification.json")["passed"])
    transform = TransformerManager()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":compile(transform.transform_cell("".join(cell["source"])), "a.ipynb", "exec")
    check("all notebook cells compile", True)
    tests = read(root / "unit_tests.json")
    check("42 unit tests passed", tests["exit_code"] == 0 and "42 passed" in (root / "unit_tests.log").read_text(encoding="utf-8"))
    receipt = read(root / "package_receipt.json")
    check("delivery ZIP hash", file_hash(Path(receipt["zip"])) == receipt["sha256"])
    with zipfile.ZipFile(receipt["zip"]) as archive:
        check("delivery ZIP no bytecode", len(archive.namelist()) == 9 and not any("__pycache__" in n for n in archive.namelist()))
        for name, identity in manifest["files"].items():check("ZIP " + name, hashlib.sha256(archive.read(model.parent.name + "/onnx/" + name)).hexdigest() == identity["sha256"])
    for path in [root / "review.html", Path("outputs/trilingual_stream_20261024/review.html")]:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r'href="([^"#]+)"', text):
            if not target.startswith(("http:", "https:")):check(str(path) + " link " + target, (path.parent / target).resolve().exists() or target == "completion_validation.json")
    static = {"ui_automation": "BLOCKED", "reason": "Computer Use could not determine current browser URL on Windows; UI automation stopped",
        "visual_or_interactive_browser_verification_claimed": False, "html_files_present": True}
    node = shutil.which("node")
    if node:
        page = Path("outputs/trilingual_stream_20261024/review.html").read_text(encoding="utf-8")
        script = re.search(r"</script><script>([\s\S]+)</script>", page).group(1)
        js = root / "stream_player_syntax.js"; js.write_text(script, encoding="utf-8")
        process = subprocess.run([node, "--check", str(js)], capture_output=True, text=True, timeout=15)
        static["javascript_syntax_check"] = process.returncode == 0
        check("player JS static syntax", process.returncode == 0)
    write_json(root / "html_static_verification.json", static)
    source = root / "source_final"; source.mkdir(exist_ok=True)
    for path in [*Path("turn_v2").glob("trilingual*.py"), *Path("tools").glob("*trilingual*.py"), Path("tools/stream_turn_review.py"), Path("tools/prepare_livekit_eot.py"), Path("tests/test_trilingual_data.py")]:
        target = source / path.name
        if target.exists():assert target.read_bytes() == path.read_bytes()
        else:shutil.copy2(path, target)
    write_json(source / "identities.json", {p.name: file_hash(p) for p in source.iterdir() if p.is_file() and p.name != "identities.json"})
    result = {"status": "TECHNICAL_PASS_QUALITY_TARGET_NOT_MET" if all(c["passed"] for c in checks) else "FAIL", "checks_passed": sum(c["passed"] for c in checks),
        "checks_total": len(checks), "checks": checks, "quality_target_met": False, "pilot_not_promoted": True,
        "old_holdout_evaluated": False, "model_sha256": manifest["files"]["model.onnx"]["sha256"], "zip_sha256": receipt["sha256"],
        "teacher_backup_sha256": backup["sha256"], "ui_verification": static}
    write_json(root / "completion_validation.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}))
    assert all(c["passed"] for c in checks), str([c for c in checks if not c["passed"]])


if __name__ == "__main__": main()
