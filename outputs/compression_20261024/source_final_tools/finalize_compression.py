"""Promote only a candidate meeting the file budget and both full-dev targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turn_v2.common import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/turn_v2_compact_20261024/onnx"))
    parser.add_argument("--protocol", type=Path, default=Path("outputs/compression_20261024/protocol.json"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    data = json.loads(Path("data_v2/frozen/manifest.json").read_text(encoding="utf-8"))
    if report["samples"] != data["splits"]["dev"]["rows"] or not report["dev_quality_target_met"]:
        raise ValueError("Full dev must reach both 90% targets before promotion")
    if not report["size_target_deployment_met"] or not report["holdout_marker_unchanged"]:
        raise ValueError("Size budget or holdout preservation check failed")
    source = Path(report["model"])
    if file_hash(source/"model.onnx") != report["model_sha256"]:
        raise ValueError("Evaluated model changed")
    if file_hash(Path(protocol["reference"])) != protocol["reference_sha256"]:
        raise ValueError("Teacher changed")
    if file_hash(Path("data_v2/frozen/holdout_consumed.json")) != protocol["holdout_marker_sha256"]:
        raise ValueError("Holdout receipt changed")
    args.output.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, args.output/path.name)
    contract_path = args.output/"turn_detector_config.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.update({"threshold": report["dev_selected_threshold"]["threshold"],
        "intra_op_threads": report["cpu"]["threads"],
        "quality_status": "SIZE_AND_FULL_DEV_TARGET_MET",
        "quality_scope": "Frozen dev selection only; consumed holdout not reopened",
        "compression_dev_report": str(args.report.as_posix())})
    write_json(contract_path, contract)
    files = {path.name: {"bytes": path.stat().st_size, "sha256": file_hash(path)}
             for path in sorted(args.output.iterdir()) if path.is_file()}
    total = sum(item["bytes"] for item in files.values())
    if total > protocol["size_budget_bytes"]:
        raise ValueError("Final deployment exceeds the size budget")
    manifest = {"status": "SIZE_AND_FULL_DEV_TARGET_MET", "model_dir": str(args.output.resolve()),
        "files": files, "deployment_bytes": total, "reference_bytes": protocol["reference_bytes"],
        "model_compression_factor": protocol["reference_bytes"]/files["model.onnx"]["bytes"],
        "deployment_compression_factor": protocol["reference_bytes"]/total,
        "dev_metrics": report["dev_selected_threshold"], "holdout_evaluated": False,
        "quality_scope": contract["quality_scope"]}
    write_json(args.output.parent/"model_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
