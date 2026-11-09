"""Bounded, dev-only INT4 storage experiment. Original INT8 is immutable."""
from pathlib import Path
import json
import subprocess
import sys
import time

from turn_v2.common import file_hash, write_json


def main():
    root = Path("outputs/quant4_20261024")
    root.mkdir(parents=True, exist_ok=False)
    source = Path("outputs/distillation_20261024/runs/distill_gru_h96_e64_l1_20261024/onnx_fp32")
    baseline = Path("models/turn_v2_distilled_min_20261024/onnx")
    reference = Path("outputs/distillation_20261024/runs/distill_gru_h96_e64_l1_20261024/quality_int8")
    controls = {str(p): file_hash(p) for p in [source / "model.onnx", baseline / "model.onnx",
                Path("data_v2/frozen/manifest.json"), Path("data_v2/frozen/dev.jsonl"),
                Path("data_v2/frozen/holdout_consumed.json")]}
    write_json(root / "protocol.json", {"scope": "Full frozen dev selection only; consumed holdout never scored",
        "target": {"accuracy": .9, "eot_f1": .9, "smaller_than_int8": True},
        "controls": controls, "baseline_model_bytes": (baseline / "model.onnx").stat().st_size,
        "quantization": "All five large weight tensors INT4; FP16 group scales; biases/norm/compute FP32",
        "initial_candidates": [{"group_size": g, "scale_method": "mse"} for g in [64, 32, 16, 8]],
        "selection": "Smallest ONNX among candidates meeting both full dev targets; threshold grid same as INT8",
        "holdout_evaluated": False})
    results = []
    for group in [64, 32, 16, 8]:
        name = "int4_g" + str(group)
        model = root / name
        commands = [
            [sys.executable, "-m", "turn_v2.quantize_student4", "--source", str(source), "--output", str(model), "--group-size", str(group)],
            [sys.executable, "-m", "turn_v2.evaluate_compressed", "--model", str(model), "--output", str(root / (name + "_quality")),
             "--teacher", str(baseline), "--reference-run", str(reference), "--threads", "2", "--select-threshold"],
        ]
        for index, command in enumerate(commands):
            started = time.monotonic()
            with (root / (name + "_" + str(index) + ".log")).open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            write_json(root / (name + "_" + str(index) + "_receipt.json"), {"command": command, "exit_code": completed.returncode,
                "elapsed_seconds": time.monotonic() - started})
            if completed.returncode:
                raise RuntimeError(f"{name} step {index} failed; inspect log")
        report = json.loads((root / (name + "_quality") / "report.json").read_text(encoding="utf-8"))
        result = {"name": name, "model_bytes": report["model_bytes"], "metrics": report["dev_selected_threshold"],
                  "quality_passed": report["dev_quality_target_met"], "report": str(root / (name + "_quality") / "report.json")}
        results.append(result)
        write_json(root / "selection.json", {"results": results, "complete": len(results) == 4})
        print(json.dumps(result), flush=True)
    if not all(file_hash(Path(p)) == sha for p, sha in controls.items()):
        raise ValueError("Frozen control changed")
    passing = [r for r in results if r["quality_passed"] and r["model_bytes"] < (baseline / "model.onnx").stat().st_size]
    write_json(root / "selection.json", {"results": results, "complete": True,
        "selected": min(passing, key=lambda r: r["model_bytes"]) if passing else None,
        "controls_unchanged": True, "holdout_evaluated": False})


if __name__ == "__main__":
    main()
