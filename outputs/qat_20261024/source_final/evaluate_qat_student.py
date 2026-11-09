"""Evaluate actual packed QAT exports and compare against frozen controls."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from turn_v2.common import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-size", type=int, choices=[32, 64], required=True)
    args = parser.parse_args()
    root = Path("outputs/qat_20261024")
    name = f"qat_gru96_g{args.group_size}_20261024" + ("_r2" if args.group_size == 64 else "")
    student = root / "runs" / name
    output = student / "onnx_int4"
    quality = student / "quality_int4"
    commands = [[sys.executable, "-m", "turn_v2.export_qat_student", "--student", str(student), "--output", str(output)],
        [sys.executable, "-m", "turn_v2.evaluate_compressed", "--model", str(output), "--output", str(quality),
         "--teacher", "models/turn_v2_distilled_min_20261024/onnx", "--reference-run",
         "outputs/distillation_20261024/runs/distill_gru_h96_e64_l1_20261024/quality_int8", "--threads", "2", "--select-threshold"]]
    for step, command in enumerate(commands):
        started = time.monotonic()
        with (student / ("evaluation_" + str(step) + ".log")).open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        write_json(student / ("evaluation_" + str(step) + "_receipt.json"), {"command": command, "exit_code": completed.returncode,
            "elapsed_seconds": time.monotonic() - started})
        assert completed.returncode == 0, "Inspect QAT evaluation log: " + str(step)
    report = json.loads((quality / "report.json").read_text(encoding="utf-8"))
    fake = np.load(student / "dev_predictions.npz")
    packed = np.load(quality / "dev_predictions.npz")
    assert np.array_equal(fake["labels"], packed["labels"])
    delta = float(np.max(np.abs(fake["prob_EOT"] - packed["prob_EOT"])))
    result = {"name": name, "group_size": args.group_size, "model_bytes": report["model_bytes"],
        "dev_metrics": report["dev_selected_threshold"], "quality_passed": report["dev_quality_target_met"],
        "gpu_fake_quant_vs_cpu_packed_max_probability_delta": delta,
        "report": str(quality / "report.json").replace("\\", "/"), "holdout_evaluated": False}
    assert delta < 1e-4, "Training fake-quant/deployment drift"
    write_json(root / ("result_g" + str(args.group_size) + ".json"), result)
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("result_g*.json"))]
    passed = [r for r in results if r["quality_passed"]]
    selection = {"complete": len(results) == 2, "results": results,
        "selected": min(passed, key=lambda r: r["model_bytes"]) if passed else None, "holdout_evaluated": False}
    write_json(root / "selection.json", selection)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
