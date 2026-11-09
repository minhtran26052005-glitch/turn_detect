"""Read the current Colab trial without interrupting its training process."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from colab_runtime import ColabRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, default=Path("outputs/training_trial_20261024"))
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    launch = json.loads((args.trial / "launch.json").read_text(encoding="utf-8"))
    receipt = json.loads("".join(output.get("text", "") for output in launch["outputs"]))
    run_name = receipt["run_name"]
    code = """import json,subprocess,re
from pathlib import Path
root=Path('/content/turn_detect_v2')
name=RUN_NAME
job=root/'jobs'/name
out=root/'runs'/name
result={'run_name':name}
for key,path in [('status',job/'status.json'),('progress',out/'progress.json'),('report',out/'training_report.json')]:
    if path.exists():
        result[key]=json.loads(path.read_text())
log=job/'train.log'
if log.exists():
    result['log_tail']=re.sub(r'\\x1b\\[[0-9;]*[A-Za-z]','',log.read_text(errors='replace')[-2800:])
result['gpu']=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
result['checkpoints']=[p.name for p in sorted((out/'checkpoints').glob('checkpoint-*'))]
print(json.dumps(result))
""".replace("RUN_NAME", repr(run_name))
    runtime = ColabRuntime()
    try:
        while True:
            try:
                response = runtime.execute(code, timeout=60, echo=False)
                if response["status"] != "ok":
                    raise RuntimeError(response)
                current = json.loads("".join(item.get("text", "") for item in response["outputs"]))
                current["observed_at_utc"] = datetime.now(timezone.utc).isoformat()
                (args.trial / "latest.json").write_text(json.dumps(current, indent=2), encoding="utf-8")
                with (args.trial / "observations.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(current) + "\n")
                state = current.get("status", {}).get("status")
                print(json.dumps({"run_name": run_name, "status": state,
                    "elapsed_seconds": current.get("status", {}).get("elapsed_seconds"),
                    "progress": current.get("progress"), "gpu": current["gpu"],
                    "checkpoints": current["checkpoints"],
                    "log_tail": current.get("log_tail", "")[-1000:],
                    "report": current.get("report")}), flush=True)
                if not args.watch or state not in ("STARTING", "RUNNING"):
                    break
            except Exception as error:
                if not args.watch:
                    raise
                print(json.dumps({"monitor_error": type(error).__name__, "training_interrupted": False}), flush=True)
                runtime.close()
            time.sleep(30)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
