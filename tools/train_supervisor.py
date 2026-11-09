"""Run one authorized Colab training job with durable status and a hard timeout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(root, run_name, command, timeout):
    job_dir = root / "jobs" / run_name
    job_dir.mkdir(parents=True, exist_ok=False)
    output = root / "runs" / run_name
    if output.exists():
        raise FileExistsError(output)
    started = time.time()
    status = {"status": "STARTING", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root), "output": str(output), "command": command,
        "hard_timeout_seconds": timeout, "supervisor_pid": os.getpid()}
    atomic_json(job_dir / "status.json", status)
    with (job_dir / "train.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"})
        status.update({"status": "RUNNING", "training_pid": child.pid})
        while child.poll() is None:
            status.update({"elapsed_seconds": time.time() - started, "heartbeat_unix": time.time()})
            progress_path = output / "progress.json"
            if progress_path.exists():
                try:
                    status["progress"] = json.loads(progress_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            atomic_json(job_dir / "status.json", status)
            if time.time() - started > timeout:
                status["status"] = "TIMEOUT"
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
            time.sleep(10)
        exit_code = child.wait()
    status.update({"exit_code": exit_code, "elapsed_seconds": time.time() - started,
                   "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    if status["status"] != "TIMEOUT":
        report = output / "training_report.json"
        if exit_code == 0 and report.exists():
            result = json.loads(report.read_text(encoding="utf-8"))
            status["status"] = "COMPLETED" if result.get("export", {}).get("passed") else "ARTIFACT_CHECK_FAILED"
            status["optimizer_steps"] = result["optimizer_steps"]
            status["dev_selected_threshold"] = result["dev_selected_threshold"]
        else:
            status["status"] = "FAILED"
    atomic_json(job_dir / "status.json", status)
    print(json.dumps(status, indent=2), flush=True)
    return 0 if status["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.run_name != Path(args.run_name).name:
        parser.error("Need a command and a simple run-name")
    raise SystemExit(run(args.root.resolve(), args.run_name, command, args.timeout))
