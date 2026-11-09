"""Execute a.ipynb's default smoke workflow in the already connected Colab kernel.

Outputs are saved back into the local notebook after each cell. No runtime is
created/restarted and no authentication token is saved in the notebook.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from colab_runtime import ColabRuntime


def run(path, timeout=900, tags=None, receipt_path=None, server_index=0, session_path=None):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    runtime = ColabRuntime(server_index, session_path)
    connection = runtime.connect()
    print(json.dumps(connection), flush=True)
    executed = []
    count = 0
    selected_count = sum(entry["cell_type"] == "code" and
        (not tags or bool(set(tags) & set(entry.get("metadata", {}).get("tags", []))))
        for entry in notebook["cells"])
    if not selected_count:
        raise ValueError("No notebook code cells matched the requested tags")
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        if tags and not set(tags) & set(cell.get("metadata", {}).get("tags", [])):
            continue
        count += 1
        tag = cell.get("metadata", {}).get("tags", [str(index)])[0]
        print(f"Executing cell {index} ({tag})", flush=True)
        result = runtime.execute("".join(cell["source"]), timeout=timeout)
        cell["execution_count"] = count
        cell["outputs"] = []
        for output in result["outputs"]:
            kind = output["output_type"]
            clean = {key: value for key, value in output.items()
                     if key in ("output_type", "name", "text", "ename", "evalue", "traceback", "data", "metadata", "execution_count")}
            if kind == "execute_result":
                clean["execution_count"] = count
            cell["outputs"].append(clean)
        cell["metadata"]["colab_verified_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        executed.append({"cell": index, "tag": tag, "status": result["status"],
                         "elapsed_seconds": result["elapsed_seconds"]})
        if result["status"] != "ok":
            break
    receipt = {"verified_at": datetime.now(timezone.utc).isoformat(), "connection": connection,
               "notebook": str(path), "cells": executed,
               "passed": all(item["status"] == "ok" for item in executed)
                         and len(executed) == selected_count, "selected_tags": tags}
    destination = receipt_path or path.parent / "outputs" / "colab_notebook_test.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if not receipt["passed"]:
        raise RuntimeError("Notebook execution failed; inspect the recorded cell output")
    runtime.close()
    print("NOTEBOOK_SMOKE_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=Path(__file__).resolve().parents[1] / "a.ipynb")
    parser.add_argument("--cell-timeout", type=int, default=900)
    parser.add_argument("--tags", nargs="+", help="Execute selected tagged cells using the existing kernel state")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--server-index", type=int, default=0)
    parser.add_argument("--session-path")
    args = parser.parse_args()
    run(args.notebook.resolve(), args.cell_timeout, args.tags, args.receipt,
        args.server_index, args.session_path)
