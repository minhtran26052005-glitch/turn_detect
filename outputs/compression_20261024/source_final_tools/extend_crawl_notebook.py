"""Refresh embedded code and append crawl/predict cells without replacing user settings."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil

from build_notebook import ROOT, crawl_predict_cells, source_bundle


def extend():
    path = ROOT / "a.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    backup = ROOT / "backups/a.before_crawl_predict.ipynb"
    if not backup.exists():
        shutil.copy2(path, backup)
    package, checksum = source_bundle()
    for entry in notebook["cells"]:
        if "bootstrap" not in entry.get("metadata", {}).get("tags", []):
            continue
        source = "".join(entry["source"])
        source, count = re.subn(r"package = base64\.b64decode\('[^']*'\)",
            "package = base64.b64decode(" + repr(package) + ")", source, count=1)
        if count != 1:
            raise ValueError("Could not locate the embedded source; notebook was not overwritten")
        source, count = re.subn(r"assert hashlib\.sha256\(package\).hexdigest\(\) == '[0-9a-f]{64}'",
            "assert hashlib.sha256(package).hexdigest() == " + repr(checksum), source, count=1)
        if count != 1:
            raise ValueError("Could not locate bundle hash; notebook was not overwritten")
        compile(source, "notebook_bootstrap", "exec")
        entry["source"] = source.splitlines(keepends=True)
        entry["execution_count"], entry["outputs"] = None, []
        entry["metadata"].pop("colab_verified_at", None)
    if not any("crawl_predict" in entry.get("metadata", {}).get("tags", []) for entry in notebook["cells"]):
        notebook["cells"][-1:-1] = crawl_predict_cells()
    for entry in notebook["cells"]:
        if entry["cell_type"] == "code":
            compile("".join(entry["source"]), "a.ipynb", "exec")
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Updated embedded package and crawl/predict cells; other settings and outputs preserved.")


if __name__ == "__main__":
    extend()
