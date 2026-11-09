"""Sync the INT4 package and verify full dev through the existing Colab kernel."""
import json
from pathlib import Path
import zipfile

from colab_runtime import ColabRuntime


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / "outputs/quant4_20261024"
    package = root / "models/turn_v2_distilled_int4_20261024"
    paths = [p for p in package.rglob("*") if p.is_file()]
    paths.extend([root / "tools/verify_student4.py", root / "turn_v2/quantize_student4.py",
                  evidence / "int4_g32_quality/dev_predictions.npz"])
    archive = evidence / "colab_int4_sync.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
        for path in paths:
            out.write(path, path.relative_to(root).as_posix())
    runtime = ColabRuntime(0, "a-jvsc-f2543605-ce4e-4c3d-a99f-688dd99e9b33d84c1b1f-d7f4-4eb7-9690-6a3514d8ec38.ipynb")
    connection = runtime.connect()
    runtime.upload(archive, "content/turn_detect_v2/colab/int4_sync_20261024.zip")
    runtime.close()
    runtime.connect()
    code = '''import hashlib, json, subprocess, sys, zipfile
from pathlib import Path
int4_sync_root = Path('/content/turn_detect_v2').resolve()
with zipfile.ZipFile(int4_sync_root/'colab/int4_sync_20261024.zip') as archive:
    for member in archive.infolist():
        target = (int4_sync_root/member.filename).resolve()
        assert target.is_relative_to(int4_sync_root)
        data = archive.read(member)
        if target.exists():
            assert target.read_bytes() == data, 'Existing INT4 sync file differs: '+member.filename
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
int4_remote_output = int4_sync_root/'outputs/quant4_20261024/colab_native_verification.json'
int4_child = subprocess.run([sys.executable, str(int4_sync_root/'tools/verify_student4.py'), '--root', str(int4_sync_root), '--output', str(int4_remote_output)], capture_output=True, text=True, timeout=60, cwd=int4_sync_root)
print(json.dumps({'exit_code':int4_child.returncode, 'stdout':int4_child.stdout, 'stderr':int4_child.stderr}))
assert int4_child.returncode == 0
'''
    result = runtime.execute(code, timeout=90, echo=False)
    (evidence / "colab_sync_receipt.json").write_text(json.dumps({"connection": connection, **result}, ensure_ascii=False, indent=2), encoding="utf-8")
    assert result["status"] == "ok", "Inspect Colab sync receipt"
    runtime.download("content/turn_detect_v2/outputs/quant4_20261024/colab_native_verification.json", evidence / "colab_native_verification.json")
    runtime.close()
    local = json.loads((evidence / "native_verification.json").read_text(encoding="utf-8"))
    remote = json.loads((evidence / "colab_native_verification.json").read_text(encoding="utf-8"))
    assert local["passed"] and remote["passed"]
    assert local["model_sha256"] == remote["model_sha256"]
    parity = {"full_dev_samples_per_platform": local["samples"],
        "windows_max_delta_vs_reference": local["full_dev_probability_delta_vs_evaluator"],
        "colab_max_delta_vs_reference": remote["full_dev_probability_delta_vs_evaluator"],
        "dev_metrics_same": local["dev_metrics"] == remote["dev_metrics"],
        "model_sha256_same": True, "passed": local["dev_metrics"] == remote["dev_metrics"]}
    (evidence / "windows_colab_parity.json").write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
    assert parity["passed"]
    print(json.dumps(parity))


if __name__ == "__main__":
    main()
