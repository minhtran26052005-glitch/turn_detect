"""Sync and verify the delivered QAT model using the existing Colab session."""
import json
from pathlib import Path
import zipfile

from colab_runtime import ColabRuntime
from qat_colab import SESSION


def main():
    root = Path.cwd().resolve()
    evidence = root / "outputs/qat_20261024"
    package = root / "models/turn_v2_distilled_qat4_20261024"
    manifest = json.loads((package / "model_manifest.json").read_text(encoding="utf-8"))
    reference = Path(manifest["selected_report"]).parent / "dev_predictions.npz"
    archive = evidence / "colab_qat_sync.zip"
    paths = [p for p in package.rglob("*") if p.is_file()] + [root / reference, root / "tools/verify_tiny_package.py"]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
        for path in paths:
            out.write(path, path.relative_to(root).as_posix())
    r = ColabRuntime(0, SESSION)
    connection = r.connect()
    r.upload(archive, "content/turn_detect_v2/colab/qat_sync_20261024.zip")
    r.close()
    r.connect()
    code = f'''from pathlib import Path
import json,subprocess,sys,zipfile
qat_root=Path('/content/turn_detect_v2').resolve()
with zipfile.ZipFile(qat_root/'colab/qat_sync_20261024.zip') as archive:
    for member in archive.infolist():
        target=(qat_root/member.filename).resolve()
        assert target.is_relative_to(qat_root)
        data=archive.read(member)
        if target.exists():assert target.read_bytes()==data, 'Existing QAT package file differs: '+member.filename
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
qat_child=subprocess.run([sys.executable,'tools/verify_tiny_package.py','--package','models/turn_v2_distilled_qat4_20261024','--reference',{reference.as_posix()!r},'--output','outputs/qat_20261024/colab_native_verification.json'],cwd=qat_root,capture_output=True,text=True,timeout=60)
print(json.dumps({{'exit_code':qat_child.returncode,'stdout':qat_child.stdout,'stderr':qat_child.stderr}}))
assert qat_child.returncode==0
'''
    result = r.execute(code, timeout=90, echo=False)
    (evidence / "colab_sync_receipt.json").write_text(json.dumps({"connection": connection, **result}, ensure_ascii=False, indent=2), encoding="utf-8")
    assert result["status"] == "ok", "Inspect Colab sync receipt"
    r.download("content/turn_detect_v2/outputs/qat_20261024/colab_native_verification.json", evidence / "colab_native_verification.json")
    r.close()
    local = json.loads((evidence / "native_verification.json").read_text(encoding="utf-8"))
    remote = json.loads((evidence / "colab_native_verification.json").read_text(encoding="utf-8"))
    parity = {"samples_per_platform": local["samples"], "windows_max_delta_vs_reference": local["max_probability_delta_vs_reference"],
        "colab_max_delta_vs_reference": remote["max_probability_delta_vs_reference"], "dev_metrics_same": local["dev_metrics"] == remote["dev_metrics"],
        "model_sha256_same": local["model_sha256"] == remote["model_sha256"],
        "passed": local["passed"] and remote["passed"] and local["dev_metrics"] == remote["dev_metrics"] and local["model_sha256"] == remote["model_sha256"]}
    (evidence / "windows_colab_parity.json").write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(parity))
    assert parity["passed"]


if __name__ == "__main__":
    main()
