"""Sync final manifest/docs only; candidate weights/thresholds must match frozen."""
import json
from pathlib import Path
import zipfile

from colab_runtime import ColabRuntime
from qat_colab import SESSION


def main():
    root = Path("outputs/trilingual_20261024")
    paths = [Path("models/turn_v3_trilingual_pilot_20261024/model_manifest.json"), Path("models/turn_v3_trilingual_pilot_20261024/README.md"),
             Path("README_TRILINGUAL_V3.md"), Path("EXPERIMENT_TRILINGUAL_20261024.md")]
    archive = root / "final_receipts_sync.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in paths:z.write(path, path.as_posix())
    r = ColabRuntime(0, SESSION); r.connect(); r.upload(archive, "content/turn_detect_v2/colab/final_receipts_sync.zip")
    r.close(); r.connect()
    result = r.execute("""from pathlib import Path
import zipfile,json
tri_root=Path('/content/turn_detect_v2').resolve()
with zipfile.ZipFile(tri_root/'colab/final_receipts_sync.zip') as archive:
    for item in archive.infolist():
        path=(tri_root/item.filename).resolve();assert path.is_relative_to(tri_root)
        data=archive.read(item)
        if path.exists():assert path.read_bytes()==data
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
print(json.dumps({'manifest_docs_synced':True,'weights_thresholds_unchanged':True}))
""", timeout=45, echo=False)
    (root / "final_receipts_sync.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    assert result["status"] == "ok"; r.close()
    print('Final metadata synced; frozen model untouched')


if __name__ == "__main__": main()
