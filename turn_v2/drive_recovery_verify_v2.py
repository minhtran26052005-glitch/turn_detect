"""CPU-only backup readback proof, using linear source-group filtering."""
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import traceback
import zipfile

ROOT = Path("outputs/translation_segmenter_v15d_20261026")
JOB = ROOT / "drive_backup_v1"


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def state(stage, **values):
    write(JOB / "verification_status_v2.json", {"at_utc": datetime.now(timezone.utc).isoformat(), "stage": stage, **values})


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def historical_fingerprint(db, lang, cycle):
    allowed = {lang + ":" + str(page) for (page,) in db.execute(
        "SELECT page_id FROM sources WHERE language=? AND added_cycle<=?", (lang, cycle))}
    checksum = hashlib.sha256()
    kept = scanned = 0
    for source_id, raw in db.execute("SELECT source_id,record_json FROM examples WHERE language=? AND split='train' ORDER BY example_id", (lang,)):
        if source_id in allowed:
            checksum.update(raw.encode("utf-8")); checksum.update(b"\n"); kept += 1
        scanned += 1
        if scanned % 50000 == 0:
            state("RECONSTRUCTING_TRAINING_FINGERPRINT", language=lang, scanned=scanned, selected=kept)
    return checksum.hexdigest(), kept


def main():
    import psutil
    import torch
    assert psutil.virtual_memory().available > 8 * 1024**3
    config = json.loads((JOB / "config.json").read_text())
    drive = Path(config["drive_path"])
    assert os.path.ismount("/content/drive")
    assert json.loads((drive / "destination_identity.json").read_text()) == config["identity"]
    manifest = json.loads((drive / "latest_manifest.json").read_text())
    assert manifest["pass_complete"] and manifest["full_recovery_available"]
    data = manifest["files"]["data/data_snapshot.zip"]
    archive = drive / data["object"]
    state("CHECKING_DATA_ARCHIVE")
    assert sha(archive) == data["sha256"]
    temp = JOB / f"verification_v2_{os.getpid()}.sqlite"
    with zipfile.ZipFile(archive) as z, z.open("state_snapshot.sqlite") as src, temp.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    db = sqlite3.connect("file:" + str(temp.resolve()) + "?mode=ro&immutable=1", uri=True)
    result = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "drive_path": str(drive),
              "manifest_captured_at_utc": manifest["captured_at_utc"],
              "data_cycle": data["source_receipt"]["cycle"], "data_zip_sha256": data["sha256"],
              "database_integrity": db.execute("PRAGMA integrity_check").fetchone()[0],
              "checkpoint_checks": {}, "scope": "backup deserialization/provenance; no training or model-quality inference"}
    try:
        for lang in ("vi", "en", "zh"):
            entry = manifest["files"][f"teachers/{lang}/latest_resume.pt"]
            path = drive / entry["object"]
            state("CHECKING_CHECKPOINT_HASH", language=lang)
            assert sha(path) == entry["sha256"]
            state("DESERIALIZING_ON_CPU", language=lang)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            receipt = entry["source_receipt"]
            for key in ("cycle", "cycle_step", "total_steps", "training_fingerprint"):
                assert saved[key] == receipt[key], (lang, key)
            assert saved["optimizer"]["state"] and saved["optimizer"]["param_groups"] and saved["model"]
            assert all(key in saved for key in ("numpy_rng", "torch_rng", "cuda_rng"))
            # Avoid a concatenation join that can scan every source per example.
            fingerprint, kept = historical_fingerprint(db, lang, saved["cycle"])
            result["checkpoint_checks"][lang] = {
                "sha256": entry["sha256"], "cycle": saved["cycle"], "cycle_step": saved["cycle_step"],
                "total_steps": saved["total_steps"], "optimizer_parameter_groups": len(saved["optimizer"]["param_groups"]),
                "optimizer_states": len(saved["optimizer"]["state"]), "rng_state_present": True,
                "training_rows_at_checkpoint_cycle": kept, "training_fingerprint": saved["training_fingerprint"],
                "reconstructed_fingerprint": fingerprint,
                "training_fingerprint_matches": fingerprint == saved["training_fingerprint"]}
            del saved
            gc.collect()
    finally:
        db.close()
        temp.unlink()
    result["all_training_fingerprints_match"] = all(x["training_fingerprint_matches"] for x in result["checkpoint_checks"].values())
    write(JOB / "recovery_verification.json", result)
    raw = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()
    with (drive / "recovery_verification.json").open("wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    assert sha(drive / "recovery_verification.json") == hashlib.sha256(raw).hexdigest()
    state("COMPLETE", all_training_fingerprints_match=result["all_training_fingerprints_match"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write(JOB / "verification_error_v2.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        state("FAILED", error=repr(exc))
        raise
