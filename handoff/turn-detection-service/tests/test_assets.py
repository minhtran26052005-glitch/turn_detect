import hashlib
import json
import zipfile
import pytest
from turn_detection import assets


def test_copy_verifies_content_and_preserves_different_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "RESERVE", 0)
    payload = b"verified object"
    digest = hashlib.sha256(payload).hexdigest()
    drive = tmp_path / "drive"
    (drive / "objects").mkdir(parents=True)
    (drive / "objects" / (digest + ".bin")).write_bytes(payload)
    entry = {"object": "objects/" + digest + ".bin", "sha256": digest, "bytes": len(payload)}
    output = tmp_path / "out" / "asset"
    assets.copy_object(drive, entry, output)
    assert output.read_bytes() == payload
    output.write_bytes(b"different")
    with pytest.raises(FileExistsError):
        assets.copy_object(drive, entry, output)
    assert output.read_bytes() == b"different"


def test_restore_training_validates_generation_and_preserves_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "RESERVE", 0)
    drive = tmp_path / "drive"
    objects = drive / "objects"
    objects.mkdir(parents=True)
    identity = {"campaign": "fixture", "nonce": "fixture"}
    (drive / "destination_identity.json").write_text(json.dumps(identity))
    database = b"small fixture database"
    db_hash = hashlib.sha256(database).hexdigest()
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("dataset.sqlite", database)
    generation = "fixture_generation"
    data = {"complete": True, "generation": generation, "database_sha256": db_hash,
            "fingerprints": {"vi": {"train": {"sha256": "train-fingerprint"}}}}
    checkpoint = {"full_epoch_generation": generation, "full_epoch_dataset_sha256": db_hash,
                  "training_fingerprint": "train-fingerprint", "weights_optimizer_rng_and_sampler": True,
                  "full_epoch_number": 1, "full_epoch_cursor": 10}

    def add(payload, receipt=None):
        digest = hashlib.sha256(payload).hexdigest()
        (objects / (digest + ".bin")).write_bytes(payload)
        entry = {"object": "objects/" + digest + ".bin", "sha256": digest, "bytes": len(payload)}
        if receipt is not None:
            entry["source_receipt"] = receipt
        return entry

    data_bytes = json.dumps(data).encode()
    prefix = f"full_epoch_v1/datasets/{generation}/"
    files = {"full_epoch_v1/current_dataset.json": add(data_bytes),
             prefix + "manifest.json": add(data_bytes), prefix + "dataset.zip": add(archive.read_bytes()),
             "teachers/vi/latest_resume.pt": add(b"fixture checkpoint", checkpoint),
             "teachers/vi/best/model.safetensors": add(b"fixture weights", {"threshold_selected_on_dev": 0.6})}
    manifest = {"pass_complete": True, "full_recovery_available": True,
                "destination_identity": identity, "files": files}
    manifest_path = drive / "latest_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "work"
    assets.restore_training(manifest_path, output, "vi")
    assets.restore_training(manifest_path, output, "vi")
    assert (output / prefix / "dataset.sqlite").read_bytes() == database
    receipt = output / "teachers/vi/resume_receipt.json"
    assert json.loads(receipt.read_text()) == checkpoint
    receipt.write_text('{"changed": true}')
    with pytest.raises(FileExistsError):
        assets.restore_training(manifest_path, output, "vi")
