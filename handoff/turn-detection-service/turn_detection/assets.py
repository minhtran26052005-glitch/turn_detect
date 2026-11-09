"""Restore verified training assets or diagnostic traces from a mounted Drive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

RESERVE = 25 * 1024**3


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json_if_absent_or_matching(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise FileExistsError(f"existing metadata differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def copy_object(drive, entry, target):
    object_name = entry["object"]
    if not re.fullmatch(r"objects/[0-9a-f]{64}\.bin", object_name):
        raise ValueError("invalid Drive object path")
    if Path(object_name).stem != entry["sha256"]:
        raise ValueError("object name and SHA-256 disagree")
    source = drive / object_name
    if source.stat().st_size != entry["bytes"]:
        raise ValueError(f"incorrect size: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == entry["bytes"] and sha256(target) == entry["sha256"]:
            return
        raise FileExistsError(f"existing file differs: {target}")
    if shutil.disk_usage(target.parent).free < entry["bytes"] + RESERVE:
        raise OSError("insufficient disk space including 25 GiB reserve")
    temporary = target.with_name(target.name + ".partial")
    with source.open("rb") as src, temporary.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024**2)
        dst.flush()
        os.fsync(dst.fileno())
    if temporary.stat().st_size != entry["bytes"] or sha256(temporary) != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {target}")
    temporary.replace(target)


def restore_training(manifest_path, run_dir, language):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (manifest.get("pass_complete") and manifest.get("full_recovery_available")):
        raise ValueError("Drive manifest is not a complete recovery backup")
    drive, files = manifest_path.parent, manifest["files"]
    identity = json.loads((drive / "destination_identity.json").read_text(encoding="utf-8"))
    if identity != manifest.get("destination_identity"):
        raise ValueError("Drive destination identity does not match the backup")
    if (run_dir / "launcher.json").exists():
        raise FileExistsError("run directory belongs to a campaign; choose a separate directory")
    current = "full_epoch_v1/current_dataset.json"
    copy_object(drive, files[current], run_dir / current)
    generation = json.loads((run_dir / current).read_text(encoding="utf-8"))["generation"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", generation):
        raise ValueError("invalid dataset generation")
    prefix = f"full_epoch_v1/datasets/{generation}/"
    for name in ("manifest.json", "dataset.zip"):
        copy_object(drive, files[prefix + name], run_dir / prefix / name)
    folder = run_dir / prefix
    dataset = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if not dataset.get("complete") or dataset.get("generation") != generation:
        raise ValueError("incomplete or mismatched dataset")
    db = folder / "dataset.sqlite"
    if not db.exists():
        with zipfile.ZipFile(folder / "dataset.zip") as archive:
            item = archive.getinfo("dataset.sqlite")
            if shutil.disk_usage(folder).free < item.file_size + RESERVE:
                raise OSError("insufficient disk space for the extracted dataset")
            temporary = folder / "dataset.sqlite.partial"
            with archive.open(item) as src, temporary.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024**2)
                dst.flush()
                os.fsync(dst.fileno())
            if sha256(temporary) != dataset["database_sha256"]:
                raise ValueError("dataset SHA-256 mismatch")
            temporary.replace(db)
    elif sha256(db) != dataset["database_sha256"]:
        raise ValueError("existing dataset SHA-256 mismatch")
    teacher = run_dir / "teachers" / language
    for logical, entry in files.items():
        if logical == f"teachers/{language}/latest_resume.pt" or logical.startswith(f"teachers/{language}/best/"):
            copy_object(drive, entry, run_dir / logical)
    checkpoint = files[f"teachers/{language}/latest_resume.pt"]["source_receipt"]
    if (checkpoint.get("full_epoch_generation") != generation
            or checkpoint.get("full_epoch_dataset_sha256") != dataset["database_sha256"]
            or checkpoint.get("training_fingerprint") != dataset["fingerprints"][language]["train"]["sha256"]
            or not checkpoint.get("weights_optimizer_rng_and_sampler")):
        raise ValueError("checkpoint does not match dataset generation")
    write_json_if_absent_or_matching(teacher / "resume_receipt.json", checkpoint)
    best = files[f"teachers/{language}/best/model.safetensors"].get("source_receipt", {})
    write_json_if_absent_or_matching(teacher / "best_report.json", best)
    print(json.dumps({"language": language, "generation": generation,
                      "epoch": checkpoint["full_epoch_number"], "cursor": checkpoint["full_epoch_cursor"]}))


def restore_traces(diagnostic_dir, output):
    spec = json.loads((Path(__file__).resolve().parents[1] / "data/test/trace_manifest.json").read_text(encoding="utf-8"))
    for language, entry in spec["files"].items():
        copy_object(diagnostic_dir, entry, output / f"test_predictions_{language}.jsonl.gz")
    print(json.dumps({"trace_dir": str(output), "languages": sorted(spec["files"])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    training = sub.add_parser("training")
    training.add_argument("--manifest", type=Path, required=True)
    training.add_argument("--run-dir", type=Path, required=True)
    training.add_argument("--language", choices=("vi", "en", "zh"), required=True)
    traces = sub.add_parser("test")
    traces.add_argument("--diagnostic-dir", type=Path, required=True)
    traces.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "training":
        restore_training(args.manifest, args.run_dir, args.language)
    else:
        restore_traces(args.diagnostic_dir, args.output)


if __name__ == "__main__":
    main()
