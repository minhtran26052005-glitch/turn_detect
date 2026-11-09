"""Stream v15d recovery files directly from Colab to the user's mounted Drive.

No GPU, OAuth extraction, Windows staging, or live-SQLite file copy is used.
Drive objects are immutable by SHA; manifest publication commits a generation.
Only objects outside the current and previous generation are pruned.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
import traceback
import uuid
import zipfile

ROOT = Path("outputs/translation_segmenter_v15d_20261026")
JOB = ROOT / "drive_backup_v1"
BLOCK = 8 * 1024 * 1024
MIN_FREE = 25 * 1024**3


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value, durable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".new-" + uuid.uuid4().hex)
    with temporary.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        if durable:
            os.fsync(stream.fileno())
    if durable and digest_file(temporary) != hashlib.sha256(raw).hexdigest():
        raise RuntimeError("Drive metadata readback hash mismatch")
    temporary.replace(path)
    if durable and path.read_bytes() != raw:
        raise RuntimeError("Drive manifest publication readback mismatch")


class Backup:
    def __init__(self, config):
        self.config = config
        self.destination = Path(config["drive_path"])
        self.files = {}
        self.verified = set()
        self.last_manifest = None
        self.last_status = 0

    def state(self, stage, **fields):
        self.last_status = time.monotonic()
        atomic_json(JOB / "status.json", {"stage": stage, "at_utc": now(),
                    "drive_path": str(self.destination), "verified_files": len(self.files), **fields})

    def check_destination(self):
        if not os.path.ismount("/content/drive"):
            raise RuntimeError("Google Drive is not mounted")
        marker = self.destination / "destination_identity.json"
        if not marker.exists() or read_json(marker) != self.config["identity"]:
            raise RuntimeError("Mounted Drive destination identity differs or is missing")
        if shutil.disk_usage(ROOT).free < MIN_FREE:
            raise RuntimeError("Colab disk is below the 25 GiB reserve")

    def load_manifest(self):
        self.check_destination()
        p = self.destination / "latest_manifest.json"
        if p.exists():
            value = read_json(p)
            if value.get("destination_identity") != self.config["identity"]:
                raise RuntimeError("Existing backup manifest belongs to another destination")
            for logical, entry in value.get("files", {}).items():
                if PurePosixPath(logical).is_absolute() or ".." in PurePosixPath(logical).parts:
                    raise RuntimeError("Invalid logical path in existing backup manifest")
                if entry.get("object") != "objects/" + entry.get("sha256", "") + ".bin":
                    raise RuntimeError("Invalid object path in existing backup manifest")
                if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
                    raise RuntimeError("Invalid object digest in existing backup manifest")
            self.last_manifest = value
            self.files = dict(value.get("files", {}))

    def put(self, logical, path, expected=None, source_receipt=None):
        """Hash and copy an open stable source handle, retaining bounded memory."""
        path = Path(path)
        self.check_destination()
        with path.open("rb") as source:
            stat = os.fstat(source.fileno())
            signature = expected or hashlib.sha256(json.dumps({
                "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "receipt": source_receipt}, sort_keys=True).encode()).hexdigest()
            previous = self.files.get(logical)
            if previous and previous.get("source_signature") == signature:
                target = self.destination / previous["object"]
                if target.exists() and target.stat().st_size == previous["bytes"]:
                    if previous["sha256"] in self.verified or digest_file(target) == previous["sha256"]:
                        self.verified.add(previous["sha256"])
                        return previous
            self.state("HASHING_SOURCE", logical_path=logical, total_bytes=stat.st_size)
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
            source.seek(0)
            if expected is not None and checksum != expected:
                raise RuntimeError("Published source SHA changed before backup: " + logical)
            current_stat = os.fstat(source.fileno())
            if (current_stat.st_size, current_stat.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError("Source was modified in place while hashing: " + logical)
            objects = self.destination / "objects"
            objects.mkdir(exist_ok=True)
            target = objects / (checksum + ".bin")
            if target.exists():
                if target.stat().st_size != stat.st_size or (checksum not in self.verified and digest_file(target) != checksum):
                    raise RuntimeError("Existing Drive object failed verification: " + checksum)
            else:
                free = shutil.disk_usage(objects).free
                if free < stat.st_size + 1024**3:
                    raise RuntimeError("Drive has insufficient space for the next recovery object")
                temporary = objects / (checksum + ".uploading")
                count = 0
                write_hash = hashlib.sha256()
                self.state("COPYING_TO_DRIVE", logical_path=logical, total_bytes=stat.st_size, completed_bytes=0)
                with temporary.open("wb") as destination:
                    while True:
                        block = source.read(BLOCK)
                        if not block:
                            break
                        destination.write(block)
                        write_hash.update(block)
                        count += len(block)
                        if time.monotonic() - self.last_status > 15:
                            self.state("COPYING_TO_DRIVE", logical_path=logical,
                                       total_bytes=stat.st_size, completed_bytes=count)
                    destination.flush()
                    os.fsync(destination.fileno())
                if count != stat.st_size or write_hash.hexdigest() != checksum:
                    raise RuntimeError("Source changed during Drive transfer: " + logical)
                self.state("VERIFYING_DRIVE_OBJECT", logical_path=logical, total_bytes=count)
                if digest_file(temporary) != checksum:
                    raise RuntimeError("Drive object readback SHA mismatch: " + logical)
                temporary.replace(target)
            self.verified.add(checksum)
            entry = {"object": "objects/" + checksum + ".bin", "sha256": checksum,
                     "bytes": stat.st_size, "source_signature": signature,
                     "verified_at_utc": now()}
            if source_receipt is not None:
                entry["source_receipt"] = source_receipt
            self.files[logical] = entry
            atomic_json(JOB / "progress_manifest.json", self.manifest(complete=False))
            atomic_json(self.destination / "in_progress_manifest.json", self.manifest(complete=False), durable=True)
            return entry

    def put_small_json(self, logical, value):
        target = JOB / (hashlib.sha256(logical.encode()).hexdigest()[:16] + ".json")
        atomic_json(target, value)
        return self.put(logical, target, expected=digest_file(target))

    def manifest(self, complete):
        required = ["data/data_snapshot.zip", "source/source_bundle.zip"]
        required += ["teachers/" + lang + "/latest_resume.pt" for lang in ("vi", "en", "zh")]
        required += ["teachers/" + lang + "/best/model.safetensors" for lang in ("vi", "en", "zh")]
        missing = [path for path in required if path not in self.files]
        return {"schema": 1, "campaign": "translation_segmenter_v15d", "captured_at_utc": now(),
                "destination_identity": self.config["identity"], "files": dict(self.files),
                "backup_policy": {"interval_seconds": self.config["interval_seconds"], "kept_generations": 2},
                "pass_complete": complete, "full_recovery_available": not missing,
                "missing_recovery_files": missing, "scope": "data, stream cursor, weights, optimizer/RNG, source, reports and available held-out assets",
                "verification": "source SHA-256, Drive fsync, Drive-mounted-file readback SHA-256"}

    def zip_source(self):
        archive = JOB / "source_bundle.zip"
        versions = {}
        for name in ("torch", "transformers", "datasets", "numpy", "sentencepiece", "jieba", "huggingface_hub"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as out:
            for source in sorted(Path("turn_v2").glob("*.py")):
                out.write(source, str(source))
            info = zipfile.ZipInfo("runtime_versions.json", date_time=(2026, 1, 1, 0, 0, 0))
            out.writestr(info, json.dumps(versions, sort_keys=True))
        self.put("source/source_bundle.zip", archive, expected=digest_file(archive))

    def save_teacher(self, language):
        folder = ROOT / "teachers" / language
        receipt_path = folder / "resume_receipt.json"
        checkpoint = folder / "latest_resume.pt"
        if receipt_path.exists() and checkpoint.exists():
            # Trainer publishes .pt with atomic replace, then publishes receipt.
            # Snapshot the open inode to avoid pairing it with a later receipt.
            before = receipt_path.read_bytes()
            receipt = json.loads(before)
            with checkpoint.open("rb") as source:
                stat = os.fstat(source.fileno())
                after = receipt_path.read_bytes()
                if before != after or stat.st_size != receipt["bytes"] or stat.st_mtime_ns > receipt_path.stat().st_mtime_ns:
                    raise RuntimeError("Checkpoint/receipt publication is in flight: " + language)
                # /proc/self/fd resolves this exact immutable inode even if the
                # trainer later replaces latest_resume.pt during the Drive copy.
                pinned = Path("/proc/self/fd") / str(source.fileno())
                self.put(f"teachers/{language}/latest_resume.pt", pinned, source_receipt=receipt)
        report_path = folder / "best_report.json"
        if report_path.exists():
            report = read_json(report_path)
            model = folder / "best" / "model.safetensors"
            self.put(f"teachers/{language}/best/model.safetensors", model, expected=report["weights_sha256"], source_receipt=report)
            for path in sorted((folder / "best").iterdir()):
                if path.is_file() and path.name != "model.safetensors":
                    self.put(f"teachers/{language}/best/{path.name}", path)

    def save_evaluation(self):
        folder = ROOT / "heldout_test_v2"
        manifest_path = folder / "manifest.json"
        if not manifest_path.exists():
            return
        manifest = read_json(manifest_path)
        self.put_small_json("heldout_test_v2/manifest.json", manifest)
        frozen = folder / "frozen"
        database = frozen / "test.sqlite"
        archive = JOB / "frozen_test_database.zip"
        if database.exists():
            if not archive.exists():
                if digest_file(database) != manifest["database_sha256"]:
                    raise RuntimeError("Frozen test database differs from evaluation manifest")
                with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as out:
                    out.write(database, "test.sqlite")
            self.put("heldout_test_v2/frozen_test_database.zip", archive, expected=digest_file(archive), source_receipt={"uncompressed_database_sha256": manifest["database_sha256"]})
        for lang, checkpoint in manifest.get("checkpoints", {}).items():
            for filename, checksum in checkpoint.get("files_sha256", {}).items():
                self.put(f"heldout_test_v2/frozen/{lang}/{filename}", frozen / lang / filename, expected=checksum)
        for name in ("report.json", "partial_report.json", "status.json", "review_samples.jsonl"):
            path = folder / name
            if path.exists():
                self.put(f"heldout_test_v2/{name}", path)
        completed = read_json(folder / "report.json") if (folder / "report.json").exists() else read_json(folder / "partial_report.json") if (folder / "partial_report.json").exists() else {}
        for lang, report in completed.get("languages", {}).items():
            path = Path(report["full_trace_remote_only"])
            self.put(f"heldout_test_v2/{path.name}", path, expected=report["full_trace_sha256"])

    def save_reports(self):
        archive = JOB / "reports_and_logs.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as out:
            for path in sorted(ROOT.rglob("*")):
                if not path.is_file() or path.suffix not in (".json", ".jsonl", ".log", ".txt"):
                    continue
                relative = path.relative_to(ROOT)
                if relative.parts[0] in ("drive_backup_v1",) or "frozen" in relative.parts or "best" in relative.parts:
                    continue
                try:
                    out.write(path, str(relative))
                except FileNotFoundError:
                    continue
        self.put("reports/reports_and_logs.zip", archive, expected=digest_file(archive))

    def prune(self, current, previous):
        referenced = {entry["object"] for value in (current, previous or {}) for entry in value.get("files", {}).values()}
        objects = self.destination / "objects"
        self.check_destination()
        count = 0
        for path in objects.iterdir():
            if re.fullmatch(r"[0-9a-f]{64}\.bin", path.name) and "objects/" + path.name not in referenced:
                path.unlink()  # Only generated, unreferenced content objects.
                count += 1
        return count

    def sync(self):
        self.check_destination()
        metadata = read_json(ROOT / "recovery/latest_snapshot.json")
        self.put("data/data_snapshot.zip", Path("/content/turn_detect_v2") / metadata["path"], expected=metadata["sha256"], source_receipt=metadata)
        self.zip_source()
        for language in ("vi", "en", "zh"):
            self.save_teacher(language)
        self.save_evaluation()
        self.save_reports()
        current = self.manifest(complete=True)
        previous = self.last_manifest
        if previous:
            atomic_json(self.destination / "previous_manifest.json", previous, durable=True)
        atomic_json(self.destination / "latest_manifest.json", current, durable=True)
        atomic_json(JOB / "latest_verified_manifest.json", current)
        self.last_manifest = current
        pruned = self.prune(current, previous)
        self.state("SYNCED", full_recovery_available=current["full_recovery_available"],
                   files=len(current["files"]), data_cycle=metadata["cycle"],
                   logical_bytes=sum(v["bytes"] for v in current["files"].values()),
                   object_bytes=sum(p.stat().st_size for p in (self.destination / "objects").glob("*.bin")),
                   pruned_objects=pruned, next_check_seconds=self.config["interval_seconds"])


def main():
    import fcntl
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB / "LOCK").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = read_json(JOB / "config.json")
        backup = Backup(config)
        loaded = False
        failures = 0
        while True:
            try:
                if not loaded:
                    backup.load_manifest()
                    loaded = True
                backup.sync()
                failures = 0
                time.sleep(config["interval_seconds"])
            except Exception as exc:
                failures += 1
                delay = min(600, 30 * 2 ** min(failures - 1, 4))
                atomic_json(JOB / "latest_error.json", {"at_utc": now(), "error": repr(exc), "traceback": traceback.format_exc(), "retry_after_seconds": delay})
                backup.state("RETRYING_AFTER_ERROR", error=repr(exc), retry_after_seconds=delay)
                time.sleep(delay)


if __name__ == "__main__":
    main()
