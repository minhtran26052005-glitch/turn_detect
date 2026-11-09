"""Recovery publication/retention tests with tiny local fixtures, no models."""
import hashlib
import json
import sqlite3
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from turn_v2 import translation_segmenter_drive_backup_v1 as backup
from turn_v2.drive_recovery_verify_v2 import historical_fingerprint


class DriveBackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="drive_backup_check_")
        self.root = Path(self.temp.name)
        self.destination = self.root / "drive"
        self.destination.mkdir()
        self.job = self.root / "job"
        self.job.mkdir()
        self.config = {"drive_path": str(self.destination), "identity": {"nonce": "test"},
                       "interval_seconds": 900}
        self.job_patch = patch.object(backup, "JOB", self.job)
        self.job_patch.start()
        self.check_patch = patch.object(backup.Backup, "check_destination")
        self.check_patch.start()

    def tearDown(self):
        self.check_patch.stop()
        self.job_patch.stop()
        self.temp.cleanup()

    def test_streamed_objects_are_verified_and_shared_by_identical_files(self):
        src = self.root / "checkpoint.pt"
        src.write_bytes(b"fixture" * 1000)
        expected = hashlib.sha256(src.read_bytes()).hexdigest()
        worker = backup.Backup(self.config)
        original = Path.read_bytes
        def bounded_read(path, *args, **kwargs):
            if path.suffix in (".pt", ".bin"):
                raise AssertionError("Large artifacts must be streamed")
            return original(path, *args, **kwargs)
        with patch.object(Path, "read_bytes", bounded_read):
            a = worker.put("teachers/vi/latest_resume.pt", src, expected=expected)
            b = worker.put("heldout/vi/model.safetensors", src, expected=expected)
        self.assertEqual(a["sha256"], expected)
        self.assertEqual(a["object"], b["object"])
        self.assertEqual(len(list((self.destination / "objects").glob("*.bin"))), 1)
        self.assertEqual(backup.digest_file(self.destination / a["object"]), expected)

    def test_corrupt_existing_object_is_rejected_after_process_restart(self):
        src = self.root / "source"
        src.write_bytes(b"good")
        expected = backup.digest_file(src)
        worker = backup.Backup(self.config)
        entry = worker.put("data", src, expected=expected)
        (self.destination / entry["object"]).write_bytes(b"evil")
        restarted = backup.Backup(self.config)
        restarted.files = {"data": entry}
        with self.assertRaisesRegex(RuntimeError, "Existing Drive object"):
            restarted.put("data", src, expected=expected)

    def test_hash_mismatch_does_not_publish_a_reference(self):
        source = self.root / "source"
        source.write_bytes(b"data")
        worker = backup.Backup(self.config)
        with self.assertRaisesRegex(RuntimeError, "Published source SHA"):
            worker.put("data", source, expected="0" * 64)
        self.assertNotIn("data", worker.files)

    def test_retention_preserves_current_previous_and_unrelated_user_files(self):
        objects = self.destination / "objects"
        objects.mkdir()
        a, b, orphan = [char * 64 + ".bin" for char in "abc"]
        for name in (a, b, orphan, "user_notes.txt"):
            (objects / name).write_text("fixture")
        current = {"files": {"a": {"object": "objects/" + a}}}
        previous = {"files": {"b": {"object": "objects/" + b}}}
        self.assertEqual(backup.Backup(self.config).prune(current, previous), 1)
        self.assertTrue((objects / a).exists())
        self.assertTrue((objects / b).exists())
        self.assertTrue((objects / "user_notes.txt").exists())
        self.assertFalse((objects / orphan).exists())

    def test_failed_metadata_readback_keeps_previous_manifest(self):
        path = self.destination / "latest_manifest.json"
        path.write_text('{"old": true}')
        with patch.object(backup, "digest_file", return_value="wrong"):
            with self.assertRaisesRegex(RuntimeError, "metadata readback"):
                backup.atomic_json(path, {"new": True}, durable=True)
        self.assertEqual(json.loads(path.read_text()), {"old": True})

    def test_manifest_cannot_escape_owned_object_directory(self):
        bad = {"destination_identity": self.config["identity"],
               "files": {"data": {"object": "../unrelated", "sha256": "a" * 64}}}
        (self.destination / "latest_manifest.json").write_text(json.dumps(bad))
        with self.assertRaisesRegex(RuntimeError, "Invalid object path"):
            backup.Backup(self.config).load_manifest()

    def test_historical_fingerprint_filters_future_pages_and_orders_exact_rows(self):
        db = sqlite3.connect(":memory:")
        db.executescript("CREATE TABLE sources(language,page_id,added_cycle);"
                         "CREATE TABLE examples(example_id,source_id,language,split,record_json);")
        db.executemany("INSERT INTO sources VALUES(?,?,?)", [("vi", "old:page", 48), ("vi", "new", 50)])
        db.executemany("INSERT INTO examples VALUES(?,?,?,?,?)", [
            ("b", "vi:old:page", "vi", "train", "tiếng việt"),
            ("a", "vi:old:page", "vi", "train", "中文"),
            ("c", "vi:new", "vi", "train", "future"),
            ("d", "vi:old:page", "vi", "dev", "dev excluded"),
        ])
        actual, count = historical_fingerprint(db, "vi", 48)
        self.assertEqual(count, 2)
        self.assertEqual(actual, hashlib.sha256("中文\ntiếng việt\n".encode()).hexdigest())
        db.close()


if __name__ == "__main__":
    unittest.main()
