"""Exercise snapshot disk preservation without loading full files into memory."""
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import translation_segmenter_v15d3_colab as control


def make_database(path, cycle, rows):
    db = sqlite3.connect(path)
    db.executescript("CREATE TABLE sources(id); CREATE TABLE examples(id); CREATE TABLE meta(key,value);")
    db.executemany("INSERT INTO sources VALUES(?)", [(i,) for i in range(rows)])
    db.executemany("INSERT INTO examples VALUES(?)", [(i,) for i in range(rows * 2)])
    db.execute("INSERT INTO meta VALUES('next_cycle',?)", (str(cycle + 1),))
    db.commit()
    db.close()


def make_archive(path, database, cycle):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(database, "state_snapshot.sqlite")
        for name, value in {"config.json": {}, "status.json": {"cycle": cycle},
                            "data_summary.json": {"cycle": cycle}}.items():
            archive.writestr(name, json.dumps(value))


class Response:
    def __init__(self, data=None):
        self.data = data
        self.status_code = 200 if data is not None else 404
        self.headers = {"content-type": "application/zip"}

    def iter_content(self, block_size):
        for i in range(0, len(self.data), 73):
            yield self.data[i:i + 73]

    def close(self):
        pass


class SnapshotStreamingTest(unittest.TestCase):
    def test_drive_only_policy_blocks_all_local_data_downloads(self):
        with tempfile.TemporaryDirectory(prefix="drive_only_policy_") as name:
            folder = Path(name)
            (folder / "storage_policy.json").write_text(json.dumps({"local_large_snapshot_downloads_enabled": False}))
            with patch.object(control, "LOCAL", folder), patch.object(control, "execute") as remote:
                with self.assertRaisesRegex(RuntimeError, "disabled by the user"):
                    control.snapshot(object())
                remote.assert_not_called()

    def test_archive_and_sqlite_are_streamed_and_old_state_is_retained(self):
        with tempfile.TemporaryDirectory(prefix="snapshot_streaming_") as name:
            folder = Path(name)
            old_db = folder / "state_snapshot.sqlite"
            make_database(old_db, 1, 2)
            old_zip = folder / "data_snapshot.zip"
            make_archive(old_zip, old_db, 1)
            old_zip_sha = control.file_sha(old_zip)
            new_db = folder / "remote.sqlite"
            make_database(new_db, 2, 3)
            incoming = folder / "remote.zip"
            make_archive(incoming, new_db, 2)
            payload = incoming.read_bytes()
            meta = {"cycle": 2, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                    "path": "outputs/test/recovery/data_snapshot.zip",
                    "scope": "DATA_AND_METADATA_ONLY_NO_MODEL_WEIGHTS",
                    "sqlite_uncompressed_bytes": new_db.stat().st_size}

            class HTTP:
                def get(self, url, **kwargs):
                    return Response(payload if "recovery/data_snapshot.zip" in url else None)

            class Runtime:
                base_url = "https://test.colab.dev/"
                http = HTTP()

            renderer = subprocess.CompletedProcess([], 0, stdout='{"rendered":true}')
            with patch.object(control, "LOCAL", folder), patch.object(control, "execute", return_value=meta), \
                    patch.object(control.subprocess, "run", return_value=renderer), \
                    patch.object(Path, "read_bytes", side_effect=AssertionError("Whole-file read is not allowed")):
                result = control.snapshot(Runtime())

            self.assertEqual(control.file_sha(folder / "data_snapshot_cycle1.zip"), old_zip_sha)
            self.assertEqual(control.file_sha(folder / "data_snapshot.zip"), meta["sha256"])
            self.assertTrue((folder / "state_snapshot_cycle1.sqlite").exists())
            self.assertEqual(result["snapshot"]["database_integrity"], "ok")
            self.assertEqual(result["snapshot"]["local_counts"], {"pages": 3, "examples": 6})
            old = sqlite3.connect(folder / "state_snapshot_cycle1.sqlite")
            self.assertEqual(old.execute("SELECT COUNT(*) FROM sources").fetchone()[0], 2)
            old.close()


if __name__ == "__main__":
    unittest.main()
