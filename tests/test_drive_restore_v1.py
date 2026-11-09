import hashlib
import io
from pathlib import Path
import sqlite3
import zipfile

import pytest

from turn_v2 import restore_drive_campaign_v1 as restore


def test_zip_traversal_and_existing_state_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(restore, 'CONTROL', tmp_path / 'control')
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('../escape', 'bad')
    with pytest.raises(ValueError):
        restore.extract_archive(archive, tmp_path / 'project')
    assert not (tmp_path / 'escape').exists()
    project = tmp_path / 'project'; project.mkdir()
    original = project / 'existing.txt'; original.write_text('user state')
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('existing.txt', 'different restored bytes')
    with pytest.raises(RuntimeError, match='overwrite'):
        restore.extract_archive(archive, project)
    assert original.read_text() == 'user state'


def test_zip_symlink_is_refused(tmp_path):
    archive = tmp_path / 'symlink.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        item = zipfile.ZipInfo('link'); item.create_system = 3
        item.external_attr = 0o120777 << 16
        z.writestr(item, '/outside')
    with pytest.raises(ValueError, match='symlink'):
        restore.extract_archive(archive, tmp_path / 'project')


def test_historical_training_fingerprint_excludes_later_source_pages():
    db = sqlite3.connect(':memory:')
    db.executescript('CREATE TABLE sources(language,page_id,added_cycle); CREATE TABLE examples(language,split,example_id,source_id,record_json);')
    db.executemany('INSERT INTO sources VALUES(?,?,?)', [('vi', 'a', 72), ('vi', 'b', 75)])
    db.executemany('INSERT INTO examples VALUES(?,?,?,?,?)', [
        ('vi', 'train', '001', 'vi:a', '{"text":"old"}'),
        ('vi', 'train', '002', 'vi:b', '{"text":"new"}'),
        ('vi', 'dev', '003', 'vi:a', '{"text":"dev"}')])
    fingerprint, rows = restore.historical_fingerprint(db, 'vi', 72)
    assert rows == 1
    assert fingerprint == hashlib.sha256(b'{"text":"old"}\n').hexdigest()


def test_corrupt_object_never_publishes_checkpoint(tmp_path, monkeypatch):
    project = tmp_path / 'project'; project.mkdir()
    drive = tmp_path / 'drive'; (drive / 'objects').mkdir(parents=True)
    expected = hashlib.sha256(b'good').hexdigest()
    (drive / 'objects' / (expected + '.bin')).write_bytes(b'evil')
    monkeypatch.setattr(restore, 'PROJECT', project)
    monkeypatch.setattr(restore, 'CONTROL', project / 'control')
    monkeypatch.setattr(restore, 'RUN', project / 'run')
    monkeypatch.setattr(restore, 'RESERVE', 0)
    target = project / 'checkpoint.pt'
    with pytest.raises(RuntimeError, match='hash mismatch'):
        restore.copy_object(drive, {'object': 'objects/' + expected + '.bin', 'sha256': expected, 'bytes': 4}, target)
    assert not target.exists()
    assert target.with_name('checkpoint.pt.restore_partial').read_bytes() == b'evil'
