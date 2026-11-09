from collections import namedtuple
import pytest
from turn_detection import data


def test_disk_reserve_includes_next_checkpoint(tmp_path, monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(data.shutil, "disk_usage", lambda _: usage(100, 0, data.RESERVE_BYTES + 9))
    data.ensure_space(tmp_path, 9)
    with pytest.raises(OSError):
        data.ensure_space(tmp_path, 10)
