"""Dataset paths and encoded memory maps."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

ROOT = Path("work")
JOB = ROOT / "full_epoch_v1"
LANGUAGES = ("vi", "en", "zh")
RECIPE = "FULL_EPOCH_CONTIGUOUS_V1"
RESERVE_BYTES = 25 * 1024**3

def now():
    return datetime.now(timezone.utc).isoformat()


def ensure_space(path, extra_bytes=0):
    if shutil.disk_usage(path).free < RESERVE_BYTES + max(0, extra_bytes):
        raise OSError("insufficient Colab disk space including 25 GiB reserve")


def load_cache(folder):
    import numpy as np
    return tuple(np.load(folder / (key + ".npy"), mmap_mode="r")
                 for key in ("ids", "lengths", "labels", "horizons"))
