"""Resume Vietnamese training without waiting for a full token-array cache.

Ordinals follow the same covering-index order as the original cache. The saved
permutation, optimizer, RNG and loss code remain in full_train_v5.
"""
import hashlib
import json
from pathlib import Path
import signal
import sqlite3
import traceback

import numpy as np

from . import translation_segmenter_full_data_v5 as data
from . import translation_segmenter_full_train_v5 as train
from .translation_segmenter_v15c import LABELS, atomic_json


class StreamedTrain:
    def __init__(self, path, expected):
        self.db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=30)
        self.db.execute('PRAGMA cache_size=-131072')
        n = expected['rows']
        self.rowids = np.empty(n, dtype=np.int64)
        self.labels = np.empty(n, dtype=np.int64)
        self.horizons = np.empty(n, dtype=np.int64)
        digest = hashlib.sha256()
        count = 0
        sql = ("SELECT rowid,example_id,input_hash,label,k FROM records INDEXED BY accepted_fingerprint_cover "
               "WHERE language=? AND split=? AND disposition='accepted' ORDER BY example_id")
        for rowid, example_id, input_hash, label, horizon in self.db.execute(sql, ('vi', 'train')):
            if count >= n or label not in LABELS or horizon not in (1, 2):
                raise RuntimeError('Invalid Vietnamese ordinal index')
            self.rowids[count], self.labels[count], self.horizons[count] = rowid, LABELS[label], horizon
            digest.update((example_id + '|' + input_hash + '|' + label + '|' + str(horizon) + '\n').encode())
            count += 1
        if count != n or digest.hexdigest() != expected['sha256']:
            raise RuntimeError('Vietnamese ordered dataset fingerprint does not match the checkpoint')
        self._key = self._ids = self._lengths = None
        self.ids = Ids(self)
        self.lengths = Lengths(self)

    def __len__(self):
        return len(self.rowids)

    def load(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        key = tuple(indices.tolist())
        if key == self._key:
            return
        wanted = self.rowids[indices].tolist()
        if len(set(wanted)) != len(wanted):
            raise RuntimeError('Duplicate sample within a microbatch')
        placeholders = ','.join('?' for _ in wanted)
        rows = self.db.execute(f'SELECT rowid,tokens,input_hash,label,k FROM records WHERE rowid IN ({placeholders})', wanted)
        found = {row[0]: row[1:] for row in rows}
        if len(found) != len(wanted):
            raise RuntimeError('Missing token row')
        tokens, lengths = [], []
        for i, rowid in enumerate(wanted):
            raw, expected_hash, label, horizon = found[rowid]
            if (not raw or len(raw) % 4 or len(raw) > 1024 or
                hashlib.sha256(raw).hexdigest() != expected_hash or
                LABELS.get(label) != self.labels[indices[i]] or horizon != self.horizons[indices[i]]):
                raise RuntimeError('Token row failed its saved checksum or metadata check')
            ids = np.frombuffer(raw, dtype='<u4').astype(np.int32)
            tokens.append(ids)
            lengths.append(len(ids))
        arr = np.zeros((len(tokens), max(lengths)), dtype=np.int32)
        for i, ids in enumerate(tokens):
            arr[i, :len(ids)] = ids
        self._key, self._ids, self._lengths = key, arr, np.asarray(lengths, dtype=np.int64)


class Ids:
    def __init__(self, stream): self.stream = stream
    def __len__(self): return len(self.stream)
    def __getitem__(self, key):
        indices, width = key
        self.stream.load(indices)
        return self.stream._ids[:, width]


class Lengths:
    def __init__(self, stream): self.stream = stream
    def __getitem__(self, indices):
        self.stream.load(indices)
        return self.stream._lengths


def run(manifest, epoch, budget):
    import fcntl
    original_cache = train.cached.cache
    stream = None

    def streamed_cache(m, language, split, *args, **kwargs):
        nonlocal stream
        if language == 'vi' and split == 'train':
            if stream is None:
                train.state('LOADING_VI_ORDERED_INDEX', language='vi', generation=m['generation'])
                path = data.JOB / 'datasets' / m['generation'] / 'dataset.sqlite'
                stream = StreamedTrain(path, m['fingerprints']['vi']['train'])
                train.state('VI_ORDERED_INDEX_READY', language='vi', generation=m['generation'], rows=len(stream))
            return stream.ids, stream.lengths, stream.labels, stream.horizons
        return original_cache(m, language, split, *args, **kwargs)

    train.cached.cache = streamed_cache
    try:
        with (data.JOB / 'GPU_LOCK').open('a') as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
            return train.train_chunk('vi', manifest, epoch, budget)
    finally:
        train.cached.cache = original_cache
        if stream is not None:
            stream.db.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation', required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--budget', type=int, default=train.CHUNK_STEPS)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, train.stop)
    signal.signal(signal.SIGINT, train.stop)
    manifest = json.loads((data.JOB / 'datasets' / args.generation / 'manifest.json').read_text())
    try:
        run(manifest, args.epoch, args.budget)
    except Exception as exc:
        atomic_json(data.JOB / 'train_error_vi_stream.json',
                    {'at_utc': data.now(), 'error': repr(exc), 'traceback': traceback.format_exc()})
        train.state('TRAIN_ERROR_PRESERVED', language='vi', error=repr(exc))
        raise
