"""Causal pause data: never place the full current utterance in context.

Scicom en/zh read-speech pauses are training material, NOT conversational gold.
Previously exposed LiveKit turns are split by original conversation and explicitly
reclassified as development data. No independent-test claim uses those rows.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import re
import time

import numpy as np
import requests

from .common import digest, file_hash, input_id, normalize, read_jsonl, write_json, write_jsonl

REPO = "Scicom-intl/semantic-vad-eot"
REVISION = "98d1445eab71e4084ccb838dbd87a205f6f2e293"


def causal_cases(row, source, allow_history=False):
    lang = row["language"]
    spans = row["silence_spans"]
    for index, span in enumerate(spans):
        if float(span["end"]) - float(span["start"]) < .2 - 1e-6:continue
        timestamp = float(span["start"]) + .2
        words = [w["word"] for w in row.get("words", []) if float(w["end"]) <= timestamp + 1e-8]
        text = ("" if lang == "zh" else " ").join(words)
        if not normalize(text):continue
        # For read speech messages contains the FULL current turn: MUST ignore.
        context = " ".join(str(m.get("content", "")) for m in row.get("messages", [])[-2:]) if allow_history else ""
        yield {"text": text, "context": context, "language": lang,
            "label": "EOT" if index == len(spans) - 1 else "NEOT", "source_id": row["id"],
            "span_index": index, "timestamp_seconds": timestamp, "source": source,
            "label_source": "final_observed_span_vs_prior_span", "score_point_seconds": .2,
            "current_turn_full_message_excluded": not allow_history}


def acquire(folder):
    folder.mkdir(parents=True, exist_ok=True)
    metadata = requests.get("https://huggingface.co/api/datasets/" + REPO, timeout=40)
    metadata.raise_for_status(); assert metadata.json()["sha"] == REVISION
    jobs = []
    for lang, total, page_count in [("en", 268249, 80), ("zh", 4821, 49)]:
        offsets = sorted(set(int(v) for v in np.linspace(0, total - 100, page_count)))
        for offset in offsets:
            jobs.append((lang, offset, min(100, total - offset)))
    def fetch(job):
        lang, offset, size = job; path = folder / f"{lang}_{offset:07d}.json"
        if path.exists():return {"path": str(path), "cached": True}
        http = requests.Session()
        time.sleep(1.)
        response = http.get("https://datasets-server.huggingface.co/rows", params={"dataset": REPO, "config": lang, "split": "train", "offset": offset, "length": size}, timeout=70)
        if response.status_code == 429:
            write_json(folder / "rate_limit.json", {"status": 429, "retry_after": response.headers.get("Retry-After"), "timestamp": time.time()})
            time.sleep(max(60., float(response.headers.get("Retry-After", "60"))))
            response = http.get("https://datasets-server.huggingface.co/rows", params={"dataset": REPO, "config": lang, "split": "train", "offset": offset, "length": size}, timeout=70)
        response.raise_for_status(); data = response.json()
        rows = [{k: item["row"][k] for k in ["id", "language", "duration", "silence_spans", "words", "messages"]} for item in data["rows"]]
        write_json(path, rows)
        return {"path": str(path), "rows": len(rows), "sha256": file_hash(path)}
    receipts = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        for i, receipt in enumerate(pool.map(fetch, jobs), 1):
            receipts.append(receipt)
            write_json(folder / "progress.json", {"pages": i, "total": len(jobs)})
            if i % 20 == 0:print(json.dumps({"pages": i, "total": len(jobs)}), flush=True)
    after = requests.get("https://huggingface.co/api/datasets/" + REPO, timeout=40)
    after.raise_for_status(); assert after.json()["sha"] == REVISION
    write_json(folder / "sources.json", {"dataset": REPO, "revision_before_after": REVISION, "license": "CC BY 4.0 (dataset card)",
        "scope": "read speech; pause labels from timings; not spontaneous human conversational gold", "files": receipts})


def acquire_parquet_metadata(folder):
    """Read only selected Parquet columns via range requests, excluding audio."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    folder.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem(token=False)
    prefix = "datasets/" + REPO + "@" + REVISION
    jobs = [("en", prefix + "/data/ml-english-00000.parquet")]
    jobs += [("zh", path) for path in sorted(fs.glob(prefix + "/data/*mandarin*.parquet"))[:12]]
    def fetch(item):
        lang, remote = item
        output = folder / (lang + "_" + Path(remote).stem + ".json")
        if not output.exists():
            localfs = HfFileSystem(token=False)
            with localfs.open(remote, "rb", block_size=65536) as stream:
                parquet = pq.ParquetFile(stream)
                table = parquet.read(columns=["id", "language", "duration", "silence_spans", "words", "messages"])
            write_json(output, table.to_pylist())
        return {"path": str(output), "source_parquet": remote, "rows": len(json.loads(output.read_text(encoding="utf-8"))), "sha256": file_hash(output)}
    receipts = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i, result in enumerate(pool.map(fetch, jobs), 1):
            receipts.append(result)
            write_json(folder / "progress.json", {"files": i, "total": len(jobs)})
            print(json.dumps({"metadata_files": i, "total": len(jobs), "rows": result["rows"]}), flush=True)
    write_json(folder / "sources.json", {"dataset": REPO, "revision": REVISION, "files": receipts,
        "audio_columns_read": False, "method": "Pinned Parquet selected columns through range reads", "license": "CC BY 4.0 dataset card",
        "scope": "Read speech pause data, not spontaneous conversational gold"})


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    splits = {k: [] for k in ["train", "dev", "sealed"]}
    raw = {}
    for path in sorted(source.glob("??_*.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):raw[row["id"]] = row
    for key, row in raw.items():
        bucket = int(digest(key)[:8], 16) % 100
        split = "train" if bucket < 80 else "dev" if bucket < 90 else "sealed"
        splits[split].extend(causal_cases(row, "scicom_read_speech", False))
    # Previously observed LiveKit is now development; retain original session
    # groups so two user turns from one call cannot leak across train/dev.
    for line in Path("data_v3/livekit_eot_20261024/turns.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        session = row["id"].split("__user_turn_")[0]
        split = "train" if int(digest(session)[:8], 16) % 100 < 65 else "dev"
        for case in causal_cases(row, "livekit_PREVIOUSLY_EXPOSED_development", True):
            case["session_group"] = session
            splits[split].append(case)
    # VI rehearsal labels are legacy derived text, no real-pause equivalence.
    rng = random.Random(20261025)
    for split, source_split, n in [("train", "train", 30000), ("dev", "dev", 1200)]:
        values = list(read_jsonl(Path("data_v2/frozen") / (source_split + ".jsonl")))
        for label, size in [("EOT", n), ("NEOT", n * 2)]:
            for row in rng.sample([r for r in values if r["label"] == label], size):
                splits[split].append({**row, "language": "vi", "source": "vi_legacy_rehearsal", "label_source": "legacy_prefix_proxy"})
    # Do not filter ambiguous hard cases by model prediction or confidence.
    seen = set(); duplicates = Counter()
    for split in ["sealed", "dev", "train"]:
        values = []
        for row in splits[split]:
            identity = row["language"] + input_id(row)
            if identity in seen:duplicates[split] += 1;continue
            seen.add(identity);values.append(row)
        rng.shuffle(values); splits[split] = values
        write_jsonl(output / (split + ".jsonl"), values)
    assert all(any(r["language"] == lang for r in splits[split]) for split in ["train", "dev"] for lang in ["vi", "en", "zh"])
    write_json(output / "manifest.json", {"scope": "DEVELOPMENT_ROUNDS; independent real VI evaluation unavailable",
        "counts": {k: dict(Counter(r["language"] + ":" + r["source"] + ":" + r["label"] for r in v)) for k, v in splits.items()},
        "split_sha256": {k: file_hash(output / (k + ".jsonl")) for k in splits}, "deduplicated": dict(duplicates),
        "causal_contract": "Visible words only; exclude full read-speech messages; current input tail; do not use hidden suffix",
        "sealed_scope": "EN/ZH held read-speech proxy; NO VI spontaneous real holdout", "previously_exposed_livekit": "training/dev explicitly, no fresh test claim",
        "sources_sha256": file_hash(source / "sources.json"), "legacy_holdout_evaluated": False})
    print(json.dumps({"rows": {k: len(v) for k, v in splits.items()}, "languages": {k: dict(Counter(r["language"] for r in v)) for k, v in splits.items()}}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["acquire", "acquire-parquet", "prepare"])
    p.add_argument("--source", type=Path, default=Path("data_v4/scicom_metadata_20261025"))
    p.add_argument("--output", type=Path, default=Path("data_v4/causal_20261025"))
    a = p.parse_args()
    if a.action == "acquire":acquire(a.source)
    elif a.action == "acquire-parquet":acquire_parquet_metadata(a.source)
    else:prepare(a.source, a.output)
