"""Pinned dialogue pilot and independent Wikipedia streaming diagnostics.

Dialogue-final turns are observed EOT; interior prefixes are continuation
proxies, not human semantic EOT labels. Official dialogue splits stay separate.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import random
import re
import time
import zipfile

import requests

from .common import digest, file_hash, input_id, normalize, read_jsonl, write_json, write_jsonl

REVISIONS = {"multiwoz": "fe0c8e65cfcd8462bd33c86e35f21addc84ca82b", "crosswoz": "df82c9fdff91b9b130f2d6b89110d3870ba6260e"}
SOURCES = {
    "multiwoz": {"url": "https://github.com/budzianowski/multiwoz", "dataset": "MultiWOZ 2.2", "language": "en", "license": "Dataset attribution CC BY 4.0 per Cambridge repository; repository code MIT", "license_url": "https://www.repository.cam.ac.uk/items/74e8d468-9442-424a-bb3b-1bb88dcb8673"},
    "crosswoz": {"url": "https://github.com/thu-coai/CrossWOZ", "dataset": "CrossWOZ", "language": "zh", "license": "Upstream repository Apache-2.0", "license_url": "https://github.com/thu-coai/CrossWOZ/blob/master/LICENSE"},
}
TITLES = {"vi": ["Hà Nội", "Đà Nẵng", "Ẩm thực Việt Nam", "Giao thông Việt Nam"],
          "en": ["London", "Public transport", "Vietnamese cuisine", "Beijing"],
          "zh": ["北京市", "上海市", "中国菜", "公共交通"]}


def units(text, language):
    if language == "zh":
        return list(re.finditer(r"[\u3400-\u9fff]|[A-Za-z0-9]+|[^\s\w]", text))
    return list(re.finditer(r"\S+", text))


def turn_rows(dataset, dialogue_id, turns, split):
    language = SOURCES[dataset]["language"]
    history = []
    for index, (speaker, text) in enumerate(turns):
        if speaker.lower() not in {"user", "usr"}:
            history.append(text)
            continue
        tokens = units(text, language)
        if not 3 <= len(tokens) <= 100:
            history.append(text)
            continue
        base = {"context": " ".join(history[-2:]), "language": language, "source_id": f"{dataset}:{dialogue_id}",
                "turn_id": index, "upstream_split": split, "source_url": SOURCES[dataset]["url"],
                "source_revision": REVISIONS[dataset], "full_turn_sha256": digest(text)}
        yield {**base, "text": text, "label": "EOT", "label_source": "observed_user_turn_end", "raw_text": text}
        seen = {normalize(text)}
        for ratio in [.5, .8]:
            end = tokens[min(len(tokens) - 2, max(0, int(len(tokens) * ratio) - 1))].end()
            prefix = text[:end].rstrip()
            if normalize(prefix) and normalize(prefix) not in seen:
                seen.add(normalize(prefix))
                yield {**base, "text": prefix, "raw_text": prefix, "label": "NEOT",
                       "label_source": "observed_continuation_prefix_proxy", "observed_suffix_sha256": digest(text[end:]),
                       "label_limit": "Prefix can be semantically complete; observed continuation is not unambiguous semantic incompleteness"}
        history.append(text)


def get_file(http, url, path):
    if path.exists():
        return path.read_bytes()
    response = http.get(url, timeout=90)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return response.content


def acquire(output):
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    urls = []
    for split, count in [("train", 5), ("dev", 2), ("test", 2)]:
        for i in range(1, count + 1):
            relative = f"data/MultiWOZ_2.2/{split}/dialogues_{i:03d}.json"
            urls.append(("multiwoz", split, f"https://raw.githubusercontent.com/budzianowski/multiwoz/{REVISIONS['multiwoz']}/{relative}", raw / f"en_{split}_{i:03d}.json"))
    for split, upstream in [("train", "train"), ("dev", "val"), ("test", "test")]:
        urls.append(("crosswoz", split, f"https://raw.githubusercontent.com/thu-coai/CrossWOZ/{REVISIONS['crosswoz']}/data/crosswoz/{upstream}.json.zip", raw / f"zh_{split}.json.zip"))
    def fetch(item):
        dataset, split, url, path = item
        http = requests.Session()
        http.headers["User-Agent"] = "TurnDetectorResearch/0.2 (bounded dialogue pilot with provenance)"
        get_file(http, url, path)
        return {"dataset": dataset, "split": split, "url": url, "path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)}
    with ThreadPoolExecutor(max_workers=6) as executor:
        receipts = list(executor.map(fetch, urls))
    write_json(output / "sources.json", {"datasets": SOURCES, "revisions": REVISIONS, "files": receipts})
    print(json.dumps({"downloaded_files": len(receipts), "bytes": sum(r["bytes"] for r in receipts)}), flush=True)


def load_dialogues(folder, dataset, split):
    if dataset == "multiwoz":
        dialogs = []
        for path in sorted((folder / "raw").glob(f"en_{split}_*.json")):
            for row in json.loads(path.read_text(encoding="utf-8")):
                dialogs.append((row["dialogue_id"], [(t["speaker"], t["utterance"]) for t in row["turns"]]))
        return dialogs
    path = folder / "raw" / f"zh_{split}.json.zip"
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".json"))
        value = json.loads(archive.read(name).decode("utf-8"))
    return [(key, [(t["role"], t["content"]) for t in row["messages"]]) for key, row in value.items()]


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    all_rows = {split: [] for split in ["train", "dev", "test"]}
    counts = {}
    for dataset in SOURCES:
        for split, max_dialogues, max_turns in [("train", 1500, 8000), ("dev", 150, 800), ("test", 150, 800)]:
            dialogues = sorted(load_dialogues(source, dataset, split), key=lambda r: digest(dataset + r[0]))[:max_dialogues]
            rows = []
            positive = 0
            for did, turns in dialogues:
                current = list(turn_rows(dataset, did, turns, split))
                if positive >= max_turns:
                    break
                rows.extend(current)
                positive += sum(r["label"] == "EOT" for r in current)
            all_rows[split].extend(rows)
            counts[dataset + ":" + split] = {"rows": len(rows), "dialogues": len({r["source_id"] for r in rows})}
    # Vietnamese uses only the already frozen TRAIN and DEV, never old holdout.
    rng = random.Random(20261024)
    for split, per_label in [("train", 8000), ("dev", 800)]:
        rows = list(read_jsonl(Path("data_v2/frozen") / (split + ".jsonl")))
        for label, count in [("EOT", per_label), ("NEOT", per_label * 2)]:
            selected = rng.sample([r for r in rows if r["label"] == label], count)
            for row in selected:
                row.update(language="vi", label_source="legacy_provided_label", upstream_split="frozen_" + split)
                all_rows[split].append(row)
    # Exact input overlap: dev/test take priority; removal never creates new rows.
    seen = set()
    rejected = Counter()
    for split in ["test", "dev", "train"]:
        clean = []
        for row in all_rows[split]:
            key = row["language"] + ":" + input_id(row)
            if key in seen:
                rejected[split] += 1
                continue
            seen.add(key)
            clean.append(row)
        rng.shuffle(clean)
        all_rows[split] = clean
        write_jsonl(output / (split + ".jsonl"), clean)
    groups = {split: {r["source_id"] for r in rows if r["language"] != "vi"} for split, rows in all_rows.items()}
    assert not (groups["train"] & groups["dev"] or groups["train"] & groups["test"] or groups["dev"] & groups["test"])
    manifest = {"scope": "TRILINGUAL_STRUCTURAL_PROXY_PILOT; not human-adjudicated semantic EOT benchmark",
        "sources": SOURCES, "source_manifest_sha256": file_hash(source / "sources.json"), "dataset_counts": counts,
        "split_sha256": {split: file_hash(output / (split + ".jsonl")) for split in all_rows},
        "counts": {split: dict(Counter(r["language"] + ":" + r["label"] for r in rows)) for split, rows in all_rows.items()},
        "exact_overlap_removed": dict(rejected), "official_dialogue_split_overlap": 0,
        "old_holdout_evaluated": False, "test_status": "SEALED_UNTIL_FINAL_SELECTION; EN/ZH only; one frozen final evaluation",
        "limitations": ["Interior prefixes can be semantically complete", "Task-oriented written dialogue differs from live ASR", "Vietnamese legacy grouping is heuristic", "No VI fresh test in this pilot"]}
    assert all(any(r["language"] == lang for r in all_rows[split]) for split in ["train", "dev"] for lang in ["vi", "en", "zh"])
    assert all(any(r["language"] == lang for r in all_rows["test"]) for lang in ["en", "zh"])
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


def crawl(output):
    output.mkdir(parents=True, exist_ok=True)
    http = requests.Session()
    http.headers["User-Agent"] = "TurnDetectorResearch/0.2 (bounded streaming punctuation diagnostic)"
    paragraphs, receipts = [], []
    for language, titles in TITLES.items():
        for title in ["|".join(titles)]:
            url = f"https://{language}.wikipedia.org/w/api.php"
            response = http.get(url, params={"action": "query", "format": "json", "formatversion": 2,
                "prop": "extracts", "explaintext": 1, "exintro": 1, "redirects": 1, "titles": title}, timeout=45)
            if response.status_code == 429:
                receipts.append({"language": language, "request_titles": title, "status": 429, "note": "Rate limit respected; no immediate retry"})
                write_json(output / "sources.partial.json", {"pages": receipts})
                continue
            response.raise_for_status()
            payload = response.json()
            for page in payload.get("query", {}).get("pages", []):
                text = page.get("extract", "")
                if len(text) < 100:
                    continue
                record = {"language": language, "title": page["title"], "source_id": f"{language}wiki:{page['pageid']}",
                    "source_url": f"https://{language}.wikipedia.org/?curid={page['pageid']}", "text": text,
                    "revision_note": "API extract retrieved and hashed at experiment time", "sha256": digest(text),
                    "license": "Wikipedia CC BY-SA terms; attribution retained", "label_status": "UNLABELED_PROSE_DIAGNOSTIC"}
                for index, paragraph in enumerate(text.splitlines()):
                    if 60 <= len(paragraph) <= 2200:
                        paragraphs.append({**record, "text": paragraph, "paragraph_index": index})
                receipts.append({k: v for k, v in record.items() if k != "text"})
            write_jsonl(output / "paragraphs.jsonl", paragraphs)
            write_json(output / "sources.partial.json", {"pages": receipts})
            time.sleep(.2)
    assert all(any(r["language"] == lang for r in paragraphs) for lang in ["vi", "en", "zh"])
    write_jsonl(output / "paragraphs.jsonl", paragraphs)
    write_json(output / "sources.json", {"pages": receipts, "paragraphs": len(paragraphs), "scope": "Streaming prose diagnostic only; punctuation is not turn gold"})
    print(json.dumps({"paragraphs": dict(Counter(r["language"] for r in paragraphs))}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["acquire", "prepare", "crawl"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", type=Path, default=Path("data_v3/sources_20261024"))
    a = p.parse_args()
    {"acquire": lambda: acquire(a.output), "prepare": lambda: prepare(a.source, a.output), "crawl": lambda: crawl(a.output)}[a.action]()
