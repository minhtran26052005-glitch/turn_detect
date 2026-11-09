"""Freeze grouped gold splits; never relabel existing evaluation data with a teacher."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .common import digest, file_hash, input_id, normalize, read_jsonl, write_json, write_jsonl


class UnionFind:
    def __init__(self):
        self.parents = {}

    def find(self, value):
        self.parents.setdefault(value, value)
        parent = self.parents[value]
        if parent != value:
            self.parents[value] = self.find(parent)
        return self.parents[value]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


def infer_groups(records, prefix_words=8):
    """Reconstruct observed EOT/prefix families; union long shared prefixes.

    Source IDs win when supplied. This is a documented heuristic for legacy
    rows lacking source IDs, not proof of independence by original dialogue.
    """
    uf, prefix_owner, exact_owner = UnionFind(), {}, {}
    prepared, anchor = [], None
    orphan_count = 0
    for number, original in enumerate(records, 1):
        if original.get("label") not in ("EOT", "NEOT"):
            raise ValueError(f"Gold row {number} needs label EOT or NEOT")
        row = {**original, "text": normalize(original["text"]),
               "context": normalize(original.get("context", "")), "origin_row": number,
               "label_source": "provided_gold", "sample_weight": 1.0}
        words = row["text"].split()
        if not words:
            row["group_id"] = "empty:" + str(number)
            row["quarantine_reason"] = "empty_text"
        elif row.get("source_id"):
            row["group_id"] = "source:" + str(row["source_id"])
        elif row["label"] == "EOT":
            row["group_id"] = "anchor:" + input_id(row)
            anchor = (words, row["group_id"])
        elif anchor and len(words) < len(anchor[0]) and anchor[0][:len(words)] == words:
            row["group_id"] = anchor[1]
        else:
            row["group_id"] = "orphan:" + input_id(row)
            orphan_count += 1
        group = row["group_id"]
        uf.find(group)
        key = input_id(row)
        if key in exact_owner:
            uf.union(group, exact_owner[key])
        else:
            exact_owner[key] = group
        # Connecting all one-word fragments would create a giant component.
        # Long prefixes link plausible siblings without claiming true source IDs.
        if len(words) >= prefix_words:
            key = digest(" ".join(words[:prefix_words]))
            if key in prefix_owner:
                uf.union(group, prefix_owner[key])
            else:
                prefix_owner[key] = group
        prepared.append(row)
    for row in prepared:
        row["group_id"] = uf.find(row["group_id"])
    return prepared, orphan_count


def split_name(group_id, seed, dev_fraction, holdout_fraction):
    bucket = int(digest(f"{seed}:{group_id}")[:16], 16) / 2**64
    return "dev" if bucket < dev_fraction else (
        "holdout" if bucket < dev_fraction + holdout_fraction else "train")


def prepare(train_path: Path, legacy_test: Path, output: Path, seed=42,
            dev_fraction=0.05, holdout_fraction=0.05, prefix_words=8):
    output = Path(output)
    if (output / "manifest.json").exists():
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        expected = {"train": file_hash(train_path), "legacy_test": file_hash(legacy_test)}
        settings = {"seed": seed, "dev_fraction": dev_fraction,
                    "holdout_fraction": holdout_fraction, "prefix_words": prefix_words}
        if manifest["input_sha256"] != expected or manifest["split_settings"] != settings:
            raise ValueError("Frozen data already exists with different inputs/settings; choose a new directory")
        for name, expected_hash in manifest["split_sha256"].items():
            if file_hash(output / f"{name}.jsonl") != expected_hash:
                raise ValueError(f"Frozen {name} split has changed")
        return manifest
    if not 0 < dev_fraction < 0.5 or not 0 < holdout_fraction < 0.5 or dev_fraction + holdout_fraction >= 1:
        raise ValueError("Need train, dev and holdout fractions > 0")
    records, orphans = infer_groups(list(read_jsonl(train_path)), prefix_words)
    legacy = list(read_jsonl(legacy_test))
    legacy_keys = {input_id(row) for row in legacy}
    legacy_prefixes = {digest(" ".join(normalize(row["text"]).split()[:prefix_words]))
                       for row in legacy if len(normalize(row["text"]).split()) >= prefix_words}
    labels = defaultdict(set)
    for row in records:
        labels[input_id(row)].add(row["label"])
    blocked_groups = set()
    for row in records:
        words = row["text"].split()
        if input_id(row) in legacy_keys or (len(words) >= prefix_words and
                digest(" ".join(words[:prefix_words])) in legacy_prefixes):
            blocked_groups.add(row["group_id"])
    splits = {"train": [], "dev": [], "holdout": []}
    quarantine, seen = [], set()
    for row in records:
        key = input_id(row)
        reason = row.get("quarantine_reason")
        if len(labels[key]) > 1:
            reason = "conflicting_labels_for_same_input"
        elif row["group_id"] in blocked_groups:
            reason = "family_overlaps_legacy_test"
        elif key in seen:
            reason = "duplicate_normalized_input"
        if reason:
            quarantine.append({**row, "quarantine_reason": reason})
            continue
        seen.add(key)
        name = split_name(row["group_id"], seed, dev_fraction, holdout_fraction)
        splits[name].append(row)
    for name, rows in splits.items():
        if set(row["label"] for row in rows) != {"EOT", "NEOT"}:
            raise ValueError(f"Split {name} needs both labels")
        write_jsonl(output / f"{name}.jsonl", rows)
    write_jsonl(output / "quarantine.jsonl", quarantine)
    manifest = {"schema": 1, "input_sha256": {"train": file_hash(train_path),
                 "legacy_test": file_hash(legacy_test)},
        "split_settings": {"seed": seed, "dev_fraction": dev_fraction,
                           "holdout_fraction": holdout_fraction, "prefix_words": prefix_words},
        "grouping": "source_id when present; otherwise preceding EOT + its prefixes, union on exact input or first 8 words",
        "grouping_limit": "Legacy rows have no original dialogue IDs; grouping is heuristic, not verified conversation independence.",
        "orphan_neot_before_union": orphans,
        "quarantine_reasons": dict(Counter(row["quarantine_reason"] for row in quarantine)),
        "splits": {name: {"rows": len(rows), "groups": len({row['group_id'] for row in rows}),
                          "labels": dict(Counter(row["label"] for row in rows))}
                   for name, rows in splits.items()},
        "split_sha256": {name: file_hash(output / f"{name}.jsonl") for name in splits},
        "targets": {"accuracy": 0.90, "eot_f1": 0.90},
        "legacy_test_status": "previously exposed benchmark; excluded from training selection and fresh target claims"}
    write_json(output / "manifest.json", manifest)
    return manifest


def augmentation_filter(silver, data_dir: Path, legacy_test: Path):
    """Silver may enter train only; reject protected inputs, groups and prefixes."""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    protected = [*read_jsonl(data_dir / "dev.jsonl"), *read_jsonl(data_dir / "holdout.jsonl"),
                 *read_jsonl(legacy_test)]
    all_gold = [*read_jsonl(data_dir / "train.jsonl"), *protected]
    exact = {input_id(row) for row in all_gold}
    groups = {row.get("source_id") for row in protected if row.get("source_id")}
    words = manifest["split_settings"]["prefix_words"]
    prefixes = {digest(" ".join(normalize(row["text"]).split()[:words]))
                for row in protected if len(normalize(row["text"]).split()) >= words}
    accepted, rejected = [], []
    for original in silver:
        row = {**original, "text": normalize(original["text"]),
               "context": normalize(original.get("context", ""))}
        key, tokens = input_id(row), row["text"].split()
        reason = None
        if row.get("label_source") != "qwen_agreement_silver" or row.get("label") not in ("EOT", "NEOT"):
            reason = "not_accepted_teacher_silver"
        elif key in exact or row.get("source_id") in groups:
            reason = "duplicate_or_protected_source"
        elif len(tokens) >= words and digest(" ".join(tokens[:words])) in prefixes:
            reason = "protected_long_prefix"
        if reason:
            rejected.append({**row, "rejection_reason": reason})
        else:
            exact.add(key)
            accepted.append({**row, "sample_weight": 0.5, "group_id": "silver:" + row["source_id"]})
    return accepted, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("train.jsonl"))
    parser.add_argument("--legacy-test", type=Path, default=Path("test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data_v2/frozen"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.train, args.legacy_test, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
