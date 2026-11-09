from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unicodedata

LABEL2ID = {"NEOT": 0, "EOT": 1}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}


def normalize(text: str) -> str:
    """ASR-style text, without a terminal-punctuation shortcut."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return re.sub(r"\s+", " ", text).strip()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def input_id(record: dict) -> str:
    return digest(normalize(record.get("context", "")) + "\0" + normalize(record["text"]))


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                record = json.loads(line)
                if not isinstance(record.get("text"), str):
                    raise ValueError(f"{path}:{number}: text must be a string")
                if not isinstance(record.get("context", ""), str):
                    raise ValueError(f"{path}:{number}: context must be a string")
                yield record


def write_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def file_hash(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encode_pair(tokenizer, context: str, text: str, max_length: int = 256):
    """One shared train/eval/export contract: preserve current text first."""
    if max_length < 3:
        raise ValueError("max_length must be >= 3")
    current = tokenizer.encode(normalize(text), add_special_tokens=False)[:max_length - 3]
    history = tokenizer.encode(normalize(context), add_special_tokens=False)
    budget = max_length - 3 - len(current)
    history = history[-budget:] if budget else []
    ids = [tokenizer.cls_token_id] + history + [tokenizer.sep_token_id] + current + [tokenizer.sep_token_id]
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}
