"""The teacher's punctuation-free, left-tail input contract."""
import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return re.sub(r"\s+", " ", text).strip()


def encode_pair(tokenizer, context: str, future: str, max_length: int = 256):
    if max_length < 3:
        raise ValueError("max_length must be at least 3")
    limit = max_length - 3
    current = tokenizer.encode(normalize(future), add_special_tokens=False)[-limit:] if limit else []
    budget = limit - len(current)
    history = tokenizer.encode(normalize(context), add_special_tokens=False)[-budget:] if budget else []
    ids = [tokenizer.cls_token_id] + history + [tokenizer.sep_token_id] + current + [tokenizer.sep_token_id]
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def validate_future(future: str):
    words = normalize(future).split()
    if len(words) not in (1, 2):
        raise ValueError("future must contain one or two space-separated words")
    return words
