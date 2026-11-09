"""Sentence-boundary teachers with 1/2-word lookahead; no LLM judge."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
import zipfile

import requests


LANGUAGES = ("vi", "en", "zh")
LABELS = {"NEOT": 0, "EOT": 1}
ID2LABEL = {value: key for key, value in LABELS.items()}
BASE = "microsoft/mdeberta-v3-base"
REVISION = "a0484667b22365f84929a935b5e50a51f71f159d"
ROOT = Path("outputs/translation_segmenter_v15c_20261026")
DATA_DIR = ROOT / "data"
DB_PATH = ROOT / "corpus.sqlite"
STOP = False
LOOKAHEAD = (1, 2)
TERMINATORS = set(".!?。！？")
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "inc", "co", "no", "e.g", "i.e", "u.s", "u.k", "a.m", "p.m"}
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
QUERIES = {
    "vi": ["khoa học", "công nghệ", "môi trường", "sức khỏe", "lịch sử", "giáo dục", "kinh tế", "văn hóa", "năng lượng", "giao thông", "động vật", "địa lý"],
    "en": ["science", "technology", "environment", "public health", "history", "education", "economics", "culture", "energy", "transport", "animals", "geography"],
    "zh": ["科学", "技术", "环境", "健康", "历史", "教育", "经济", "文化", "能源", "交通", "动物", "地理"],
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest_text(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path: Path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def status(stage: str, **fields):
    atomic_json(ROOT / "status.json", {
        "status": "RUNNING", "stage": stage, "heartbeat_utc": utc_now(),
        "pid": os.getpid(), "campaign": "translation_sentence_boundary_lookahead_v15c",
        "qwen_judge_enabled": False, "student_training_enabled": False,
        "quality_target_certified": False, **fields,
    })


def request_stop(*_):
    global STOP
    STOP = True


class Corpus:
    def __init__(self, path=DB_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS sources(
          language TEXT NOT NULL, page_id TEXT NOT NULL, title TEXT NOT NULL,
          url TEXT NOT NULL, text TEXT NOT NULL, text_sha256 TEXT NOT NULL,
          split TEXT NOT NULL, added_cycle INTEGER NOT NULL,
          PRIMARY KEY(language,page_id));
        CREATE TABLE IF NOT EXISTS examples(
          example_id TEXT PRIMARY KEY, language TEXT NOT NULL, split TEXT NOT NULL,
          label TEXT NOT NULL, lookahead_words INTEGER NOT NULL, source_id TEXT NOT NULL,
          record_json TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS examples_lookup ON examples(language,split,label,lookahead_words);
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        """)
        self.db.commit()

    def close(self):
        self.db.commit(); self.db.close()

    def known(self, language, page_id):
        return self.db.execute("SELECT 1 FROM sources WHERE language=? AND page_id=?", (language, str(page_id))).fetchone() is not None

    def add_page(self, language, page, cycle, jieba):
        page_id = str(page.get("pageid", "")); title = page.get("title", "")
        text = page.get("extract", "").strip(); url = page.get("fullurl", "")
        if not page_id or len(text) < 500 or self.known(language, page_id):
            return 0
        bucket = int(hashlib.sha256(f"{language}:{page_id}".encode()).hexdigest()[:8], 16) % 100
        split = "train" if bucket < 80 else "dev" if bucket < 90 else "test"
        source_id = f"{language}:{page_id}"
        self.db.execute("INSERT INTO sources VALUES(?,?,?,?,?,?,?,?)",
                        (language, page_id, title, url, text, digest_text(text), split, cycle))
        samples = examples_from_page(language, page_id, title, url, text, split, jieba)
        self.db.executemany("INSERT OR IGNORE INTO examples VALUES(?,?,?,?,?,?,?)", [
            (r["example_id"], language, split, r["label"], r["lookahead_words"], source_id,
             json.dumps(r, ensure_ascii=False, separators=(",", ":"))) for r in samples
        ])
        return len(samples)

    def rows(self, language, split):
        return [json.loads(row[0]) for row in self.db.execute(
            "SELECT record_json FROM examples WHERE language=? AND split=? ORDER BY example_id", (language, split))]

    def counts(self):
        return [dict(zip(("language", "split", "label", "lookahead_words", "samples"), row))
                for row in self.db.execute("SELECT language,split,label,lookahead_words,COUNT(*) FROM examples GROUP BY 1,2,3,4 ORDER BY 1,2,3,4")]

    def page_counts(self):
        return {lang: {split: count for split, count in self.db.execute(
            "SELECT split,COUNT(*) FROM sources WHERE language=? GROUP BY split", (lang,))} for lang in LANGUAGES}

    def fingerprint(self, language):
        h = hashlib.sha256()
        for (value,) in self.db.execute("SELECT record_json FROM examples WHERE language=? AND split='train' ORDER BY example_id", (language,)):
            h.update(value.encode("utf-8")); h.update(b"\n")
        return h.hexdigest()


def sentence_spans(text):
    """Return only punctuation-terminated sentences; avoid decimals/abbreviations."""
    start = 0
    for i, char in enumerate(text):
        if char not in TERMINATORS:
            continue
        if char == ".":
            before = text[i - 1] if i else ""
            after = text[i + 1] if i + 1 < len(text) else ""
            if before == "." or after == "." or (before.isdigit() and after.isdigit()):
                continue
            if after and not after.isspace() and after not in '\"”’»)]}':
                continue
            previous = re.search(r"([\w.]+)$", text[start:i], re.UNICODE)
            if previous and previous.group(1).strip(".").lower() in ABBREVIATIONS:
                continue
        end = i + 1
        while end < len(text) and text[end] in '\"”’»)]}':
            end += 1
        if end == len(text) or text[end].isspace() or char in "。！？":
            sent = text[start:end].strip()
            if len(sent) >= 25:
                yield sent
            start = end
            while start < len(text) and text[start].isspace():
                start += 1


def words(text, language, jieba):
    if language == "zh":
        values = jieba.lcut(text, HMM=False)
        result = []
        for value in values:
            cleaned = "".join(ch for ch in value if ch.isalnum() or "\u3400" <= ch <= "\u9fff")
            if cleaned:
                result.append(cleaned)
        return result
    return WORD_RE.findall(text)


def detokenize(values):
    return " ".join(values)


def make_example(language, page_id, paragraph, sentence_index, kind, k, cut, left, right, split, title, url,
                 target_mark=""):
    key = f"{language}|{page_id}|{paragraph}|{sentence_index}|{kind}|{k}|{cut}|{target_mark}"
    example_id = hashlib.sha256(key.encode("utf-8")).hexdigest()
    label = "EOT" if kind == "sentence_boundary" else "NEOT"
    return {
        "example_id": example_id, "language": language, "context": detokenize(left),
        "text": detokenize(right), "label": label, "lookahead_words": k,
        "target_punctuation": target_mark,
        "boundary_type": kind, "source_id": f"{language}:{page_id}",
        "source_title": title, "source_url": url, "paragraph_index": paragraph,
        "sentence_index": sentence_index, "cut_after_word": cut,
        "label_source": "TERMINAL_PUNCTUATION_EOT_INTERNAL_PUNCTUATION_NEOT",
        "punctuation_in_model_input": False,
        "is_human_gold": False,
    }


def terminal_mark(sentence):
    match = re.search(r"([.!?。！？])[\"”’»)\]}]*$", sentence)
    return match.group(1) if match else ""


def examples_from_page(language, page_id, title, url, text, split, jieba):
    result = []
    for paragraph_index, paragraph in enumerate(re.split(r"\n+", text)):
        sents = []
        for sentence in sentence_spans(paragraph):
            mark = terminal_mark(sentence)
            body = re.sub(r"([.!?。！？])[\"”’»)\]}]*$", "", sentence).strip()
            tokens = words(body, language, jieba)
            if len(tokens) >= 5:
                sents.append({"tokens": tokens, "body": body, "terminal_mark": mark})
        if len(sents) < 2:
            continue
        for sent_index, sent in enumerate(sents[:-1]):
            current = sent["tokens"]
            next_words = sents[sent_index + 1]["tokens"]
            for k in LOOKAHEAD:
                if len(current) >= 4 and len(next_words) >= k:
                    result.append(make_example(language, page_id, paragraph_index, sent_index,
                        "sentence_boundary", k, len(current), current, next_words[:k], split, title, url,
                        target_mark=sent["terminal_mark"]))
            # Negative samples are real internal punctuation boundaries.
            # The punctuation is supervision metadata only; input is plain words.
            for match in re.finditer(r"[,，;；:：、]", sent["body"]):
                left = words(sent["body"][:match.start()], language, jieba)
                right = words(sent["body"][match.end():], language, jieba)
                if len(left) < 2:
                    continue
                for k in LOOKAHEAD:
                    if len(right) >= k:
                        result.append(make_example(language, page_id, paragraph_index, sent_index,
                            "internal_punctuation", k, len(left), left, right[:k], split, title, url,
                            target_mark=match.group(0)))
    return result


def ensure_jieba():
    try:
        import jieba
        return jieba
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "jieba==0.42.1"])
        import jieba
        return jieba


class WikiCrawler:
    def __init__(self, min_interval=1.1):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "TurnDetectorTranslationSegmentation/0.1 (bounded Wikipedia sentence corpus)"
        self.min_interval = min_interval
        self.last_request = 0.0
        self.errors = []

    def pages(self, language, query, limit, continuation=None):
        url = f"https://{language}.wikipedia.org/w/api.php"
        params = {
            "action": "query", "generator": "search", "gsrsearch": query,
            "gsrnamespace": 0, "gsrlimit": limit,
            "prop": "extracts|info", "explaintext": 1, "exchars": 12000,
            "inprop": "url", "format": "json", "formatversion": 2, "redirects": 1,
        }
        cursor = dict(continuation or {})
        collected = {}
        calls = 0
        while len(collected) < limit and calls < 10:
            wait = self.min_interval - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            request_params = {**params, **cursor}
            payload = None
            calls += 1
            for attempt in range(5):
                try:
                    response = self.session.get(url, params=request_params, timeout=60)
                    self.last_request = time.monotonic()
                    if response.status_code == 429 or response.status_code >= 500:
                        retry_after = response.headers.get("Retry-After")
                        delay = min(60, float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt * 2)
                        self.errors.append({"language": language, "query": query, "http": response.status_code,
                                            "retry": attempt + 1, "cursor": cursor})
                        time.sleep(delay)
                        continue
                    response.raise_for_status()
                    payload = response.json()
                    break
                except (requests.RequestException, ValueError) as exc:
                    self.errors.append({"language": language, "query": query, "error": str(exc),
                                        "retry": attempt + 1, "cursor": cursor})
                    if attempt < 4:
                        time.sleep(min(60, 2 ** attempt * 2))
            if payload is None:
                return list(collected.values()), cursor, False
            page_values = payload.get("query", {}).get("pages", [])
            if isinstance(page_values, dict):
                page_values = list(page_values.values())
            for page in page_values:
                page_id = str(page.get("pageid", ""))
                if page_id:
                    collected[page_id] = page
            next_cursor = payload.get("continue", {})
            if not next_cursor:
                return list(collected.values())[:limit], {}, True
            cursor = next_cursor
        return list(collected.values())[:limit], cursor, False


def collect_cycle(cycle, queries_per_language=4, pages_per_query=25):
    import shutil
    if shutil.disk_usage(ROOT).free < 25 * 2**30:
        status("WAITING_FOR_REMOTE_DISK", free_GiB=shutil.disk_usage(ROOT).free / 2**30)
        return False
    jieba = ensure_jieba()
    crawler = WikiCrawler(); corpus = Corpus()
    counts_before = corpus.page_counts(); added_pages = Counter(); added_examples = Counter()
    cursor_progress = Counter()
    try:
        for language in LANGUAGES:
            if STOP: break
            bank = QUERIES[language]
            offset = ((cycle - 1) * queries_per_language) % len(bank)
            selected = [bank[(offset + j) % len(bank)] for j in range(queries_per_language)]
            for query in selected:
                if STOP: break
                cursor_key = f"search_cursor:{language}:{hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]}"
                stored = corpus.db.execute("SELECT value FROM meta WHERE key=?", (cursor_key,)).fetchone()
                if stored and stored[0] == "__exhausted__":
                    cursor_progress[language + ":exhausted"] += 1
                    continue
                cursor = json.loads(stored[0]) if stored else {}
                status("CRAWLING_SENTENCE_CORPUS", cycle=cycle, language=language, query=query,
                       qwen_judge_enabled=False, pages_added=dict(added_pages), example_counts=dict(added_examples))
                pages, next_cursor, exhausted = crawler.pages(language, query, pages_per_query, cursor)
                cursor_progress[language + ":pages_fetched"] += len(pages)
                for page in pages:
                    if corpus.known(language, page.get("pageid", "")):
                        continue
                    count = corpus.add_page(language, page, cycle, jieba)
                    if count:
                        added_pages[language] += 1; added_examples[language] += count
                if exhausted:
                    corpus.db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (cursor_key, "__exhausted__"))
                elif next_cursor:
                    corpus.db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
                                      (cursor_key, json.dumps(next_cursor, ensure_ascii=False, sort_keys=True)))
                corpus.db.commit()
        all_counts = corpus.counts(); page_counts = corpus.page_counts()
        receipt = {"cycle": cycle, "at_utc": utc_now(), "queries_per_language": queries_per_language,
                   "pages_per_query": pages_per_query, "added_pages": dict(added_pages),
                   "added_examples": dict(added_examples), "page_counts_by_split": page_counts,
                    "example_counts": all_counts, "http_errors": crawler.errors,
                    "search_cursor_progress": dict(cursor_progress),
                   "label_source": "PUNCTUATION_DERIVED_SENTENCE_BOUNDARY", "qwen_used": False,
                   "split_is_source_group_disjoint": True}
        atomic_json(ROOT / "cycles" / f"{cycle:06d}" / "crawl_receipt.json", receipt)
        atomic_json(ROOT / "data_summary.json", receipt)
        return True
    finally:
        corpus.close()


def load_rows(language, split):
    corpus = Corpus()
    try:
        return corpus.rows(language, split)
    finally:
        corpus.close()


def encode(rows, tokenizer):
    import numpy as np
    from .monolingual_data import encode_tail
    ids = np.full((len(rows), 256), tokenizer.pad_token_id, np.int32)
    lengths = np.zeros(len(rows), np.int64)
    for i, row in enumerate(rows):
        value = encode_tail(tokenizer, row["context"], row["text"], max_length=256)["input_ids"]
        ids[i, :len(value)] = value; lengths[i] = len(value)
    labels = np.array([LABELS[row["label"]] for row in rows], np.int64)
    horizons = np.array([row["lookahead_words"] for row in rows], np.int64)
    return ids, lengths, labels, horizons


def predict(model, encoded, batch_size=48):
    import numpy as np
    import torch
    ids, lengths, _, _ = encoded
    result = np.zeros(len(ids), np.float32); model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(ids), batch_size):
            stop = min(len(ids), start + batch_size); width = int(lengths[start:stop].max())
            x = torch.from_numpy(ids[start:stop, :width].astype(np.int64)).cuda()
            mask = torch.arange(width, device="cuda")[None] < torch.from_numpy(lengths[start:stop]).cuda()[:, None]
            result[start:stop] = model(input_ids=x, attention_mask=mask.long()).logits.float().softmax(-1)[:, 1].cpu().numpy()
    return result


def metrics(labels, probs, threshold):
    from .metrics import classification_metrics
    return classification_metrics(labels, probs, threshold)


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def eval_report(model, encoded, rows, language, cycle, stage):
    import numpy as np
    labels = encoded[2]; probs = predict(model, encoded)
    threshold_candidates = np.linspace(0.05, 0.95, 181)
    candidates = []
    for candidate in threshold_candidates:
        scores = []
        for k in LOOKAHEAD:
            ids = np.array([i for i, row in enumerate(rows) if row["lookahead_words"] == k], dtype=np.int64)
            if len(ids):
                m = metrics(labels[ids], probs[ids], float(candidate))
                scores.extend([m["accuracy"], m["eot_f1"]])
            period_ids = np.array([i for i, row in enumerate(rows)
                                   if row["lookahead_words"] == k and row["label"] == "EOT"
                                   and row.get("target_punctuation") in {".", "。"}], dtype=np.int64)
            internal_ids = np.array([i for i, row in enumerate(rows)
                                     if row["lookahead_words"] == k and row["label"] == "NEOT"
                                     and row.get("target_punctuation") in {",", "，", ";", "；", ":", "：", "、"}], dtype=np.int64)
            if len(period_ids):
                scores.append(float((probs[period_ids] >= candidate).mean()))
            if len(internal_ids):
                scores.append(float((probs[internal_ids] < candidate).mean()))
        score = min(scores) if scores else 0.0
        false_eot = int(((labels == 0) & (probs >= candidate)).sum())
        candidates.append((score, -false_eot, -abs(float(candidate) - 0.5), float(candidate)))
    selection_score, _, _, threshold = max(candidates)
    calibrated = metrics(labels, probs, threshold)
    by_horizon = {}
    for k in LOOKAHEAD:
        idx = np.flatnonzero(encoded[3] == k)
        by_horizon[str(k)] = metrics(labels[idx], probs[idx], threshold)
    period_idx = np.array([i for i, row in enumerate(rows)
                           if row["label"] == "EOT" and row.get("target_punctuation") in {".", "。"}], dtype=np.int64)
    ascii_period_idx = np.array([i for i, row in enumerate(rows)
                                 if row["label"] == "EOT" and row.get("target_punctuation") == "."], dtype=np.int64)
    fullstop_idx = np.array([i for i, row in enumerate(rows)
                             if row["label"] == "EOT" and row.get("target_punctuation") in {".", "。"}], dtype=np.int64)
    period_eot_recall = float((probs[period_idx] >= threshold).mean()) if len(period_idx) else None
    ascii_period_eot_recall = float((probs[ascii_period_idx] >= threshold).mean()) if len(ascii_period_idx) else None
    fullstop_eot_recall = float((probs[fullstop_idx] >= threshold).mean()) if len(fullstop_idx) else None
    period_recall_by_horizon = {}
    for k in LOOKAHEAD:
        idx = np.array([i for i in period_idx if rows[i]["lookahead_words"] == k], dtype=np.int64)
        period_recall_by_horizon[str(k)] = {
            "targets": int(len(idx)),
            "eot_recall": float((probs[idx] >= threshold).mean()) if len(idx) else None,
        }
    internal_idx = np.array([i for i, row in enumerate(rows)
                             if row["label"] == "NEOT" and row.get("target_punctuation") in
                             {",", "，", ";", "；", ":", "：", "、"}], dtype=np.int64)
    internal_no_eot_rate = float((probs[internal_idx] < threshold).mean()) if len(internal_idx) else None
    value = {"language": language, "cycle": cycle, "stage": stage, "dev_samples": len(rows),
             "threshold_selected_on_dev": threshold, "dev_accuracy": calibrated["accuracy"],
             "dev_eot_f1": calibrated["eot_f1"], "by_lookahead_words": by_horizon,
             "terminal_period_eot_recall": period_eot_recall,
             "ascii_dot_eot_recall": ascii_period_eot_recall,
             "terminal_period_eot_recall_by_lookahead": period_recall_by_horizon,
             "terminal_fullstop_eot_recall_including_zh_period": fullstop_eot_recall,
             "terminal_period_targets": len(period_idx), "terminal_fullstop_targets": len(fullstop_idx),
             "internal_punctuation_no_eot_rate": internal_no_eot_rate,
             "internal_punctuation_targets": int(len(internal_idx)),
             "selection_score_min_accuracy_f1_period_recall_and_internal_punctuation_specificity": selection_score,
             "test_evaluated": False, "quality_target_certified": False,
             "label_scope": "TERMINAL_PUNCTUATION_EOT_INTERNAL_PUNCTUATION_NEOT_NOT_SPEECH_GOLD",
             "punctuation_in_model_input": False}
    return value, probs


def train_teacher(language, cycle, steps=100, microbatch=32):
    import numpy as np
    import torch
    import transformers.utils as hf_utils
    import transformers.utils.import_utils as hf_imports
    hf_utils.is_torchvision_available = hf_imports.is_torchvision_available = lambda: False
    hf_utils.is_librosa_available = hf_imports.is_librosa_available = lambda: False
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "Connected Colab BF16 GPU required"
    torch.set_num_threads(2); torch.manual_seed(20261026 + LANGUAGES.index(language))
    corpus = Corpus()
    try:
        train = corpus.rows(language, "train"); dev = corpus.rows(language, "dev")
        train_fingerprint = corpus.fingerprint(language)
    finally:
        corpus.close()
    def enough(rows):
        counts = Counter((r["label"], r["lookahead_words"]) for r in rows)
        return all(counts[(label, k)] >= 100 for label in LABELS for k in LOOKAHEAD)
    if not enough(train) or not enough(dev):
        atomic_json(ROOT / "teachers" / language / "deferred.json", {
            "cycle": cycle, "train_samples": len(train), "dev_samples": len(dev),
            "reason": "Need >=100 train and dev examples for each label x lookahead horizon", "qwen_used": False})
        return

    folder = ROOT / "teachers" / language; folder.mkdir(parents=True, exist_ok=True)
    status("LOADING_PRETRAINED_SEGMENTER", cycle=cycle, language=language, qwen_judge_enabled=False,
           train_samples=len(train), dev_samples=len(dev), microbatch=microbatch)
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, revision=REVISION, num_labels=2, label2id=LABELS,
        id2label={0: "NEOT", 1: "EOT"}, dtype=torch.float32).cuda()
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW([
        {"params": [p for n,p in model.named_parameters() if not n.startswith(("classifier.", "pooler."))], "lr": 1e-5},
        {"params": [p for n,p in model.named_parameters() if n.startswith(("classifier.", "pooler."))], "lr": 5e-5},
    ], weight_decay=0.01)
    train_x = encode(train, tokenizer); dev_x = encode(dev, tokenizer)
    rng = np.random.default_rng(20261026 + cycle + LANGUAGES.index(language))
    checkpoint = folder / "latest_resume.pt"; total_steps = 0; start_step = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True); optimizer.load_state_dict(saved["optimizer"])
        total_steps = int(saved["total_steps"])
        if saved.get("cycle") == cycle:
            assert saved.get("training_fingerprint") == train_fingerprint, "Dataset changed inside an interrupted cycle"
            start_step = int(saved["cycle_step"]); rng.bit_generator.state = saved["numpy_rng"]
            torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state(saved["cuda_rng"])
        del saved; gc.collect()
    identity = {"language":language,"base":BASE,"revision":REVISION,
                "parameters":sum(p.numel() for p in model.parameters()),
                "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
                "task":"sentence boundary before 1/2 future words","label_order":ID2LABEL,
                "input":"punctuation-free words: [CLS] left text [SEP] 1-or-2 future words [SEP]",
                "target":"EOT at terminal punctuation; NEOT at internal punctuation",
                "punctuation_in_model_input":False,"qwen_judge_used":False,
                "source_sha256":file_sha(Path(__file__))}
    atomic_json(folder / "identity.json", identity)

    def save_resume(step):
        tmp = folder / "latest_resume.tmp"
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"cycle":cycle,
                    "cycle_step":step,"total_steps":total_steps,"training_fingerprint":train_fingerprint,
                    "numpy_rng":rng.bit_generator.state,"torch_rng":torch.get_rng_state(),
                    "cuda_rng":torch.cuda.get_rng_state()},tmp)
        tmp.replace(checkpoint)
        atomic_json(folder / "resume_receipt.json", {"cycle":cycle,"cycle_step":step,"total_steps":total_steps,
                    "training_fingerprint":train_fingerprint,"bytes":checkpoint.stat().st_size,
                    "weights_and_optimizer_remote_only":True,"checkpoint":str(checkpoint)})

    if not (folder / "pretrained_baseline.json").exists():
        baseline, probs = eval_report(model, dev_x, dev, language, cycle, "pretrained_baseline")
        baseline["classification_head"] = "NEW_RANDOM_SENTENCE_BOUNDARY_HEAD"
        atomic_json(folder / "pretrained_baseline.json", baseline)
    pools = {(label,k): np.array([i for i,r in enumerate(train)
             if r["label"]==label and r["lookahead_words"]==k],dtype=np.int64)
             for label in LABELS for k in LOOKAHEAD}
    torch.cuda.reset_peak_memory_stats()
    accumulation = 2
    for step in range(start_step + 1, steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses=[]
        for _ in range(accumulation):
            per_group = max(1, microbatch // 4)
            indices=np.concatenate([rng.choice(pools[key],per_group,replace=True) for key in pools]);rng.shuffle(indices)
            width=int(train_x[1][indices].max())
            ids=torch.from_numpy(train_x[0][indices,:width].astype(np.int64)).cuda()
            mask=torch.arange(width,device="cuda")[None] < torch.from_numpy(train_x[1][indices]).cuda()[:,None]
            labels=torch.from_numpy(train_x[2][indices]).cuda()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits=model(input_ids=ids,attention_mask=mask.long()).logits.float()
                loss=torch.nn.functional.cross_entropy(logits,labels,label_smoothing=.01)/accumulation
            assert torch.isfinite(loss);loss.backward();losses.append(float(loss.detach())*accumulation)
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);assert torch.isfinite(grad)
        warmup=min(1.0,(total_steps+1)/30)
        for group,base_lr in zip(optimizer.param_groups,[1e-5,5e-5]):group["lr"]=base_lr*warmup
        optimizer.step();total_steps+=1
        if step%10==0 or step==steps:
            state={"cycle":cycle,"language":language,"step":step,"steps":steps,"total_steps":total_steps,
                   "loss":float(np.mean(losses)),"train_samples":len(train),"dev_samples":len(dev),
                   "parameters":identity["parameters"],"GPU_peak_GiB":torch.cuda.max_memory_allocated()/2**30,
                   "qwen_judge_enabled":False}
            atomic_json(folder/"progress.json",state);status("TRAINING_LOOKAHEAD_SEGMENTER",**state)
            print(json.dumps(state),flush=True)
        if step%25==0 or step==steps or STOP:save_resume(step)
        if STOP:
            del model,optimizer;gc.collect();torch.cuda.empty_cache();return
    report, probs = eval_report(model,dev_x,dev,language,cycle,"trained")
    threshold = report["threshold_selected_on_dev"]
    trace = []
    for row, prob in zip(dev, probs):
        predicted = "EOT" if float(prob) >= threshold else "NEOT"
        trace.append({**row, "p_eot": float(prob), "predicted": predicted,
                      "correct": predicted == row["label"],
                      "punctuation_in_model_input": False})
    write_jsonl(folder / "dev_trace.jsonl", trace)
    old = json.loads((folder/"best_report.json").read_text()) if (folder/"best_report.json").exists() else None
    score_key = "selection_score_min_accuracy_f1_period_recall_and_internal_punctuation_specificity"
    if old is None or report[score_key] > old[score_key]:
        model.save_pretrained(folder/"best",safe_serialization=True);tokenizer.save_pretrained(folder/"best")
        report["weights_sha256"]=file_sha(folder/"best/model.safetensors");report["path"]=str(folder/"best")
        atomic_json(folder/"best_report.json",report)
    atomic_json(folder/"last_training_report.json",report)
    atomic_json(ROOT/"cycles"/f"{cycle:06d}"/(language+"_dev.json"),report)
    del model,optimizer,tokenizer;gc.collect();torch.cuda.empty_cache()


def snapshot(corpus, cycle):
    folder=ROOT/"recovery";folder.mkdir(parents=True,exist_ok=True)
    db_snapshot=folder/"state_snapshot.sqlite"
    dst=sqlite3.connect(db_snapshot);corpus.db.backup(dst);dst.close()
    archive=folder/"metadata_snapshot.zip";tmp=folder/"metadata_snapshot.tmp"
    paths=[db_snapshot,ROOT/"config.json",ROOT/"status.json",ROOT/"data_summary.json"]
    paths+=list((ROOT/"teachers").glob("*/*.json"))+list((ROOT/"teachers").glob("*/best/*.json"))
    paths+=list((ROOT/"teachers").glob("*/dev_trace.jsonl"))
    paths += [Path(__file__)]
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            if p.is_file():z.write(p,str(p))
    tmp.replace(archive)
    atomic_json(folder/"latest_snapshot.json",{"cycle":cycle,"path":str(archive),"bytes":archive.stat().st_size,
        "sha256":file_sha(archive),"scope":"METADATA_ONLY_NOT_A_FULL_TEACHER_RECOVERY",
        "teacher_weights_backed_up_externally":False})


def run_child(phase, cycle, language, args):
    cmd=[sys.executable,"-u","-m","turn_v2.translation_segmenter_v15c","--phase",phase,
         "--cycle",str(cycle),"--steps",str(args.steps),"--microbatch",str(args.microbatch),
         "--queries-per-language",str(args.queries_per_language),"--pages-per-query",str(args.pages_per_query)]
    if language:cmd += ["--language",language]
    name=language or "crawl";log=ROOT/"cycles"/f"{cycle:06d}"/(name+".log")
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("a",encoding="utf-8") as stream:
        proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
        atomic_json(ROOT/"child.json",{"pid":proc.pid,"command":cmd,"start_ticks":(Path("/proc")/str(proc.pid)/"stat").read_text().split()[21],"log":str(log)})
        sent=False
        while proc.poll() is None:
            if (ROOT/"STOP_REQUESTED").exists():request_stop()
            if STOP and not sent:proc.send_signal(signal.SIGTERM);sent=True
            time.sleep(2)
        if proc.returncode and not STOP:raise RuntimeError(f"child failed: {phase}/{language} rc={proc.returncode}; see {log}")


def orchestrate(args):
    import shutil
    ROOT.mkdir(parents=True,exist_ok=True)
    config={"base":BASE,"revision":REVISION,"languages":list(LANGUAGES),"qwen_judge_enabled":False,
            "task":"translation sentence boundary with 1/2 future words",
            "target_definition":"EOT at sentence-final punctuation; NEOT at internal punctuation; punctuation is label metadata only",
            "punctuation_in_model_input":False,
            "split":"page-grouped 80/10/10 train/dev/test; test not used for checkpoint selection",
            "lookahead_words":list(LOOKAHEAD),"test_used_for_selection":False,
            "steps_per_language_per_cycle":args.steps,"microbatch":args.microbatch,
            "queries_per_language":args.queries_per_language,"pages_per_query":args.pages_per_query,
            "source_sha256":file_sha(Path(__file__))}
    if (ROOT/"config.json").exists():
        old=json.loads((ROOT/"config.json").read_text())
        assert old==config,"Config changed; start a versioned run rather than silently resuming"
    else:atomic_json(ROOT/"config.json",config)
    failures=0
    while not STOP:
        if (ROOT/"STOP_REQUESTED").exists():request_stop();break
        if shutil.disk_usage(ROOT).free < 25*2**30:
            status("WAITING_FOR_REMOTE_DISK",free_GiB=shutil.disk_usage(ROOT).free/2**30);time.sleep(30);continue
        corpus=Corpus()
        saved_cycle=corpus.db.execute("SELECT value FROM meta WHERE key='next_cycle'").fetchone()
        cycle=int(saved_cycle[0]) if saved_cycle else 1
        (ROOT/"cycles"/f"{cycle:06d}").mkdir(parents=True,exist_ok=True)
        try:
            # The collector is deterministic and uses public sentence punctuation only.
            run_child("collect",cycle,None,args)
            if STOP:break
            for language in LANGUAGES:
                if STOP:break
                run_child("train",cycle,language,args)
            if STOP:break
            summary={"cycle":cycle,"qwen_judge_enabled":False,"source_pages":corpus.page_counts(),
                     "example_counts":corpus.counts(),"languages":{},"quality_target_certified":False}
            for language in LANGUAGES:
                path=ROOT/"teachers"/language/"last_training_report.json"
                if path.exists():summary["languages"][language]=json.loads(path.read_text())
            atomic_json(ROOT/"cycles"/f"{cycle:06d}"/"summary.json",summary)
            atomic_json(ROOT/"status.json",{"status":"CYCLE_COMPLETE_CONTINUING","cycle":cycle,
                "at_utc":utc_now(),"qwen_judge_enabled":False,"summary":summary})
            corpus.db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('next_cycle',?)",(str(cycle+1),))
            corpus.db.commit()
            snapshot(corpus,cycle);failures=0
        except Exception as exc:
            failures+=1
            atomic_json(ROOT/"latest_error.json",{"at_utc":utc_now(),"error":str(exc),"traceback":traceback.format_exc()})
            status("RETRYING_AFTER_ERROR",error=str(exc),backoff_seconds=min(600,30*2**min(failures,4)))
            deadline=time.time()+min(600,30*2**min(failures,4))
            while time.time()<deadline and not STOP:
                if (ROOT/"STOP_REQUESTED").exists():request_stop()
                time.sleep(2)
        finally:
            corpus.close()
    atomic_json(ROOT/"status.json",{"status":"STOPPED_BY_REQUEST" if STOP else "STOPPED",
        "at_utc":utc_now(),"qwen_judge_enabled":False,"preserved_checkpoints":True})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=["run","collect","train"],default="run")
    parser.add_argument("--language",choices=LANGUAGES)
    parser.add_argument("--cycle",type=int,default=1)
    parser.add_argument("--steps",type=int,default=100)
    parser.add_argument("--microbatch",type=int,default=32)
    parser.add_argument("--queries-per-language",type=int,default=4)
    parser.add_argument("--pages-per-query",type=int,default=25)
    args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,request_stop)
    if args.phase=="run":orchestrate(args)
    elif args.phase=="collect":collect_cycle(args.cycle,args.queries_per_language,args.pages_per_query)
    else:
        assert args.language
        train_teacher(args.language,args.cycle,args.steps,args.microbatch)


if __name__=="__main__":main()
