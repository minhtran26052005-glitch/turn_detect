"""Continuous multilingual sentence-boundary crawl with durable HF provenance."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import traceback

from . import translation_segmenter_v15c as trainer


LANGUAGES = ("vi", "en", "zh")
ROOT = Path("outputs/translation_segmenter_v15d_20261026")
DB_PATH = ROOT / "corpus.sqlite"
STOP = False
DOCS_PER_SOURCE_CYCLE = 100
WIKI_PAGES_PER_LANGUAGE = 10
MIN_FREE_BYTES = 25 * 1024**3

SOURCES = (
    {"key": "hf:fineweb2:vie_Latn", "language": "vi", "dataset": "HuggingFaceFW/fineweb-2",
     "config": "vie_Latn", "split": "train", "license": "ODC-By-1.0", "kind": "web"},
    {"key": "hf:fineweb:en", "language": "en", "dataset": "HuggingFaceFW/fineweb",
     "config": None, "split": "train", "license": "ODC-By-1.0", "kind": "web"},
    {"key": "hf:industrycorpus2_news_media:en", "language": "en", "dataset": "BAAI/IndustryCorpus2_news_media",
     "config": None, "split": "train", "license": "Apache-2.0", "kind": "news_media"},
    {"key": "hf:fineweb2:cmn_Hani", "language": "zh", "dataset": "HuggingFaceFW/fineweb-2",
     "config": "cmn_Hani", "split": "train", "license": "ODC-By-1.0", "kind": "web"},
    {"key": "hf:industrycorpus2_news_media:zh", "language": "zh", "dataset": "BAAI/IndustryCorpus2_news_media",
     "config": None, "split": "train", "license": "Apache-2.0", "kind": "news_media"},
)


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def status(stage, **fields):
    atomic_json(ROOT / "status.json", {"campaign": "translation_segmenter_v15d",
        "status": "RUNNING", "stage": stage, "heartbeat_utc": now(), "pid": os.getpid(),
        "qwen_judge_enabled": False, "quality_target_certified": False, **fields})


def request_stop(*_):
    global STOP
    STOP = True
    trainer.STOP = True


BaseCorpus = trainer.Corpus


class Corpus(BaseCorpus):
    def __init__(self, path=DB_PATH):
        super().__init__(path)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS crawl_observations(
          source_key TEXT NOT NULL, language TEXT NOT NULL, document_id TEXT NOT NULL,
          content_sha256 TEXT NOT NULL, disposition TEXT NOT NULL, reason TEXT NOT NULL,
          title TEXT NOT NULL, url TEXT NOT NULL, text_chars INTEGER NOT NULL,
          examples INTEGER NOT NULL, first_cycle INTEGER NOT NULL, last_cycle INTEGER NOT NULL,
          seen_count INTEGER NOT NULL DEFAULT 1,
          PRIMARY KEY(source_key,document_id));
        CREATE INDEX IF NOT EXISTS crawl_observations_source
          ON crawl_observations(language,source_key,disposition);
        CREATE TABLE IF NOT EXISTS crawl_rejected_text(
          source_key TEXT NOT NULL, language TEXT NOT NULL, document_id TEXT NOT NULL,
          title TEXT NOT NULL, url TEXT NOT NULL, text TEXT NOT NULL,
          reason TEXT NOT NULL, cycle INTEGER NOT NULL,
          PRIMARY KEY(source_key,document_id));
        CREATE TABLE IF NOT EXISTS source_provenance(
          language TEXT NOT NULL, page_id TEXT NOT NULL, source_key TEXT NOT NULL,
          dataset TEXT NOT NULL, license TEXT NOT NULL, kind TEXT NOT NULL,
          PRIMARY KEY(language,page_id));
        """)
        self.db.commit()

    def meta_get(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def meta_set(self, key, value):
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        self.db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, value))

    def add_crawled_document(self, spec, item, cycle, jieba):
        language, source_key = spec["language"], spec["key"]
        text = str(item.get("text") or item.get("content") or "").strip()
        if not text:
            return "rejected_empty", 0
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        url = str(item.get("url") or item.get("source_url") or
                  f"https://huggingface.co/datasets/{spec['dataset']}")
        upstream_id = str(item.get("_id") or item.get("id") or item.get("url") or text_hash)
        document_id = hashlib.sha256(upstream_id.encode("utf-8")).hexdigest()
        title = str(item.get("title") or text.splitlines()[0][:160] or upstream_id)
        existing = self.db.execute("""SELECT disposition,examples FROM crawl_observations
            WHERE source_key=? AND document_id=?""", (source_key, document_id)).fetchone()
        if existing:
            self.db.execute("""UPDATE crawl_observations SET seen_count=seen_count+1,last_cycle=?
                WHERE source_key=? AND document_id=?""", (cycle, source_key, document_id))
            return "duplicate_id", 0
        duplicate = self.db.execute("""SELECT 1 FROM sources
            WHERE language=? AND text_sha256=? LIMIT 1""", (language, text_hash)).fetchone()
        if duplicate:
            disposition, reason, examples = "duplicate_content", "same text already stored", 0
        elif len(text) < 500:
            disposition, reason, examples = "rejected_short", "text shorter than 500 characters", 0
            self.db.execute("INSERT OR IGNORE INTO crawl_rejected_text VALUES(?,?,?,?,?,?,?,?)",
                (source_key, language, document_id, title, url, text, reason, cycle))
        else:
            page_id = f"{source_key}:{upstream_id}"
            page = {"pageid": page_id, "title": title, "extract": text, "fullurl": url}
            examples = BaseCorpus.add_page(self, language, page, cycle, jieba)
            saved = self.db.execute("SELECT 1 FROM sources WHERE language=? AND page_id=?",
                                    (language, page_id)).fetchone()
            if saved:
                disposition = "accepted" if examples else "saved_no_examples"
                reason = "" if examples else "saved text; no sentence pairs met extraction rules"
                self.db.execute("INSERT OR REPLACE INTO source_provenance VALUES(?,?,?,?,?,?)",
                    (language, page_id, source_key, spec["dataset"], spec["license"], spec["kind"]))
            else:
                disposition, reason = "duplicate_id", "source page id already stored"
                examples = 0
        self.db.execute("""INSERT INTO crawl_observations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,1)""",
            (source_key, language, document_id, text_hash, disposition, reason, title, url,
             len(text), examples, cycle, cycle))
        return disposition, examples

    def crawler_counts(self):
        return [dict(zip(("language", "source_key", "disposition", "documents", "examples"), row))
            for row in self.db.execute("""SELECT language,source_key,disposition,COUNT(*),SUM(examples)
                FROM crawl_observations GROUP BY 1,2,3 ORDER BY 1,2,3""")]


trainer.ROOT = ROOT
trainer.DATA_DIR = ROOT / "data"
trainer.DB_PATH = DB_PATH
trainer.Corpus = Corpus
trainer.status = status


class HFStreams:
    def __init__(self, corpus):
        from datasets import load_dataset
        self.load_dataset = load_dataset
        self.corpus = corpus
        self.loaded = {}
        self.errors = {}

    def get(self, spec):
        key = spec["key"]
        if key in self.loaded:
            return self.loaded[key]
        if self.errors.get(key, 0) > time.time():
            raise RuntimeError("source retry is in backoff")
        status("RESOLVING_HF_STREAM", source=key, dataset=spec["dataset"], config=spec["config"])
        kwargs = {"split": spec["split"], "streaming": True, "token": False}
        if spec["config"]:
            kwargs["name"] = spec["config"]
        dataset = self.load_dataset(spec["dataset"], **kwargs)
        saved = self.corpus.meta_get("hf_state:" + key)
        if saved:
            dataset.load_state_dict(json.loads(saved))
        iterator = iter(dataset)
        self.loaded[key] = (dataset, iterator)
        return self.loaded[key]

    def take(self, spec, cycle, limit, jieba):
        fetched = Counter()
        try:
            dataset, iterator = self.get(spec)
            for index in range(limit):
                if STOP or (ROOT / "STOP_REQUESTED").exists():
                    request_stop(); break
                try:
                    item = next(iterator)
                except StopIteration:
                    self.corpus.meta_set("hf_exhausted:" + spec["key"], "true")
                    break
                disposition, examples = self.corpus.add_crawled_document(spec, item, cycle, jieba)
                fetched[disposition] += 1
                self.corpus.meta_set("hf_state:" + spec["key"], dataset.state_dict())
                self.corpus.db.commit()
                if (index + 1) % 10 == 0:
                    status("CRAWLING_HF_STREAM", cycle=cycle, source=spec["key"],
                        source_progress=index + 1, source_limit=limit,
                        source_counts=self.corpus.crawler_counts(),
                        pages=self.corpus.page_counts(), examples=self.corpus.counts())
            self.errors.pop(spec["key"], None)
        except Exception as exc:
            self.loaded.pop(spec["key"], None)
            delay = min(600, 30 * 2 ** min(self.errors.get(spec["key"] + ":tries", 0), 4))
            self.errors[spec["key"]] = time.time() + delay
            self.errors[spec["key"] + ":tries"] = self.errors.get(spec["key"] + ":tries", 0) + 1
            fetched["error"] += 1
            atomic_json(ROOT / "latest_error.json", {"at_utc": now(), "source": spec["key"],
                "error": repr(exc), "backoff_seconds": delay, "traceback": traceback.format_exc()})
        self.corpus.db.commit()
        return dict(fetched)


def crawl_wikipedia(corpus, cycle, jieba):
    import shutil
    crawler = trainer.WikiCrawler(min_interval=1.5)
    pages_added, examples_added, fetched_counts = Counter(), Counter(), Counter()
    for language in LANGUAGES:
        if STOP: break
        bank = trainer.QUERIES[language]
        query = bank[(cycle - 1) % len(bank)]
        cursor_key = f"search_cursor:{language}:{hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]}"
        stored = corpus.meta_get(cursor_key)
        if stored == "__exhausted__":
            fetched_counts[language + ":exhausted"] += 1; continue
        continuation = json.loads(stored) if stored else {}
        try:
            pages, next_cursor, exhausted = crawler.pages(language, query, WIKI_PAGES_PER_LANGUAGE, continuation)
            for page in pages:
                page_id = str(page.get("pageid", "")); text = str(page.get("extract") or "").strip()
                if not page_id: continue
                spec = {"key": "wiki:" + language, "language": language, "dataset": "Wikipedia API",
                        "license": "CC-BY-SA-4.0", "kind": "encyclopedia"}
                disposition, count = corpus.add_crawled_document(spec,
                    {"id": page_id, "text": text, "title": page.get("title", ""), "url": page.get("fullurl", "")},
                    cycle, jieba)
                fetched_counts[language + ":" + disposition] += 1
                if disposition in ("accepted", "saved_no_examples"):
                    corpus.db.execute("UPDATE source_provenance SET dataset='Wikipedia API',license='CC-BY-SA-4.0',kind='encyclopedia' WHERE source_key=?",
                                      ("wiki:" + language,))
                if disposition == "accepted": pages_added[language] += 1; examples_added[language] += count
                corpus.db.commit()
            corpus.meta_set(cursor_key, "__exhausted__" if exhausted else next_cursor)
            corpus.db.commit()
        except Exception as exc:
            crawler.errors.append({"language": language, "query": query, "error": repr(exc)})
    return {"pages_added": dict(pages_added), "examples_added": dict(examples_added),
            "results": dict(fetched_counts), "errors": crawler.errors}


def run_child(phase, cycle, language, args):
    cmd = [sys.executable, "-u", "-m", "turn_v2.translation_segmenter_v15d", "--phase", phase,
           "--cycle", str(cycle), "--steps", str(args.steps), "--microbatch", str(args.microbatch)]
    if language: cmd += ["--language", language]
    log = ROOT / "cycles" / f"{cycle:06d}" / f"{phase}_{language or 'all'}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT)
        ticks = (Path("/proc") / str(proc.pid) / "stat").read_text().split()[21]
        atomic_json(ROOT / "child.json", {"pid": proc.pid, "start_ticks": ticks, "command": cmd,
                                           "module": "turn_v2.translation_segmenter_v15d", "log": str(log)})
        sent = False
        while proc.poll() is None:
            if (ROOT / "STOP_REQUESTED").exists(): request_stop()
            if STOP and not sent: proc.send_signal(signal.SIGTERM); sent = True
            time.sleep(2)
        if proc.returncode and not STOP:
            raise RuntimeError(f"child failed {phase}/{language}, rc={proc.returncode}, log={log}")


def write_summary(corpus, cycle, hf_stats, wiki_stats):
    summary = {"cycle": cycle, "at_utc": now(), "qwen_used": False,
        "label_source": "terminal punctuation EOT; internal punctuation NEOT",
        "punctuation_in_model_input": False, "added_pages_wikipedia": wiki_stats["pages_added"],
        "added_examples_wikipedia": wiki_stats["examples_added"], "hf_source_batches": hf_stats,
        "source_counts": corpus.crawler_counts(), "page_counts_by_split": corpus.page_counts(),
        "example_counts": corpus.counts(), "wikipedia_errors": wiki_stats["errors"],
        "quality_target_certified": False, "labels_are_human_gold": False}
    atomic_json(ROOT / "data_summary.json", summary)
    atomic_json(ROOT / "cycles" / f"{cycle:06d}" / "crawl_receipt.json", summary)
    return summary


def make_data_snapshot(corpus, cycle):
    recovery = ROOT / "recovery"; recovery.mkdir(parents=True, exist_ok=True)
    db_file = recovery / "state_snapshot.sqlite"
    destination = sqlite3.connect(db_file); corpus.db.backup(destination); destination.close()
    archive = recovery / "data_snapshot.zip"; temp = recovery / "data_snapshot.tmp"
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as output:
        output.write(db_file, "state_snapshot.sqlite")
        for name in ("config.json", "status.json", "data_summary.json"):
            path = ROOT / name
            if path.is_file(): output.write(path, name)
    temp.replace(archive)
    atomic_json(recovery / "latest_snapshot.json", {"cycle": cycle,
        "path": archive.as_posix(), "bytes": archive.stat().st_size, "sha256": file_sha(archive),
        "scope": "DATA_AND_METADATA_ONLY_NO_MODEL_WEIGHTS"})


def train_child(language, cycle, args):
    run_child("train", cycle, language, args)


def orchestrate(args):
    import shutil
    ROOT.mkdir(parents=True, exist_ok=True)
    config = {"campaign": "translation_segmenter_v15d", "languages": list(LANGUAGES),
        "lookahead_words": [1, 2], "punctuation_in_model_input": False, "qwen_judge_enabled": False,
        "docs_per_hf_source_cycle": args.docs_per_source, "wiki_pages_per_language_cycle": WIKI_PAGES_PER_LANGUAGE,
        "datasets": list(SOURCES), "gated_dataset_skipped": "fsnaix/vietnamese-corpus-large; public-card probe returned gated",
        "base_model": trainer.BASE, "base_revision": trainer.REVISION,
        "steps": args.steps, "microbatch": args.microbatch,
        "source_sha256": file_sha(Path(__file__))}
    path = ROOT / "config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != config:
            raise RuntimeError("v15d config differs from saved run; refusing to overwrite existing state")
    else: atomic_json(path, config)
    failures = 0
    with _open_corpus() as corpus:
        streams = HFStreams(corpus)
        jieba = trainer.ensure_jieba()
        while not STOP:
            if (ROOT / "STOP_REQUESTED").exists(): request_stop(); break
            free = shutil.disk_usage(ROOT).free
            if free < MIN_FREE_BYTES:
                status("WAITING_FOR_REMOTE_DISK", free_bytes=free); time.sleep(30); continue
            saved_cycle = corpus.meta_get("next_cycle")
            cycle = int(saved_cycle or 1)
            (ROOT / "cycles" / f"{cycle:06d}").mkdir(parents=True, exist_ok=True)
            try:
                hf_stats = {}
                for spec in SOURCES:
                    if STOP: break
                    if corpus.meta_get("hf_exhausted:" + spec["key"]) == "true":
                        hf_stats[spec["key"]] = {"exhausted": True}; continue
                    hf_stats[spec["key"]] = streams.take(spec, cycle, args.docs_per_source, jieba)
                wiki_stats = crawl_wikipedia(corpus, cycle, jieba)
                corpus.db.commit()
                summary = write_summary(corpus, cycle, hf_stats, wiki_stats)
                status("CRAWL_BATCH_SAVED", cycle=cycle, summary=summary,
                       source_counts=summary["source_counts"], remote_disk_free_bytes=free)
                corpus.meta_set("next_cycle", str(cycle + 1)); corpus.db.commit()
                make_data_snapshot(corpus, cycle)
                if STOP: break
                for language in LANGUAGES:
                    if STOP: break
                    train_child(language, cycle, args)
                if STOP: break
                atomic_json(ROOT / "cycles" / f"{cycle:06d}" / "summary.json", summary)
                status("CYCLE_COMPLETE_CONTINUING", cycle=cycle, summary=summary,
                       next_cycle=cycle + 1, remote_disk_free_bytes=shutil.disk_usage(ROOT).free)
                failures = 0
            except Exception as exc:
                failures += 1
                delay = min(600, 30 * 2 ** min(failures, 4))
                atomic_json(ROOT / "latest_error.json", {"at_utc": now(), "error": repr(exc),
                    "traceback": traceback.format_exc(), "backoff_seconds": delay})
                status("RETRYING_AFTER_ERROR", error=repr(exc), backoff_seconds=delay)
                deadline = time.time() + delay
                while time.time() < deadline and not STOP:
                    if (ROOT / "STOP_REQUESTED").exists(): request_stop()
                    time.sleep(2)
    atomic_json(ROOT / "status.json", {"campaign": "translation_segmenter_v15d",
        "status": "STOPPED_BY_REQUEST" if STOP else "STOPPED", "at_utc": now(),
        "qwen_judge_enabled": False, "preserved_sqlite_and_checkpoints": True})


class _CorpusContext:
    def __enter__(self): self.value = Corpus(); return self.value
    def __exit__(self, *_): self.value.close()


def _open_corpus(): return _CorpusContext()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("run", "train"), default="run")
    parser.add_argument("--language", choices=LANGUAGES)
    parser.add_argument("--cycle", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--docs-per-source", type=int, default=DOCS_PER_SOURCE_CYCLE)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    trainer.STOP = False
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, request_stop)
    if args.phase == "train":
        if not args.language: raise ValueError("--language required for train phase")
        trainer.train_teacher(args.language, args.cycle, args.steps, args.microbatch)
    else:
        orchestrate(args)


if __name__ == "__main__": main()
