"""One held-out prose evaluation on Colab, with immutable inputs/checkpoints.

This companion waits for all current teachers, leaves campaign training alone,
and uses only thresholds already selected on dev. No test result is fed back to
the trainer. Full predictions remain on Colab; a compact review can be fetched.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time
import traceback

ROOT = Path("outputs/translation_segmenter_v15d_20261026")
JOB = ROOT / "heldout_test_v2"
LANGUAGES = ("vi", "en", "zh")
MIN_FREE = 25 * 1024**3
LAST_STATUS = 0.0


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def status(stage, **fields):
    global LAST_STATUS
    LAST_STATUS = time.monotonic()
    atomic_json(JOB / "status.json", {"stage": stage, "at_utc": now(), **fields})


def process_matches(record):
    try:
        argv = record["command"]
        module = argv[argv.index("-m") + 1]
        proc = Path("/proc") / str(record["pid"])
        return (module.encode() + b"\0" in (proc / "cmdline").read_bytes()
                and (proc / "stat").read_text().split()[21] == record["start_ticks"])
    except (OSError, KeyError, ValueError, IndexError):
        return False


def training_active():
    try:
        child = read_json(ROOT / "child.json")
        argv = child.get("command", [])
        return process_matches(child) and "--phase" in argv and argv[argv.index("--phase") + 1] == "train"
    except (OSError, ValueError, IndexError):
        return False


def wait_for_gpu(model=None, language=None):
    """Release VRAM promptly when the campaign starts a training child."""
    paused = False
    while training_active():
        if model is not None and not paused:
            import torch
            model.to("cpu")
            torch.cuda.empty_cache()
            paused = True
        if time.monotonic() - LAST_STATUS > 25:
            status("WAITING_FOR_CAMPAIGN_GPU", language=language)
        time.sleep(5)
    if paused:
        model.to("cuda")
    if shutil.disk_usage(ROOT).free < MIN_FREE:
        raise RuntimeError("Remote free disk fell below the 25 GiB reserve")


def await_checkpoints(minimum_cycle, timeout):
    deadline = time.monotonic() + timeout
    while True:
        reports = {}
        for lang in LANGUAGES:
            try:
                report = read_json(ROOT / "teachers" / lang / "best_report.json")
                weights = ROOT / "teachers" / lang / "best" / "model.safetensors"
                if (report.get("stage") == "trained" and report.get("cycle", 0) >= minimum_cycle
                        and report.get("weights_sha256") and weights.is_file()):
                    reports[lang] = report
            except (OSError, ValueError):
                pass
        if len(reports) == 3 and not training_active():
            return
        parent = read_json(ROOT / "launcher.json")
        if not process_matches(parent):
            raise RuntimeError("Campaign parent exited before all test checkpoints were available")
        if time.monotonic() > deadline:
            raise TimeoutError("No complete set of trained teachers within the evaluation wait limit")
        status("WAITING_FOR_THREE_CHECKPOINTS", minimum_cycle=minimum_cycle,
               ready_languages=sorted(reports), training_child_active=training_active())
        time.sleep(15)


def freeze_inputs():
    """Copy checkpoints and make a WAL-aware database snapshot before scoring."""
    wait_for_gpu()
    if shutil.disk_usage(ROOT).free < MIN_FREE + 8 * 1024**3:
        raise RuntimeError("Not enough space to freeze test data/checkpoints above the reserve")
    frozen = JOB / "frozen"
    frozen.mkdir(exist_ok=False)
    manifests = {}
    for lang in LANGUAGES:
        report_path = ROOT / "teachers" / lang / "best_report.json"
        before = report_path.read_bytes()
        report = json.loads(before)
        source = ROOT / "teachers" / lang / "best"
        target = frozen / lang
        shutil.copytree(source, target)
        if report_path.read_bytes() != before:
            raise RuntimeError(f"Checkpoint publication changed during freeze: {lang}; preserve files for review")
        actual = file_sha(target / "model.safetensors")
        if actual != report["weights_sha256"]:
            raise RuntimeError(f"Frozen checkpoint SHA mismatch for {lang}")
        atomic_json(frozen / f"{lang}_dev_report.json", report)
        manifests[lang] = {
            "checkpoint_cycle": report["cycle"], "weights_sha256": actual,
            "threshold_from_dev": float(report["threshold_selected_on_dev"]),
            "files_sha256": {p.name: file_sha(p) for p in sorted(target.iterdir()) if p.is_file()},
        }
    snapshot = frozen / "test.sqlite"
    source = sqlite3.connect("file:" + str((ROOT / "corpus.sqlite").resolve()) + "?mode=ro", uri=True)
    destination = sqlite3.connect(snapshot)
    try:
        source.backup(destination, pages=16384, sleep=0.05)
    finally:
        destination.close()
        source.close()
    db = sqlite3.connect("file:" + str(snapshot.resolve()) + "?mode=ro", uri=True)
    try:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Frozen test database integrity check failed")
        groups = db.execute("SELECT COUNT(*) FROM (SELECT source_id FROM examples GROUP BY source_id HAVING COUNT(DISTINCT split)>1)").fetchone()[0]
        if groups:
            raise RuntimeError(f"Source groups cross train/dev/test: {groups}")
        next_cycle = db.execute("SELECT value FROM meta WHERE key='next_cycle'").fetchone()[0]
        counts = [dict(zip(("language", "split", "rows"), row)) for row in db.execute(
            "SELECT language,split,COUNT(*) FROM examples GROUP BY 1,2 ORDER BY 1,2")]
    finally:
        db.close()
    manifest = {"frozen_at_utc": now(), "database_sha256": file_sha(snapshot),
                "database_integrity": "ok", "database_next_cycle": int(next_cycle),
                "split_counts": counts, "source_split_leakage_groups": groups,
                "checkpoints": manifests, "code_sha256": file_sha(Path(__file__)),
                "source_encoder_sha256": file_sha(Path("turn_v2/monolingual_data.py")),
                "test_used_for_threshold_or_checkpoint_selection": False}
    atomic_json(JOB / "manifest.json", manifest)
    return manifest


def token_hash(ids):
    import numpy as np
    return hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).digest()


def audit_test(db, lang, rows, encoded, tokenizer):
    """Exclude exact source and actual 256-token input overlap with train/dev."""
    from .monolingual_data import encode_tail
    import numpy as np
    ids, lengths = encoded[:2]
    signatures = [token_hash(ids[i, :int(lengths[i])]) for i in range(len(rows))]
    wanted = set(signatures)
    leaked = set()
    count = 0
    for (raw,) in db.execute("SELECT record_json FROM examples WHERE language=? AND split IN ('train','dev')", (lang,)):
        if count % 512 == 0:
            wait_for_gpu(language=lang)
        row = json.loads(raw)
        signature = token_hash(encode_tail(tokenizer, row["context"], row["text"], 256)["input_ids"])
        if signature in wanted:
            leaked.add(signature)
        count += 1
        if count % 10000 == 0:
            status("AUDITING_ENCODED_INPUT_LEAKAGE", language=lang, scanned_non_test_rows=count,
                   overlapping_test_inputs=len(leaked))
    source_rows = list(db.execute("SELECT page_id,text_sha256,split FROM sources WHERE language=?", (lang,)))
    non_test_hashes = {digest for _, digest, split in source_rows if split != "test"}
    leaked_sources = {lang + ":" + str(page) for page, digest, split in source_rows
                      if split == "test" and digest in non_test_hashes}
    clean = np.asarray([i for i, row in enumerate(rows)
                        if signatures[i] not in leaked and row["source_id"] not in leaked_sources], dtype=np.int64)
    if not len(clean):
        raise RuntimeError(f"No independent test rows remain for {lang}")
    audit = {"non_test_rows_scanned": count, "test_rows_before_audit": len(rows),
             "overlapping_encoded_test_input_keys": len(leaked),
             "test_source_groups_with_exact_non_test_text": len(leaked_sources),
             "test_rows_after_audit": int(len(clean)),
             "removed_rows": len(rows) - int(len(clean)),
             "unique_test_model_input_keys": len(wanted),
             "policy": "remove overlap with train/dev before scoring; keep all labels and within-test duplicates"}
    return clean, audit


def infer(model, encoded, lang):
    import numpy as np
    import torch
    ids, lengths = encoded[:2]
    probs = np.zeros(len(ids), dtype=np.float32)
    model.eval()
    for start in range(0, len(ids), 48):
        wait_for_gpu(model=model, language=lang)
        stop = min(len(ids), start + 48)
        width = int(lengths[start:stop].max())
        x = torch.from_numpy(ids[start:stop, :width].astype(np.int64)).cuda()
        lens = torch.from_numpy(lengths[start:stop]).cuda()
        mask = torch.arange(width, device="cuda")[None] < lens[:, None]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x, attention_mask=mask.long()).logits.float()
            probs[start:stop] = logits.softmax(-1)[:, 1].cpu().numpy()
        del x, lens, mask, logits
        if start % 2400 == 0:
            status("SCORING_FROZEN_TEST", language=lang, completed=stop, total=len(ids))
    return probs


def cluster_intervals(rows, indices, labels, predictions):
    import numpy as np
    groups = {}
    for i in indices:
        row = rows[int(i)]
        group = groups.setdefault(row["source_id"], [0, 0, 0, 0])
        # TP, TN, FP, FN
        group[0 if labels[i] and predictions[i] else 1 if not labels[i] and not predictions[i]
              else 2 if not labels[i] else 3] += 1
    values = np.asarray(list(groups.values()), dtype=np.int64)
    rng = np.random.default_rng(20261026)
    total = values[rng.integers(0, len(values), size=(500, len(values)))].sum(axis=1)
    tp, tn, fp, fn = total.T
    accuracy = (tp + tn) / np.maximum(total.sum(axis=1), 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return {"method": "500 bootstrap draws over source pages; 95% percentile interval",
            "source_groups": len(groups), "accuracy": np.quantile(accuracy, [.025, .975]).tolist(),
            "eot_f1": np.quantile(f1, [.025, .975]).tolist()}


def score_language(db, lang, manifest):
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .common import normalize
    from .metrics import classification_metrics
    from .translation_segmenter_v15c import encode

    model_dir = JOB / "frozen" / lang
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    packed = db.execute("SELECT record_json FROM examples WHERE language=? AND split='test' ORDER BY example_id", (lang,)).fetchall()
    records_sha = hashlib.sha256()
    rows = []
    for (raw,) in packed:
        records_sha.update(raw.encode()); records_sha.update(b"\n")
        row = json.loads(raw)
        if row.get("punctuation_in_model_input") is not False:
            raise RuntimeError("Unexpected input protocol in frozen test")
        rows.append(row)
    del packed
    status("ENCODING_FROZEN_TEST", language=lang, test_rows=len(rows))
    encoded = encode(rows, tokenizer)
    clean, audit = audit_test(db, lang, rows, encoded, tokenizer)
    wait_for_gpu(language=lang)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).cuda()
    probs = infer(model, encoded, lang)
    threshold = manifest["checkpoints"][lang]["threshold_from_dev"]
    labels, horizons = encoded[2:]
    predictions = (probs >= threshold).astype(np.int64)
    overall = classification_metrics(labels[clean], probs[clean], threshold)
    by_horizon = {str(k): classification_metrics(labels[idx], probs[idx], threshold)
                  for k in (1, 2) if len(idx := clean[horizons[clean] == k])}
    period = np.asarray([i for i in clean if rows[int(i)]["label"] == "EOT"
                         and rows[int(i)].get("target_punctuation") in {".", "。"}], dtype=np.int64)
    internal = np.asarray([i for i in clean if rows[int(i)]["label"] == "NEOT"
                           and rows[int(i)].get("target_punctuation") in {",", "，", ";", "；", ":", "：", "、"}], dtype=np.int64)
    clean_set = set(clean.tolist())
    trace_path = JOB / f"test_predictions_{lang}.jsonl.gz"
    sampled = Counter()
    with gzip.open(trace_path, "wt", encoding="utf-8") as stream, (JOB / "review_samples.jsonl").open("a", encoding="utf-8") as review:
        for i, row in enumerate(rows):
            predicted = "EOT" if predictions[i] else "NEOT"
            value = {**row, "context": normalize(row["context"]), "text": normalize(row["text"]),
                     "predicted": predicted, "p_eot": float(probs[i]),
                     "correct": predicted == row["label"], "included_in_clean_test": i in clean_set,
                     "threshold_from_dev": threshold}
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            key = (row["label"], row["lookahead_words"], value["correct"])
            if i in clean_set and sampled[key] < 15:
                review.write(json.dumps(value, ensure_ascii=False) + "\n")
                sampled[key] += 1
    result = {**manifest["checkpoints"][lang], "threshold_selected_on_test": False,
              "test_rows_sha256": records_sha.hexdigest(), "leakage_audit": audit,
              "metrics": overall, "raw_test_metrics_before_overlap_filter": classification_metrics(labels, probs, threshold),
              "by_lookahead_words": by_horizon, "source_cluster_95pct_intervals": cluster_intervals(rows, clean, labels, predictions),
              "terminal_period_eot_recall": float((probs[period] >= threshold).mean()) if len(period) else None,
              "terminal_period_targets": int(len(period)),
              "internal_punctuation_no_eot_rate": float((probs[internal] < threshold).mean()) if len(internal) else None,
              "internal_punctuation_targets": int(len(internal)),
              "test_evaluated": True, "prose_test_accuracy_and_f1_90": min(overall["accuracy"], overall["eot_f1"]) >= .9,
              "full_trace_remote_only": str(trace_path), "full_trace_sha256": file_sha(trace_path)}
    del model, tokenizer, encoded, rows, labels, probs, predictions
    gc.collect(); torch.cuda.empty_cache()
    return result


def run(args):
    import fcntl
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB / "LOCK").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (JOB / "report.json").exists():
            raise RuntimeError("One-shot report already exists; preserve it instead of rerunning test")
        await_checkpoints(args.minimum_cycle, args.wait_seconds)
        status("FREEZING_DATA_AND_CHECKPOINTS")
        manifest = freeze_inputs()
        import torch
        import transformers.utils as hf_utils
        import transformers.utils.import_utils as hf_imports
        hf_utils.is_torchvision_available = hf_imports.is_torchvision_available = lambda: False
        hf_utils.is_librosa_available = hf_imports.is_librosa_available = lambda: False
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        torch.set_num_threads(1)
        report = {"created_at_utc": now(), "scope": "PUNCTUATION_DERIVED_PROSE_HELDOUT_NOT_HUMAN_AUDIO_GOLD",
                  "protocol": "lowercase punctuation-free text; 1/2 word lookahead; frozen checkpoints and data",
                  "threshold_policy": "dev-selected thresholds frozen before test; test never selects checkpoints or thresholds",
                  "quality_target_certified": False, "manifest": manifest, "languages": {}, "complete": False}
        db = sqlite3.connect("file:" + str((JOB / "frozen/test.sqlite").resolve()) + "?mode=ro", uri=True)
        try:
            for lang in LANGUAGES:
                report["languages"][lang] = score_language(db, lang, manifest)
                atomic_json(JOB / "partial_report.json", report)
        finally:
            db.close()
        report.update({"complete": True, "completed_at_utc": now(),
                       "all_three_prose_test_targets_met": all(x["prose_test_accuracy_and_f1_90"] for x in report["languages"].values())})
        atomic_json(JOB / "report.json", report)
        status("COMPLETE", report="report.json", languages=list(report["languages"]),
               all_three_prose_test_targets_met=report["all_three_prose_test_targets_met"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-cycle", type=int, default=45)
    parser.add_argument("--wait-seconds", type=int, default=7200)
    args = parser.parse_args()
    try:
        run(args)
    except BlockingIOError:
        raise SystemExit("Another evaluator holds the job lock; existing state was preserved")
    except Exception as exc:
        JOB.mkdir(parents=True, exist_ok=True)
        atomic_json(JOB / "error.json", {"at_utc": now(), "error": repr(exc), "traceback": traceback.format_exc()})
        status("FAILED", error=repr(exc))
        raise
