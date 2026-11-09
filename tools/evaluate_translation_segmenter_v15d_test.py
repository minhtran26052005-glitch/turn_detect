"""Run a one-shot held-out test on the Colab v15d cycle-42 best checkpoints.

The test split is read from the frozen cycle-42 SQLite snapshot. Thresholds are
fixed from each language's existing dev report; this tool never tunes on test.
It evaluates on Colab and transfers only the small JSON metrics report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from teacher_first_colab import campaign_runtime


LOCAL = Path("outputs/translation_segmenter_v15d_20261026")
SESSION = "a-jvsc-c56e5860-53dd-42ab-9046-4b7355b137a0c830cfa6-577a-4f10-ad66-a2ccf59eca47.ipynb"


REMOTE_CODE = r'''import gc, hashlib, json, os, sqlite3, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

root = Path("/content/turn_detect_v2")
os.chdir(root)
sys.path.insert(0, str(root))
run = root / "outputs/translation_segmenter_v15d_20261026"
snapshot_db = run / "recovery/state_snapshot.sqlite"
snapshot_meta = json.loads((run / "recovery/latest_snapshot.json").read_text())
assert snapshot_meta["scope"] == "DATA_AND_METADATA_ONLY_NO_MODEL_WEIGHTS"
snapshot_cycle = int(snapshot_meta["cycle"])
result_path = run / f"test_eval_cycle{snapshot_cycle}.json"
assert not result_path.exists(), "One-shot test report already exists; refusing to re-evaluate the holdout"

launcher = json.loads((run / "launcher.json").read_text())
lp = Path("/proc") / str(launcher["pid"])
assert (lp / "cmdline").exists()
cmd = (lp / "cmdline").read_bytes()
ticks = (lp / "stat").read_text().split()[21]
expected = launcher["command"][launcher["command"].index("-m") + 1]
assert expected.encode() + b"\x00" in cmd and ticks == launcher["start_ticks"], "Campaign parent identity changed"

assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "Expected connected Colab BF16 GPU"
torch.set_num_threads(2)
import transformers.utils as hf_utils
import transformers.utils.import_utils as hf_imports
hf_utils.is_torchvision_available = hf_imports.is_torchvision_available = lambda: False
hf_utils.is_librosa_available = hf_imports.is_librosa_available = lambda: False
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from turn_v2.translation_segmenter_v15c import encode
from turn_v2.metrics import classification_metrics

db = sqlite3.connect("file:" + str(snapshot_db) + "?mode=ro", uri=True)
assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
next_cycle = int(db.execute("SELECT value FROM meta WHERE key='next_cycle'").fetchone()[0])
assert next_cycle == snapshot_cycle + 1, f"Snapshot database/cycle mismatch (next_cycle={next_cycle})"
leaks = db.execute("SELECT COUNT(*) FROM (SELECT source_id FROM examples GROUP BY source_id HAVING COUNT(DISTINCT split)>1)").fetchone()[0]
assert leaks == 0, f"Source/page leakage across train/dev/test: {leaks}"

def pause_for_campaign_training():
    """Yield the GPU if the always-on parent enters a training phase."""
    paused = False
    while True:
        try:
            stage = json.loads((run / "status.json").read_text()).get("stage", "")
        except Exception:
            stage = ""
        if not stage.startswith("TRAINING"):
            if paused:
                model.to("cuda")
                torch.cuda.empty_cache()
            return
        if not paused:
            model.to("cpu")
            torch.cuda.empty_cache()
            paused = True
        time.sleep(5)

def wait_for_training_window():
    while True:
        try:
            stage = json.loads((run / "status.json").read_text()).get("stage", "")
        except Exception:
            stage = ""
        if not stage.startswith("TRAINING"):
            return
        time.sleep(5)

def predict_with_yield(model, encoded):
    ids, lengths, _, _ = encoded
    probs = np.zeros(len(ids), dtype=np.float32)
    model.eval()
    for start in range(0, len(ids), 48):
        pause_for_campaign_training()
        stop = min(len(ids), start + 48)
        width = int(lengths[start:stop].max())
        x = torch.from_numpy(ids[start:stop, :width].astype(np.int64)).cuda()
        lens = torch.from_numpy(lengths[start:stop]).cuda()
        mask = torch.arange(width, device="cuda")[None] < lens[:, None]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x, attention_mask=mask.long()).logits.float()
            probs[start:stop] = logits.softmax(-1)[:, 1].cpu().numpy()
    return probs

def score_subset(labels, probs, threshold, indices):
    return classification_metrics(labels[indices], probs[indices], threshold)

output = {
    "campaign": "translation_segmenter_v15d",
    "evaluation": "one-shot held-out prose test; page-grouped split",
    "cycle": snapshot_cycle,
    "at_utc": datetime.now(timezone.utc).isoformat(),
    "snapshot_sha256": snapshot_meta["sha256"],
    "snapshot_database_integrity": "ok",
    "source_split_leakage_groups": leaks,
    "threshold_policy": "fixed per-language threshold selected on dev; no threshold/model selection on test",
    "punctuation_in_model_input": False,
    "qwen_used": False,
    "label_scope": "punctuation-derived prose; not conversational or human/audio gold",
    "languages": {},
    "quality_target_certified": False,
}

internal_marks = {",", "，", ";", "；", ":", "：", "、"}
period_marks = {".", "。"}
terminal_marks = {".", "!", "?", "。", "！", "？"}
for lang in ("vi", "en", "zh"):
    wait_for_training_window()
    report_path = run / "teachers" / lang / "best_report.json"
    report = json.loads(report_path.read_text())
    assert report.get("stage") == "trained" and report.get("cycle", 0) <= snapshot_cycle, f"Invalid best checkpoint for {lang}"
    assert report.get("test_evaluated") is False, f"Test was already evaluated for {lang}"
    threshold = float(report["threshold_selected_on_dev"])
    model_dir = run / "teachers" / lang / "best"
    weights = model_dir / "model.safetensors"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert digest == report["weights_sha256"], f"Best checkpoint hash mismatch for {lang}"

    packed = db.execute("SELECT example_id,record_json FROM examples WHERE language=? AND split='test' ORDER BY example_id", (lang,)).fetchall()
    rows = [json.loads(item[1]) for item in packed]
    row_hash = hashlib.sha256()
    for example_id, record in packed:
        row_hash.update(example_id.encode()); row_hash.update(b"\n")
    assert rows and all(row.get("punctuation_in_model_input") is False for row in rows)
    encoded = encode(rows, AutoTokenizer.from_pretrained(model_dir, local_files_only=True))
    labels = encoded[2]
    wait_for_training_window()
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).cuda()
    probs = predict_with_yield(model, encoded)
    overall = classification_metrics(labels, probs, threshold)
    by_lookahead = {}
    for k in (1, 2):
        idx = np.asarray([i for i, row in enumerate(rows) if row["lookahead_words"] == k], dtype=np.int64)
        by_lookahead[str(k)] = classification_metrics(labels[idx], probs[idx], threshold)
    dot_idx = np.asarray([i for i, row in enumerate(rows) if row["label"] == "EOT" and row.get("target_punctuation") in period_marks], dtype=np.int64)
    ascii_idx = np.asarray([i for i, row in enumerate(rows) if row["label"] == "EOT" and row.get("target_punctuation") == "."], dtype=np.int64)
    terminal_idx = np.asarray([i for i, row in enumerate(rows) if row["label"] == "EOT" and row.get("target_punctuation") in terminal_marks], dtype=np.int64)
    internal_idx = np.asarray([i for i, row in enumerate(rows) if row["label"] == "NEOT" and row.get("target_punctuation") in internal_marks], dtype=np.int64)
    item = {
        "checkpoint_cycle": report["cycle"], "checkpoint_sha256": digest,
        "test_rows": len(rows), "test_rows_sha256": row_hash.hexdigest(),
        "threshold_from_dev": threshold, "metrics": overall, "by_lookahead_words": by_lookahead,
        "terminal_period_eot_recall": float((probs[dot_idx] >= threshold).mean()) if len(dot_idx) else None,
        "ascii_dot_eot_recall": float((probs[ascii_idx] >= threshold).mean()) if len(ascii_idx) else None,
        "all_terminal_punctuation_eot_recall": float((probs[terminal_idx] >= threshold).mean()) if len(terminal_idx) else None,
        "terminal_period_targets": int(len(dot_idx)),
        "internal_punctuation_no_eot_rate": float((probs[internal_idx] < threshold).mean()) if len(internal_idx) else None,
        "internal_punctuation_targets": int(len(internal_idx)),
        "checkpoint_selected_on_test": False,
        "test_evaluated": True,
    }
    output["languages"][lang] = item
    atomic_tmp = result_path.with_suffix(".partial")
    atomic_tmp.write_text(json.dumps({**output, "partial": True}, ensure_ascii=False, indent=2) + "\n")
    atomic_tmp.replace(result_path)
    print(json.dumps({"language": lang, "test_rows": len(rows), "metrics": overall}, ensure_ascii=False), flush=True)
    del model, encoded, labels, probs, rows, packed
    gc.collect(); torch.cuda.empty_cache()

db.close()
result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print("TEST_RESULT=" + json.dumps(output, ensure_ascii=False), flush=True)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-path", default=SESSION)
    args = parser.parse_args()
    LOCAL.mkdir(parents=True, exist_ok=True)
    runtime = campaign_runtime(args.session_path)
    try:
        connection = runtime.connect()
        if connection.get("execution_state") != "idle":
            raise RuntimeError(f"Refusing to queue test evaluation into busy kernel: {connection}")
        response = runtime.execute(REMOTE_CODE, timeout=3600, echo=False)
        if response.get("status") != "ok":
            errors = [item for item in response.get("outputs", []) if item.get("output_type") == "error"]
            raise RuntimeError("Colab held-out evaluation cell failed: " + json.dumps(errors, ensure_ascii=False))
        text = "".join(item.get("text", "") for item in response.get("outputs", [])
                       if item.get("output_type") == "stream")
        marker = next((line[len("TEST_RESULT="):] for line in text.splitlines()
                       if line.startswith("TEST_RESULT=")), None)
        if marker is None:
            raise RuntimeError("Colab test cell returned without a final JSON receipt")
        result = json.loads(marker)
        target = LOCAL / f"test_eval_cycle{result['cycle']}.json"
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"connection": connection, "result": result,
                          "local_report": str(target)}, ensure_ascii=False))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
