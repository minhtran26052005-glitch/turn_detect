"""Checkpoint I/O and DEV evaluation for the boundary classifier."""
import hashlib
import json
from pathlib import Path

LABELS = {"NEOT": 0, "EOT": 1}
LOOKAHEAD = (1, 2)

def atomic_json(path: Path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha(path: Path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
