from __future__ import annotations

import numpy as np


def probabilities(logits):
    logits = np.asarray(logits, dtype=np.float64)
    exponent = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return (exponent / exponent.sum(axis=-1, keepdims=True))[:, 1]


def classification_metrics(labels, probs, threshold=0.5):
    labels, probs = np.asarray(labels), np.asarray(probs)
    if not len(labels) or labels.shape != probs.shape:
        raise ValueError("Need equally sized nonempty labels and probabilities")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 (NEOT) or 1 (EOT)")
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("Invalid probabilities")
    predictions = (probs >= threshold).astype(int)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    accuracy = (tp + tn) / len(labels)
    return {"samples": len(labels), "threshold": float(threshold), "accuracy": accuracy,
            "eot_precision": precision, "eot_recall": recall, "eot_f1": f1,
            "joint_score": min(accuracy, f1), "false_eot": fp, "missed_eot": fn,
            "false_eot_rate": fp / max(fp + tn, 1), "confusion_matrix_NEOT_EOT": [[tn, fp], [fn, tp]]}


def calibrate_on_dev(labels, probs):
    """Fixed grid; caller must supply dev predictions, never holdout predictions."""
    candidates = [classification_metrics(labels, probs, float(value))
                  for value in np.linspace(0.05, 0.95, 181)]
    return max(candidates, key=lambda item: (item["joint_score"], -item["false_eot"],
                                           -abs(item["threshold"] - 0.5)))
