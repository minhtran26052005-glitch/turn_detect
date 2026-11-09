"""
Inference-only core for text-based turn detection using ONNX Runtime.

Key difference from naive tokenizer(text, truncation=True):
  HuggingFace default truncation cuts from the RIGHT, discarding the current
  utterance (text) which is the most important signal. This module tokenizes
  context and text separately, then LEFT-truncates context tokens so that
  the full text is always preserved within max_length.

Input format: [CLS] <context_tokens> [SEP] <text_tokens> [SEP] [PAD...]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from tqdm import tqdm


def build_input_ids(
    tokenizer,
    context: str,
    text: str,
    max_length: int,
) -> tuple[list[int], list[int]]:
    """
    Tokenize context + text and assemble input_ids / attention_mask.

    Context is left-truncated (oldest tokens dropped) when the combined
    sequence exceeds max_length. The text portion is never truncated.

    Returns:
        input_ids      — list of int, length == max_length
        attention_mask — list of int, length == max_length
    """
    ctx_ids  = tokenizer.encode(context.strip().lower(), add_special_tokens=False)
    text_ids = tokenizer.encode(text.strip().lower(),    add_special_tokens=False)

    # Layout: [CLS] ctx [SEP] text [SEP]  →  3 special tokens
    # If text alone already fills the window, truncate it from the right (last resort)
    max_text = max_length - 3
    if len(text_ids) > max_text:
        text_ids = text_ids[:max_text]

    budget = max_length - len(text_ids) - 3
    if len(ctx_ids) > budget:
        ctx_ids = ctx_ids[-budget:] if budget > 0 else []

    cls = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id
    pad = tokenizer.pad_token_id

    input_ids = [cls] + ctx_ids + [sep] + text_ids + [sep]
    attention_mask = [1] * len(input_ids)

    pad_len = max_length - len(input_ids)
    input_ids      += [pad] * pad_len
    attention_mask += [0]   * pad_len

    return input_ids, attention_mask


class TurnDetectorONNX:
    """Text-based turn-end detector backed by an ONNX model."""

    def __init__(self, model_path: str, max_length: int = 256, device: str = "cpu"):
        self.max_length = max_length

        print(f"Loading ONNX model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        onnx_path = next(Path(model_path).glob("*.onnx"))

        device = device.lower()
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            raise ValueError(f"Unsupported device: {device!r} (expected 'cpu' or 'cuda')")

        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        print(f"Model loaded: {onnx_path.name} (providers={self.session.get_providers()})")

    def predict(self, context: str, text: str) -> dict[str, Any]:
        """Single-sample inference."""
        ids, mask = build_input_ids(self.tokenizer, context, text, self.max_length)
        logits = self.session.run(
            ["logits"],
            {
                "input_ids":      np.array([ids],  dtype=np.int64),
                "attention_mask": np.array([mask], dtype=np.int64),
            },
        )[0][0]
        probs = _softmax(logits)
        return {
            "predicted_class": int(np.argmax(probs)),
            "prob_EOT": float(probs[1]),
        }

    def predict_batch(
        self,
        records: list[dict],
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        """
        Batch inference.

        Args:
            records: list of {"context": str, "text": str}

        Returns:
            list of {"predicted_class": int, "prob_EOT": float}
        """
        results = []
        for i in tqdm(range(0, len(records), batch_size), desc="Running inference"):
            batch = records[i : i + batch_size]
            ids_list, mask_list = [], []
            for rec in batch:
                ids, mask = build_input_ids(
                    self.tokenizer,
                    rec.get("context", ""),
                    rec.get("text", ""),
                    self.max_length,
                )
                ids_list.append(ids)
                mask_list.append(mask)

            logits_batch = self.session.run(
                ["logits"],
                {
                    "input_ids":      np.array(ids_list,  dtype=np.int64),
                    "attention_mask": np.array(mask_list, dtype=np.int64),
                },
            )[0]
            for logits in logits_batch:
                probs = _softmax(logits)
                results.append({
                    "predicted": int(np.argmax(probs)),
                    "prob": float(probs[1]),
                })
        return results


def _softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()
