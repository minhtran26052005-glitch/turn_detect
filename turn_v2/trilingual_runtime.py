"""Standalone NumPy/Tokenizers/ORT inference with explicit vi/en/zh thresholds."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def normalize(text):
    value = unicodedata.normalize("NFKC", text).lower()
    value = "".join(" " if unicodedata.category(c).startswith("P") else c for c in value)
    return re.sub(r"\s+", " ", value).strip()


class TinyOnnxDetector:
    def __init__(self, path, threads=None):
        self.path = Path(path)
        self.contract = json.loads((self.path / "turn_detector_config.json").read_text(encoding="utf-8"))
        self.tokenizer = Tokenizer.from_file(str(self.path / "tokenizer.json"))
        settings = json.loads((self.path / "tokenizer_config.json").read_text(encoding="utf-8"))
        def special(name):
            value = settings[name]
            if isinstance(value, dict):value = value["content"]
            result = self.tokenizer.token_to_id(value)
            if result is None:raise ValueError("Missing special token: " + name)
            return result
        self.cls_id, self.sep_id, self.pad_id = [special(n) for n in ["cls_token", "sep_token", "pad_token"]]
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or self.contract.get("intra_op_threads", 2)
        options.inter_op_num_threads = 1
        for k, v in self.contract.get("onnx_session_options", {}).items():options.add_session_config_entry(k, str(v))
        self.session = ort.InferenceSession(str(self.path / "model.onnx"), options, providers=["CPUExecutionProvider"])

    def _encode(self, text, context):
        maximum = self.contract["max_length"]
        original = self.tokenizer.encode(normalize(text), add_special_tokens=False).ids
        current = original[:maximum - 3]
        history = self.tokenizer.encode(normalize(context), add_special_tokens=False).ids
        budget = maximum - 3 - len(current)
        history = history[-budget:] if budget else []
        ids = [self.cls_id] + history + [self.sep_id] + current + [self.sep_id]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}, len(original) > maximum - 3

    def encode(self, text, context=""):
        return self._encode(text, context)[0]

    def predict(self, text, context="", language=None):
        if language not in {"vi", "en", "zh"}:
            raise ValueError("Pass language='vi', 'en', or 'zh'; automatic language detection is not included")
        threshold = self.contract["language_thresholds"][language]
        encoded, truncated = self._encode(text, context)
        size = len(encoded["input_ids"])
        target = min(self.contract["max_length"], ((size + 7) // 8) * 8)
        arrays = {k: np.asarray([v + [self.pad_id if k == "input_ids" else 0] * (target - size)], np.int64) for k, v in encoded.items()}
        logits = self.session.run(["logits"], arrays)[0].astype(np.float64)
        exp = np.exp(logits - logits.max(-1, keepdims=True))
        probability = float((exp / exp.sum(-1, keepdims=True))[0, 1])
        return {"language": language, "prediction": "EOT" if probability >= threshold else "NEOT", "prob_EOT": probability,
                "threshold": threshold, "input_truncated": truncated, "quality_status": self.contract.get("quality_status"), "smoke_only": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--language", choices=["vi", "en", "zh"], required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--context", default="")
    p.add_argument("--threads", type=int)
    a = p.parse_args()
    print(json.dumps(TinyOnnxDetector(a.model, a.threads).predict(a.text, a.context, a.language), ensure_ascii=False, indent=2))
