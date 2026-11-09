"""Inference using the new model's saved preprocessing and dev-selected threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from .common import encode_pair
from .metrics import probabilities


class TurnDetectorV2:
    def __init__(self, model_path, device="cpu", threads=None):
        self.path = Path(model_path)
        self.contract = json.loads((self.path / "turn_detector_config.json").read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        onnx_files = list(self.path.glob("*.onnx"))
        self.onnx = bool(onnx_files)
        if self.onnx:
            import onnxruntime as ort
            if device != "cpu":
                raise ValueError("This V2 ONNX runner currently uses CPU; use the PyTorch model for CUDA")
            options = ort.SessionOptions()
            for key, value in self.contract.get("onnx_session_options", {}).items():
                options.add_session_config_entry(key, str(value))
            selected_threads = threads if threads is not None else self.contract.get("intra_op_threads")
            if selected_threads:
                options.intra_op_num_threads = selected_threads
                options.inter_op_num_threads = 1
            self.model = ort.InferenceSession(str(onnx_files[0]), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        else:
            import torch
            from transformers import AutoModelForSequenceClassification
            if device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable")
            self.device = device
            self.model = AutoModelForSequenceClassification.from_pretrained(self.path).to(device).eval()

    def predict(self, text, context=""):
        encoded = encode_pair(self.tokenizer, context, text, self.contract["max_length"])
        multiple = self.contract.get("padding_to_multiple_of")
        length = len(encoded["input_ids"])
        target = min(self.contract["max_length"], ((length + multiple - 1) // multiple) * multiple) if multiple else self.contract["max_length"]
        pad = target - length
        arrays = {"input_ids": np.array([encoded["input_ids"] + [self.tokenizer.pad_token_id] * pad], dtype=np.int64),
                  "attention_mask": np.array([encoded["attention_mask"] + [0] * pad], dtype=np.int64)}
        if self.onnx:
            logits = self.model.run(["logits"], arrays)[0]
        else:
            import torch
            with torch.inference_mode():
                logits = self.model(**{key: torch.from_numpy(value).to(self.device)
                                      for key, value in arrays.items()}).logits.float().cpu().numpy()
        prob = float(probabilities(logits)[0])
        return {"prediction": "EOT" if prob >= self.contract["threshold"] else "NEOT",
                "prob_EOT": prob, "threshold": self.contract["threshold"],
                "smoke_only": self.contract["smoke_only"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, help="ONNX CPU intra-op threads; defaults to the model contract")
    args = parser.parse_args()
    print(json.dumps(TurnDetectorV2(args.model, args.device, args.threads).predict(args.text, args.context),
                     ensure_ascii=False, indent=2))
