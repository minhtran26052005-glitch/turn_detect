"""Export FP32 ONNX and measure PyTorch parity; quantization is deliberately separate."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import encode_pair, write_json
from .metrics import probabilities


def export_model(model_path: Path, output: Path, parity_records):
    import onnx
    import onnxruntime as ort
    output.mkdir(parents=True, exist_ok=True)
    contract = json.loads((model_path / "turn_detector_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).float().cpu().eval()

    class LogitsOnly(torch.nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier

        def forward(self, input_ids, attention_mask):
            return self.classifier(input_ids=input_ids, attention_mask=attention_mask).logits

    # export() restores the wrapper's previous training flag recursively. The
    # wrapper itself must be in eval mode, or it re-enables classifier dropout.
    wrapper = LogitsOnly(model).eval()

    def inputs(row):
        encoded = encode_pair(tokenizer, row.get("context", ""), row["text"], contract["max_length"])
        pad = contract["max_length"] - len(encoded["input_ids"])
        return {"input_ids": torch.tensor([encoded["input_ids"] + [tokenizer.pad_token_id] * pad]),
                "attention_mask": torch.tensor([encoded["attention_mask"] + [0] * pad])}

    sample = inputs(parity_records[0])
    path = output / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(wrapper, (sample["input_ids"], sample["attention_mask"]), str(path),
            input_names=["input_ids", "attention_mask"], output_names=["logits"], opset_version=18,
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                          "attention_mask": {0: "batch", 1: "sequence"}, "logits": {0: "batch"}},
            dynamo=False, external_data=False)
    onnx.checker.check_model(str(path))
    if wrapper.training or model.training:
        raise RuntimeError("Export unexpectedly re-enabled training mode")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    deltas, classes_equal = [], True
    for row in parity_records:
        values = inputs(row)
        with torch.no_grad():
            expected = wrapper(**values).numpy()
        actual = session.run(["logits"], {key: tensor.numpy() for key, tensor in values.items()})[0]
        deltas.append(float(np.abs(probabilities(expected) - probabilities(actual)).max()))
        classes_equal &= bool(np.array_equal(expected.argmax(-1), actual.argmax(-1)))
    tokenizer.save_pretrained(output)
    shutil.copy2(model_path / "config.json", output / "config.json")
    shutil.copy2(model_path / "turn_detector_config.json", output / "turn_detector_config.json")
    report = {"format": "FP32_ONNX", "samples": len(parity_records), "batch_size": 1,
              "max_abs_prob_EOT_delta": max(deltas), "argmax_classes_equal": classes_equal,
              "passed": max(deltas) < 1e-4 and classes_equal,
              "quantized": False, "note": "INT8 requires separate dev/holdout parity and batch-sensitivity evaluation"}
    write_json(output / "parity_report.json", report)
    if not report["passed"]:
        raise RuntimeError("ONNX parity failed; model was not accepted for deployment")
    return report
