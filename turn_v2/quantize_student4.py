"""Pack GRU student weights as real INT4; keep execution in FP32.

Uses ONNX opset 21 blocked DequantizeLinear and two signed nibbles per byte.
Group scales use FP16 storage, followed by a Cast back to FP32. Quantization
parameters are derived from weights only, never dev/holdout activations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import ml_dtypes
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from .common import file_hash, write_json
from .compress import sort_graph_with_captures


def quantize_groups(values, group_size, scale_method="mse"):
    """Return signed INT4 codes and FP16 scales for last-axis groups."""
    width = values.shape[-1]
    groups = (width + group_size - 1) // group_size
    padded = np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, groups * group_size - width)])
    blocks = padded.reshape(*values.shape[:-1], groups, group_size)
    valid = np.arange(groups * group_size).reshape(groups, group_size) < width
    valid = valid.reshape((1,) * (values.ndim - 1) + valid.shape)
    scale = np.maximum(np.max(blocks, axis=-1) / 7, -np.min(blocks, axis=-1) / 8)
    scale = np.maximum(scale, np.finfo(np.float16).tiny)
    best_scale = scale.astype(np.float16)
    best_error = np.full(scale.shape, np.inf, np.float32)
    # Fixed weight-only search, with rounded deployment scales in the objective.
    for fraction in ([1.] if scale_method == "max" else [.8, .85, .9, .95, 1.]):
        candidate = np.maximum(scale * fraction, np.finfo(np.float16).tiny).astype(np.float16)
        codes = np.clip(np.rint(blocks / candidate[..., None].astype(np.float32)), -8, 7)
        error = (((codes * candidate[..., None].astype(np.float32) - blocks) ** 2) * valid).sum(-1)
        better = error < best_error
        best_error = np.minimum(best_error, error)
        best_scale = np.where(better, candidate, best_scale)
    codes = np.clip(np.rint(blocks / best_scale[..., None].astype(np.float32)), -8, 7)
    codes = codes.reshape(*values.shape[:-1], groups * group_size)[..., :width]
    restored_scale = np.repeat(best_scale.astype(np.float32), group_size, axis=-1)[..., :width]
    restored = codes * restored_scale
    return codes.astype(ml_dtypes.int4), best_scale, restored


def export(args):
    args.output.mkdir(parents=True, exist_ok=False)
    source_model = args.source / "model.onnx"
    graph = onnx.version_converter.convert_version(onnx.load(str(source_model)), 21)
    weights, nodes, records = [], [], []
    for tensor in graph.graph.initializer:
        if tensor.data_type != TensorProto.FLOAT or len(tensor.dims) < 2 or (len(tensor.dims) == 2 and tensor.dims[0] == 1):
            weights.append(tensor)
            continue
        values = numpy_helper.to_array(tensor)
        codes, scale, restored = quantize_groups(values, args.group_size, args.scale_method)
        name = tensor.name + "_int4"
        packed = numpy_helper.from_array(codes, name)
        if packed.data_type != TensorProto.INT4 or len(packed.raw_data) != (values.size + 1) // 2:
            raise ValueError("INT4 packing failed")
        weights.extend([packed, numpy_helper.from_array(scale, name + "_scale")])
        nodes.extend([
            helper.make_node("DequantizeLinear", [name, name + "_scale"], [name + "_half"],
                             axis=values.ndim - 1, block_size=args.group_size),
            helper.make_node("Cast", [name + "_half"], [tensor.name], to=TensorProto.FLOAT),
        ])
        # DQ outputs FP16 before Cast. Include its rounding in parity evidence.
        restored = restored.astype(np.float16).astype(np.float32)
        records.append({"name": tensor.name, "shape": list(values.shape), "elements": values.size,
                        "packed_bytes": len(packed.raw_data), "scale_bytes": scale.nbytes,
                        "weight_mse": float(np.mean((restored - values) ** 2))})
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(weights)
    original_nodes = list(graph.graph.node)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes + original_nodes)
    sort_graph_with_captures(graph.graph)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save_model(graph, str(args.output / "model.onnx"))
    for filename in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        shutil.copy2(args.source / filename, args.output / filename)
    contract = json.loads((args.source / "turn_detector_config.json").read_text(encoding="utf-8"))
    contract.update(storage_precision="INT4", compute_precision="FP32", quantized=True,
                    quantization_group_size=args.group_size, quantization_scale_method=args.scale_method,
                    quantization_scale_precision="FP16", quality_status="NOT_YET_EVALUATED",
                    onnx_opset=21, onnx_session_options={"session.disable_quant_qdq": "1"})
    contract.pop("compression_dev_report", None)
    write_json(args.output / "turn_detector_config.json", contract)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_quant_qdq", "1")
    session = ort.InferenceSession(str(args.output / "model.onnx"), options, providers=["CPUExecutionProvider"])
    # Smoke all dynamic shapes and padded batches, independently of dev labels.
    rng = np.random.default_rng(42)
    vocabulary_size = json.loads((args.source / "config.json").read_text(encoding="utf-8"))["vocab_size"]
    padding_delta = 0.
    for batch, length in [(1, 8), (3, 24), (2, 256)]:
        ids = rng.integers(4, vocabulary_size, (batch, length), dtype=np.int64)
        mask = np.ones_like(ids)
        logits = session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        extended = {"input_ids": np.pad(ids, ((0, 0), (0, 8))),
                    "attention_mask": np.pad(mask, ((0, 0), (0, 8)))}
        after = session.run(None, extended)[0]
        padding_delta = max(padding_delta, float(np.max(np.abs(after - logits))))
        if not np.isfinite(logits).all() or padding_delta > 1e-4:
            raise ValueError("Dynamic shapes/right-padding parity failed")
    report = {"source_sha256": file_hash(source_model), "model_sha256": file_hash(args.output / "model.onnx"),
              "storage_precision": "INT4", "execution_precision": "FP32", "group_size": args.group_size,
              "scale_precision": "FP16", "scale_method": args.scale_method, "tensors": records,
              "packed_elements": sum(r["elements"] for r in records),
              "packed_weight_bytes": sum(r["packed_bytes"] for r in records),
              "scale_bytes": sum(r["scale_bytes"] for r in records),
              "onnx_bytes": (args.output / "model.onnx").stat().st_size,
              "right_padding_max_logit_delta": padding_delta, "technical_checks_passed": True,
              "holdout_evaluated": False}
    write_json(args.output.parent / (args.output.name + "_export.json"), report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="FP32 ONNX student directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, choices=[8, 16, 32, 64], default=32)
    parser.add_argument("--scale-method", choices=["mse", "max"], default="mse")
    export(parser.parse_args())
