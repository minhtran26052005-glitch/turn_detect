"""Export QAT's exact INT4 codes/scales; never refit quantization after QAT."""
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
import torch

from .common import encode_pair, file_hash, read_jsonl, write_json
from .compress import sort_graph_with_captures
from .distill_teacher import padded
from .metrics import probabilities
from .qat_student import WEIGHTS, pack_parameter
from .tiny_student import load_student


def onnx_gru_order(array, hidden):
    # PyTorch (reset, update, new) -> ONNX (update, reset, hidden).
    return np.concatenate([array[hidden:2 * hidden], array[:hidden], array[2 * hidden:]], axis=0)[None]


def export(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    model = load_student(args.student).float().eval()
    params = torch.load(args.student / "quant_params.pt", map_location="cpu", weights_only=True)
    group_size = params["group_size"]
    packed = {}
    with torch.no_grad():
        for name in WEIGHTS:
            parent, key = name.rsplit(".", 1)
            tensor = getattr(model.get_submodule(parent), key)
            codes, scales, restored = pack_parameter(tensor.numpy(), params["scales"][name].numpy(), group_size)
            tensor.copy_(torch.from_numpy(restored))
            packed[name] = (codes, scales, restored)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.student)
    rows = list(read_jsonl(args.dev))[:24]
    arrays = padded([encode_pair(tok, r.get("context", ""), r["text"], 256) for r in rows[:2]], tok.pad_token_id)
    reference_path = args.student / "qat_reference_fp32.onnx"
    assert not reference_path.exists()
    with torch.no_grad():
        torch.onnx.export(model, (torch.from_numpy(arrays["input_ids"]), torch.from_numpy(arrays["attention_mask"])),
            str(reference_path), input_names=["input_ids", "attention_mask"], output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"}, "logits": {0: "batch"}},
            opset_version=18, dynamo=False, external_data=False)
    graph = onnx.version_converter.convert_version(onnx.load(str(reference_path)), 21)
    gru = [n for n in graph.graph.node if n.op_type == "GRU"]
    assert len(gru) == 1
    mapping = {name: packed[name] for name in ["embedding.weight", "head.0.weight", "head.3.weight"]}
    for position, name in [(1, "gru.weight_ih_l0"), (2, "gru.weight_hh_l0")]:
        mapping[gru[0].input[position]] = tuple(onnx_gru_order(array, model.gru.hidden_size) for array in packed[name])
    initializers, nodes, records = [], [], []
    for tensor in graph.graph.initializer:
        if tensor.name not in mapping:
            initializers.append(tensor)
            continue
        codes, scales, restored = mapping[tensor.name]
        # Reject gate ordering, export transposition, or quantizer drift.
        if not np.array_equal(numpy_helper.to_array(tensor), restored):
            raise ValueError("Baked QAT weight differs from packed decode: " + tensor.name)
        name = tensor.name + "_qat_int4"
        quantized = numpy_helper.from_array(codes.astype(ml_dtypes.int4), name)
        assert quantized.data_type == TensorProto.INT4
        assert len(quantized.raw_data) == (codes.size + 1) // 2
        initializers.extend([quantized, numpy_helper.from_array(scales.astype(np.float16), name + "_scale")])
        nodes.extend([helper.make_node("DequantizeLinear", [name, name + "_scale"], [name + "_half"], axis=codes.ndim - 1, block_size=group_size),
                      helper.make_node("Cast", [name + "_half"], [tensor.name], to=TensorProto.FLOAT)])
        records.append({"onnx_name": tensor.name, "shape": list(codes.shape), "elements": codes.size,
            "packed_bytes": len(quantized.raw_data), "scale_bytes": scales.nbytes, "baked_weight_decode_exact": True})
    assert len(records) == 5
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(initializers)
    original_nodes = list(graph.graph.node)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes + original_nodes)
    sort_graph_with_captures(graph.graph)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save_model(graph, str(args.output / "model.onnx"))
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        shutil.copy2(args.student / name, args.output / name)
    contract = json.loads((args.student / "turn_detector_config.json").read_text(encoding="utf-8"))
    contract.update(storage_precision="INT4", compute_precision="FP32", quantized=True, training_method="INT4_QAT_WITH_DISTILLATION",
                    quantization_group_size=group_size, quantization_scale_precision="FP16", quantization_scale_policy="FROZEN_DURING_QAT_AND_EXPORT",
                    onnx_opset=21, onnx_session_options={"session.disable_quant_qdq": "1"}, quality_status="PACKED_QAT_EXPORT_PENDING_FULL_DEV_EVALUATION")
    contract.pop("compression_dev_report", None)
    write_json(args.output / "turn_detector_config.json", contract)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_quant_qdq", "1")
    session = ort.InferenceSession(str(args.output / "model.onnx"), options, providers=["CPUExecutionProvider"])
    deltas = []
    padding_delta = 0.
    with torch.no_grad():
        for batch_size in [1, 3, 8, 24]:
            arr = padded([encode_pair(tok, r.get("context", ""), r["text"], 256) for r in rows[:batch_size]], tok.pad_token_id)
            expected = model(**{k: torch.from_numpy(v) for k, v in arr.items()}).numpy()
            actual = session.run(["logits"], arr)[0]
            deltas.append(float(np.max(np.abs(probabilities(expected) - probabilities(actual)))))
            longer = {k: np.pad(v, ((0, 0), (0, 32))) for k, v in arr.items()}
            after = session.run(["logits"], longer)[0]
            padding_delta = max(padding_delta, float(np.max(np.abs(probabilities(actual) - probabilities(after)))))
    result = {"training_method": "INT4_QAT_WITH_DISTILLATION", "group_size": group_size, "tensors": records,
        "model_sha256": file_hash(args.output / "model.onnx"), "model_bytes": (args.output / "model.onnx").stat().st_size,
        "latent_student_sha256": file_hash(args.student / "student.pt"), "quant_params_sha256": file_hash(args.student / "quant_params.pt"),
        "frozen_scales_used_without_refit": True, "baked_weight_decode_exact": True,
        "max_probability_delta_pytorch_vs_packed_onnx": max(deltas), "right_padding_probability_delta": padding_delta,
        "technical_parity_passed": max(deltas) < 1e-4 and padding_delta < 1e-5, "holdout_evaluated": False}
    write_json(args.student / "qat_export_report.json", result)
    print(json.dumps(result), flush=True)
    assert result["technical_parity_passed"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dev", type=Path, default=Path("data_v2/frozen/dev.jsonl"))
    export(p.parse_args())
