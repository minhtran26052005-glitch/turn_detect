"""Post-training compression of the frozen V2 model; vocabulary comes from train only."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gc
import json
from pathlib import Path
import shutil
import time

from .common import file_hash, normalize, read_jsonl, write_json


def sort_graph_with_captures(graph, outer_names=()):
    """Sort nodes including the implicit outer-scope inputs of If subgraphs."""
    import onnx
    def subgraphs(node):
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield attribute.g
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                yield from attribute.graphs
    def free_names(value):
        defined = {item.name for item in value.input} | {item.name for item in value.initializer}
        defined.update(name for node in value.node for name in node.output if name)
        referenced = {name for node in value.node for name in node.input if name}
        referenced.update(item.name for item in value.output)
        for node in value.node:
            for child in subgraphs(node):
                referenced.update(free_names(child))
        return referenced - defined
    defined = {item.name for item in graph.input} | {item.name for item in graph.initializer}
    all_names = set(outer_names) | defined | {name for node in graph.node for name in node.output if name}
    dependencies = []
    for node in graph.node:
        required = {name for name in node.input if name}
        for child in subgraphs(node):
            required.update(free_names(child))
            sort_graph_with_captures(child, all_names)
        dependencies.append((node, required))
    available, ordered = set(outer_names) | defined, []
    while dependencies:
        pending = []
        for node, required in dependencies:
            if required <= available:
                ordered.append(node)
                available.update(name for name in node.output if name)
            else:
                pending.append((node, required))
        if len(pending) == len(dependencies):
            raise ValueError("Unresolved graph inputs or cyclic graph dependencies")
        dependencies = pending
    del graph.node[:]
    graph.node.extend(ordered)


def vocabulary(teacher: Path, train: Path, work: Path):
    from tokenizers import Tokenizer
    work.mkdir(parents=True, exist_ok=True)
    if (work / "vocabulary_report.json").exists():
        raise FileExistsError("Vocabulary already frozen; use its receipt or choose another work directory")
    original = Tokenizer.from_file(str(teacher / "tokenizer.json"))
    source = json.loads((teacher / "tokenizer.json").read_text(encoding="utf-8"))
    if source["model"]["type"] != "Unigram":
        raise ValueError("This recipe supports the supplied Unigram tokenizer only")
    counts, pending, records = Counter(), [], 0
    for row in read_jsonl(train):
        pending.extend([normalize(row.get("context", "")), normalize(row["text"])])
        records += 1
        if len(pending) >= 1024:
            for encoded in original.encode_batch(pending, add_special_tokens=False):
                counts.update(encoded.ids)
            pending = []
    for encoded in original.encode_batch(pending, add_special_tokens=False):
        counts.update(encoded.ids)
    vocab = source["model"]["vocab"]
    original_base_size = len(vocab)
    # HF registers [MASK] outside the Unigram list, but it has an embedding row.
    for token in sorted(source.get("added_tokens", []), key=lambda entry: entry["id"]):
        if token["id"] >= len(vocab):
            if token["id"] != len(vocab):
                raise ValueError("Non-contiguous added token IDs require a separate mapping")
            vocab.append([token["content"], 0.0])
    # Preserve single-character fallbacks, byte pieces, and all registered tokens.
    # Removed multi-character pieces can resegment into retained pieces rather than UNK.
    characters = {index for index, (token, _) in enumerate(vocab)
                  if len(token.replace("\u2581", "")) <= 1 or token.startswith("<0x")}
    registered = {token["id"] for token in source.get("added_tokens", [])}
    kept = sorted(set(counts) | characters | registered | {0, 1, 2, 3})
    remap = {old: new for new, old in enumerate(kept)}
    compact = copy.deepcopy(source)
    compact["model"]["vocab"] = [vocab[index] for index in kept]
    compact["model"]["unk_id"] = remap[source["model"]["unk_id"]]
    for token in compact.get("added_tokens", []):
        token["id"] = remap[token["id"]]
    processor = compact.get("post_processor", {})
    for token in processor.get("special_tokens", {}).values():
        token["ids"] = [remap[index] for index in token["ids"]]
    tokenizer = Tokenizer.from_str(json.dumps(compact, ensure_ascii=False))
    fp32 = work / "fp32"
    fp32.mkdir(exist_ok=True)
    tokenizer.save(str(fp32 / "tokenizer.json"))
    for name in ("tokenizer_config.json", "config.json", "turn_detector_config.json"):
        shutil.copy2(teacher / name, fp32 / name)
    configuration = json.loads((fp32 / "config.json").read_text(encoding="utf-8"))
    configuration["vocab_size"] = len(kept)
    write_json(fp32 / "config.json", configuration)
    tokenizer_config = json.loads((fp32 / "tokenizer_config.json").read_text(encoding="utf-8"))
    if tokenizer_config.get("added_tokens_decoder"):
        tokenizer_config["added_tokens_decoder"] = {
            str(remap[int(index)]): value for index, value in tokenizer_config["added_tokens_decoder"].items()}
    write_json(fp32 / "tokenizer_config.json", tokenizer_config)
    # Verify segmentation for every training example, including original ID ordering.
    pending, changed, tokens = [], 0, 0
    def verify(texts):
        nonlocal changed, tokens
        before = original.encode_batch(texts, add_special_tokens=False)
        after = tokenizer.encode_batch(texts, add_special_tokens=False)
        for left, right in zip(before, after, strict=True):
            changed += [kept[index] for index in right.ids] != left.ids
            tokens += len(left.ids)
    for row in read_jsonl(train):
        pending.extend([normalize(row.get("context", "")), normalize(row["text"])])
        if len(pending) >= 1024:
            verify(pending)
            pending = []
    verify(pending)
    write_json(work / "kept_original_ids.json", kept)
    report = {"train_sha256": file_hash(train), "teacher_tokenizer_sha256": file_hash(teacher / "tokenizer.json"),
        "source": "frozen train only; no dev/holdout tokens used to construct the vocabulary",
        "train_records": records, "observed_train_tokens": tokens,
        "original_tokenizer_vocab": original_base_size, "retained_vocab": len(kept),
        "observed_unique_train_ids": len(counts), "character_and_byte_ids": len(characters),
        "segmentation_changed_train": changed, "tokenizer_bytes": (fp32 / "tokenizer.json").stat().st_size}
    write_json(work / "vocabulary_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if changed:
        raise RuntimeError("Compaction changed training segmentation; inspect before building a candidate")


def prune(teacher: Path, work: Path):
    import numpy as np
    import onnx
    from onnx import numpy_helper
    kept = json.loads((work / "kept_original_ids.json").read_text(encoding="utf-8"))
    original_config = json.loads((teacher / "config.json").read_text(encoding="utf-8"))
    graph = onnx.load(str(teacher / "model.onnx"))
    matches = [tensor for tensor in graph.graph.initializer
               if list(tensor.dims) == [original_config["vocab_size"], original_config["hidden_size"]]]
    if len(matches) != 1:
        raise ValueError("Expected one identifiable word embedding initializer")
    embedding = matches[0]
    reduced = np.array(numpy_helper.to_array(embedding)[kept], copy=True)
    embedding.CopyFrom(numpy_helper.from_array(reduced, name=embedding.name))
    path = work / "fp32/model.onnx"
    onnx.save_model(graph, str(path))
    onnx.checker.check_model(str(path))
    report = {"method": "Unigram vocabulary compaction; all encoder weights and 12 layers preserved",
        "source_onnx_sha256": file_hash(teacher / "model.onnx"),
        "original_vocab": original_config["vocab_size"], "retained_vocab": len(kept),
        "original_onnx_bytes": (teacher / "model.onnx").stat().st_size,
        "fp32_onnx_bytes": path.stat().st_size, "fp32_onnx_sha256": file_hash(path)}
    contract_path = work / "fp32/turn_detector_config.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.update({"compression": report, "quality_status": "DEV_NOT_EVALUATED", "padding_to_multiple_of": 8})
    write_json(contract_path, contract)
    write_json(work / "pruning_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def quantize(work: Path, weight_type: str):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    from onnxruntime.quantization import quantize_dynamic, QuantType
    fp32 = work / "fp32"
    output = work / ("int8_" + weight_type.lower() + "_no_constfold")
    output.mkdir(exist_ok=False)
    for path in fp32.glob("*.json"):
        shutil.copy2(path, output / path.name)
    # Row-wise symmetric embedding quantization keeps a separate scale per token.
    model = onnx.load(str(fp32 / "model.onnx"))
    config = json.loads((fp32 / "config.json").read_text(encoding="utf-8"))
    embedding = next(value for value in model.graph.initializer
        if list(value.dims) == [config["vocab_size"], config["hidden_size"]])
    values = numpy_helper.to_array(embedding)
    scale = np.maximum(np.abs(values).max(axis=1) / 127, np.finfo(np.float32).tiny).astype(np.float32)
    quantized = np.clip(np.rint(values / scale[:, None]), -127, 127).astype(np.int8)
    embedding_name = embedding.name
    embedding.CopyFrom(numpy_helper.from_array(quantized, name=embedding_name))
    prefix = "compact_word_embedding"
    model.graph.initializer.extend([numpy_helper.from_array(scale, prefix + "_scale"),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), prefix + "_axes")])
    nodes, matched = [], 0
    for node in model.graph.node:
        if node.op_type == "Gather" and node.input[0] == embedding_name:
            matched += 1
            old_output = node.output[0]
            node.output[0] = prefix + "_int8"
            nodes.extend([node,
                helper.make_node("Cast", [prefix + "_int8"], [prefix + "_float"], to=TensorProto.FLOAT),
                helper.make_node("Gather", [prefix + "_scale", node.input[1]], [prefix + "_gather_scale"], axis=0),
                helper.make_node("Unsqueeze", [prefix + "_gather_scale", prefix + "_axes"], [prefix + "_broadcast_scale"]),
                helper.make_node("Mul", [prefix + "_float", prefix + "_broadcast_scale"], [old_output])])
        else:
            nodes.append(node)
    if matched != 1:
        raise ValueError("Expected exactly one word embedding Gather")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    intermediate = work / "embedding_int8.onnx"
    onnx.save_model(model, str(intermediate))
    del model, values, quantized, embedding, nodes, node, scale
    gc.collect()
    # Eliminate constant Identity aliases before quantization, otherwise tied
    # attention weights remain duplicated in FP32 as well as INT8.
    import onnxruntime as ort
    preprocessed = work / "embedding_int8_no_constfold.onnx"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.optimized_model_filepath = str(preprocessed)
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    # ConstantFolding expands projected relative positions into 24 large FP32
    # tables. Keep them as shared computations so the deployed file stays small.
    session = ort.InferenceSession(str(intermediate), options, providers=["CPUExecutionProvider"],
        disabled_optimizers={"ConstantFolding"}, enable_fallback=0)
    del session
    gc.collect()
    destination = output / "model.onnx"
    quantize_dynamic(str(preprocessed), str(destination), per_channel=True,
        weight_type=getattr(QuantType, weight_type), reduce_range=weight_type == "QInt8",
        op_types_to_quantize=["MatMul"], extra_options={"MatMulConstBOnly": True,
            "WeightSymmetric": weight_type == "QInt8"})
    # ORT 1.23's quantizer sorts only explicit edges and can move If before a
    # tensor captured by its branch. Restore the actual dependency order.
    converted = onnx.load(str(destination))
    sort_graph_with_captures(converted.graph)
    onnx.save_model(converted, str(destination))
    del converted
    onnx.checker.check_model(str(destination))
    contract_path = output / "turn_detector_config.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["compression"]["quantization"] = {"dense_weights": weight_type, "per_channel": True,
        "reduce_range": weight_type == "QInt8", "embedding": "symmetric int8 per token row",
        "preprocessing": "ORT BASIC identity elimination; ConstantFolding disabled to preserve shared relative-position computation",
        "subgraph_capture_order_repaired": True}
    contract["quantized"] = True
    write_json(contract_path, contract)
    report = {"candidate": str(output), "model_bytes": destination.stat().st_size,
        "deployment_bytes": sum(path.stat().st_size for path in output.iterdir() if path.is_file()),
        "onnx_sha256": file_hash(destination), "recipe": contract["compression"]}
    write_json(work / (output.name + "_size.json"), report)
    print(json.dumps(report, indent=2), flush=True)


def weight_only(work: Path):
    """INT8 linear weights with FP32 activations, relative embeddings and head."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    output = work / "weight_only_int8"
    output.mkdir(exist_ok=False)
    for path in (work / "fp32").glob("*.json"):
        shutil.copy2(path, output/path.name)
    model = onnx.load(str(work / "embedding_int8_no_constfold.onnx"))
    matmul_weights = {node.input[1] for node in model.graph.node if node.op_type == "MatMul"}
    initializers, dequantizers, converted = [], [], []
    for tensor in model.graph.initializer:
        keep_float = "rel_embeddings" in tensor.name or ".pooler." in tensor.name or tensor.name.endswith("classifier.weight")
        if tensor.data_type != TensorProto.FLOAT or len(tensor.dims) != 2 or keep_float:
            initializers.append(tensor)
            continue
        values = numpy_helper.to_array(tensor)
        axis = 1 if tensor.name in matmul_weights else 0
        scale = np.maximum(np.abs(values).max(axis=1-axis)/127, np.finfo(np.float32).tiny).astype(np.float32)
        broadcast = scale[None,:] if axis == 1 else scale[:,None]
        quantized = np.clip(np.rint(values/broadcast),-127,127).astype(np.int8)
        prefix = tensor.name + "_weight_only"
        initializers.extend([numpy_helper.from_array(quantized,prefix+"_int8"),
            numpy_helper.from_array(scale,prefix+"_scale"),
            numpy_helper.from_array(np.zeros(scale.shape,dtype=np.int8),prefix+"_zero")])
        dequantizers.append(helper.make_node("DequantizeLinear",
            [prefix+"_int8",prefix+"_scale",prefix+"_zero"],[tensor.name],axis=axis))
        converted.append({"name":tensor.name,"shape":list(tensor.dims),"axis":axis})
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    nodes = dequantizers + list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    sort_graph_with_captures(model.graph)
    destination = output/"model.onnx"
    onnx.save_model(model,str(destination))
    onnx.checker.check_model(str(destination))
    contract_path = output/"turn_detector_config.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["compression"]["quantization"] = {"dense_weights":"symmetric per-channel INT8 with DequantizeLinear",
        "activations":"FP32", "relative_embeddings_and_head":"FP32",
        "embedding":"symmetric INT8 per token row", "quantized_linear_tensors":len(converted)}
    contract["quantized"] = True
    write_json(contract_path,contract)
    report = {"candidate":str(output),"model_bytes":destination.stat().st_size,
        "deployment_bytes":sum(path.stat().st_size for path in output.iterdir() if path.is_file()),
        "onnx_sha256":file_hash(destination),"recipe":contract["compression"],"quantized_tensors":converted}
    write_json(work/"weight_only_int8_size.json",report)
    print(json.dumps({key:value for key,value in report.items() if key!="quantized_tensors"},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["vocabulary", "prune", "quantize", "weight_only"])
    parser.add_argument("--teacher", type=Path, default=Path("models/turn_v2_trial_20261024/onnx"))
    parser.add_argument("--train", type=Path, default=Path("data_v2/frozen/train.jsonl"))
    parser.add_argument("--work", type=Path, default=Path("outputs/compression_20261024/work"))
    parser.add_argument("--weight-type", choices=["QInt8", "QUInt8"], default="QUInt8")
    args = parser.parse_args()
    started = time.monotonic()
    if args.stage == "vocabulary":
        vocabulary(args.teacher, args.train, args.work)
    elif args.stage == "prune":
        prune(args.teacher, args.work)
    elif args.stage == "quantize":
        quantize(args.work, args.weight_type)
    else:
        weight_only(args.work)
    print(json.dumps({"stage": args.stage, "elapsed_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
