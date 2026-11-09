"""Regression for the real ORT quantizer failure on implicit If captures."""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
import pytest

from turn_v2.compress import sort_graph_with_captures


def captured_model():
    branch = helper.make_graph([helper.make_node("Identity", ["captured"], ["branch_output"])],
        "branch", [], [helper.make_tensor_value_info("branch_output", TensorProto.FLOAT, [1])])
    conditional = helper.make_node("If", ["condition"], ["output"], then_branch=branch, else_branch=branch)
    producer = helper.make_node("Add", ["x", "one"], ["captured"])
    graph = helper.make_graph([conditional, producer], "capture_order_regression",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1]),
         helper.make_tensor_value_info("condition", TensorProto.BOOL, [])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        [numpy_helper.from_array(np.array([1], dtype=np.float32), "one")])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)


def test_capture_producer_is_sorted_before_if_and_both_branches_run():
    model = captured_model()
    with pytest.raises(onnx.checker.ValidationError):
        onnx.checker.check_model(model)
    sort_graph_with_captures(model.graph)
    onnx.checker.check_model(model)
    assert [node.op_type for node in model.graph.node] == ["Add", "If"]
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for condition in (False, True):
        actual = session.run(None, {"x": np.array([2], dtype=np.float32), "condition": np.array(condition)})[0]
        np.testing.assert_array_equal(actual, np.array([3], dtype=np.float32))


def test_capture_dependency_cycle_is_rejected():
    model = captured_model()
    model.graph.node[1].input[0] = "output"
    with pytest.raises(ValueError, match="cyclic"):
        sort_graph_with_captures(model.graph)


def test_undefined_outer_capture_is_rejected():
    model = captured_model()
    del model.graph.node[1]
    with pytest.raises(ValueError, match="Unresolved"):
        sort_graph_with_captures(model.graph)
