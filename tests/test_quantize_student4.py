"""Verify INT4 packing and the scales that the ONNX runtime actually consumes."""
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from turn_v2.quantize_student4 import quantize_groups


class Int4Tests(unittest.TestCase):
    def test_signed_nibbles_partial_groups_and_runtime_reconstruction(self):
        values = np.random.default_rng(7).normal(size=(3, 19)).astype(np.float32)
        values[0, :4] = [-8., -1., 0., 7.]
        codes, scales, restored = quantize_groups(values, 8)
        tensor = numpy_helper.from_array(codes, "packed")
        self.assertEqual(tensor.data_type, TensorProto.INT4)
        self.assertEqual(len(tensor.raw_data), (values.size + 1) // 2)
        np.testing.assert_array_equal(numpy_helper.to_array(tensor), codes)
        self.assertTrue((codes.astype(np.int8) < 0).any())
        graph = helper.make_graph([
            helper.make_node("DequantizeLinear", ["packed", "scale"], ["half"], axis=1, block_size=8),
            helper.make_node("Cast", ["half"], ["actual"], to=TensorProto.FLOAT)], "int4_test", [],
            [helper.make_tensor_value_info("actual", TensorProto.FLOAT, list(values.shape))],
            [tensor, numpy_helper.from_array(scales, "scale")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
        onnx.checker.check_model(model, full_check=True)
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        actual = session.run(None, {})[0]
        np.testing.assert_array_equal(actual, restored.astype(np.float16).astype(np.float32))

    def test_zero_groups_and_weight_only_mse_search(self):
        zero = np.zeros((2, 16), np.float32)
        codes, scales, restored = quantize_groups(zero, 16)
        self.assertTrue(np.isfinite(scales).all())
        np.testing.assert_array_equal(codes.astype(np.int8), zero.astype(np.int8))
        np.testing.assert_array_equal(restored, zero)
        values = np.random.default_rng(3).normal(size=(5, 35)).astype(np.float32)
        _, _, mse = quantize_groups(values, 16, "mse")
        _, _, maximum = quantize_groups(values, 16, "max")
        self.assertLessEqual(float(((values - mse) ** 2).sum()), float(((values - maximum) ** 2).sum()) + 1e-6)


if __name__ == "__main__":
    unittest.main()
