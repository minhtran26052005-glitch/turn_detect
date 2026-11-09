import unittest

import numpy as np
import torch

from turn_v2.qat_student import GroupInt4FakeQuant, WEIGHTS, latent_state, pack_parameter, prepare_qat
from turn_v2.quantize_student4 import quantize_groups
from turn_v2.tiny_student import TinyTurnGRU


class QatTests(unittest.TestCase):
    def test_forward_matches_packed_4bit_with_half_rounding(self):
        weight = torch.tensor(np.random.default_rng(5).normal(size=(3, 35)), dtype=torch.float32, requires_grad=True)
        _, scale, _ = quantize_groups(weight.detach().numpy(), 32)
        quantizer = GroupInt4FakeQuant(torch.from_numpy(scale), 32)
        actual = quantizer(weight)
        codes, stored_scale, expected = pack_parameter(weight.detach().numpy(), scale, 32)
        np.testing.assert_array_equal(actual.detach().numpy(), expected)
        self.assertTrue((codes >= -8).all() and (codes <= 7).all())
        self.assertEqual(stored_scale.dtype, np.float16)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(weight.grad).all())
        self.assertGreater(float(weight.grad.abs().sum()), 0.)

    def test_every_matrix_receives_gradient_and_latent_checkpoint_loads(self):
        torch.manual_seed(1)
        torch.set_num_threads(2)
        model = TinyTurnGRU(vocab_size=17, embedding_size=8, hidden_size=12)
        prepare_qat(model, 32)
        ids = torch.randint(1, 17, (3, 8))
        mask = torch.ones_like(ids)
        model(ids, mask).square().sum().backward()
        for name in WEIGHTS:
            parent, key = name.rsplit(".", 1)
            grad = getattr(model.get_submodule(parent).parametrizations, key).original.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0.)
        state = latent_state(model)
        self.assertFalse(any("parametrizations" in n for n in state))
        restored = TinyTurnGRU(vocab_size=17, embedding_size=8, hidden_size=12)
        restored.load_state_dict(state, strict=True)

    def test_clipped_ste_and_zero_group(self):
        weight = torch.tensor([[0., .1, 20., -20.]], requires_grad=True)
        fake = GroupInt4FakeQuant(torch.ones((1, 1), dtype=torch.float16), 4)(weight)
        np.testing.assert_array_equal(fake.detach().numpy(), [[0., 0., 7., -8.]])
        fake.sum().backward()
        np.testing.assert_array_equal(weight.grad.numpy(), [[1., 1., 0., 0.]])
        zero = torch.zeros((2, 32))
        _, scale, _ = quantize_groups(zero.numpy(), 32)
        self.assertTrue(torch.isfinite(GroupInt4FakeQuant(torch.from_numpy(scale), 32)(zero)).all())


if __name__ == "__main__":
    unittest.main()
