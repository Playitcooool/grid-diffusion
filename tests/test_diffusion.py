import unittest

import torch

from diffusion_learning.diffusion import Schedule, TinyDenoiser, q_sample, sample, seed_everything
from diffusion_learning.patterns import PATTERN_NAMES, all_patterns


class DiffusionTests(unittest.TestCase):
    def test_patterns_shape_and_seed(self):
        seed_everything(7)
        a = torch.randn(3)
        seed_everything(7)
        b = torch.randn(3)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(all_patterns().shape, (len(PATTERN_NAMES), 1, 16, 16))

    def test_forward_diffusion_adds_more_noise_later(self):
        schedule = Schedule.linear(steps=10)
        x0 = all_patterns()[0:1]
        noise = torch.ones_like(x0)
        early = q_sample(x0, torch.tensor([0]), schedule, noise)
        late = q_sample(x0, torch.tensor([9]), schedule, noise)
        self.assertGreater(torch.mean(torch.abs(late - x0)).item(), torch.mean(torch.abs(early - x0)).item())

    def test_one_training_step_has_finite_loss_and_gradients(self):
        schedule = Schedule.linear(steps=10)
        model = TinyDenoiser(hidden=32, emb_dim=8)
        x0 = all_patterns()[0:2]
        labels = torch.tensor([0, 1])
        t = torch.tensor([1, 7])
        noise = torch.randn_like(x0)
        loss = torch.nn.functional.mse_loss(model(q_sample(x0, t, schedule, noise), t, labels), noise)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_sampler_shape_without_gui(self):
        schedule = Schedule.linear(steps=12)
        model = TinyDenoiser(hidden=32, emb_dim=8)
        xs, pred = sample(model, schedule, 2, sample_steps=5, seed=0)
        self.assertEqual(xs.shape, (6, 1, 16, 16))
        self.assertEqual(pred.shape, xs.shape)
        target = all_patterns()[2]
        start = torch.linalg.vector_norm(xs[0] - target)
        final = torch.linalg.vector_norm(xs[-1] - target)
        self.assertLess(final, start)


if __name__ == "__main__":
    unittest.main()
