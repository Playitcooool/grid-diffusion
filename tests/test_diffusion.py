import unittest

import torch

from diffusion_learning.diffusion import Schedule, TinyUNet, q_sample, sample, seed_everything, train
from diffusion_learning.app import step_explanation
from diffusion_learning.patterns import GRID_SIZE, PATTERN_NAMES, all_patterns


class DiffusionTests(unittest.TestCase):
    def test_step_explanations_cover_sampling_phases(self):
        explanations = [step_explanation(i, 10) for i in (0, 4, 9)]
        self.assertEqual(len(set(explanations)), 3)
        self.assertTrue(all(explanations))

    def test_patterns_shape_and_seed(self):
        seed_everything(7)
        a = torch.randn(3)
        seed_everything(7)
        b = torch.randn(3)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(all_patterns().shape, (len(PATTERN_NAMES), 1, GRID_SIZE, GRID_SIZE))

    def test_each_pattern_has_foreground_and_background(self):
        patterns = all_patterns()
        for name, pattern in zip(PATTERN_NAMES, patterns):
            with self.subTest(name=name):
                self.assertTrue(torch.any(pattern == -1))
                self.assertTrue(torch.any(pattern == 1))

    def test_forward_diffusion_adds_more_noise_later(self):
        schedule = Schedule.linear(steps=10)
        x0 = all_patterns()[0:1]
        noise = torch.ones_like(x0)
        early = q_sample(x0, torch.tensor([0]), schedule, noise)
        late = q_sample(x0, torch.tensor([9]), schedule, noise)
        self.assertEqual(early.shape, x0.shape)
        self.assertGreater(torch.mean(torch.abs(late - x0)).item(), torch.mean(torch.abs(early - x0)).item())

    def test_cosine_schedule_is_finite_and_decreasing(self):
        schedule = Schedule.cosine(steps=20)
        self.assertTrue(torch.isfinite(schedule.betas).all())
        self.assertTrue(torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1]))

    def test_unet_forward_has_finite_gradients(self):
        schedule = Schedule.linear(steps=10)
        model = TinyUNet(base=8, emb_dim=16)
        x0 = all_patterns()[0:2]
        labels = torch.tensor([0, 1])
        t = torch.tensor([1, 7])
        noise = torch.randn_like(x0)
        loss = torch.nn.functional.mse_loss(model(q_sample(x0, t, schedule, noise), t, labels), noise)
        loss.backward()
        self.assertEqual(model(x0, t, labels).shape, x0.shape)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_train_returns_finite_loss_and_ema(self):
        schedule = Schedule.linear(steps=8)
        model = TinyUNet(base=8, emb_dim=16)
        losses, ema_model = train(model, all_patterns(), schedule, steps=1, seed=0, batch_size=4)
        self.assertEqual(len(losses), 1)
        self.assertTrue(torch.isfinite(torch.tensor(losses)).all())
        self.assertTrue(all(torch.isfinite(p).all() for p in ema_model.parameters()))

    def test_samplers_shape_without_gui(self):
        schedule = Schedule.linear(steps=12)
        model = TinyUNet(base=8, emb_dim=16)
        for sampler in ("ddpm", "ddim"):
            with self.subTest(sampler=sampler):
                xs, pred = sample(model, schedule, 2, sample_steps=5, seed=0, sampler=sampler, guidance_scale=1.0)
                self.assertEqual(xs.shape, (6, 1, 1, GRID_SIZE, GRID_SIZE))
                self.assertEqual(pred.shape, xs.shape)
                self.assertTrue(torch.isfinite(xs).all())
                self.assertTrue(torch.isfinite(pred).all())

    def test_cfg_sampling_runs_without_nans(self):
        schedule = Schedule.linear(steps=8)
        model = TinyUNet(base=8, emb_dim=16)
        xs, _ = sample(model, schedule, 1, sample_steps=4, seed=0, sampler="ddim", guidance_scale=2.0)
        self.assertTrue(torch.isfinite(xs).all())


if __name__ == "__main__":
    unittest.main()
