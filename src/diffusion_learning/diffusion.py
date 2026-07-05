from __future__ import annotations

import math
import copy
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .patterns import GRID_SIZE, PATTERN_NAMES


@dataclass(frozen=True)
class Schedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    alpha_bars_prev: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    posterior_variance: torch.Tensor
    posterior_log_variance_clipped: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor

    @classmethod
    def linear(cls, steps: int = 1_000, device: torch.device | str = "cpu") -> "Schedule":
        betas = torch.linspace(1e-4, 0.02, steps, device=device)
        return cls.from_betas(betas)

    @classmethod
    def cosine(cls, steps: int = 1_000, device: torch.device | str = "cpu") -> "Schedule":
        s = 0.008
        x = torch.linspace(0, steps, steps + 1, device=device)
        alpha_bars = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5).square()
        alpha_bars = alpha_bars / alpha_bars[0]
        betas = (1 - alpha_bars[1:] / alpha_bars[:-1]).clamp(1e-4, 0.999)
        return cls.from_betas(betas)

    @classmethod
    def from_betas(cls, betas: torch.Tensor) -> "Schedule":
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        return cls(
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            alpha_bars_prev=alpha_bars_prev,
            sqrt_recip_alphas=torch.sqrt(1.0 / alphas),
            posterior_variance=posterior_variance,
            posterior_log_variance_clipped=posterior_variance.clamp_min(1e-20).log(),
            posterior_mean_coef1=betas * alpha_bars_prev.sqrt() / (1.0 - alpha_bars),
            posterior_mean_coef2=(1.0 - alpha_bars_prev) * alphas.sqrt() / (1.0 - alpha_bars),
        )

    @property
    def steps(self) -> int:
        return int(self.betas.numel())


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: Schedule, noise: torch.Tensor | None = None) -> torch.Tensor:
    noise = torch.randn_like(x0) if noise is None else noise
    ab = schedule.alpha_bars[t].view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device) * (-math.log(10_000.0) / max(half - 1, 1)))
    angles = t.float()[:, None] * freqs[None]
    emb = torch.cat([angles.sin(), angles.cos()], dim=1)
    return F.pad(emb, (0, dim - emb.shape[1]))


def _groups(channels: int) -> int:
    for group in (8, 4, 2):
        if channels % group == 0:
            return group
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    null_label = len(PATTERN_NAMES)

    def __init__(self, base: int = 32, emb_dim: int = 64) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.label_emb = nn.Embedding(len(PATTERN_NAMES) + 1, emb_dim)
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim * 4), nn.SiLU(), nn.Linear(emb_dim * 4, emb_dim))
        self.in_conv = nn.Conv2d(1, base, 3, padding=1)
        self.down1 = ResBlock(base, base, emb_dim)
        self.downsample1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.down2 = ResBlock(base * 2, base * 2, emb_dim)
        self.downsample2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)
        self.mid = ResBlock(base * 4, base * 4, emb_dim)
        self.upsample2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.up2 = ResBlock(base * 4, base * 2, emb_dim)
        self.upsample1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.up1 = ResBlock(base * 2, base, emb_dim)
        self.out_norm = nn.GroupNorm(_groups(base), base)
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.emb_dim)) + self.label_emb(labels)
        h0 = self.in_conv(x)
        h1 = self.down1(h0, emb)
        h2 = self.down2(self.downsample1(h1), emb)
        h3 = self.mid(self.downsample2(h2), emb)
        h = self.upsample2(h3)
        h = self.up2(torch.cat([h, h2], dim=1), emb)
        h = self.upsample1(h)
        h = self.up1(torch.cat([h, h1], dim=1), emb)
        return self.out_conv(F.silu(self.out_norm(h)))


def train(
    model: TinyUNet,
    data: torch.Tensor,
    schedule: Schedule,
    *,
    steps: int,
    seed: int,
    lr: float = 2e-4,
    batch_size: int = 64,
    ema_decay: float = 0.999,
    cond_drop: float = 0.1,
) -> tuple[list[float], TinyUNet]:
    seed_everything(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    losses: list[float] = []

    for _ in range(steps):
        labels = torch.randint(0, data.shape[0], (batch_size,), device=data.device)
        x0 = data[labels]
        train_labels = labels.masked_fill(torch.rand(labels.shape, device=data.device) < cond_drop, TinyUNet.null_label)
        t = torch.randint(0, schedule.steps, (x0.shape[0],), device=data.device)
        noise = torch.randn_like(x0)
        loss = F.mse_loss(model(q_sample(x0, t, schedule, noise), t, train_labels), noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p.mul_(ema_decay).add_(p, alpha=1.0 - ema_decay)
        losses.append(float(loss.detach().cpu()))

    return losses, ema_model


def _guided_eps(model: TinyUNet, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    if guidance_scale == 1.0:
        return model(x, t, labels)
    nulls = torch.full_like(labels, TinyUNet.null_label)
    eps_uncond = model(x, t, nulls)
    eps_cond = model(x, t, labels)
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


@torch.no_grad()
def sample(
    model: TinyUNet,
    schedule: Schedule,
    label: int,
    *,
    sample_steps: int,
    seed: int,
    sampler: str = "ddpm",
    guidance_scale: float = 2.0,
    eta: float = 0.0,
    num_samples: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = schedule.betas.device
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((num_samples, 1, GRID_SIZE, GRID_SIZE), generator=gen, device=device)
    labels = torch.full((num_samples,), label, device=device, dtype=torch.long)
    xs = []
    pred_x0s = []

    if sampler == "ddpm":
        record_times = set(torch.linspace(schedule.steps - 1, 0, sample_steps, device=device).long().tolist())
        for t_scalar in torch.arange(schedule.steps - 1, -1, -1, device=device):
            t = t_scalar.repeat(num_samples)
            eps = _guided_eps(model, x, t, labels, guidance_scale)
            ab = schedule.alpha_bars[t].view(-1, 1, 1, 1)
            pred_x0 = ((x - (1.0 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            if int(t_scalar) in record_times:
                xs.append(x.detach().cpu())
                pred_x0s.append(pred_x0.detach().cpu())
            coef1 = schedule.posterior_mean_coef1[t].view(-1, 1, 1, 1)
            coef2 = schedule.posterior_mean_coef2[t].view(-1, 1, 1, 1)
            mean = coef1 * pred_x0 + coef2 * x
            variance = schedule.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn(x.shape, generator=gen, device=device) if int(t_scalar) > 0 else torch.zeros_like(x)
            x = mean + variance.sqrt() * noise
        xs.append(x.detach().cpu())
        pred_x0s.append(x.detach().cpu())
        return torch.stack(xs), torch.stack(pred_x0s)

    if sampler != "ddim":
        raise ValueError(f"unknown sampler {sampler!r}")

    times = torch.linspace(schedule.steps - 1, 0, sample_steps, device=device).long().unique_consecutive()
    for i, t_scalar in enumerate(times):
        t = t_scalar.repeat(num_samples)
        eps = _guided_eps(model, x, t, labels, guidance_scale)
        ab = schedule.alpha_bars[t].view(-1, 1, 1, 1)
        pred_x0 = ((x - (1.0 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
        xs.append(x.detach().cpu())
        pred_x0s.append(pred_x0.detach().cpu())

        if i == len(times) - 1:
            x = pred_x0
            continue

        prev_t = times[i + 1].repeat(num_samples)
        ab_prev = schedule.alpha_bars[prev_t].view(-1, 1, 1, 1)
        sigma = eta * ((1.0 - ab_prev) / (1.0 - ab)).sqrt() * (1.0 - ab / ab_prev).clamp_min(0).sqrt()
        noise = torch.randn(x.shape, generator=gen, device=device)
        x = ab_prev.sqrt() * pred_x0 + (1.0 - ab_prev - sigma.square()).clamp_min(0).sqrt() * eps + sigma * noise

    xs.append(x.detach().cpu())
    pred_x0s.append(x.detach().cpu())
    return torch.stack(xs), torch.stack(pred_x0s)
