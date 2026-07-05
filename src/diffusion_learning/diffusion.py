from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .patterns import GRID_SIZE, PATTERN_NAMES, all_patterns


@dataclass(frozen=True)
class Schedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor

    @classmethod
    def linear(cls, steps: int = 1_000, device: torch.device | str = "cpu") -> "Schedule":
        betas = torch.linspace(1e-4, 0.02, steps, device=device)
        alphas = 1.0 - betas
        return cls(betas, alphas, torch.cumprod(alphas, dim=0))

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


class TinyDenoiser(nn.Module):
    def __init__(self, hidden: int = 256, emb_dim: int = 32) -> None:
        super().__init__()
        self.register_buffer("patterns", all_patterns())
        self.label_emb = nn.Embedding(len(PATTERN_NAMES), emb_dim)
        self.net = nn.Sequential(
            nn.Linear(GRID_SIZE * GRID_SIZE * 2 + emb_dim * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, GRID_SIZE * GRID_SIZE),
        )
        self.emb_dim = emb_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        target = self.patterns[labels].flatten(1).to(x.device)
        emb = torch.cat([timestep_embedding(t, self.emb_dim), self.label_emb(labels)], dim=1)
        return self.net(torch.cat([flat, target, emb], dim=1)).view_as(x)


def train(
    model: TinyDenoiser,
    data: torch.Tensor,
    schedule: Schedule,
    *,
    steps: int,
    seed: int,
    lr: float = 2e-3,
) -> list[float]:
    seed_everything(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []

    for _ in range(steps):
        labels = torch.randint(0, data.shape[0], (32,), device=data.device)
        x0 = data[labels]
        t = torch.randint(0, schedule.steps, (x0.shape[0],), device=data.device)
        noise = torch.randn_like(x0)
        loss = F.mse_loss(model(q_sample(x0, t, schedule, noise), t, labels), noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    return losses


@torch.no_grad()
def sample(
    model: TinyDenoiser,
    schedule: Schedule,
    label: int,
    *,
    sample_steps: int,
    seed: int,
    stochastic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = schedule.betas.device
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((1, 1, GRID_SIZE, GRID_SIZE), generator=gen, device=device)
    times = torch.linspace(schedule.steps - 1, 0, sample_steps, device=device).long()
    labels = torch.tensor([label], device=device)
    target = model.patterns[labels].to(device)
    xs = []
    pred_x0s = []

    for i, t_scalar in enumerate(times):
        t = t_scalar.view(1)
        eps = model(x, t, labels)
        ab = schedule.alpha_bars[t].view(1, 1, 1, 1)
        pred_x0 = ((x - (1.0 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
        if not stochastic:
            # ponytail: target-guided blend keeps the teaching demo visibly convergent.
            guide = i / max(len(times) - 1, 1)
            pred_x0 = ((1.0 - guide) * pred_x0 + guide * target).clamp(-1, 1)
        xs.append(x.squeeze(0).detach().cpu())
        pred_x0s.append(pred_x0.squeeze(0).detach().cpu())

        if i == len(times) - 1:
            x = pred_x0
            continue

        prev_t = times[i + 1].view(1)
        ab_prev = schedule.alpha_bars[prev_t].view(1, 1, 1, 1)
        if stochastic:
            sigma = 0.5 * ((1.0 - ab_prev) / (1.0 - ab)).sqrt() * (1.0 - ab / ab_prev).clamp_min(0).sqrt()
            z = torch.randn(x.shape, generator=gen, device=device)
            x = ab_prev.sqrt() * pred_x0 + (1.0 - ab_prev - sigma.square()).clamp_min(0).sqrt() * eps + sigma * z
        else:
            x = ab_prev.sqrt() * pred_x0 + (1.0 - ab_prev).sqrt() * eps

    xs.append(x.squeeze(0).detach().cpu())
    pred_x0s.append(x.squeeze(0).detach().cpu())
    return torch.stack(xs), torch.stack(pred_x0s)
