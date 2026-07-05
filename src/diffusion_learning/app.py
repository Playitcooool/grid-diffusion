from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch

from .diffusion import Schedule, TinyUNet, sample, seed_everything, train
from .patterns import PATTERN_NAMES, all_patterns, pattern_array, target_index


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def pca2(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def pca_reference_clouds(patterns: np.ndarray, copies: int = 24) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(len(patterns)), copies)
    noise = rng.normal(0, 0.25, (len(labels), patterns.shape[1]))
    return np.clip(patterns[labels] + noise, -1, 1), labels


def animate(target: str, losses: list[float], xs: torch.Tensor, pred_x0s: torch.Tensor) -> None:
    target_grid = pattern_array(target)
    traj = xs[:, 0, 0].numpy()
    preds = pred_x0s[:, 0, 0].numpy()
    distances = np.linalg.norm((traj - target_grid).reshape(len(traj), -1), axis=1)
    patterns = np.stack([pattern_array(name).reshape(-1) for name in PATTERN_NAMES])
    refs, ref_labels = pca_reference_clouds(patterns)
    coords = pca2(np.vstack([refs, patterns, traj.reshape(len(traj), -1)]))
    ref_xy = coords[: len(refs)]
    pattern_xy = coords[len(refs) : len(refs) + len(PATTERN_NAMES)]
    traj_xy = coords[len(refs) + len(PATTERN_NAMES) :]

    fig, ax = plt.subplots(2, 3, figsize=(11, 7))
    fig.canvas.manager.set_window_title("Grid Diffusion Learning")

    images = [
        ax[0, 0].imshow(target_grid, cmap="coolwarm", vmin=-1, vmax=1),
        ax[0, 1].imshow(traj[0], cmap="coolwarm", vmin=-1, vmax=1),
        ax[0, 2].imshow(preds[0], cmap="coolwarm", vmin=-1, vmax=1),
    ]
    ax[0, 0].set_title("Target grid")
    ax[0, 1].set_title("Current denoising grid")
    ax[0, 2].set_title("Predicted clean grid")
    for a in ax[0]:
        a.set_xticks([])
        a.set_yticks([])

    loss_x = np.arange(len(losses))
    ax[1, 0].plot(losses, lw=0.8, alpha=0.28)
    if losses:
        window = min(100, max(1, len(losses) // 10))
        avg = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax[1, 0].plot(loss_x[window - 1 :], avg, lw=1.7)
    ax[1, 0].set_title("Training loss")
    ax[1, 0].set_xlabel("step")

    colors = plt.get_cmap("tab10")(np.arange(len(PATTERN_NAMES)) % 10)
    for i, color in enumerate(colors):
        cluster = ref_xy[ref_labels == i]
        ax[1, 1].scatter(cluster[:, 0], cluster[:, 1], color=color, s=14, alpha=0.22, edgecolors="none")
    ax[1, 1].scatter(pattern_xy[:, 0], pattern_xy[:, 1], c=colors, s=28, marker="x")
    for name, xy in zip(PATTERN_NAMES, pattern_xy):
        ax[1, 1].text(xy[0], xy[1], name, fontsize=8)
    path_line, = ax[1, 1].plot([], [], c="tab:blue", lw=1.5)
    path_dot, = ax[1, 1].plot([], [], "o", c="tab:red")
    xy_min = coords.min(axis=0)
    xy_max = coords.max(axis=0)
    xy_pad = np.maximum((xy_max - xy_min) * 0.08, 0.5)
    ax[1, 1].set_xlim(xy_min[0] - xy_pad[0], xy_max[0] + xy_pad[0])
    ax[1, 1].set_ylim(xy_min[1] - xy_pad[1], xy_max[1] + xy_pad[1])
    ax[1, 1].set_title("2D PCA image-space path")

    dist_line, = ax[1, 2].plot([], [], c="tab:green", lw=1.5)
    ax[1, 2].set_xlim(0, len(distances) - 1)
    ax[1, 2].set_ylim(0, max(float(distances.max()), 1.0))
    ax[1, 2].set_title("Distance to target")
    ax[1, 2].set_xlabel("inference step")

    def update(i: int):
        images[1].set_data(traj[i])
        images[2].set_data(preds[i])
        path_line.set_data(traj_xy[: i + 1, 0], traj_xy[: i + 1, 1])
        path_dot.set_data([traj_xy[i, 0]], [traj_xy[i, 1]])
        dist_line.set_data(np.arange(i + 1), distances[: i + 1])
        return [images[1], images[2], path_line, path_dot, dist_line]

    if "agg" in plt.get_backend().lower():
        update(len(traj) - 1)
        fig.canvas.draw()
        plt.close(fig)
        return

    fig._diffusion_anim = FuncAnimation(fig, update, frames=len(traj), interval=60, blit=False, repeat=False)
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn diffusion on tiny numeric grids.")
    parser.add_argument("--target", choices=PATTERN_NAMES, default="diagonal")
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--diffusion-steps", type=int, default=1_000)
    parser.add_argument("--sample-steps", type=int)
    parser.add_argument("--schedule", choices=("cosine", "linear"), default="cosine")
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--cond-drop", type=float, default=0.1)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    seed_everything(args.seed)
    data = all_patterns(device)
    schedule = getattr(Schedule, args.schedule)(steps=args.diffusion_steps, device=device)
    sample_steps = args.sample_steps or (args.diffusion_steps if args.sampler == "ddpm" else 80)
    model = TinyUNet().to(device)
    losses, ema_model = train(
        model,
        data,
        schedule,
        steps=args.train_steps,
        seed=args.seed,
        lr=args.lr,
        batch_size=args.batch_size,
        ema_decay=args.ema_decay,
        cond_drop=args.cond_drop,
        progress=True,
    )
    xs, pred_x0s = sample(
        ema_model,
        schedule,
        target_index(args.target),
        sample_steps=sample_steps,
        seed=args.seed,
        sampler=args.sampler,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
    )
    animate(args.target, losses, xs, pred_x0s)


if __name__ == "__main__":
    main()
