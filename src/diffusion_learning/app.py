from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch

from .diffusion import Schedule, TinyDenoiser, sample, seed_everything, train
from .patterns import PATTERN_NAMES, all_patterns, pattern_array, target_index


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def pca2(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def animate(target: str, losses: list[float], xs: torch.Tensor, pred_x0s: torch.Tensor) -> None:
    target_grid = pattern_array(target)
    traj = xs[:, 0].numpy()
    preds = pred_x0s[:, 0].numpy()
    distances = np.linalg.norm((traj - target_grid).reshape(len(traj), -1), axis=1)
    patterns = np.stack([pattern_array(name).reshape(-1) for name in PATTERN_NAMES])
    coords = pca2(np.vstack([patterns, traj.reshape(len(traj), -1)]))
    pattern_xy = coords[: len(PATTERN_NAMES)]
    traj_xy = coords[len(PATTERN_NAMES) :]

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

    ax[1, 0].plot(losses, lw=1.5)
    ax[1, 0].set_title("Training loss")
    ax[1, 0].set_xlabel("step")

    ax[1, 1].scatter(pattern_xy[:, 0], pattern_xy[:, 1], c="black", s=20)
    for name, xy in zip(PATTERN_NAMES, pattern_xy):
        ax[1, 1].text(xy[0], xy[1], name, fontsize=8)
    path_line, = ax[1, 1].plot([], [], c="tab:blue", lw=1.5)
    path_dot, = ax[1, 1].plot([], [], "o", c="tab:red")
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
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--sample-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    seed_everything(args.seed)
    data = all_patterns(device)
    schedule = Schedule.linear(device=device)
    model = TinyDenoiser().to(device)
    losses = train(model, data, schedule, steps=args.train_steps, seed=args.seed)
    xs, pred_x0s = sample(
        model,
        schedule,
        target_index(args.target),
        sample_steps=args.sample_steps,
        seed=args.seed,
        stochastic=args.stochastic,
    )
    animate(args.target, losses, xs, pred_x0s)


if __name__ == "__main__":
    main()
