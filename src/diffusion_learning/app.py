from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import numpy as np
import torch

from .diffusion import Schedule, TinyUNet, sample, seed_everything, train
from .patterns import PATTERN_NAMES, all_patterns, pattern_array, target_index

DEFAULT_CHECKPOINT = "diffusion-demo.pt"


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def pca(points: np.ndarray, dims: int = 3) -> np.ndarray:
    # Project 1024-pixel grids into a tiny coordinate system for plotting.
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:dims].T


def pca_reference_clouds(patterns: np.ndarray, copies: int = 24) -> tuple[np.ndarray, np.ndarray]:
    # Around each ideal pattern, make a small cloud so PCA has local structure.
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(len(patterns)), copies)
    noise = rng.normal(0, 0.12, (len(labels), patterns.shape[1]))
    return np.clip(patterns[labels] + noise, 0, 1), labels


def probability_grid(grid: np.ndarray) -> np.ndarray:
    # Model images live in [-1, 1]; probability space is easier to compare.
    return np.clip((grid + 1.0) * 0.5, 0.0, 1.0)


def smooth_path(points: np.ndarray, factor: int = 8) -> tuple[np.ndarray, np.ndarray]:
    frames = np.arange(len(points))
    dense_frames = np.linspace(0, len(points) - 1, max(2, (len(points) - 1) * factor + 1))
    dense = np.column_stack([np.interp(dense_frames, frames, points[:, dim]) for dim in range(points.shape[1])])
    return dense_frames, dense


def density_surface(points: np.ndarray, size: int = 35) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Simple Gaussian kernel density over the PCA plane. This is only a teaching
    # visualization, not part of the diffusion model.
    xy = points[:, :2]
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    pad = np.maximum((xy_max - xy_min) * 0.08, 0.5)
    x = np.linspace(xy_min[0] - pad[0], xy_max[0] + pad[0], size)
    y = np.linspace(xy_min[1] - pad[1], xy_max[1] + pad[1], size)
    xx, yy = np.meshgrid(x, y)
    sigma = max(float(np.linalg.norm(xy_max - xy_min)) / 8.0, 1.0)
    d2 = (xx[..., None] - xy[:, 0]) ** 2 + (yy[..., None] - xy[:, 1]) ** 2
    zz = np.exp(-d2 / (2.0 * sigma * sigma)).mean(axis=2)
    zz = zz / max(float(zz.max()), 1e-12)
    return xx, yy, zz


def step_explanation(frame: int, total: int) -> str:
    """Describe what the learner should notice at this point in sampling."""
    progress = frame / max(total - 1, 1)
    if progress < 0.34:
        return "Early: x_t is mostly noise; the clean estimate is still uncertain."
    if progress < 0.67:
        return "Middle: large-scale structure appears as the model removes noise."
    return "Late: small corrections sharpen the sample toward the chosen class."


def animate(target: str, losses: list[float], xs: torch.Tensor, pred_x0s: torch.Tensor, output: str | None = None) -> None:
    target_grid = pattern_array(target)
    traj = xs[:, 0, 0].numpy()
    preds = pred_x0s[:, 0, 0].numpy()
    distances = np.linalg.norm((traj - target_grid).reshape(len(traj), -1), axis=1)
    patterns = np.stack([probability_grid(pattern_array(name)).reshape(-1) for name in PATTERN_NAMES])
    refs, ref_labels = pca_reference_clouds(patterns)
    # Plot the model's clean estimates in probability space; the noisy x_t path
    # is less useful for seeing whether the sampler is moving toward a class.
    traj_probs = probability_grid(preds).reshape(len(preds), -1)
    coords = pca(np.vstack([refs, patterns, traj_probs]))
    ref_xyz = coords[: len(refs)]
    pattern_xyz = coords[len(refs) : len(refs) + len(PATTERN_NAMES)]
    traj_xy = coords[len(refs) + len(PATTERN_NAMES) :, :2]
    surface_x, surface_y, surface_z = density_surface(np.vstack([ref_xyz[:, :2], pattern_xyz[:, :2], traj_xy]))
    dense_frames, dense_traj_xy = smooth_path(traj_xy)
    dense_path_z = np.interp(dense_frames, np.arange(len(traj_xy)), np.full(len(traj_xy), 1.08))

    fig, ax = plt.subplots(2, 3, figsize=(11, 7))
    fig.subplots_adjust(bottom=0.24, hspace=0.32)
    ax[1, 1].remove()
    ax_path = fig.add_subplot(2, 3, 5, projection="3d")
    fig.canvas.manager.set_window_title("Grid Diffusion Learning")

    images = [
        ax[0, 0].imshow(target_grid, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest"),
        ax[0, 1].imshow(traj[0], cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest"),
        ax[0, 2].imshow(preds[0], cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest"),
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
    ax_path.plot_surface(surface_x, surface_y, surface_z, cmap="viridis", alpha=0.42, linewidth=0, antialiased=True)
    for i, color in enumerate(colors):
        cluster = ref_xyz[ref_labels == i]
        ax_path.scatter(cluster[:, 0], cluster[:, 1], np.full(len(cluster), 1.02), color=color, s=8, alpha=0.18, edgecolors="none")
    ax_path.scatter(pattern_xyz[:, 0], pattern_xyz[:, 1], np.full(len(pattern_xyz), 1.04), c=colors, s=28, marker="x")
    target_xyz = pattern_xyz[target_index(target)]
    ax_path.text(target_xyz[0], target_xyz[1], 1.08, target, fontsize=9, fontweight="bold")
    path_line, = ax_path.plot([], [], [], c="tab:blue", lw=1.5)
    path_dot, = ax_path.plot([], [], [], "o", c="tab:red")
    xy_min = coords[:, :2].min(axis=0)
    xy_max = coords[:, :2].max(axis=0)
    xy_pad = np.maximum((xy_max - xy_min) * 0.08, 0.5)
    ax_path.set_xlim(xy_min[0] - xy_pad[0], xy_max[0] + xy_pad[0])
    ax_path.set_ylim(xy_min[1] - xy_pad[1], xy_max[1] + xy_pad[1])
    ax_path.set_zlim(0, 1.12)
    ax_path.set_title("PCA probability-density path")

    dist_line, = ax[1, 2].plot([], [], c="tab:green", lw=1.5)
    ax[1, 2].set_xlim(0, len(distances) - 1)
    ax[1, 2].set_ylim(0, max(float(distances.max()), 1.0))
    ax[1, 2].set_title("Distance to target")
    ax[1, 2].set_xlabel("inference step")

    lesson = fig.text(0.5, 0.135, step_explanation(0, len(traj)), ha="center", fontsize=9)

    def update(i: int):
        images[1].set_data(traj[i])
        images[2].set_data(preds[i])
        upto = np.searchsorted(dense_frames, i, side="right")
        path_line.set_data_3d(dense_traj_xy[:upto, 0], dense_traj_xy[:upto, 1], dense_path_z[:upto])
        path_dot.set_data_3d([traj_xy[i, 0]], [traj_xy[i, 1]], [1.08])
        dist_line.set_data(np.arange(i + 1), distances[: i + 1])
        lesson.set_text(step_explanation(i, len(traj)))
        return [images[1], images[2], path_line, path_dot, dist_line]

    if "agg" in plt.get_backend().lower() and output is None:
        # Headless test mode: render one frame to catch plotting errors.
        update(len(traj) - 1)
        fig.canvas.draw()
        plt.close(fig)
        return

    slider_ax = fig.add_axes((0.29, 0.045, 0.42, 0.025))
    frame_slider = Slider(slider_ax, "Denoising step", 0, len(traj) - 1, valinit=0, valstep=1)
    previous = Button(fig.add_axes((0.04, 0.035, 0.07, 0.045)), "Prev")
    play = Button(fig.add_axes((0.12, 0.035, 0.07, 0.045)), "Pause")
    following = Button(fig.add_axes((0.74, 0.035, 0.07, 0.045)), "Next")
    restart = Button(fig.add_axes((0.82, 0.035, 0.09, 0.045)), "Restart")

    def set_playing(playing: bool) -> None:
        (animation.resume if playing else animation.pause)()
        play.label.set_text("Pause" if playing else "Play")

    def show_frame(value: float) -> None:
        set_playing(False)
        update(int(value))
        fig.canvas.draw_idle()

    def move(delta: int) -> None:
        frame_slider.set_val(np.clip(int(frame_slider.val) + delta, 0, len(traj) - 1))

    def toggle(_event: object) -> None:
        set_playing(play.label.get_text() == "Play")
        fig.canvas.draw_idle()

    def start_over(_event: object) -> None:
        animation.frame_seq = animation.new_frame_seq()
        frame_slider.set_val(0)
        set_playing(True)
        fig.canvas.draw_idle()

    def animate_frame(i: int):
        frame_slider.eventson = False
        frame_slider.set_val(i)
        frame_slider.eventson = True
        return update(i)

    frame_slider.on_changed(show_frame)
    previous.on_clicked(lambda _event: move(-1))
    following.on_clicked(lambda _event: move(1))
    restart.on_clicked(start_over)
    play.on_clicked(toggle)
    if output:
        frame_slider.eventson = False
        frame_slider.set_val(len(traj) - 1)
        frame_slider.eventson = True
        update(len(traj) - 1)
        fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"saved visualization to {output}", file=sys.stderr)
        return
    animation = FuncAnimation(fig, animate_frame, frames=len(traj), interval=60, blit=False, repeat=False)
    # Keep interactive objects alive for the lifetime of the Matplotlib window.
    fig._diffusion_controls = (animation, frame_slider, previous, play, following, restart)
    plt.show()


def save_checkpoint(path: str, ema_model: TinyUNet, losses: list[float], schedule: str, diffusion_steps: int) -> None:
    # Save only what inference needs: EMA weights plus plotting/schedule metadata.
    torch.save(
        {
            "ema_model": ema_model.state_dict(),
            "losses": losses,
            "schedule": schedule,
            "diffusion_steps": diffusion_steps,
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device) -> tuple[TinyUNet, list[float], str, int]:
    # Checkpoints are device-neutral; map tensors to the requested runtime device.
    checkpoint = torch.load(path, map_location=device)
    model = TinyUNet().to(device)
    model.load_state_dict(checkpoint["ema_model"])
    model.eval()
    return model, checkpoint.get("losses", []), checkpoint.get("schedule", "cosine"), checkpoint.get("diffusion_steps", 1_000)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a tiny diffusion model, then explore its denoising path.",
        epilog="Tip: train once with --mode train, then vary inference flags with --mode infer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("run", "train", "infer"), default="run", help="run both phases, or only one")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="model file to save or load")
    parser.add_argument("--target", choices=PATTERN_NAMES, default="diagonal", help="class to generate")
    parser.add_argument("--train-steps", type=int, default=10_000, help="optimizer updates")
    parser.add_argument("--diffusion-steps", type=int, default=1_000, help="noise levels in the learned process")
    parser.add_argument("--sample-steps", type=int, help="recorded DDPM frames or DDIM denoising steps")
    parser.add_argument("--schedule", choices=("cosine", "linear"), default="cosine", help="how noise grows during training")
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm", help="faithful DDPM or faster DDIM inference")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="training examples per update")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="smoothing for sampling weights")
    parser.add_argument("--cond-drop", type=float, default=0.1, help="fraction of labels hidden during CFG training")
    parser.add_argument("--guidance-scale", type=float, default=2.0, help="strength of class conditioning at inference")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM randomness; zero is deterministic")
    parser.add_argument("--seed", type=int, default=0, help="reproducible training and sampling seed")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or another PyTorch device")
    parser.add_argument("--output", help="save a PNG instead of opening an interactive window")
    args = parser.parse_args()

    device = choose_device(args.device)
    seed_everything(args.seed)

    if args.mode in ("run", "train"):
        # Training mode creates a fresh model and stores its EMA copy.
        data = all_patterns(device)
        schedule = getattr(Schedule, args.schedule)(steps=args.diffusion_steps, device=device)
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
        save_checkpoint(args.checkpoint, ema_model, losses, args.schedule, args.diffusion_steps)
        print(f"saved checkpoint to {args.checkpoint}", file=sys.stderr)
        if args.mode == "train":
            return
    else:
        # Inference mode skips training entirely and reuses the checkpoint.
        ema_model, losses, checkpoint_schedule, checkpoint_steps = load_checkpoint(args.checkpoint, device)
        args.schedule = checkpoint_schedule
        args.diffusion_steps = checkpoint_steps
        schedule = getattr(Schedule, args.schedule)(steps=args.diffusion_steps, device=device)

    sample_steps = args.sample_steps or (args.diffusion_steps if args.sampler == "ddpm" else 80)
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
    output = args.output
    if output or os.environ.get("CI") or os.environ.get("CODEX_CI"):
        plt.switch_backend("Agg")
        output = output or "diffusion-result.png"
    animate(args.target, losses, xs, pred_x0s, output)


if __name__ == "__main__":
    main()
