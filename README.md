# Grid Diffusion Learning

Tiny diffusion demo using `16x16` numeric grids instead of image datasets. It trains a small conditional PyTorch denoiser on five synthetic patterns, then animates reverse diffusion from noise toward one target pattern.

## Quickstart

```bash
uv sync
uv run diffusion-demo --target diagonal --train-steps 1500 --sample-steps 80 --seed 0
```

Targets: `dot`, `bar`, `diagonal`, `box`, `cross`.

Useful flags:

```bash
uv run diffusion-demo --target cross --train-steps 800 --sample-steps 60 --device cpu
uv run diffusion-demo --target box --stochastic
```

`--device auto` uses Apple MPS when available, otherwise CPU.

## What It Teaches

The clean grid is `x_0`. Forward diffusion adds Gaussian noise:

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

The model learns to predict `epsilon` from the noisy grid `x_t`, timestep `t`, and target pattern id. During reverse diffusion, the predicted noise gives an estimate of the clean grid:

```text
pred_x0 = (x_t - sqrt(1 - alpha_bar_t) * pred_epsilon) / sqrt(alpha_bar_t)
```

By default sampling is deterministic DDIM-style, which makes the path easier to see. `--stochastic` adds noise during reverse sampling.

## Animation Panels

- Target grid: the selected clean pattern.
- Current grid: the current reverse-diffusion state.
- Predicted clean grid: the model's current `x_0` estimate.
- Training loss: loss over training steps.
- PCA path: a 2D projection of grid space, with the denoising trajectory moving through it.
- Distance: Euclidean distance from the current grid to the target over inference steps.

## Tests

```bash
uv run python -m unittest
```
