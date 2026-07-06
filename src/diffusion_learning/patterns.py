from __future__ import annotations

import numpy as np
import torch

PATTERN_NAMES = ("dot", "bar", "diagonal", "box", "cross", "ring", "checker", "zigzag", "spiral", "corners")
GRID_SIZE = 32


def _empty() -> np.ndarray:
    # Diffusion data uses -1 as background and +1 as foreground.
    return np.full((GRID_SIZE, GRID_SIZE), -1.0, dtype=np.float32)


def pattern_array(name: str) -> np.ndarray:
    # Hand-built patterns replace an image dataset so the demo stays tiny.
    grid = _empty()
    c = GRID_SIZE // 2
    yy, xx = np.indices((GRID_SIZE, GRID_SIZE))

    if name == "dot":
        grid[c - 3 : c + 3, c - 3 : c + 3] = 1.0
    elif name == "bar":
        grid[:, c - 3 : c + 3] = 1.0
    elif name == "diagonal":
        grid[np.abs(yy - xx) <= 1] = 1.0
    elif name == "box":
        grid[6:26, 6:9] = 1.0
        grid[6:26, 23:26] = 1.0
        grid[6:9, 6:26] = 1.0
        grid[23:26, 6:26] = 1.0
    elif name == "cross":
        grid[c - 2 : c + 2, :] = 1.0
        grid[:, c - 2 : c + 2] = 1.0
    elif name == "ring":
        r = np.hypot(xx - (c - 0.5), yy - (c - 0.5))
        grid[(r >= 9) & (r <= 12)] = 1.0
    elif name == "checker":
        grid[((xx // 4) + (yy // 4)) % 2 == 0] = 1.0
    elif name == "zigzag":
        points = np.array([4, 12, 20, 28, 20, 12, 4])
        for x0, x1, y in zip(points, points[1:], range(3, 28, 4)):
            xs = np.linspace(x0, x1, 5).round().astype(int)
            for x, yy0 in zip(xs, range(y, y + 5)):
                grid[max(0, yy0 - 1) : min(GRID_SIZE, yy0 + 2), max(0, x - 1) : min(GRID_SIZE, x + 2)] = 1.0
    elif name == "spiral":
        for inset in range(5, 16, 4):
            grid[inset, inset : GRID_SIZE - inset] = 1.0
            grid[inset : GRID_SIZE - inset, GRID_SIZE - inset - 1] = 1.0
            grid[GRID_SIZE - inset - 1, inset + 3 : GRID_SIZE - inset] = 1.0
            grid[inset + 4 : GRID_SIZE - inset, inset + 3] = 1.0
    elif name == "corners":
        grid[3:10, 3:10] = 1.0
        grid[3:10, 22:29] = 1.0
        grid[22:29, 3:10] = 1.0
        grid[22:29, 22:29] = 1.0
    else:
        raise ValueError(f"unknown target {name!r}; choose one of {', '.join(PATTERN_NAMES)}")

    return grid


def all_patterns(device: torch.device | str = "cpu") -> torch.Tensor:
    data = np.stack([pattern_array(name) for name in PATTERN_NAMES])
    return torch.tensor(data[:, None, :, :], device=device)


def target_index(name: str) -> int:
    return PATTERN_NAMES.index(name)
