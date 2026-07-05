from __future__ import annotations

import numpy as np
import torch

PATTERN_NAMES = ("dot", "bar", "diagonal", "box", "cross")
GRID_SIZE = 16


def _empty() -> np.ndarray:
    return np.full((GRID_SIZE, GRID_SIZE), -1.0, dtype=np.float32)


def pattern_array(name: str) -> np.ndarray:
    grid = _empty()
    c = GRID_SIZE // 2

    if name == "dot":
        grid[c - 2 : c + 2, c - 2 : c + 2] = 1.0
    elif name == "bar":
        grid[:, c - 2 : c + 2] = 1.0
    elif name == "diagonal":
        for i in range(GRID_SIZE):
            grid[max(0, i - 1) : min(GRID_SIZE, i + 2), i] = 1.0
    elif name == "box":
        grid[3:13, 3] = 1.0
        grid[3:13, 12] = 1.0
        grid[3, 3:13] = 1.0
        grid[12, 3:13] = 1.0
    elif name == "cross":
        grid[c - 1 : c + 1, :] = 1.0
        grid[:, c - 1 : c + 1] = 1.0
    else:
        raise ValueError(f"unknown target {name!r}; choose one of {', '.join(PATTERN_NAMES)}")

    return grid


def all_patterns(device: torch.device | str = "cpu") -> torch.Tensor:
    data = np.stack([pattern_array(name) for name in PATTERN_NAMES])
    return torch.tensor(data[:, None, :, :], device=device)


def target_index(name: str) -> int:
    return PATTERN_NAMES.index(name)
