import random
from pathlib import Path

import numpy as np


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value < 1e-12:
        return np.ones_like(values, dtype=float)
    return (values - min_value) / (max_value - min_value)
