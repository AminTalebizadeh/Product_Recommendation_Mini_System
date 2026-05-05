from __future__ import annotations


def precision_at_k(actual: list[str], predicted: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    predicted_at_k = predicted[:k]
    if not predicted_at_k:
        return 0.0
    hits = len(set(actual).intersection(predicted_at_k))
    return hits / k


def recall_at_k(actual: list[str], predicted: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not actual:
        return 0.0
    predicted_at_k = predicted[:k]
    hits = len(set(actual).intersection(predicted_at_k))
    return hits / len(set(actual))


def hit_rate_at_k(actual: list[str], predicted: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    predicted_at_k = predicted[:k]
    hits = len(set(actual).intersection(predicted_at_k))
    return 1.0 if hits > 0 else 0.0
