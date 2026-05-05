from app.evaluation.metrics import hit_rate_at_k, precision_at_k, recall_at_k


def test_precision_at_k() -> None:
    actual = ["P1", "P2", "P3"]
    predicted = ["P2", "P4", "P5"]
    assert precision_at_k(actual, predicted, 3) == 1 / 3


def test_recall_at_k() -> None:
    actual = ["P1", "P2", "P3"]
    predicted = ["P2", "P4", "P5"]
    assert recall_at_k(actual, predicted, 3) == 1 / 3


def test_hit_rate_at_k() -> None:
    actual = ["P1", "P2", "P3"]
    predicted = ["P4", "P2", "P5"]
    assert hit_rate_at_k(actual, predicted, 3) == 1.0
