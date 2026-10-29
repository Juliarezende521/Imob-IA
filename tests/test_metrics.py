import pytest

from imob_ai.evaluation.metrics import (
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k() -> None:
    assert precision_at_k(["A", "B", "C"], {"A", "C"}, 3) == pytest.approx(2 / 3)


def test_precision_uses_k_when_fewer_results_are_returned() -> None:
    assert precision_at_k(["A"], {"A"}, 5) == pytest.approx(0.2)


def test_recall_at_k_with_multiple_relevant_items() -> None:
    assert recall_at_k(["A", "X", "B"], {"A", "B", "C"}, 3) == pytest.approx(2 / 3)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["X", "A", "B"], {"A", "B"}) == pytest.approx(0.5)


def test_hit_at_k() -> None:
    assert hit_at_k(["X", "A"], {"A"}, 1) == 0.0
    assert hit_at_k(["X", "A"], {"A"}, 2) == 1.0


def test_perfect_result() -> None:
    retrieved = ["A", "B"]
    relevant = {"A", "B"}

    assert precision_at_k(retrieved, relevant, 1) == 1.0
    assert recall_at_k(retrieved, relevant, 2) == 1.0
    assert reciprocal_rank(retrieved, relevant) == 1.0


def test_no_relevant_retrieved() -> None:
    retrieved = ["X", "Y"]
    relevant = {"A"}

    assert precision_at_k(retrieved, relevant, 2) == 0.0
    assert recall_at_k(retrieved, relevant, 2) == 0.0
    assert reciprocal_rank(retrieved, relevant) == 0.0
    assert hit_at_k(retrieved, relevant, 2) == 0.0


@pytest.mark.parametrize("metric", [precision_at_k, recall_at_k, hit_at_k])
def test_non_positive_k_is_rejected(metric: object) -> None:
    with pytest.raises(ValueError, match="K deve ser positivo"):
        metric([], {"A"}, 0)  # type: ignore[operator]
