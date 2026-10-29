import pytest

from imob_ai.data import load_catalog
from imob_ai.evaluation.dataset import load_evaluation_cases
from imob_ai.hybrid.calibration import WeightResult, calibrate_weights, select_weight


class EmptyHybridService:
    def search(
        self, catalog: object, query: object, lexical_weight: float, rrf_k: int
    ) -> object:
        class Result:
            items: list[object] = []

        return Result()


def test_calibration_rejects_test_split() -> None:
    catalog = load_catalog("data/synthetic/properties.json")
    cases = load_evaluation_cases("data/evaluation/retrieval_queries.json", catalog)

    with pytest.raises(ValueError, match="somente casos de development"):
        calibrate_weights(EmptyHybridService(), catalog, cases)  # type: ignore[arg-type]


def test_weight_selection_excludes_individual_extremes() -> None:
    results = [
        WeightResult(0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        WeightResult(0.4, 0.6, 0.9, 0.8, 0.7, 1.0, 0.5),
        WeightResult(1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ]

    assert select_weight(results).lexical_weight == 0.4


def test_weight_selection_uses_frozen_metric_order() -> None:
    results = [
        WeightResult(0.4, 0.6, 0.9, 0.7, 0.8, 1.0, 0.5),
        WeightResult(0.5, 0.5, 0.9, 0.8, 0.6, 1.0, 0.5),
    ]

    assert select_weight(results).lexical_weight == 0.5
