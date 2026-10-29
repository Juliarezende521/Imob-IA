from collections.abc import Sequence

import pytest

from imob_ai.data import load_catalog
from imob_ai.evaluation.dataset import load_evaluation_cases
from imob_ai.evaluation.schemas import EvaluationCase
from imob_ai.semantic.threshold import (
    ThresholdResult,
    calibrate_threshold,
    select_threshold,
)


class EmptyService:
    def search(self, catalog: object, query: object, threshold: float) -> object:
        class Result:
            items: list[object] = []

        return Result()


def test_calibration_rejects_test_split() -> None:
    catalog = load_catalog("data/synthetic/properties.json")
    cases = load_evaluation_cases("data/evaluation/retrieval_queries.json", catalog)

    with pytest.raises(ValueError, match="somente casos de development"):
        calibrate_threshold(EmptyService(), catalog, cases, [0.5])  # type: ignore[arg-type]


def test_calibration_accepts_only_development() -> None:
    catalog = load_catalog("data/synthetic/properties.json")
    cases: Sequence[EvaluationCase] = load_evaluation_cases(
        "data/evaluation/retrieval_queries.json", catalog
    )
    development = [case for case in cases if case.split.value == "development"]

    results = calibrate_threshold(
        EmptyService(),
        catalog,
        development,
        [0.4, 0.5],  # type: ignore[arg-type]
    )

    assert [result.threshold for result in results] == [0.4, 0.5]


def test_selects_best_balanced_threshold() -> None:
    results = [
        ThresholdResult(0.3, 0.9, 0.2, 0.327),
        ThresholdResult(0.5, 0.8, 0.8, 0.8),
        ThresholdResult(0.7, 0.4, 1.0, 0.571),
    ]

    assert select_threshold(results).threshold == 0.5


def test_threshold_tie_prefers_more_conservative_value() -> None:
    results = [
        ThresholdResult(0.1, 0.8, 0.8, 0.8),
        ThresholdResult(0.2, 0.8, 0.8, 0.8),
    ]

    assert select_threshold(results).threshold == 0.2
