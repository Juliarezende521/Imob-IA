"""Calibração do limiar usando exclusivamente desenvolvimento."""

from collections.abc import Sequence
from dataclasses import dataclass

from imob_ai.data import Property
from imob_ai.evaluation.runner import evaluate_cases
from imob_ai.evaluation.schemas import EvaluationCase
from imob_ai.semantic.config import THRESHOLD_CANDIDATES
from imob_ai.semantic.service import SemanticSearchService


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    recall_at_5: float
    no_result_accuracy: float
    balanced_score: float


def calibrate_threshold(
    service: SemanticSearchService,
    catalog: Sequence[Property],
    cases: Sequence[EvaluationCase],
    candidates: Sequence[float] = THRESHOLD_CANDIDATES,
) -> list[ThresholdResult]:
    """Avalia candidatos sem permitir casos do split test."""
    development = [case for case in cases if case.split.value == "development"]
    if not development or len(development) != len(cases):
        raise ValueError("Calibração aceita somente casos de development.")

    results: list[ThresholdResult] = []
    for threshold in candidates:

        def retrieve(items: Sequence[Property], case: EvaluationCase) -> list[str]:
            query = case.filters.model_copy(update={"text": case.query, "limit": None})
            search_result = service.search(items, query, threshold=threshold)
            return [match.property.id for match in search_result.items]

        report = evaluate_cases(catalog, development, retrieve)
        recall = report.overall.recall_at_5 or 0.0
        no_result = report.overall.no_result_accuracy or 0.0
        balanced = (
            2 * recall * no_result / (recall + no_result) if recall + no_result else 0.0
        )
        results.append(ThresholdResult(threshold, recall, no_result, balanced))
    return results


def select_threshold(results: Sequence[ThresholdResult]) -> ThresholdResult:
    """Maximiza equilíbrio; desempata por recall e maior limiar."""
    if not results:
        raise ValueError("Nenhum candidato de limiar foi avaliado.")
    return max(
        results,
        key=lambda result: (
            result.balanced_score,
            result.recall_at_5,
            result.threshold,
        ),
    )
