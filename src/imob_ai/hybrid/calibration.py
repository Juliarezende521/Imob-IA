"""Calibração de pesos Weighted RRF somente em desenvolvimento."""

from collections.abc import Sequence
from dataclasses import dataclass

from imob_ai.data import Property
from imob_ai.evaluation.runner import evaluate_cases
from imob_ai.evaluation.schemas import EvaluationCase
from imob_ai.hybrid.config import RRF_K, WEIGHT_CANDIDATES
from imob_ai.hybrid.service import HybridSearchService


@dataclass(frozen=True)
class WeightResult:
    lexical_weight: float
    semantic_weight: float
    mrr: float
    recall_at_5: float
    precision_at_5: float
    hit_rate_at_5: float
    no_result_accuracy: float


def calibrate_weights(
    service: HybridSearchService,
    catalog: Sequence[Property],
    cases: Sequence[EvaluationCase],
    candidates: Sequence[float] = WEIGHT_CANDIDATES,
) -> list[WeightResult]:
    """Avalia a grade e rejeita qualquer presença do split test."""
    if not cases or any(case.split.value != "development" for case in cases):
        raise ValueError("Calibração aceita somente casos de development.")

    results: list[WeightResult] = []
    for lexical_weight in candidates:

        def retrieve(items: Sequence[Property], case: EvaluationCase) -> list[str]:
            query = case.filters.model_copy(update={"text": case.query, "limit": None})
            result = service.search(items, query, lexical_weight, RRF_K)
            return [match.property.id for match in result.items]

        metrics = evaluate_cases(catalog, cases, retrieve).overall
        results.append(
            WeightResult(
                lexical_weight=lexical_weight,
                semantic_weight=1 - lexical_weight,
                mrr=metrics.mrr or 0.0,
                recall_at_5=metrics.recall_at_5 or 0.0,
                precision_at_5=metrics.precision_at_5 or 0.0,
                hit_rate_at_5=metrics.hit_rate_at_5 or 0.0,
                no_result_accuracy=metrics.no_result_accuracy or 0.0,
            )
        )
    return results


def select_weight(results: Sequence[WeightResult]) -> WeightResult:
    """Aplica a ordem de métricas congelada e exige pesos híbridos."""
    hybrid = [result for result in results if 0 < result.lexical_weight < 1]
    if not hybrid:
        raise ValueError("Nenhuma configuração híbrida foi avaliada.")
    return max(
        hybrid,
        key=lambda result: (
            result.mrr,
            result.recall_at_5,
            result.precision_at_5,
            result.hit_rate_at_5,
            result.no_result_accuracy,
            -abs(result.lexical_weight - 0.5),
        ),
    )
