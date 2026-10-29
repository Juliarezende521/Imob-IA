"""Métricas puras para avaliar rankings de recuperação."""

from collections.abc import Sequence


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Calcula relevantes nas primeiras K posições dividido por K."""
    _validate_k(k)
    hits = sum(property_id in relevant for property_id in retrieved[:k])
    return hits / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Calcula relevantes nas primeiras K posições dividido pelo total relevante."""
    _validate_k(k)
    if not relevant:
        raise ValueError("Recall exige pelo menos um item relevante.")
    hits = sum(property_id in relevant for property_id in retrieved[:k])
    return hits / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Retorna o inverso da posição do primeiro resultado relevante."""
    if not relevant:
        raise ValueError("MRR exige pelo menos um item relevante.")
    for rank, property_id in enumerate(retrieved, start=1):
        if property_id in relevant:
            return 1 / rank
    return 0.0


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Retorna 1 quando existe ao menos um relevante nas primeiras K posições."""
    _validate_k(k)
    return float(any(property_id in relevant for property_id in retrieved[:k]))


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("K deve ser positivo.")
