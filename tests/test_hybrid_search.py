from collections.abc import Sequence

import pytest

from imob_ai.data import load_catalog
from imob_ai.hybrid.service import HybridSearchService
from imob_ai.search import PropertySearchQuery
from imob_ai.semantic.service import SemanticSearchService


class HybridFakeEncoder:
    @property
    def identity(self) -> str:
        return "hybrid-fake"

    @property
    def dimension(self) -> int:
        return 2

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_candidate_absent_from_lexical_has_no_artificial_rank() -> None:
    catalog = load_catalog("data/synthetic/properties.json")[:2]
    service = HybridSearchService(SemanticSearchService(HybridFakeEncoder()))

    result = service.search(
        catalog,
        PropertySearchQuery(text="termo-semantico-inexistente"),
        lexical_weight=0.5,
    )

    assert result.items
    assert all(item.lexical_rank is None for item in result.items)
    assert all(item.lexical_score is None for item in result.items)
    assert all(item.lexical_contribution == 0 for item in result.items)
    assert all(item.semantic_rank is not None for item in result.items)


def test_rrf_contributions_are_inspectable() -> None:
    catalog = load_catalog("data/synthetic/properties.json")
    service = HybridSearchService(SemanticSearchService(HybridFakeEncoder()))

    result = service.search(
        catalog,
        PropertySearchQuery(text="casa", limit=1),
        lexical_weight=0.6,
    )
    match = result.items[0]

    assert match.lexical_rank is not None
    assert match.semantic_rank is not None
    assert match.lexical_contribution == pytest.approx(0.6 / (60 + match.lexical_rank))
    assert match.semantic_contribution == pytest.approx(
        0.4 / (60 + match.semantic_rank)
    )
    assert match.hybrid_score == pytest.approx(
        match.lexical_contribution + match.semantic_contribution
    )


def test_hybrid_applies_filters_and_excludes_unavailable() -> None:
    catalog = load_catalog("data/synthetic/properties.json")
    service = HybridSearchService(SemanticSearchService(HybridFakeEncoder()))

    result = service.search(
        catalog,
        PropertySearchQuery(text="casa", city="Franca"),
        lexical_weight=0.5,
    )

    assert result.items
    assert all(item.property.city == "Franca" for item in result.items)
    assert all(item.property.available for item in result.items)


def test_hybrid_respects_limit() -> None:
    catalog = load_catalog("data/synthetic/properties.json")
    service = HybridSearchService(SemanticSearchService(HybridFakeEncoder()))

    result = service.search(
        catalog,
        PropertySearchQuery(text="casa", limit=2),
        lexical_weight=0.5,
    )

    assert len(result.items) == 2


def test_hybrid_returns_empty_after_filters() -> None:
    catalog = load_catalog("data/synthetic/properties.json")
    service = HybridSearchService(SemanticSearchService(HybridFakeEncoder()))

    result = service.search(
        catalog,
        PropertySearchQuery(text="casa", city="Cidade Inexistente"),
        lexical_weight=0.5,
    )

    assert result.items == []


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_hybrid_rejects_invalid_weight(weight: float) -> None:
    service = HybridSearchService(SemanticSearchService(HybridFakeEncoder()))

    with pytest.raises(ValueError, match="Peso lexical"):
        service.search(
            load_catalog("data/synthetic/properties.json"),
            PropertySearchQuery(text="casa"),
            lexical_weight=weight,
        )
