from collections.abc import Sequence

import pytest

from imob_ai.data import Property, load_catalog
from imob_ai.search import PropertySearchQuery
from imob_ai.semantic.service import SemanticSearchService


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    @property
    def identity(self) -> str:
        return "fake-encoder"

    @property
    def dimension(self) -> int:
        return 2

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            if text == "consulta principal":
                vectors.append([1.0, 0.0])
            elif "Casa térrea" in text:
                vectors.append([1.0, 0.0])
            elif "Apartamento compacto" in text:
                vectors.append([0.8, 0.6])
            else:
                vectors.append([0.0, 1.0])
        return vectors


@pytest.fixture
def catalog() -> list[Property]:
    return load_catalog("data/synthetic/properties.json")


def test_filters_before_semantic_ranking(catalog: list[Property]) -> None:
    encoder = FakeEncoder()
    service = SemanticSearchService(encoder)
    query = PropertySearchQuery(text="consulta principal", city="Franca")

    result = service.search(catalog, query)

    assert result.items
    assert all(item.property.city == "Franca" for item in result.items)
    encoded_documents = encoder.calls[1]
    assert len(encoded_documents) == result.total_candidates


def test_excludes_unavailable_by_default(catalog: list[Property]) -> None:
    result = SemanticSearchService(FakeEncoder()).search(
        catalog, PropertySearchQuery(text="consulta principal")
    )

    assert all(item.property.available for item in result.items)


def test_orders_by_descending_similarity_with_deterministic_tiebreaker(
    catalog: list[Property],
) -> None:
    base = catalog[0]
    custom = [
        base.model_copy(update={"id": "B"}),
        base.model_copy(update={"id": "A"}),
        catalog[1],
    ]
    result = SemanticSearchService(FakeEncoder()).search(
        custom,
        PropertySearchQuery(text="consulta principal", include_unavailable=True),
    )

    assert [item.property.id for item in result.items] == ["A", "B", "IMOB-002"]


def test_applies_limit(catalog: list[Property]) -> None:
    result = SemanticSearchService(FakeEncoder()).search(
        catalog, PropertySearchQuery(text="consulta principal", limit=2)
    )

    assert len(result.items) == 2


def test_returns_empty_after_structured_filters(catalog: list[Property]) -> None:
    result = SemanticSearchService(FakeEncoder()).search(
        catalog,
        PropertySearchQuery(text="consulta principal", city="Cidade Inexistente"),
    )

    assert result.items == []
    assert result.total_candidates == 0


def test_threshold_can_remove_results(catalog: list[Property]) -> None:
    service = SemanticSearchService(FakeEncoder())
    query = PropertySearchQuery(text="consulta principal")

    raw = service.search(catalog, query, threshold=None)
    restricted = service.search(catalog, query, threshold=0.9)

    assert len(raw.items) > len(restricted.items)
    assert all(item.semantic_score >= 0.9 for item in restricted.items)


def test_document_embeddings_are_cached(catalog: list[Property]) -> None:
    encoder = FakeEncoder()
    service = SemanticSearchService(encoder)
    query = PropertySearchQuery(text="consulta principal", city="Franca")

    service.search(catalog, query)
    first_document_batch = encoder.calls[1]
    service.search(catalog, query)

    assert first_document_batch
    assert len(encoder.calls) == 3
