"""Busca semântica exata em memória."""

from collections.abc import Sequence
from math import isfinite, sqrt
from typing import Optional

from imob_ai.data import Property
from imob_ai.search import PropertySearchQuery, search_properties
from imob_ai.semantic.document import build_property_document
from imob_ai.semantic.encoder import TextEncoder, validate_embeddings
from imob_ai.semantic.schemas import SemanticMatch, SemanticSearchResult


class SemanticSearchService:
    """Aplica filtros antes de ranquear candidatos por cosseno."""

    def __init__(self, encoder: TextEncoder) -> None:
        self._encoder = encoder
        self._document_cache: dict[tuple[str, str], list[float]] = {}

    def search(
        self,
        catalog: Sequence[Property],
        query: PropertySearchQuery,
        threshold: Optional[float] = None,
    ) -> SemanticSearchResult:
        if not query.text:
            raise ValueError("Busca semântica exige texto livre.")
        if threshold is not None and not -1 <= threshold <= 1:
            raise ValueError("O limiar deve estar entre -1 e 1.")

        filter_query = query.model_copy(update={"text": None, "limit": None})
        filtered = search_properties(catalog, filter_query)
        candidates = [item.property for item in filtered.items]
        if not candidates:
            return SemanticSearchResult(
                items=[],
                total_candidates=0,
                threshold=threshold,
                model_name=self._encoder.identity,
            )

        query_vector = self._encoder.encode([query.text])[0]
        validate_embeddings([query_vector], 1, self._encoder.dimension)
        document_vectors = self._document_vectors(candidates)

        matches = [
            SemanticMatch(
                property=property_,
                semantic_score=cosine_similarity(query_vector, vector),
            )
            for property_, vector in zip(candidates, document_vectors)
        ]
        if threshold is not None:
            matches = [match for match in matches if match.semantic_score >= threshold]
        matches.sort(key=lambda match: (-match.semantic_score, match.property.id))
        if query.limit is not None:
            matches = matches[: query.limit]
        return SemanticSearchResult(
            items=matches,
            total_candidates=len(candidates),
            threshold=threshold,
            model_name=self._encoder.identity,
        )

    def _document_vectors(self, properties: Sequence[Property]) -> list[list[float]]:
        documents = [build_property_document(property_) for property_ in properties]
        missing = [
            document
            for document in documents
            if (self._encoder.identity, document) not in self._document_cache
        ]
        if missing:
            encoded = self._encoder.encode(missing)
            validate_embeddings(encoded, len(missing), self._encoder.dimension)
            for document, vector in zip(missing, encoded):
                self._document_cache[(self._encoder.identity, document)] = vector
        return [
            self._document_cache[(self._encoder.identity, document)]
            for document in documents
        ]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Calcula cosseno e rejeita vetores inválidos ou incompatíveis."""
    if len(left) != len(right):
        raise ValueError("Embeddings possuem dimensões incompatíveis.")
    if not left:
        raise ValueError("Embedding não pode ser vazio.")
    if not all(isfinite(float(value)) for value in [*left, *right]):
        raise ValueError("Embedding contém NaN ou infinito.")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Embedding com norma zero não permite cosseno.")
    score = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, score))
