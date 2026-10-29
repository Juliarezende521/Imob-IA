"""Serviço de Weighted Reciprocal Rank Fusion."""

from collections.abc import Sequence

from imob_ai.data import Property
from imob_ai.hybrid.config import RRF_K
from imob_ai.hybrid.schemas import HybridMatch, HybridSearchResult
from imob_ai.search import PropertySearchQuery, search_properties
from imob_ai.semantic.config import SEMANTIC_THRESHOLD
from imob_ai.semantic.service import SemanticSearchService


class HybridSearchService:
    """Combina rankings existentes sem misturar suas escalas de score."""

    def __init__(self, semantic_service: SemanticSearchService) -> None:
        self._semantic_service = semantic_service

    def search(
        self,
        catalog: Sequence[Property],
        query: PropertySearchQuery,
        lexical_weight: float,
        rrf_k: int = RRF_K,
    ) -> HybridSearchResult:
        if not query.text:
            raise ValueError("Busca híbrida exige texto livre.")
        if not 0 <= lexical_weight <= 1:
            raise ValueError("Peso lexical deve estar entre 0 e 1.")
        if rrf_k <= 0:
            raise ValueError("RRF k deve ser positivo.")
        semantic_weight = 1 - lexical_weight
        full_query = query.model_copy(update={"limit": None})

        lexical = search_properties(catalog, full_query)
        semantic = self._semantic_service.search(catalog, full_query, threshold=None)
        lexical_by_id = {
            match.property.id: (rank, match)
            for rank, match in enumerate(lexical.items, start=1)
        }
        semantic_by_id = {
            match.property.id: (rank, match)
            for rank, match in enumerate(semantic.items, start=1)
        }

        matches: list[HybridMatch] = []
        for property_id in sorted(set(lexical_by_id) | set(semantic_by_id)):
            lexical_entry = lexical_by_id.get(property_id)
            semantic_entry = semantic_by_id.get(property_id)
            lexical_rank = lexical_entry[0] if lexical_entry else None
            semantic_rank = semantic_entry[0] if semantic_entry else None
            lexical_score = lexical_entry[1].lexical_score if lexical_entry else None
            semantic_score = (
                semantic_entry[1].semantic_score if semantic_entry else None
            )

            has_evidence = lexical_rank is not None or (
                semantic_score is not None and semantic_score >= SEMANTIC_THRESHOLD
            )
            if not has_evidence:
                continue

            lexical_contribution = (
                lexical_weight / (rrf_k + lexical_rank)
                if lexical_rank is not None
                else 0.0
            )
            semantic_contribution = (
                semantic_weight / (rrf_k + semantic_rank)
                if semantic_rank is not None
                else 0.0
            )
            property_ = (
                lexical_entry[1].property
                if lexical_entry
                else semantic_entry[1].property  # type: ignore[index]
            )
            matches.append(
                HybridMatch(
                    property=property_,
                    lexical_rank=lexical_rank,
                    semantic_rank=semantic_rank,
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                    lexical_contribution=lexical_contribution,
                    semantic_contribution=semantic_contribution,
                    hybrid_score=lexical_contribution + semantic_contribution,
                )
            )

        matches.sort(
            key=lambda match: (
                -match.hybrid_score,
                -(match.semantic_score if match.semantic_score is not None else -1),
                -(match.lexical_score if match.lexical_score is not None else -1),
                match.property.id,
            )
        )
        if query.limit is not None:
            matches = matches[: query.limit]
        return HybridSearchResult(
            items=matches,
            total_candidates=len(semantic.items),
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
            rrf_k=rrf_k,
        )
