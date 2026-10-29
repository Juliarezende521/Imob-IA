"""Resultados explicáveis da fusão de rankings."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from imob_ai.data import Property


class HybridMatch(BaseModel):
    """Imóvel com posições, scores originais e contribuições RRF."""

    model_config = ConfigDict(frozen=True)

    property: Property
    lexical_rank: Optional[int] = Field(default=None, gt=0)
    semantic_rank: Optional[int] = Field(default=None, gt=0)
    lexical_score: Optional[int] = Field(default=None, ge=0)
    semantic_score: Optional[float] = Field(default=None, ge=-1, le=1)
    lexical_contribution: float = Field(ge=0)
    semantic_contribution: float = Field(ge=0)
    hybrid_score: float = Field(ge=0)


class HybridSearchResult(BaseModel):
    """Ranking fundido após filtros, evidência e limite."""

    model_config = ConfigDict(frozen=True)

    items: list[HybridMatch]
    total_candidates: int = Field(ge=0)
    lexical_weight: float = Field(ge=0, le=1)
    semantic_weight: float = Field(ge=0, le=1)
    rrf_k: int = Field(gt=0)
