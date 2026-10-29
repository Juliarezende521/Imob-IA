"""Resultados tipados da busca semântica."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from imob_ai.data import Property


class SemanticMatch(BaseModel):
    """Imóvel e sua similaridade cosseno com a consulta."""

    model_config = ConfigDict(frozen=True)

    property: Property
    semantic_score: float = Field(ge=-1, le=1)


class SemanticSearchResult(BaseModel):
    """Ranking semântico depois dos filtros e do limiar."""

    model_config = ConfigDict(frozen=True)

    items: list[SemanticMatch]
    total_candidates: int = Field(ge=0)
    threshold: Optional[float]
    model_name: str
