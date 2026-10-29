"""Contratos de entrada e saída da busca de imóveis."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from imob_ai.data.schemas import Property, PropertyPurpose, PropertyType
from imob_ai.search.normalization import normalize_text


class SearchSort(str, Enum):
    """Ordenações disponíveis para a busca."""

    RELEVANCE = "relevancia"
    PRICE_ASC = "menor_preco"
    PRICE_DESC = "maior_preco"
    AREA_ASC = "menor_area"
    AREA_DESC = "maior_area"


class PropertySearchQuery(BaseModel):
    """Filtros opcionais combinados com lógica AND."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", frozen=True)

    purpose: Optional[PropertyPurpose] = None
    property_type: Optional[PropertyType] = None
    city: Optional[str] = Field(default=None, min_length=1)
    neighborhood: Optional[str] = Field(default=None, min_length=1)
    min_price: Optional[float] = Field(default=None, ge=0)
    max_price: Optional[float] = Field(default=None, ge=0)
    min_bedrooms: Optional[int] = Field(default=None, ge=0)
    min_bathrooms: Optional[int] = Field(default=None, ge=0)
    min_parking_spaces: Optional[int] = Field(default=None, ge=0)
    min_area_m2: Optional[float] = Field(default=None, ge=0)
    max_area_m2: Optional[float] = Field(default=None, ge=0)
    pets_allowed: Optional[bool] = None
    furnished: Optional[bool] = None
    required_features: list[str] = Field(default_factory=list)
    include_unavailable: bool = False
    text: Optional[str] = Field(default=None, min_length=1)
    sort: SearchSort = SearchSort.RELEVANCE
    limit: Optional[int] = Field(default=None, gt=0, le=50)

    @field_validator("required_features")
    @classmethod
    def normalize_required_features(cls, features: list[str]) -> list[str]:
        """Normaliza características e descarta valores vazios e repetidos."""
        normalized: list[str] = []
        for feature in features:
            value = normalize_text(feature)
            if value and value not in normalized:
                normalized.append(value)
        return normalized

    @model_validator(mode="after")
    def validate_ranges(self) -> "PropertySearchQuery":
        """Garante que os limites mínimos não ultrapassem os máximos."""
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("min_price não pode ser maior que max_price.")
        if (
            self.min_area_m2 is not None
            and self.max_area_m2 is not None
            and self.min_area_m2 > self.max_area_m2
        ):
            raise ValueError("min_area_m2 não pode ser maior que max_area_m2.")
        return self


class SearchMatch(BaseModel):
    """Imóvel acompanhado das evidências produzidas pelo baseline."""

    model_config = ConfigDict(frozen=True)

    property: Property
    lexical_score: int = Field(ge=0)
    matched_terms: list[str]
    matched_criteria: list[str]


class SearchResult(BaseModel):
    """Resultado completo antes e depois da aplicação do limite."""

    model_config = ConfigDict(frozen=True)

    items: list[SearchMatch]
    total_before_limit: int = Field(ge=0)
    sort: SearchSort
