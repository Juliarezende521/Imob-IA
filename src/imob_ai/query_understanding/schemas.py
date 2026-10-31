"""Contratos tipados da interpretação de consultas imobiliárias."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from imob_ai.data.schemas import PropertyPurpose, PropertyType


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExtractedPropertyFilters(FrozenModel):
    """Filtros extraídos que possuem equivalência explícita na busca atual."""

    purpose: Optional[PropertyPurpose] = None
    property_type: Optional[PropertyType] = None
    city: Optional[str] = None
    neighborhood: Optional[str] = None
    max_price: Optional[float] = Field(default=None, ge=0)
    min_bedrooms: Optional[int] = Field(default=None, ge=0)
    min_bathrooms: Optional[int] = Field(default=None, ge=0)
    min_parking_spaces: Optional[int] = Field(default=None, ge=0)
    pets_allowed: Optional[bool] = None
    furnished: Optional[bool] = None


class RecognizedConstraint(FrozenModel):
    field: str
    value: Any
    source_text: str
    corrected_from: Optional[str] = None


class UnresolvedConstraint(FrozenModel):
    field: str
    value: str
    reason: str


class QueryInterpretation(FrozenModel):
    original_query: str
    normalized_query: str
    extracted_filters: ExtractedPropertyFilters
    remaining_text: Optional[str] = None
    recognized_constraints: list[RecognizedConstraint] = Field(default_factory=list)
    unresolved_constraints: list[UnresolvedConstraint] = Field(default_factory=list)
    requires_clarification: bool = False
