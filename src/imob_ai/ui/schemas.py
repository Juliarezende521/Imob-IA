"""Modelos locais que validam os contratos HTTP da API."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthLive(StrictModel):
    status: str


class HealthReady(StrictModel):
    status: str
    classifier_ready: bool
    catalog_ready: bool
    configuration_valid: bool


class Metadata(StrictModel):
    api_version: str
    default_classifier: str
    default_search: str
    available_intents: list[str]
    synthetic_property_count: int = Field(ge=0)
    synthetic_data: bool
    demonstration_notice: str
    transformer: str


class Classification(StrictModel):
    intent: str
    model_score: float = Field(ge=0, le=1)
    second_intent: str
    second_model_score: float = Field(ge=0, le=1)
    margin: float = Field(ge=0, le=1)
    requires_human_review: bool
    review_reason: Optional[str] = None
    model: str


class PropertyPurpose(str, Enum):
    RENT = "aluguel"
    SALE = "venda"


class PropertyType(str, Enum):
    HOUSE = "casa"
    APARTMENT = "apartamento"
    COMMERCIAL = "comercial"
    LAND = "terreno"


class SearchSort(str, Enum):
    RELEVANCE = "relevancia"
    PRICE_ASC = "menor_preco"
    PRICE_DESC = "maior_preco"
    AREA_ASC = "menor_area"
    AREA_DESC = "maior_area"


class PropertySearchInput(StrictModel):
    text: Optional[str] = Field(default=None, min_length=1, max_length=500)
    purpose: Optional[PropertyPurpose] = None
    property_type: Optional[PropertyType] = None
    city: Optional[str] = Field(default=None, min_length=1, max_length=100)
    neighborhood: Optional[str] = Field(default=None, min_length=1, max_length=100)
    min_price: Optional[float] = Field(default=None, ge=0)
    max_price: Optional[float] = Field(default=None, ge=0)
    min_area_m2: Optional[float] = Field(default=None, ge=0)
    max_area_m2: Optional[float] = Field(default=None, ge=0)
    bedrooms: Optional[int] = Field(default=None, ge=0)
    bathrooms: Optional[int] = Field(default=None, ge=0)
    parking_spaces: Optional[int] = Field(default=None, ge=0)
    pets_allowed: Optional[bool] = None
    furnished: Optional[bool] = None
    available: Optional[bool] = True
    sort: SearchSort = SearchSort.RELEVANCE
    limit: int = Field(default=10, gt=0, le=50)

    @model_validator(mode="after")
    def validate_ranges(self) -> "PropertySearchInput":
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("O preço mínimo não pode ser maior que o máximo.")
        if (
            self.min_area_m2 is not None
            and self.max_area_m2 is not None
            and self.min_area_m2 > self.max_area_m2
        ):
            raise ValueError("A área mínima não pode ser maior que a máxima.")
        return self


class Property(StrictModel):
    id: str
    purpose: PropertyPurpose
    property_type: PropertyType
    city: str
    neighborhood: str
    price: float = Field(gt=0)
    bedrooms: int = Field(ge=0)
    bathrooms: int = Field(ge=0)
    parking_spaces: int = Field(ge=0)
    area_m2: float = Field(gt=0)
    pets_allowed: bool
    furnished: bool
    features: list[str]
    description: str
    available: bool


class PropertySearchItem(StrictModel):
    property: Property
    lexical_score: int = Field(ge=0)
    matched_terms: list[str]
    matched_criteria: list[str]


class PropertySearchResult(StrictModel):
    applied_filters: dict[str, Any]
    total: int = Field(ge=0)
    results: list[PropertySearchItem]
    search_engine: str
    synthetic_data: bool


class ExtractedPropertyFilters(StrictModel):
    purpose: Optional[PropertyPurpose] = None
    property_type: Optional[PropertyType] = None
    city: Optional[str] = None
    neighborhood: Optional[str] = None
    max_price: Optional[float] = None
    min_bedrooms: Optional[int] = None
    min_bathrooms: Optional[int] = None
    min_parking_spaces: Optional[int] = None
    pets_allowed: Optional[bool] = None
    furnished: Optional[bool] = None


class RecognizedConstraint(StrictModel):
    field: str
    value: Any
    source_text: str
    corrected_from: Optional[str] = None


class UnresolvedConstraint(StrictModel):
    field: str
    value: str
    reason: str


class QueryInterpretation(StrictModel):
    original_query: str
    normalized_query: str
    extracted_filters: ExtractedPropertyFilters
    remaining_text: Optional[str] = None
    recognized_constraints: list[RecognizedConstraint]
    unresolved_constraints: list[UnresolvedConstraint]
    requires_clarification: bool


class AssistantResult(StrictModel):
    classification: Classification
    action: str
    message: str
    search: Optional[PropertySearchResult] = None
    interpreted_query: Optional[QueryInterpretation] = None
    requires_clarification: bool = False
    generative_ai_used: bool
