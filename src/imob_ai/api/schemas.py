"""Contratos públicos da API local."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from imob_ai.api.config import (
    CLASSIFIER_NAME,
    DEFAULT_SEARCH_LIMIT,
    MAX_MESSAGE_LENGTH,
    MAX_SEARCH_LIMIT,
)
from imob_ai.data.schemas import Property, PropertyPurpose, PropertyType
from imob_ai.search.schemas import PropertySearchQuery, SearchSort


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class MessageRequest(StrictModel):
    message: str = Field(
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description=(
            "Mensagem sintética curta, em português, cuja intenção será classificada."
        ),
        examples=["Quero visitar o apartamento amanhã"],
    )


class ClassificationResponse(StrictModel):
    intent: str
    model_score: float = Field(ge=0, le=1)
    second_intent: str
    second_model_score: float = Field(ge=0, le=1)
    margin: float = Field(ge=0, le=1)
    requires_human_review: bool
    review_reason: Optional[str] = None
    model: str = CLASSIFIER_NAME


class PropertySearchRequest(StrictModel):
    text: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=500,
        description="Termos livres usados pelo ranking lexical.",
        examples=["casa com quintal"],
    )
    purpose: Optional[PropertyPurpose] = None
    property_type: Optional[PropertyType] = None
    city: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Cidade fictícia presente no catálogo sintético.",
        examples=["Franca"],
    )
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
    limit: int = Field(
        default=DEFAULT_SEARCH_LIMIT,
        gt=0,
        le=MAX_SEARCH_LIMIT,
        description="Número máximo de resultados retornados.",
        examples=[5],
    )

    @model_validator(mode="after")
    def validate_ranges(self) -> "PropertySearchRequest":
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

    def to_query(self, fallback_text: Optional[str] = None) -> PropertySearchQuery:
        return PropertySearchQuery(
            text=self.text or fallback_text,
            purpose=self.purpose,
            property_type=self.property_type,
            city=self.city,
            neighborhood=self.neighborhood,
            min_price=self.min_price,
            max_price=self.max_price,
            min_area_m2=self.min_area_m2,
            max_area_m2=self.max_area_m2,
            min_bedrooms=self.bedrooms,
            min_bathrooms=self.bathrooms,
            min_parking_spaces=self.parking_spaces,
            pets_allowed=self.pets_allowed,
            furnished=self.furnished,
            include_unavailable=self.available is False,
            sort=self.sort,
            limit=self.limit,
        )

    def applied_filters(self, fallback_text: Optional[str] = None) -> dict[str, Any]:
        values = self.model_dump(exclude_none=True, mode="json")
        if "text" not in values and fallback_text:
            values["text"] = fallback_text
        return values


class PropertySearchItem(StrictModel):
    property: Property
    lexical_score: int = Field(ge=0)
    matched_terms: list[str]
    matched_criteria: list[str]


class PropertySearchResponse(StrictModel):
    applied_filters: dict[str, Any]
    total: int = Field(ge=0)
    results: list[PropertySearchItem]
    search_engine: str = "lexical"
    synthetic_data: bool = True


class AssistantRequest(MessageRequest):
    search_filters: Optional[PropertySearchRequest] = None


class RouteAction(str, Enum):
    SEARCH_PROPERTIES = "search_properties"
    HUMAN_REVIEW = "human_review"
    IDENTIFY_PROPERTY = "identify_property"
    PROVIDE_GENERIC_DOCUMENT_GUIDANCE = "provide_generic_document_guidance"
    HUMAN_FINANCIAL_SUPPORT = "human_financial_support"
    HUMAN_MAINTENANCE_SUPPORT = "human_maintenance_support"
    HUMAN_SUPPORT = "human_support"
    OUT_OF_SCOPE = "out_of_scope"


class AssistantResponse(StrictModel):
    classification: ClassificationResponse
    action: RouteAction
    message: str
    search: Optional[PropertySearchResponse] = None
    generative_ai_used: bool = False


class ReadyResponse(StrictModel):
    status: str
    classifier_ready: bool
    catalog_ready: bool
    configuration_valid: bool


class ErrorBody(StrictModel):
    code: str
    message: str
    details: Optional[dict[str, Any]] = None


class ErrorResponse(StrictModel):
    error: ErrorBody


class MetaResponse(StrictModel):
    api_version: str
    default_classifier: str
    default_search: str
    available_intents: list[str]
    synthetic_property_count: int
    synthetic_data: bool
    demonstration_notice: str
    transformer: str
