"""Schemas dos julgamentos e resultados da avaliação."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from imob_ai.search.schemas import PropertySearchQuery


class EvaluationSplit(str, Enum):
    """Divisões com papéis distintos no ciclo experimental."""

    DEVELOPMENT = "development"
    TEST = "test"


class QueryCategory(str, Enum):
    """Categorias controladas das consultas de avaliação."""

    LITERAL = "literal"
    SYNONYM = "synonym"
    PARAPHRASE = "paraphrase"
    IMPLICIT_INTENT = "implicit_intent"
    NUMERIC_FILTER = "numeric_filter"
    MULTIPLE_CONSTRAINTS = "multiple_constraints"
    SPECIFIC_FEATURE = "specific_feature"
    AMBIGUOUS = "ambiguous"
    NO_RESULT = "no_result"
    ACCENT_VARIATION = "accent_variation"
    CASE_VARIATION = "case_variation"
    LEXICAL_CHALLENGE = "lexical_challenge"


class EvaluationCase(BaseModel):
    """Consulta com julgamento manual de relevância."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    filters: PropertySearchQuery = Field(default_factory=PropertySearchQuery)
    relevant_property_ids: list[str]
    split: EvaluationSplit
    category: QueryCategory
    notes: str = Field(min_length=1)
    expected_no_results: bool = False

    @field_validator("relevant_property_ids")
    @classmethod
    def reject_duplicate_relevant_ids(cls, ids: list[str]) -> list[str]:
        if len(ids) != len(set(ids)):
            raise ValueError("relevant_property_ids não pode conter repetições.")
        return ids

    @model_validator(mode="after")
    def validate_relevance_contract(self) -> "EvaluationCase":
        if self.filters.text is not None:
            raise ValueError("O texto deve estar em query, não em filters.text.")
        if self.filters.limit is not None:
            raise ValueError("O limite não pode ser fixado nos casos de avaliação.")
        if self.expected_no_results and self.relevant_property_ids:
            raise ValueError("Caso sem resultado não pode conter IDs relevantes.")
        if not self.expected_no_results and not self.relevant_property_ids:
            raise ValueError("Caso de recuperação precisa conter IDs relevantes.")
        return self


class QueryEvaluation(BaseModel):
    """Resultado calculado para uma consulta."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    split: EvaluationSplit
    category: QueryCategory
    relevant_ids: list[str]
    retrieved_ids: list[str]
    precision_at_1: Optional[float] = None
    precision_at_3: Optional[float] = None
    precision_at_5: Optional[float] = None
    recall_at_1: Optional[float] = None
    recall_at_3: Optional[float] = None
    recall_at_5: Optional[float] = None
    reciprocal_rank: Optional[float] = None
    hit_at_1: Optional[float] = None
    hit_at_3: Optional[float] = None
    hit_at_5: Optional[float] = None
    no_result_correct: Optional[bool] = None


class AggregateMetrics(BaseModel):
    """Médias de um grupo de consultas."""

    model_config = ConfigDict(frozen=True)

    total_cases: int = Field(ge=0)
    retrieval_cases: int = Field(ge=0)
    no_result_cases: int = Field(ge=0)
    precision_at_1: Optional[float] = None
    precision_at_3: Optional[float] = None
    precision_at_5: Optional[float] = None
    recall_at_1: Optional[float] = None
    recall_at_3: Optional[float] = None
    recall_at_5: Optional[float] = None
    mrr: Optional[float] = None
    hit_rate_at_1: Optional[float] = None
    hit_rate_at_3: Optional[float] = None
    hit_rate_at_5: Optional[float] = None
    no_result_accuracy: Optional[float] = None


class EvaluationReport(BaseModel):
    """Relatório completo e determinístico do baseline."""

    model_config = ConfigDict(frozen=True)

    queries: list[QueryEvaluation]
    overall: AggregateMetrics
    by_split: dict[str, AggregateMetrics]
    by_category: dict[str, AggregateMetrics]
