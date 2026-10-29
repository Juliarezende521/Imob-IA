"""Contratos do dataset de classificação de intenções."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Intent(str, Enum):
    SEARCH_PROPERTY = "BUSCA_IMOVEL"
    SCHEDULE_VISIT = "AGENDAR_VISITA"
    PROPERTY_QUESTION = "DUVIDA_IMOVEL"
    RENTAL_DOCUMENTATION = "DOCUMENTACAO_LOCACAO"
    RENTAL_FINANCE = "FINANCEIRO_LOCACAO"
    PROPERTY_MAINTENANCE = "MANUTENCAO_IMOVEL"
    HUMAN_SERVICE = "ATENDIMENTO_HUMANO"
    OUT_OF_SCOPE = "FORA_DE_ESCOPO"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class IntentMessage(BaseModel):
    """Mensagem sintética com um único rótulo principal."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    intent: Intent
    split: DatasetSplit
    scenario_family: str = Field(min_length=1)
    difficulty: Difficulty
    notes: Optional[str] = None
