"""Schema validado para os imóveis do catálogo sintético."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PropertyPurpose(str, Enum):
    """Finalidades aceitas para um imóvel."""

    RENT = "aluguel"
    SALE = "venda"


class PropertyType(str, Enum):
    """Tipos de imóvel disponíveis no catálogo."""

    HOUSE = "casa"
    APARTMENT = "apartamento"
    COMMERCIAL = "comercial"
    LAND = "terreno"


class Property(BaseModel):
    """Representação validada de um imóvel fictício."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    purpose: PropertyPurpose
    property_type: PropertyType
    city: str = Field(min_length=1)
    neighborhood: str = Field(min_length=1)
    price: float = Field(gt=0)
    bedrooms: int = Field(ge=0)
    bathrooms: int = Field(ge=0)
    parking_spaces: int = Field(ge=0)
    area_m2: float = Field(gt=0)
    pets_allowed: bool
    furnished: bool
    features: list[str]
    description: str = Field(min_length=1)
    available: bool

    @field_validator("features")
    @classmethod
    def normalize_features(cls, features: list[str]) -> list[str]:
        """Normaliza espaços e caixa, remove vazios e repetições."""
        normalized: list[str] = []

        for feature in features:
            value = " ".join(feature.strip().lower().split())
            if value and value not in normalized:
                normalized.append(value)

        return normalized
