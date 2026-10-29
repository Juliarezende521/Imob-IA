import pytest
from pydantic import ValidationError

from imob_ai.data.schemas import Property


@pytest.fixture
def valid_property_data() -> dict[str, object]:
    return {
        "id": "IMOB-TESTE",
        "purpose": "aluguel",
        "property_type": "casa",
        "city": "Franca",
        "neighborhood": "Jardim Aurora",
        "price": 2000,
        "bedrooms": 3,
        "bathrooms": 2,
        "parking_spaces": 1,
        "area_m2": 120,
        "pets_allowed": True,
        "furnished": False,
        "features": [" Quintal amplo ", "quintal   amplo", "Área verde"],
        "description": "Casa fictícia com espaço para animais.",
        "available": True,
    }


def test_property_normalizes_features(valid_property_data: dict[str, object]) -> None:
    property_ = Property.model_validate(valid_property_data)

    assert property_.features == ["quintal amplo", "área verde"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("price", 0),
        ("bedrooms", -1),
        ("bathrooms", -1),
        ("parking_spaces", -1),
        ("area_m2", -10),
        ("description", "   "),
    ],
)
def test_property_rejects_invalid_values(
    valid_property_data: dict[str, object], field: str, value: object
) -> None:
    valid_property_data[field] = value

    with pytest.raises(ValidationError):
        Property.model_validate(valid_property_data)


def test_property_rejects_invalid_type(
    valid_property_data: dict[str, object],
) -> None:
    valid_property_data["property_type"] = "galpão"

    with pytest.raises(ValidationError):
        Property.model_validate(valid_property_data)


def test_property_rejects_invalid_purpose(
    valid_property_data: dict[str, object],
) -> None:
    valid_property_data["purpose"] = "temporada"

    with pytest.raises(ValidationError):
        Property.model_validate(valid_property_data)


def test_property_rejects_missing_required_field(
    valid_property_data: dict[str, object],
) -> None:
    del valid_property_data["city"]

    with pytest.raises(ValidationError):
        Property.model_validate(valid_property_data)
