from pathlib import Path

import pytest

from imob_ai.data.catalog import load_catalog
from imob_ai.data.schemas import PropertyPurpose, PropertyType
from imob_ai.query_understanding import interpret_property_query

CATALOG = tuple(load_catalog(Path("data/synthetic/properties.json")))


def parse(query, catalog=CATALOG):
    return interpret_property_query(query, catalog)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("ALUGUEL", PropertyPurpose.RENT),
        ("quero alugar", PropertyPurpose.RENT),
        ("para locação", PropertyPurpose.RENT),
        ("desejo comprar", PropertyPurpose.SALE),
        ("imóvel à venda", PropertyPurpose.SALE),
    ],
)
def test_extracts_purpose_aliases(query, expected):
    assert parse(query).extracted_filters.purpose == expected


def test_absent_and_contradictory_purpose():
    assert parse("imóvel com quintal").extracted_filters.purpose is None
    result = parse("quero alugar ou comprar")
    assert result.extracted_filters.purpose is None
    assert result.requires_clarification
    assert result.unresolved_constraints[0].reason == "valores_contraditorios"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("casa", PropertyType.HOUSE),
        ("casas", PropertyType.HOUSE),
        ("apartamento", PropertyType.APARTMENT),
        ("apartamentos", PropertyType.APARTMENT),
        ("apê", PropertyType.APARTMENT),
        ("terreno", PropertyType.LAND),
        ("ponto comercial", PropertyType.COMMERCIAL),
        ("sala comercial", PropertyType.COMMERCIAL),
    ],
)
def test_extracts_property_type_aliases(query, expected):
    assert parse(query).extracted_filters.property_type == expected


def test_conservative_typo_is_explained():
    result = parse("quero uma acasa")
    assert result.extracted_filters.property_type == PropertyType.HOUSE
    constraint = next(
        item for item in result.recognized_constraints if item.field == "property_type"
    )
    assert constraint.corrected_from == "acasa"


def test_unknown_and_contradictory_types():
    assert parse("quero um loft").extracted_filters.property_type is None
    result = parse("casa ou terreno")
    assert result.extracted_filters.property_type is None
    assert result.requires_clarification


@pytest.mark.parametrize(
    ("query", "city"),
    [
        ("em franca", "Franca"),
        ("EM SÃO PAULO", "São Paulo"),
        ("em sao paulo", "São Paulo"),
        ("em ribeirao preto", "Ribeirão Preto"),
    ],
)
def test_extracts_canonical_city_ignoring_case_and_accents(query, city):
    assert parse(query).extracted_filters.city == city


def test_prefers_longest_catalog_location_name():
    base = CATALOG[0]
    custom = (
        base.model_copy(update={"id": "A", "city": "Rio"}),
        base.model_copy(update={"id": "B", "city": "Rio Claro"}),
    )
    assert parse("casa em Rio Claro", custom).extracted_filters.city == "Rio Claro"


def test_extracts_known_neighborhood_and_city():
    result = parse("casa no Jardim Aurora em Franca")
    assert result.extracted_filters.city == "Franca"
    assert result.extracted_filters.neighborhood == "Jardim Aurora"
    assert not result.requires_clarification


def test_unknown_neighborhood_requires_clarification():
    result = parse("casa no Jardim Secreto em Franca")
    assert result.extracted_filters.city == "Franca"
    assert result.requires_clarification
    assert result.unresolved_constraints[0].field == "neighborhood"
    assert result.unresolved_constraints[0].value == "Jardim Secreto"


def test_unknown_city_requires_clarification():
    result = parse("casa em Sorocaba")
    assert result.extracted_filters.city is None
    assert result.requires_clarification
    assert result.unresolved_constraints[0].field == "city"


def test_does_not_confuse_features_with_location():
    condominium = parse("casa em condomínio com quintal")
    area = parse("casa com área gourmet")
    assert not condominium.requires_clarification
    assert condominium.extracted_filters.city is None
    assert "condomínio" in condominium.remaining_text
    assert not area.requires_clarification
    assert area.extracted_filters.neighborhood is None


def test_normalization_handles_punctuation_spacing_and_accents():
    result = parse("  CASA,   para LOCAÇÃO... em São   Paulo!  ")
    assert result.normalized_query == "casa para locacao em sao paulo"
    assert result.extracted_filters.purpose == PropertyPurpose.RENT
    assert result.extracted_filters.city == "São Paulo"


def test_remaining_text_preserves_compound_feature_and_removes_filters():
    result = parse("casa para alugar em Franca com quintal e área gourmet")
    assert result.remaining_text == "quintal área gourmet"
    assert "casa" not in result.remaining_text
    assert "Franca" not in result.remaining_text


def test_only_filters_leave_no_remaining_text():
    result = parse("casa para alugar em Franca")
    assert result.remaining_text is None


def test_query_without_filters_remains_available_for_lexical_search():
    result = parse("varanda ensolarada")
    assert result.remaining_text == "varanda ensolarada"


def test_extracts_price_with_brazilian_thousands_separator():
    result = parse("imóvel até R$ 2.500")
    assert result.extracted_filters.max_price == 2500


def test_extracts_safe_numeric_filters():
    result = parse("a partir de 2 quartos com 2 banheiros e com garagem")
    assert result.extracted_filters.min_bedrooms == 2
    assert result.extracted_filters.min_bathrooms == 2
    assert result.extracted_filters.min_parking_spaces == 1


@pytest.mark.parametrize(
    ("query", "field", "expected"),
    [
        ("que aceite animais", "pets_allowed", True),
        ("aceita pet", "pets_allowed", True),
        ("mobiliado", "furnished", True),
        ("não mobiliado", "furnished", False),
    ],
)
def test_extracts_safe_boolean_filters(query, field, expected):
    assert getattr(parse(query).extracted_filters, field) is expected


def test_regression_city_petropolis_does_not_become_loose_text():
    result = parse("eu alugar um acasa no city petropolis em franca")
    assert result.extracted_filters.purpose == PropertyPurpose.RENT
    assert result.extracted_filters.property_type == PropertyType.HOUSE
    assert result.extracted_filters.city == "Franca"
    assert result.remaining_text is None
    assert result.requires_clarification
    assert result.unresolved_constraints[0].value == "city petropolis"
