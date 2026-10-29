from pathlib import Path

import pytest
from pydantic import ValidationError

from imob_ai.data import Property, load_catalog
from imob_ai.search import PropertySearchQuery, SearchSort, search_properties

CATALOG_PATH = Path("data/synthetic/properties.json")


@pytest.fixture(scope="module")
def catalog() -> list[Property]:
    return load_catalog(CATALOG_PATH)


def test_query_without_filters_returns_only_available(
    catalog: list[Property],
) -> None:
    result = search_properties(catalog, PropertySearchQuery())

    assert result.total_before_limit == 40
    assert all(item.property.available for item in result.items)


def test_unavailable_properties_are_excluded_by_default(
    catalog: list[Property],
) -> None:
    result = search_properties(catalog, PropertySearchQuery())

    assert not any(not item.property.available for item in result.items)


def test_unavailable_properties_can_be_included(catalog: list[Property]) -> None:
    result = search_properties(catalog, PropertySearchQuery(include_unavailable=True))

    assert result.total_before_limit == 48
    assert any(not item.property.available for item in result.items)


@pytest.mark.parametrize(
    ("field", "value", "property_field"),
    [
        ("purpose", "aluguel", "purpose"),
        ("property_type", "casa", "property_type"),
        ("city", "Franca", "city"),
        ("neighborhood", "Jardim Aurora", "neighborhood"),
        ("pets_allowed", True, "pets_allowed"),
        ("furnished", True, "furnished"),
    ],
)
def test_exact_filters(
    catalog: list[Property], field: str, value: object, property_field: str
) -> None:
    query = PropertySearchQuery.model_validate({field: value})
    result = search_properties(catalog, query)

    assert result.items
    assert all(
        getattr(item.property, property_field).value == value
        if hasattr(getattr(item.property, property_field), "value")
        else getattr(item.property, property_field) == value
        for item in result.items
    )


def test_city_filter_ignores_accents_and_case(catalog: list[Property]) -> None:
    result = search_properties(catalog, PropertySearchQuery(city="RIBEIRAO  PRETO"))

    assert result.items
    assert {item.property.city for item in result.items} == {"Ribeirão Preto"}


def test_price_range(catalog: list[Property]) -> None:
    result = search_properties(
        catalog, PropertySearchQuery(min_price=2000, max_price=4000)
    )

    assert result.items
    assert all(2000 <= item.property.price <= 4000 for item in result.items)


@pytest.mark.parametrize(
    ("field", "minimum", "property_field"),
    [
        ("min_bedrooms", 3, "bedrooms"),
        ("min_bathrooms", 2, "bathrooms"),
        ("min_parking_spaces", 2, "parking_spaces"),
    ],
)
def test_minimum_quantity_filters(
    catalog: list[Property], field: str, minimum: int, property_field: str
) -> None:
    result = search_properties(
        catalog, PropertySearchQuery.model_validate({field: minimum})
    )

    assert result.items
    assert all(
        getattr(item.property, property_field) >= minimum for item in result.items
    )


def test_area_range(catalog: list[Property]) -> None:
    result = search_properties(
        catalog, PropertySearchQuery(min_area_m2=80, max_area_m2=140)
    )

    assert result.items
    assert all(80 <= item.property.area_m2 <= 140 for item in result.items)


def test_required_feature_is_normalized(catalog: list[Property]) -> None:
    result = search_properties(
        catalog, PropertySearchQuery(required_features=[" AREA   VERDE "])
    )

    assert result.items
    assert all("área verde" in item.property.features for item in result.items)


def test_multiple_required_features_use_and_logic(catalog: list[Property]) -> None:
    result = search_properties(
        catalog,
        PropertySearchQuery(required_features=["varanda", "área verde"]),
    )

    assert result.items
    assert all(
        {"varanda", "área verde"}.issubset(item.property.features)
        for item in result.items
    )


def test_combines_multiple_filters_with_and_logic(catalog: list[Property]) -> None:
    query = PropertySearchQuery(
        purpose="aluguel",
        property_type="casa",
        city="Belo Horizonte",
        min_bedrooms=2,
        pets_allowed=True,
    )
    result = search_properties(catalog, query)

    assert result.items
    assert all(
        item.property.purpose.value == "aluguel"
        and item.property.property_type.value == "casa"
        and item.property.city == "Belo Horizonte"
        and item.property.bedrooms >= 2
        and item.property.pets_allowed
        for item in result.items
    )


def test_no_results(catalog: list[Property]) -> None:
    result = search_properties(catalog, PropertySearchQuery(city="Cidade Inexistente"))

    assert result.items == []
    assert result.total_before_limit == 0


@pytest.mark.parametrize(
    "data",
    [
        {"min_price": 2000, "max_price": 1000},
        {"min_area_m2": 100, "max_area_m2": 50},
    ],
)
def test_invalid_ranges_are_rejected(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PropertySearchQuery.model_validate(data)


@pytest.mark.parametrize("limit", [0, -1, 51])
def test_invalid_limit_is_rejected(limit: int) -> None:
    with pytest.raises(ValidationError):
        PropertySearchQuery(limit=limit)


def test_text_search_matches_description(catalog: list[Property]) -> None:
    result = search_properties(catalog, PropertySearchQuery(text="ventilação cruzada"))

    assert result.items
    assert any("ventilacao" in item.matched_terms for item in result.items)


def test_text_search_matches_features(catalog: list[Property]) -> None:
    result = search_properties(catalog, PropertySearchQuery(text="bicicletário"))

    assert result.items
    assert all("bicicletario" in item.matched_terms for item in result.items)


def test_text_search_ignores_accents(catalog: list[Property]) -> None:
    result = search_properties(catalog, PropertySearchQuery(text="area verde"))

    assert result.items
    assert any(item.lexical_score == 2 for item in result.items)


def test_lexical_score_counts_distinct_matched_terms(
    catalog: list[Property],
) -> None:
    result = search_properties(
        catalog, PropertySearchQuery(text="apartamento area verde area")
    )

    assert result.items[0].lexical_score == 3
    assert result.items[0].matched_terms == ["apartamento", "area", "verde"]


def test_relevance_sort_has_deterministic_tiebreakers(
    catalog: list[Property],
) -> None:
    base = catalog[0]
    custom_catalog = [
        base.model_copy(update={"id": "B", "price": 1000}),
        base.model_copy(update={"id": "C", "price": 2000}),
        base.model_copy(update={"id": "A", "price": 1000}),
    ]
    result = search_properties(
        custom_catalog,
        PropertySearchQuery(text="casa", include_unavailable=True),
    )

    assert [item.property.id for item in result.items] == ["A", "B", "C"]


@pytest.mark.parametrize(
    ("sort", "field", "reverse"),
    [
        (SearchSort.PRICE_ASC, "price", False),
        (SearchSort.PRICE_DESC, "price", True),
        (SearchSort.AREA_ASC, "area_m2", False),
        (SearchSort.AREA_DESC, "area_m2", True),
    ],
)
def test_explicit_sort_orders(
    catalog: list[Property], sort: SearchSort, field: str, reverse: bool
) -> None:
    result = search_properties(catalog, PropertySearchQuery(sort=sort))
    values = [getattr(item.property, field) for item in result.items]

    assert values == sorted(values, reverse=reverse)


def test_limit_is_applied_after_counting(catalog: list[Property]) -> None:
    result = search_properties(catalog, PropertySearchQuery(limit=3))

    assert len(result.items) == 3
    assert result.total_before_limit == 40


def test_search_does_not_modify_catalog(catalog: list[Property]) -> None:
    original_order = [item.id for item in catalog]
    original_content = [item.model_dump() for item in catalog]

    search_properties(
        catalog,
        PropertySearchQuery(text="casa", sort=SearchSort.PRICE_DESC),
    )

    assert [item.id for item in catalog] == original_order
    assert [item.model_dump() for item in catalog] == original_content


@pytest.mark.parametrize(
    "data",
    [
        {"purpose": "temporada"},
        {"property_type": "galpão"},
        {"min_bedrooms": -1},
        {"unknown_filter": True},
    ],
)
def test_invalid_query_is_rejected(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PropertySearchQuery.model_validate(data)
