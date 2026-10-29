"""Serviço determinístico de busca estruturada e lexical."""

from collections.abc import Callable, Sequence

from imob_ai.data.schemas import Property
from imob_ai.search.normalization import normalize_text, tokenize
from imob_ai.search.schemas import (
    PropertySearchQuery,
    SearchMatch,
    SearchResult,
    SearchSort,
)


def search_properties(
    catalog: Sequence[Property], query: PropertySearchQuery
) -> SearchResult:
    """Filtra e ordena imóveis sem modificar o catálogo recebido."""
    matches: list[SearchMatch] = []
    terms = tokenize(query.text) if query.text else []

    for property_ in catalog:
        if not _matches_structured_filters(property_, query):
            continue

        matched_terms = _find_matched_terms(property_, terms)
        if terms and not matched_terms:
            continue

        matches.append(
            SearchMatch(
                property=property_,
                lexical_score=len(matched_terms),
                matched_terms=matched_terms,
                matched_criteria=_matched_criteria(query),
            )
        )

    ordered = sorted(matches, key=_sort_key(query.sort))
    total_before_limit = len(ordered)
    if query.limit is not None:
        ordered = ordered[: query.limit]

    return SearchResult(
        items=ordered,
        total_before_limit=total_before_limit,
        sort=query.sort,
    )


def _matches_structured_filters(
    property_: Property, query: PropertySearchQuery
) -> bool:
    if not query.include_unavailable and not property_.available:
        return False
    if query.purpose is not None and property_.purpose != query.purpose:
        return False
    if (
        query.property_type is not None
        and property_.property_type != query.property_type
    ):
        return False
    if query.city is not None and normalize_text(property_.city) != normalize_text(
        query.city
    ):
        return False
    if query.neighborhood is not None and normalize_text(
        property_.neighborhood
    ) != normalize_text(query.neighborhood):
        return False
    if query.min_price is not None and property_.price < query.min_price:
        return False
    if query.max_price is not None and property_.price > query.max_price:
        return False
    if query.min_bedrooms is not None and property_.bedrooms < query.min_bedrooms:
        return False
    if query.min_bathrooms is not None and property_.bathrooms < query.min_bathrooms:
        return False
    if (
        query.min_parking_spaces is not None
        and property_.parking_spaces < query.min_parking_spaces
    ):
        return False
    if query.min_area_m2 is not None and property_.area_m2 < query.min_area_m2:
        return False
    if query.max_area_m2 is not None and property_.area_m2 > query.max_area_m2:
        return False
    if query.pets_allowed is not None and property_.pets_allowed != query.pets_allowed:
        return False
    if query.furnished is not None and property_.furnished != query.furnished:
        return False

    normalized_features = {normalize_text(value) for value in property_.features}
    if not all(feature in normalized_features for feature in query.required_features):
        return False
    return True


def _find_matched_terms(property_: Property, terms: list[str]) -> list[str]:
    searchable_text = normalize_text(
        " ".join(
            [
                property_.purpose.value,
                property_.property_type.value,
                property_.city,
                property_.neighborhood,
                *property_.features,
                property_.description,
            ]
        )
    )
    searchable_tokens = set(tokenize(searchable_text))
    return [term for term in terms if term in searchable_tokens]


def _matched_criteria(query: PropertySearchQuery) -> list[str]:
    criteria = ["disponibilidade"] if not query.include_unavailable else []
    fields = {
        "purpose": "finalidade",
        "property_type": "tipo",
        "city": "cidade",
        "neighborhood": "bairro",
        "min_price": "preço mínimo",
        "max_price": "preço máximo",
        "min_bedrooms": "dormitórios mínimos",
        "min_bathrooms": "banheiros mínimos",
        "min_parking_spaces": "vagas mínimas",
        "min_area_m2": "área mínima",
        "max_area_m2": "área máxima",
        "pets_allowed": "aceita animais",
        "furnished": "mobiliado",
    }
    for field, label in fields.items():
        if getattr(query, field) is not None:
            criteria.append(label)
    if query.required_features:
        criteria.append("características obrigatórias")
    if query.text:
        criteria.append("texto livre")
    return criteria


def _sort_key(sort: SearchSort) -> Callable[[SearchMatch], tuple[object, ...]]:
    if sort == SearchSort.PRICE_ASC:
        return lambda match: (match.property.price, match.property.id)
    if sort == SearchSort.PRICE_DESC:
        return lambda match: (-match.property.price, match.property.id)
    if sort == SearchSort.AREA_ASC:
        return lambda match: (match.property.area_m2, match.property.id)
    if sort == SearchSort.AREA_DESC:
        return lambda match: (-match.property.area_m2, match.property.id)
    return lambda match: (
        -match.lexical_score,
        match.property.price,
        match.property.id,
    )
