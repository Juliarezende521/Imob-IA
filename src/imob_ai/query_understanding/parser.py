"""Parser determinístico e explicável de consultas imobiliárias."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from imob_ai.data.schemas import Property
from imob_ai.query_understanding.schemas import (
    ExtractedPropertyFilters,
    QueryInterpretation,
    RecognizedConstraint,
    UnresolvedConstraint,
)
from imob_ai.query_understanding.vocabulary import (
    CONSERVATIVE_CORRECTIONS,
    NON_LOCATION_TERMS,
    PROPERTY_TYPE_ALIASES,
    PURPOSE_ALIASES,
    STOPWORDS,
)
from imob_ai.search.normalization import normalize_text

TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ÿ0-9]+")


@dataclass(frozen=True)
class Token:
    original: str
    normalized: str


def _tokens(value: str) -> list[Token]:
    return [
        Token(match.group(0), normalize_text(match.group(0)))
        for match in TOKEN_PATTERN.finditer(value)
    ]


def _alias_tokens(aliases: dict[str, object]) -> list[tuple[list[str], object, str]]:
    ordered = []
    for alias, value in aliases.items():
        ordered.append((alias.split(), value, alias))
    return sorted(ordered, key=lambda item: len(item[0]), reverse=True)


def _find_phrases(
    tokens: list[Token], aliases: dict[str, object]
) -> list[tuple[int, int, object, str]]:
    matches: list[tuple[int, int, object, str]] = []
    occupied: set[int] = set()
    normalized = [token.normalized for token in tokens]
    for phrase, value, alias in _alias_tokens(aliases):
        size = len(phrase)
        for start in range(len(tokens) - size + 1):
            indexes = set(range(start, start + size))
            if indexes & occupied:
                continue
            if normalized[start : start + size] == phrase:
                matches.append((start, start + size, value, alias))
                occupied.update(indexes)
    return sorted(matches)


def _canonical_aliases(values: Sequence[str]) -> dict[str, str]:
    return {normalize_text(value): value for value in values}


def _source(tokens: list[Token], start: int, end: int) -> str:
    return " ".join(token.original for token in tokens[start:end])


def _consume_context(tokens: list[Token], consumed: set[int], start: int) -> None:
    for index in range(max(0, start - 2), start):
        if tokens[index].normalized in {"em", "no", "na", "bairro", "regiao"}:
            consumed.add(index)


def _extract_controlled(
    tokens: list[Token],
    aliases: dict[str, object],
    field: str,
    consumed: set[int],
    recognized: list[RecognizedConstraint],
    unresolved: list[UnresolvedConstraint],
) -> Optional[object]:
    matches = _find_phrases(tokens, aliases)
    if not matches:
        return None
    distinct = {match[2] for match in matches}
    for start, end, value, alias in matches:
        consumed.update(range(start, end))
        correction = CONSERVATIVE_CORRECTIONS.get(alias)
        recognized.append(
            RecognizedConstraint(
                field=field,
                value=getattr(value, "value", value),
                source_text=_source(tokens, start, end),
                corrected_from=alias if correction else None,
            )
        )
    if len(distinct) > 1:
        unresolved.append(
            UnresolvedConstraint(
                field=field,
                value=" / ".join(
                    sorted(getattr(value, "value", value) for value in distinct)
                ),
                reason="valores_contraditorios",
            )
        )
        return None
    return next(iter(distinct))


def _extract_catalog_value(
    tokens: list[Token],
    canonical_values: Sequence[str],
    field: str,
    consumed: set[int],
    recognized: list[RecognizedConstraint],
) -> tuple[Optional[str], Optional[tuple[int, int]]]:
    aliases = _canonical_aliases(canonical_values)
    matches = _find_phrases(tokens, aliases)
    if not matches:
        return None, None
    start, end, value, _ = max(matches, key=lambda item: item[1] - item[0])
    consumed.update(range(start, end))
    _consume_context(tokens, consumed, start)
    recognized.append(
        RecognizedConstraint(
            field=field,
            value=value,
            source_text=_source(tokens, start, end),
        )
    )
    return value, (start, end)


def _location_candidate(tokens: list[Token], start: int, end: int) -> str:
    candidate = [token.original for token in tokens[start:end]]
    return " ".join(candidate).strip()


def _detect_unknown_location(
    tokens: list[Token],
    city_span: Optional[tuple[int, int]],
    city: Optional[str],
    neighborhood: Optional[str],
    consumed: set[int],
    unresolved: list[UnresolvedConstraint],
) -> None:
    normalized = [token.normalized for token in tokens]
    if neighborhood is None and city_span is not None:
        city_start = city_span[0]
        for marker in ("bairro", "regiao", "no", "na"):
            indexes = [
                index
                for index, token in enumerate(normalized[:city_start])
                if token == marker
            ]
            if not indexes:
                continue
            start = indexes[-1] + 1
            end = city_start
            while end > start and normalized[end - 1] in {"em", "de"}:
                end -= 1
            value = _location_candidate(tokens, start, end)
            if value and normalize_text(value) not in NON_LOCATION_TERMS:
                consumed.update(range(indexes[-1], city_start))
                unresolved.append(
                    UnresolvedConstraint(
                        field="neighborhood",
                        value=value,
                        reason="nao_encontrado_no_catalogo",
                    )
                )
                return

    if city is None:
        for index, token in enumerate(normalized[:-1]):
            if token != "em":
                continue
            end = index + 1
            while end < len(tokens) and normalized[end] not in {
                "com",
                "para",
                "que",
                "ate",
            }:
                end += 1
            value = _location_candidate(tokens, index + 1, end)
            normalized_value = normalize_text(value)
            if (
                value
                and normalized_value not in NON_LOCATION_TERMS
                and not normalized_value.startswith("condominio")
            ):
                consumed.update(range(index, end))
                unresolved.append(
                    UnresolvedConstraint(
                        field="city",
                        value=value,
                        reason="nao_encontrada_no_catalogo",
                    )
                )
                return


def _number(tokens: list[Token], index: int) -> tuple[Optional[float], int]:
    if index >= len(tokens) or not tokens[index].normalized.isdigit():
        return None, index
    pieces = [tokens[index].normalized]
    end = index + 1
    if (
        end < len(tokens)
        and len(tokens[end].normalized) == 3
        and tokens[end].normalized.isdigit()
    ):
        pieces.append(tokens[end].normalized)
        end += 1
    return float("".join(pieces)), end


def _extract_numeric_and_boolean(
    tokens: list[Token],
    consumed: set[int],
    recognized: list[RecognizedConstraint],
) -> dict[str, object]:
    normalized = [token.normalized for token in tokens]
    values: dict[str, object] = {}

    for index in range(len(tokens)):
        price_start = None
        if normalized[index] == "ate":
            price_start = index + 1
        elif normalized[index : index + 2] == ["no", "maximo"]:
            price_start = index + 2
        if price_start is not None:
            if price_start < len(tokens) and normalized[price_start] == "r":
                price_start += 1
            value, end = _number(tokens, price_start)
            if value is not None:
                values["max_price"] = value
                consumed.update(range(index, end))
                if end < len(tokens) and normalized[end] == "reais":
                    consumed.add(end)
                    end += 1
                recognized.append(
                    RecognizedConstraint(
                        field="max_price",
                        value=value,
                        source_text=_source(tokens, index, end),
                    )
                )
                break

    quantity_patterns = [
        (["a", "partir", "de"], "quartos", "min_bedrooms"),
        (["pelo", "menos"], "quartos", "min_bedrooms"),
        (["pelo", "menos"], "dormitorios", "min_bedrooms"),
        (["com"], "banheiros", "min_bathrooms"),
    ]
    for prefix, noun, field in quantity_patterns:
        for index in range(len(tokens) - len(prefix) - 1):
            if normalized[index : index + len(prefix)] != prefix:
                continue
            number_index = index + len(prefix)
            value, end = _number(tokens, number_index)
            if value is None or end >= len(tokens) or normalized[end] != noun:
                continue
            integer = int(value)
            values[field] = integer
            consumed.update(range(index, end + 1))
            recognized.append(
                RecognizedConstraint(
                    field=field,
                    value=integer,
                    source_text=_source(tokens, index, end + 1),
                )
            )

    phrase_fields = [
        (("com", "garagem"), "min_parking_spaces", 1),
        (("aceita", "animais"), "pets_allowed", True),
        (("aceita", "pet"), "pets_allowed", True),
        (("aceite", "animais"), "pets_allowed", True),
        (("aceite", "pet"), "pets_allowed", True),
        (("nao", "mobiliado"), "furnished", False),
        (("nao", "mobiliada"), "furnished", False),
        (("mobiliado",), "furnished", True),
        (("mobiliada",), "furnished", True),
    ]
    occupied = set(consumed)
    for phrase, field, value in phrase_fields:
        size = len(phrase)
        for start in range(len(tokens) - size + 1):
            indexes = set(range(start, start + size))
            if indexes & occupied or tuple(normalized[start : start + size]) != phrase:
                continue
            values[field] = value
            consumed.update(indexes)
            occupied.update(indexes)
            recognized.append(
                RecognizedConstraint(
                    field=field,
                    value=value,
                    source_text=_source(tokens, start, start + size),
                )
            )
            break
    return values


def _remaining_text(tokens: list[Token], consumed: set[int]) -> Optional[str]:
    remaining = [
        token.original.casefold()
        for index, token in enumerate(tokens)
        if index not in consumed
        and token.normalized not in STOPWORDS
        and not token.normalized.isdigit()
        and token.normalized not in {"reais", "partir", "menos", "maximo"}
    ]
    return " ".join(remaining) or None


def interpret_property_query(
    query: str, catalog: Sequence[Property]
) -> QueryInterpretation:
    """Interpreta filtros sem relaxamento ou aproximação silenciosa."""
    tokens = _tokens(query)
    consumed: set[int] = set()
    recognized: list[RecognizedConstraint] = []
    unresolved: list[UnresolvedConstraint] = []

    purpose = _extract_controlled(
        tokens,
        PURPOSE_ALIASES,
        "purpose",
        consumed,
        recognized,
        unresolved,
    )
    property_type = _extract_controlled(
        tokens,
        PROPERTY_TYPE_ALIASES,
        "property_type",
        consumed,
        recognized,
        unresolved,
    )
    cities = sorted({property_.city for property_ in catalog}, key=len, reverse=True)
    city, city_span = _extract_catalog_value(
        tokens, cities, "city", consumed, recognized
    )
    neighborhoods = sorted(
        {
            property_.neighborhood
            for property_ in catalog
            if city is None or property_.city == city
        },
        key=len,
        reverse=True,
    )
    neighborhood, _ = _extract_catalog_value(
        tokens, neighborhoods, "neighborhood", consumed, recognized
    )
    if (
        city
        and neighborhood
        and not any(
            item.city == city and item.neighborhood == neighborhood for item in catalog
        )
    ):
        unresolved.append(
            UnresolvedConstraint(
                field="neighborhood",
                value=neighborhood,
                reason="nao_disponivel_na_cidade",
            )
        )
    _detect_unknown_location(
        tokens,
        city_span,
        city,
        neighborhood,
        consumed,
        unresolved,
    )
    extra = _extract_numeric_and_boolean(tokens, consumed, recognized)
    filters = ExtractedPropertyFilters(
        purpose=purpose,
        property_type=property_type,
        city=city,
        neighborhood=neighborhood,
        **extra,
    )
    return QueryInterpretation(
        original_query=query,
        normalized_query=" ".join(token.normalized for token in tokens),
        extracted_filters=filters,
        remaining_text=_remaining_text(tokens, consumed),
        recognized_constraints=recognized,
        unresolved_constraints=unresolved,
        requires_clarification=bool(unresolved),
    )
