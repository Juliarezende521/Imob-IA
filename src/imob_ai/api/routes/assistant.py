"""Roteamento seguro, não generativo e com filtros estritos."""

from typing import Optional

from fastapi import APIRouter, Depends

from imob_ai.api.dependencies import AppServices, require_services
from imob_ai.api.routes.intents import response_from_classification
from imob_ai.api.routes.properties import execute_search
from imob_ai.api.schemas import (
    AssistantRequest,
    AssistantResponse,
    PropertySearchRequest,
    RouteAction,
)
from imob_ai.api.services.routing import decide_route
from imob_ai.query_understanding import interpret_property_query
from imob_ai.query_understanding.schemas import (
    QueryInterpretation,
    UnresolvedConstraint,
)
from imob_ai.search.normalization import normalize_text

router = APIRouter(prefix="/v1/assistant", tags=["assistant"])


def _same_text(left: str, right: str) -> bool:
    return normalize_text(left) == normalize_text(right)


def _prefer_explicit(explicit, extracted):
    return explicit if explicit is not None else extracted


def _with_explicit_conflicts(
    interpretation: QueryInterpretation,
    explicit: Optional[PropertySearchRequest],
) -> QueryInterpretation:
    if explicit is None:
        return interpretation
    unresolved = list(interpretation.unresolved_constraints)
    extracted = interpretation.extracted_filters
    comparisons = {
        "purpose": (extracted.purpose, explicit.purpose),
        "property_type": (extracted.property_type, explicit.property_type),
        "city": (extracted.city, explicit.city),
        "neighborhood": (extracted.neighborhood, explicit.neighborhood),
    }
    for field, (parsed, supplied) in comparisons.items():
        if parsed is None or supplied is None:
            continue
        parsed_value = getattr(parsed, "value", parsed)
        supplied_value = getattr(supplied, "value", supplied)
        if not _same_text(str(parsed_value), str(supplied_value)):
            unresolved.append(
                UnresolvedConstraint(
                    field=field,
                    value=f"{parsed_value} / {supplied_value}",
                    reason="conflito_com_filtro_explicito",
                )
            )
    return interpretation.model_copy(
        update={
            "unresolved_constraints": unresolved,
            "requires_clarification": bool(unresolved),
        }
    )


def _search_request(
    interpretation: QueryInterpretation,
    explicit: Optional[PropertySearchRequest],
) -> PropertySearchRequest:
    supplied = explicit or PropertySearchRequest()
    extracted = interpretation.extracted_filters
    return PropertySearchRequest(
        text=supplied.text or interpretation.remaining_text,
        purpose=supplied.purpose or extracted.purpose,
        property_type=supplied.property_type or extracted.property_type,
        city=supplied.city or extracted.city,
        neighborhood=supplied.neighborhood or extracted.neighborhood,
        min_price=supplied.min_price,
        max_price=_prefer_explicit(supplied.max_price, extracted.max_price),
        min_area_m2=supplied.min_area_m2,
        max_area_m2=supplied.max_area_m2,
        bedrooms=_prefer_explicit(supplied.bedrooms, extracted.min_bedrooms),
        bathrooms=_prefer_explicit(supplied.bathrooms, extracted.min_bathrooms),
        parking_spaces=_prefer_explicit(
            supplied.parking_spaces, extracted.min_parking_spaces
        ),
        pets_allowed=(
            supplied.pets_allowed
            if supplied.pets_allowed is not None
            else extracted.pets_allowed
        ),
        furnished=(
            supplied.furnished
            if supplied.furnished is not None
            else extracted.furnished
        ),
        available=supplied.available,
        sort=supplied.sort,
        limit=supplied.limit,
    )


def _clarification_message(
    interpretation: QueryInterpretation, services: AppServices
) -> str:
    constraint = interpretation.unresolved_constraints[0]
    labels = {
        "neighborhood": "bairro",
        "city": "cidade",
        "purpose": "finalidade",
        "property_type": "tipo de imóvel",
    }
    label = labels.get(constraint.field, "restrição")
    options: list[str] = []
    if constraint.field == "neighborhood" and interpretation.extracted_filters.city:
        city = interpretation.extracted_filters.city
        options = sorted(
            {item.neighborhood for item in services.catalog if item.city == city}
        )[:5]
    elif constraint.field == "city":
        options = sorted({item.city for item in services.catalog})[:6]
    base = (
        f"Não encontrei ou não pude resolver o {label} “{constraint.value}” no "
        "catálogo sintético. Confira ou reformule esse critério."
    )
    if options:
        return f"{base} Opções disponíveis: {', '.join(options)}."
    return base


@router.post(
    "/route",
    response_model=AssistantResponse,
    summary="Classifica e direciona uma solicitação sem gerar conteúdo",
)
def route(
    request: AssistantRequest,
    services: AppServices = Depends(require_services),
) -> AssistantResponse:
    classification = services.classifier.classify(request.message)
    decision = decide_route(classification.intent, classification.requires_human_review)
    search = None
    message = decision.message
    action = decision.action
    interpretation = None
    requires_clarification = False
    if decision.action == RouteAction.SEARCH_PROPERTIES:
        interpretation = interpret_property_query(request.message, services.catalog)
        interpretation = _with_explicit_conflicts(
            interpretation, request.search_filters
        )
        requires_clarification = interpretation.requires_clarification
        if requires_clarification:
            action = RouteAction.CLARIFY_SEARCH
            message = _clarification_message(interpretation, services)
        else:
            filters = _search_request(interpretation, request.search_filters)
            search = execute_search(filters, services)
            if not search.results:
                message = "Nenhum imóvel corresponde a todos os critérios informados."
    return AssistantResponse(
        classification=response_from_classification(classification),
        action=action,
        message=message,
        search=search,
        interpreted_query=interpretation,
        requires_clarification=requires_clarification,
    )
