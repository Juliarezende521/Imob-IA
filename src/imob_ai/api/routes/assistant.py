"""Roteamento seguro e não generativo."""

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

router = APIRouter(prefix="/v1/assistant", tags=["assistant"])


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
    if decision.action == RouteAction.SEARCH_PROPERTIES:
        filters = request.search_filters or PropertySearchRequest()
        search = execute_search(filters, services, fallback_text=request.message)
        if not search.results:
            message = "Nenhum imóvel sintético corresponde aos critérios informados."
    return AssistantResponse(
        classification=response_from_classification(classification),
        action=decision.action,
        message=message,
        search=search,
    )
