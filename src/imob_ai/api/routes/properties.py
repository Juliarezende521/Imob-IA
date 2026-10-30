"""Busca estruturada e lexical no catálogo sintético."""

from typing import Optional

from fastapi import APIRouter, Depends

from imob_ai.api.dependencies import AppServices, require_services
from imob_ai.api.schemas import (
    PropertySearchItem,
    PropertySearchRequest,
    PropertySearchResponse,
)
from imob_ai.search.service import search_properties

router = APIRouter(prefix="/v1/properties", tags=["properties"])


def execute_search(
    request: PropertySearchRequest,
    services: AppServices,
    fallback_text: Optional[str] = None,
) -> PropertySearchResponse:
    query = request.to_query(fallback_text=fallback_text)
    result = search_properties(services.catalog, query)
    items = result.items
    if request.available is False:
        items = [item for item in items if not item.property.available]
    return PropertySearchResponse(
        applied_filters=request.applied_filters(fallback_text),
        total=len(items),
        results=[
            PropertySearchItem.model_validate(item.model_dump()) for item in items
        ],
    )


@router.post(
    "/search",
    response_model=PropertySearchResponse,
    summary="Busca imóveis fictícios com filtros e texto livre",
)
def search(
    request: PropertySearchRequest,
    services: AppServices = Depends(require_services),
) -> PropertySearchResponse:
    return execute_search(request, services)
