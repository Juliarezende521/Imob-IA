"""Metadados públicos sem caminhos ou informações pessoais."""

from fastapi import APIRouter, Depends

from imob_ai.api.config import API_VERSION, CLASSIFIER_NAME, SEARCH_ENGINE_NAME
from imob_ai.api.dependencies import AppServices, require_services
from imob_ai.api.schemas import MetaResponse
from imob_ai.intent.schemas import Intent

router = APIRouter(prefix="/v1", tags=["metadata"])


@router.get("/meta", response_model=MetaResponse, summary="Metadados da demonstração")
def metadata(
    services: AppServices = Depends(require_services),
) -> MetaResponse:
    return MetaResponse(
        api_version=API_VERSION,
        default_classifier=CLASSIFIER_NAME,
        default_search=SEARCH_ENGINE_NAME,
        available_intents=[intent.value for intent in Intent],
        synthetic_property_count=len(services.catalog),
        synthetic_data=True,
        demonstration_notice="API local para demonstração com dados sintéticos.",
        transformer="experimento_documentado_nao_carregado",
    )
