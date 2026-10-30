"""Liveness e readiness."""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from imob_ai.api.dependencies import RuntimeState, get_runtime
from imob_ai.api.schemas import ReadyResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", summary="Verifica se o processo está ativo")
def live() -> dict[str, str]:
    return {"status": "alive"}


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={503: {"description": "Serviços internos ainda não estão prontos"}},
    summary="Verifica classificador, catálogo e configuração",
)
def ready(runtime: RuntimeState = Depends(get_runtime)):
    classifier_ready = runtime.services is not None
    catalog_ready = bool(runtime.services and runtime.services.catalog)
    content = {
        "status": "ready" if runtime.ready else "not_ready",
        "classifier_ready": classifier_ready,
        "catalog_ready": catalog_ready,
        "configuration_valid": runtime.configuration_valid,
    }
    if not runtime.ready:
        return JSONResponse(status_code=503, content=content)
    return ReadyResponse(**content)
