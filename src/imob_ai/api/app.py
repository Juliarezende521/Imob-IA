"""Factory da aplicação FastAPI local."""

import logging
import time
from collections.abc import Callable
from typing import Optional

from fastapi import FastAPI, Request, Response

from imob_ai.api.config import API_VERSION, ApiConfig
from imob_ai.api.dependencies import AppServices
from imob_ai.api.errors import install_error_handlers
from imob_ai.api.lifespan import build_lifespan, default_service_factory
from imob_ai.api.routes import assistant, health, intents, metadata, properties

logger = logging.getLogger("imob_ai.api.requests")


def create_app(
    service_factory: Optional[Callable[[], AppServices]] = None,
    config: Optional[ApiConfig] = None,
) -> FastAPI:
    api_config = config or ApiConfig()
    factory = service_factory or (lambda: default_service_factory(api_config))
    app = FastAPI(
        title="Imob AI — API local",
        description=(
            "Demonstração local com classificação tradicional, busca lexical e "
            "dados inteiramente sintéticos. Não usa LLM nem Transformer em runtime."
        ),
        version=API_VERSION,
        lifespan=build_lifespan(factory),
        openapi_tags=[
            {"name": "health", "description": "Estado do processo e serviços"},
            {"name": "metadata", "description": "Metadados públicos"},
            {"name": "intents", "description": "Classificação de intenções"},
            {"name": "properties", "description": "Busca lexical de imóveis"},
            {"name": "assistant", "description": "Roteamento seguro"},
        ],
    )
    install_error_handlers(app)

    @app.middleware("http")
    async def request_metrics(request: Request, call_next) -> Response:
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            content_length = request.headers.get("content-length", "unknown")
            logger.info(
                "request endpoint=%s status=%s duration_ms=%.3f body_bytes=%s",
                request.url.path,
                status,
                duration_ms,
                content_length,
            )

    app.include_router(health.router)
    app.include_router(metadata.router)
    app.include_router(intents.router)
    app.include_router(properties.router)
    app.include_router(assistant.router)
    return app
