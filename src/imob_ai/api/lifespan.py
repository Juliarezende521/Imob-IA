"""Inicialização única do classificador e catálogo."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from imob_ai.api.config import ApiConfig
from imob_ai.api.dependencies import AppServices, RuntimeState
from imob_ai.api.services.classifier import (
    UncertaintyPolicy,
    build_classifier_service,
)
from imob_ai.data.catalog import load_catalog

logger = logging.getLogger("imob_ai.api")
ServiceFactory = Callable[[], AppServices]


def default_service_factory(config: ApiConfig) -> AppServices:
    config.validate()
    policy = UncertaintyPolicy.load(config.policy_path)
    classifier = build_classifier_service(policy)
    catalog = tuple(load_catalog(config.catalog_path))
    return AppServices(classifier=classifier, catalog=catalog)


def build_lifespan(service_factory: ServiceFactory):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = RuntimeState()
        app.state.runtime = runtime
        try:
            runtime.services = service_factory()
            runtime.configuration_valid = True
        except Exception as error:
            runtime.initialization_error = type(error).__name__
            logger.error(
                "api_initialization_failed error_type=%s", type(error).__name__
            )
        yield
        runtime.services = None
        runtime.configuration_valid = False

    return lifespan
