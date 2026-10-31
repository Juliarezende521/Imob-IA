"""Cliente HTTP tipado usado exclusivamente pela interface."""

import logging
from typing import Any, Optional, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from imob_ai.ui.config import UiConfig
from imob_ai.ui.schemas import (
    AssistantResult,
    Classification,
    HealthLive,
    HealthReady,
    Metadata,
    PropertySearchInput,
    PropertySearchResult,
)

logger = logging.getLogger("imob_ai.ui.http")
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class ApiClientError(RuntimeError):
    """Erro público e seguro ao acessar a API."""

    def __init__(self, message: str, technical_detail: str) -> None:
        super().__init__(message)
        self.public_message = message
        self.technical_detail = technical_detail


class ApiUnavailableError(ApiClientError):
    pass


class ApiTimeoutError(ApiClientError):
    pass


class ApiResponseError(ApiClientError):
    pass


class ImobAiApiClient:
    def __init__(
        self,
        config: UiConfig,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=config.api_url,
            timeout=config.timeout_seconds,
            transport=transport,
            follow_redirects=False,
            trust_env=False,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ImobAiApiClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        response_model: type[ResponseModel],
        payload: Optional[dict[str, Any]] = None,
    ) -> ResponseModel:
        try:
            response = self._client.request(method, path, json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            logger.warning("API timeout endpoint=%s", path)
            raise ApiTimeoutError(
                "A API demorou demais para responder.", type(exc).__name__
            ) from exc
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "API status_error endpoint=%s status=%s", path, exc.response.status_code
            )
            raise ApiResponseError(
                "A API recusou a solicitação.",
                f"HTTP {exc.response.status_code}",
            ) from exc
        except httpx.RequestError as exc:
            logger.warning("API unavailable endpoint=%s", path)
            raise ApiUnavailableError(
                "Não foi possível conectar à API local.", type(exc).__name__
            ) from exc

        try:
            return response_model.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            logger.warning("API invalid_response endpoint=%s", path)
            raise ApiResponseError(
                "A API retornou uma resposta incompatível.", type(exc).__name__
            ) from exc

    def health_live(self) -> HealthLive:
        return self._request("GET", "/health/live", HealthLive)

    def health_ready(self) -> HealthReady:
        return self._request("GET", "/health/ready", HealthReady)

    def metadata(self) -> Metadata:
        return self._request("GET", "/v1/meta", Metadata)

    def classify(self, message: str) -> Classification:
        return self._request(
            "POST", "/v1/intents/classify", Classification, {"message": message}
        )

    def search(self, filters: PropertySearchInput) -> PropertySearchResult:
        return self._request(
            "POST",
            "/v1/properties/search",
            PropertySearchResult,
            filters.model_dump(mode="json", exclude_none=True),
        )

    def route(
        self,
        message: str,
        search_filters: Optional[PropertySearchInput] = None,
    ) -> AssistantResult:
        payload: dict[str, Any] = {"message": message}
        if search_filters is not None:
            payload["search_filters"] = search_filters.model_dump(
                mode="json", exclude_none=True
            )
        return self._request("POST", "/v1/assistant/route", AssistantResult, payload)
