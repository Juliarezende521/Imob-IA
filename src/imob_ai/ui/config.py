"""Configuração segura da interface local."""

import os
from dataclasses import dataclass
from urllib.parse import urlparse

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 5.0
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class UiConfigError(ValueError):
    """Indica configuração inválida da interface."""


@dataclass(frozen=True)
class UiConfig:
    api_url: str = DEFAULT_API_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        normalized = self.api_url.rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme != "http" or parsed.hostname not in LOCAL_HOSTS:
            raise UiConfigError("IMOB_AI_API_URL deve apontar para uma API HTTP local.")
        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise UiConfigError(
                "A URL da API não pode conter credenciais ou parâmetros."
            )
        if self.timeout_seconds <= 0:
            raise UiConfigError("O timeout deve ser positivo.")
        object.__setattr__(self, "api_url", normalized)

    @classmethod
    def from_env(cls) -> "UiConfig":
        return cls(api_url=os.getenv("IMOB_AI_API_URL", DEFAULT_API_URL))
