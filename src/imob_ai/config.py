"""Configuração da aplicação carregada a partir do ambiente."""

from dataclasses import dataclass
from os import environ

SUPPORTED_ENVIRONMENTS = frozenset({"development", "test", "production"})


@dataclass(frozen=True)
class AppConfig:
    """Valores de configuração independentes do código-fonte."""

    environment: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Cria a configuração e rejeita ambientes desconhecidos."""
        environment = environ.get("APP_ENV", "development").strip().lower()

        if environment not in SUPPORTED_ENVIRONMENTS:
            supported = ", ".join(sorted(SUPPORTED_ENVIRONMENTS))
            raise ValueError(
                f"APP_ENV inválido: {environment!r}. Valores aceitos: {supported}."
            )

        return cls(environment=environment)
