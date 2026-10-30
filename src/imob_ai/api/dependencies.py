"""Estado de runtime e dependências injetáveis."""

from dataclasses import dataclass
from typing import Optional

from fastapi import Request

from imob_ai.api.services.classifier import ClassifierService
from imob_ai.data.schemas import Property


@dataclass(frozen=True)
class AppServices:
    classifier: ClassifierService
    catalog: tuple[Property, ...]


@dataclass
class RuntimeState:
    services: Optional[AppServices] = None
    initialization_error: Optional[str] = None
    configuration_valid: bool = False

    @property
    def ready(self) -> bool:
        return (
            self.services is not None
            and bool(self.services.catalog)
            and self.configuration_valid
        )


class ServiceNotReadyError(RuntimeError):
    pass


def get_runtime(request: Request) -> RuntimeState:
    return request.app.state.runtime


def require_services(request: Request) -> AppServices:
    runtime = get_runtime(request)
    if not runtime.ready or runtime.services is None:
        raise ServiceNotReadyError("Serviço ainda não está pronto.")
    return runtime.services
