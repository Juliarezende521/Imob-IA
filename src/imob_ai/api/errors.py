"""Erros públicos estáveis, sem detalhes internos."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from imob_ai.api.dependencies import ServiceNotReadyError


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceNotReadyError)
    async def service_not_ready(
        request: Request, error: ServiceNotReadyError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "service_not_ready",
                    "message": "O serviço ainda não está pronto.",
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        safe_details = [
            {"location": list(item["loc"]), "type": item["type"]}
            for item in error.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "A requisição contém dados inválidos.",
                    "details": {"validation": safe_details},
                }
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "Ocorreu um erro interno inesperado.",
                }
            },
        )
