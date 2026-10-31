import json
import logging

import httpx
import pytest

from imob_ai.ui.api_client import (
    ApiResponseError,
    ApiTimeoutError,
    ApiUnavailableError,
    ImobAiApiClient,
)
from imob_ai.ui.config import UiConfig
from imob_ai.ui.schemas import PropertySearchInput

CLASSIFICATION = {
    "intent": "BUSCA_IMOVEL",
    "model_score": 0.8,
    "second_intent": "DUVIDA_IMOVEL",
    "second_model_score": 0.1,
    "margin": 0.7,
    "requires_human_review": False,
    "review_reason": None,
    "model": "tfidf_logistic_regression",
}

PROPERTY = {
    "id": "IMOB-001",
    "purpose": "aluguel",
    "property_type": "casa",
    "city": "Franca",
    "neighborhood": "Centro",
    "price": 1800.0,
    "bedrooms": 2,
    "bathrooms": 1,
    "parking_spaces": 1,
    "area_m2": 80.0,
    "pets_allowed": True,
    "furnished": False,
    "features": ["quintal"],
    "description": "Casa sintética com quintal.",
    "available": True,
}


def response(data, status=200):
    return httpx.Response(status, json=data)


def client_for(handler):
    return ImobAiApiClient(UiConfig(), transport=httpx.MockTransport(handler))


def test_health_success():
    def handler(request):
        if request.url.path == "/health/live":
            return response({"status": "alive"})
        return response(
            {
                "status": "ready",
                "classifier_ready": True,
                "catalog_ready": True,
                "configuration_valid": True,
            }
        )

    with client_for(handler) as client:
        assert client.health_live().status == "alive"
        assert client.health_ready().status == "ready"


def test_api_unavailable():
    def handler(request):
        raise httpx.ConnectError("offline", request=request)

    with client_for(handler) as client, pytest.raises(ApiUnavailableError):
        client.health_live()


def test_api_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with client_for(handler) as client, pytest.raises(ApiTimeoutError):
        client.health_live()


def test_http_error_does_not_log_message(caplog):
    secret_message = "mensagem que não pode aparecer no log"

    def handler(request):
        return response({"error": {"code": "invalid"}}, status=422)

    with caplog.at_level(logging.WARNING), client_for(handler) as client:
        with pytest.raises(ApiResponseError):
            client.classify(secret_message)
    assert secret_message not in caplog.text


def test_invalid_json():
    def handler(request):
        return httpx.Response(200, content=b"not-json")

    with client_for(handler) as client, pytest.raises(ApiResponseError):
        client.metadata()


def test_incompatible_response_schema():
    def handler(request):
        return response({"intent": "BUSCA_IMOVEL"})

    with client_for(handler) as client, pytest.raises(ApiResponseError):
        client.classify("Procuro uma casa")


def test_valid_classification_request():
    def handler(request):
        assert json.loads(request.content) == {"message": "Procuro uma casa"}
        return response(CLASSIFICATION)

    with client_for(handler) as client:
        result = client.classify("Procuro uma casa")
    assert result.intent == "BUSCA_IMOVEL"
    assert result.model_score == 0.8


def test_valid_metadata():
    def handler(request):
        return response(
            {
                "api_version": "1.0.0",
                "default_classifier": "tfidf_logistic_regression",
                "default_search": "lexical",
                "available_intents": ["BUSCA_IMOVEL"],
                "synthetic_property_count": 50,
                "synthetic_data": True,
                "demonstration_notice": "Dados sintéticos.",
                "transformer": "experimento_documentado_nao_carregado",
            }
        )

    with client_for(handler) as client:
        metadata = client.metadata()
    assert metadata.default_search == "lexical"
    assert metadata.synthetic_data is True


@pytest.mark.parametrize(
    "items",
    [
        [],
        [
            {
                "property": PROPERTY,
                "lexical_score": 2,
                "matched_terms": ["casa"],
                "matched_criteria": ["cidade"],
            }
        ],
    ],
)
def test_valid_search_and_empty_search(items):
    def handler(request):
        payload = json.loads(request.content)
        assert payload["city"] == "Franca"
        return response(
            {
                "applied_filters": payload,
                "total": len(items),
                "results": items,
                "search_engine": "lexical",
                "synthetic_data": True,
            }
        )

    with client_for(handler) as client:
        result = client.search(PropertySearchInput(city="Franca"))
    assert len(result.results) == len(items)


def test_valid_routing():
    def handler(request):
        return response(
            {
                "classification": CLASSIFICATION,
                "action": "search_properties",
                "message": "Busca lexical executada.",
                "search": {
                    "applied_filters": {"city": "Franca"},
                    "total": 0,
                    "results": [],
                    "search_engine": "lexical",
                    "synthetic_data": True,
                },
                "generative_ai_used": False,
            }
        )

    with client_for(handler) as client:
        result = client.route("Procuro uma casa", PropertySearchInput(city="Franca"))
    assert result.action == "search_properties"
    assert result.generative_ai_used is False


def test_app_smoke_without_browser_or_real_api(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from imob_ai.ui import api_client
    from imob_ai.ui.schemas import HealthLive, HealthReady

    monkeypatch.setattr(
        api_client.ImobAiApiClient,
        "health_live",
        lambda self: HealthLive(status="alive"),
    )
    monkeypatch.setattr(
        api_client.ImobAiApiClient,
        "health_ready",
        lambda self: HealthReady(
            status="ready",
            classifier_ready=True,
            catalog_ready=True,
            configuration_valid=True,
        ),
    )
    app_path = "src/imob_ai/ui/app.py"
    app = AppTest.from_file(app_path).run(timeout=10)
    assert not app.exception
    assert app.title[0].value == "ImobAI"


def test_app_presents_uncertainty_score_and_human_route(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from imob_ai.ui import api_client
    from imob_ai.ui.schemas import AssistantResult, HealthLive, HealthReady

    monkeypatch.setattr(
        api_client.ImobAiApiClient,
        "health_live",
        lambda self: HealthLive(status="alive"),
    )
    monkeypatch.setattr(
        api_client.ImobAiApiClient,
        "health_ready",
        lambda self: HealthReady(
            status="ready",
            classifier_ready=True,
            catalog_ready=True,
            configuration_valid=True,
        ),
    )
    result = AssistantResult(
        classification={
            **CLASSIFICATION,
            "model_score": 0.2,
            "requires_human_review": True,
            "review_reason": "score_below_threshold",
        },
        action="human_review",
        message="A solicitação deve ser revisada por uma pessoa.",
        search=None,
        generative_ai_used=False,
    )
    monkeypatch.setattr(
        api_client.ImobAiApiClient,
        "route",
        lambda self, message, search_filters=None: result,
    )

    app = AppTest.from_file("src/imob_ai/ui/app.py").run(timeout=10)
    app.text_area[0].input("Mensagem sintética ambígua")
    app.button[0].click().run(timeout=10)

    assert not app.exception
    assert any(
        metric.label == "model_score" and metric.value == "0.200"
        for metric in app.metric
    )
    assert any("não uma probabilidade calibrada" in item.value for item in app.caption)
    assert any("incerto" in item.value for item in app.warning)
    assert any("atendimento humano" in item.value for item in app.info)


def test_app_prevents_empty_message_submission(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from imob_ai.ui import api_client
    from imob_ai.ui.schemas import HealthLive, HealthReady

    monkeypatch.setattr(
        api_client.ImobAiApiClient,
        "health_live",
        lambda self: HealthLive(status="alive"),
    )
    monkeypatch.setattr(
        api_client.ImobAiApiClient,
        "health_ready",
        lambda self: HealthReady(
            status="ready",
            classifier_ready=True,
            catalog_ready=True,
            configuration_valid=True,
        ),
    )
    app = AppTest.from_file("src/imob_ai/ui/app.py").run(timeout=10)
    app.button[0].click().run(timeout=10)

    assert not app.exception
    assert any("Digite uma mensagem" in item.value for item in app.warning)
