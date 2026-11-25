import logging
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from imob_ai.api.app import create_app
from imob_ai.api.dependencies import AppServices
from imob_ai.api.services.classifier import Classification
from imob_ai.data.catalog import load_catalog

CATALOG = tuple(load_catalog(Path("data/synthetic/properties.json")))


class FakeClassifier:
    def __init__(self, intent="BUSCA_IMOVEL", review=False):
        self.intent = intent
        self.review = review
        self.calls = 0
        self.fit_count = 1

    def classify(self, message):
        self.calls += 1
        return Classification(
            intent=self.intent,
            model_score=0.7 if not self.review else 0.2,
            second_intent="DUVIDA_IMOVEL",
            second_model_score=0.15,
            margin=0.55 if not self.review else 0.01,
            requires_human_review=self.review,
            review_reason=(
                "margin_below_operational_threshold" if self.review else None
            ),
        )


def client_for(classifier=None):
    selected = classifier or FakeClassifier()
    services = AppServices(classifier=selected, catalog=CATALOG)
    return TestClient(create_app(service_factory=lambda: services)), selected


def test_liveness_and_readiness():
    client, _ = client_for()
    with client:
        assert client.get("/health/live").json() == {"status": "alive"}
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json() == {
            "status": "ready",
            "classifier_ready": True,
            "catalog_ready": True,
            "configuration_valid": True,
        }


def test_initialization_failure_keeps_liveness_but_not_readiness():
    def fail():
        raise RuntimeError("private initialization detail")

    with TestClient(create_app(service_factory=fail)) as client:
        assert client.get("/health/live").status_code == 200
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"
        assert "private" not in response.text


def test_classification_response_and_model_name():
    client, _ = client_for()
    with client:
        response = client.post(
            "/v1/intents/classify", json={"message": "Mensagem sintética"}
        )
    assert response.status_code == 200
    assert response.json() == {
        "intent": "BUSCA_IMOVEL",
        "model_score": 0.7,
        "second_intent": "DUVIDA_IMOVEL",
        "second_model_score": 0.15,
        "margin": 0.55,
        "requires_human_review": False,
        "review_reason": None,
        "model": "tfidf_logistic_regression",
    }


@pytest.mark.parametrize("message", ["", "   ", "x" * 501])
def test_invalid_messages_are_rejected(message):
    client, _ = client_for()
    with client:
        response = client.post("/v1/intents/classify", json={"message": message})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_uncertain_classification_exposes_score_margin_and_reason():
    client, _ = client_for(FakeClassifier(review=True))
    with client:
        body = client.post(
            "/v1/intents/classify", json={"message": "Mensagem ambígua"}
        ).json()
    assert body["model_score"] == 0.2
    assert body["margin"] == 0.01
    assert body["requires_human_review"] is True
    assert body["review_reason"] == "margin_below_operational_threshold"


def test_property_search_text_filters_limit_and_synthetic_flag():
    client, _ = client_for()
    with client:
        response = client.post(
            "/v1/properties/search",
            json={"text": "casa", "city": "Franca", "limit": 2},
        )
    body = response.json()
    assert response.status_code == 200
    assert body["synthetic_data"] is True
    assert body["search_engine"] == "lexical"
    assert len(body["results"]) <= 2
    assert body["total"] >= len(body["results"])
    assert all(item["property"]["city"] == "Franca" for item in body["results"])


def test_property_search_unavailable_and_no_results():
    client, _ = client_for()
    with client:
        unavailable = client.post(
            "/v1/properties/search", json={"available": False, "limit": 50}
        ).json()
        empty = client.post(
            "/v1/properties/search", json={"city": "Cidade Sintética Inexistente"}
        ).json()
    assert unavailable["results"]
    assert all(not item["property"]["available"] for item in unavailable["results"])
    assert empty["total"] == 0
    assert empty["results"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {"min_price": 2000, "max_price": 1000},
        {"min_area_m2": 100, "max_area_m2": 50},
        {"limit": 51},
        {"unknown": True},
    ],
)
def test_invalid_search_is_rejected(payload):
    client, _ = client_for()
    with client:
        response = client.post("/v1/properties/search", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("intent", "action"),
    [
        ("AGENDAR_VISITA", "human_support"),
        ("DUVIDA_IMOVEL", "identify_property"),
        ("DOCUMENTACAO_LOCACAO", "provide_generic_document_guidance"),
        ("FINANCEIRO_LOCACAO", "human_financial_support"),
        ("MANUTENCAO_IMOVEL", "human_maintenance_support"),
        ("ATENDIMENTO_HUMANO", "human_support"),
        ("FORA_DE_ESCOPO", "out_of_scope"),
    ],
)
def test_safe_routes_for_non_search_intents(intent, action):
    client, _ = client_for(FakeClassifier(intent=intent))
    with client:
        body = client.post(
            "/v1/assistant/route", json={"message": "Mensagem sintética"}
        ).json()
    assert body["action"] == action
    assert body["search"] is None
    assert body["generative_ai_used"] is False


def test_search_route_and_optional_filters():
    client, _ = client_for(FakeClassifier(intent="BUSCA_IMOVEL"))
    with client:
        response = client.post(
            "/v1/assistant/route",
            json={
                "message": "Procuro casa sintética em Franca",
                "search_filters": {"city": "Franca", "property_type": "casa"},
            },
        )
    body = response.json()
    assert body["action"] == "search_properties"
    assert body["search"]["synthetic_data"] is True


def test_assistant_applies_extracted_filters_with_strict_and_policy():
    client, _ = client_for(FakeClassifier(intent="BUSCA_IMOVEL"))
    with client:
        body = client.post(
            "/v1/assistant/route",
            json={"message": "Quero uma casa para alugar em Franca."},
        ).json()
    assert body["action"] == "search_properties"
    assert body["requires_clarification"] is False
    assert body["interpreted_query"]["extracted_filters"] == {
        "purpose": "aluguel",
        "property_type": "casa",
        "city": "Franca",
        "neighborhood": None,
        "max_price": None,
        "min_bedrooms": None,
        "min_bathrooms": None,
        "min_parking_spaces": None,
        "pets_allowed": None,
        "furnished": None,
    }
    assert body["search"]["results"]
    assert all(
        item["property"]["purpose"] == "aluguel"
        and item["property"]["property_type"] == "casa"
        and item["property"]["city"] == "Franca"
        for item in body["search"]["results"]
    )


def test_assistant_uses_remaining_text_only_inside_filtered_candidates():
    client, _ = client_for(FakeClassifier(intent="BUSCA_IMOVEL"))
    with client:
        body = client.post(
            "/v1/assistant/route",
            json={"message": "Quero casa em Franca com quintal."},
        ).json()
    assert body["interpreted_query"]["remaining_text"] == "quintal"
    assert body["search"]["results"]
    assert all(
        item["property"]["property_type"] == "casa"
        and item["property"]["city"] == "Franca"
        for item in body["search"]["results"]
    )


def test_assistant_respects_known_neighborhood():
    client, _ = client_for(FakeClassifier(intent="BUSCA_IMOVEL"))
    with client:
        body = client.post(
            "/v1/assistant/route",
            json={"message": "Quero imóvel no Centro em Franca."},
        ).json()
    assert body["requires_clarification"] is False
    assert all(
        item["property"]["neighborhood"] == "Centro"
        and item["property"]["city"] == "Franca"
        for item in body["search"]["results"]
    )


def test_unknown_location_requests_clarification_without_search():
    client, _ = client_for(FakeClassifier(intent="BUSCA_IMOVEL"))
    with client:
        body = client.post(
            "/v1/assistant/route",
            json={"message": "Quero casa em Sorocaba."},
        ).json()
    assert body["action"] == "clarify_search"
    assert body["requires_clarification"] is True
    assert body["search"] is None
    assert "Sorocaba" in body["message"]


def test_observed_regression_never_returns_campinas_or_land():
    client, _ = client_for(FakeClassifier(intent="BUSCA_IMOVEL"))
    message = "eu alugar um acasa no city petropolis em franca"
    with client:
        body = client.post("/v1/assistant/route", json={"message": message}).json()
    filters = body["interpreted_query"]["extracted_filters"]
    unresolved = body["interpreted_query"]["unresolved_constraints"]
    assert filters["purpose"] == "aluguel"
    assert filters["property_type"] == "casa"
    assert filters["city"] == "Franca"
    assert unresolved[0]["value"] == "city petropolis"
    assert body["requires_clarification"] is True
    assert body["search"] is None


def test_valid_filters_with_no_matches_return_honest_zero():
    client, _ = client_for(FakeClassifier(intent="BUSCA_IMOVEL"))
    with client:
        body = client.post(
            "/v1/assistant/route",
            json={"message": "Procuro apartamento à venda em São Paulo."},
        ).json()
    assert body["requires_clarification"] is False
    assert body["search"]["results"] == []
    assert body["search"]["total"] == 0
    assert "todos os critérios" in body["message"]


def test_conflicting_explicit_and_extracted_location_does_not_relax_filter():
    client, _ = client_for(FakeClassifier(intent="BUSCA_IMOVEL"))
    with client:
        body = client.post(
            "/v1/assistant/route",
            json={
                "message": "Quero casa em Franca.",
                "search_filters": {"city": "Campinas"},
            },
        ).json()
    assert body["requires_clarification"] is True
    assert body["search"] is None


def test_uncertain_route_goes_directly_to_human_review():
    client, _ = client_for(FakeClassifier(review=True))
    with client:
        body = client.post(
            "/v1/assistant/route", json={"message": "Mensagem ambígua"}
        ).json()
    assert body["action"] == "human_review"
    assert body["search"] is None


def test_metadata_and_openapi_are_safe():
    client, _ = client_for()
    with client:
        meta = client.get("/v1/meta").json()
        schema = client.get("/openapi.json")
        docs = client.get("/docs")
    assert meta["default_classifier"] == "tfidf_logistic_regression"
    assert meta["default_search"] == "lexical"
    assert meta["synthetic_property_count"] == 48
    assert meta["transformer"] == "experimento_documentado_nao_carregado"
    assert schema.status_code == 200
    assert docs.status_code == 200
    assert "/Users/" not in schema.text
    assert "challenge_messages" not in schema.text


def test_cors_is_disabled_and_extra_fields_are_rejected():
    client, _ = client_for()
    with client:
        cors = client.options(
            "/v1/intents/classify",
            headers={"Origin": "https://example.invalid"},
        )
        extra = client.post(
            "/v1/intents/classify",
            json={"message": "Mensagem sintética", "extra": "no"},
        )
    assert "access-control-allow-origin" not in cors.headers
    assert extra.status_code == 422


def test_logs_do_not_contain_message_content(caplog):
    secret_message = "texto-integral-nao-deve-aparecer"
    client, _ = client_for()
    caplog.set_level(logging.INFO, logger="imob_ai.api.requests")
    with client:
        client.post("/v1/intents/classify", json={"message": secret_message})
    assert secret_message not in caplog.text
    assert "/v1/intents/classify" in caplog.text


def test_requests_need_no_network(monkeypatch):
    def deny_network(*args, **kwargs):
        raise AssertionError("Rede não permitida")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    client, _ = client_for()
    with client:
        assert client.get("/health/ready").status_code == 200
        assert (
            client.post(
                "/v1/intents/classify", json={"message": "Mensagem sintética"}
            ).status_code
            == 200
        )


def test_repeated_and_concurrent_requests_reuse_classifier_and_catalog():
    classifier = FakeClassifier()
    client, _ = client_for(classifier)
    original_catalog = tuple(item.id for item in CATALOG)
    with client:

        def call():
            return client.post(
                "/v1/intents/classify", json={"message": "Mensagem sintética"}
            ).status_code

        with ThreadPoolExecutor(max_workers=4) as executor:
            statuses = list(executor.map(lambda _: call(), range(12)))
    assert statuses == [200] * 12
    assert classifier.fit_count == 1
    assert classifier.calls == 12
    assert tuple(item.id for item in CATALOG) == original_catalog


def _route_message(message):
    client, _ = client_for(FakeClassifier(intent="BUSCA_IMOVEL"))
    with client:
        return client.post("/v1/assistant/route", json={"message": message}).json()


@pytest.mark.parametrize(
    ("message", "label", "source"),
    [
        ("Quero casa com até 3 quartos.", "quartos", "até 3 quartos"),
        ("Quero apartamento com até 2 banheiros.", "banheiros", "até 2 banheiros"),
        ("Quero apartamento com até 1 vaga.", "vagas de garagem", "até 1 vaga"),
        ("Quero terreno com até 2 mil m².", "área", "até 2 mil m"),
    ],
)
def test_unsupported_upper_bound_has_specific_message(message, label, source):
    body = _route_message(message)
    assert body["action"] == "clarify_search"
    assert body["search"] is None
    assert body["message"] == (
        f"Entendi que você informou um limite máximo para {label} "
        f"(“{source}”), mas esse filtro ainda não é suportado. "
        "Reformule a busca sem esse limite."
    )
    assert "catálogo" not in body["message"]


def test_unknown_city_message_uses_feminine_article():
    body = _route_message("Quero casa em Sorocaba.")
    assert body["message"] == (
        "Não encontrei ou não pude resolver a cidade “Sorocaba” no catálogo "
        "sintético. Confira ou reformule esse critério. Opções disponíveis: "
        "Belo Horizonte, Campinas, Curitiba, Franca, Ribeirão Preto, São Paulo."
    )


def test_unknown_neighborhood_message_uses_masculine_article():
    body = _route_message("Quero casa no Jardim Secreto em Franca.")
    assert body["message"].startswith(
        "Não encontrei ou não pude resolver o bairro “Jardim Secreto” no "
        "catálogo sintético. Confira ou reformule esse critério. "
        "Opções disponíveis: "
    )


def test_contradictory_purpose_message_uses_feminine_article():
    body = _route_message("Quero alugar ou comprar uma casa.")
    assert body["message"] == (
        "Não encontrei ou não pude resolver a finalidade “aluguel / venda” no "
        "catálogo sintético. Confira ou reformule esse critério."
    )


def test_contradictory_property_type_message_uses_masculine_article():
    body = _route_message("Quero casa ou terreno.")
    assert body["message"] == (
        "Não encontrei ou não pude resolver o tipo de imóvel “casa / terreno” "
        "no catálogo sintético. Confira ou reformule esse critério."
    )


UNAVAILABLE = [item for item in CATALOG if not item.available]
AVAILABLE = [item for item in CATALOG if item.available]


def _search(payload):
    client, _ = client_for()
    with client:
        response = client.post("/v1/properties/search", json=payload)
    assert response.status_code == 200
    return response.json()


def test_catalog_has_unavailable_properties_after_small_limit():
    assert len(UNAVAILABLE) > 3
    assert CATALOG[0].available


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_unavailable_filter_runs_before_limit(limit):
    body = _search({"available": False, "limit": limit})
    assert len(body["results"]) == limit
    assert all(not item["property"]["available"] for item in body["results"])
    assert body["total"] == len(UNAVAILABLE)


@pytest.mark.parametrize(
    ("limit", "returned"),
    [
        (50, len(UNAVAILABLE)),
        (len(UNAVAILABLE), len(UNAVAILABLE)),
        (len(UNAVAILABLE) - 1, len(UNAVAILABLE) - 1),
    ],
)
def test_unavailable_limit_above_equal_and_below_total(limit, returned):
    body = _search({"available": False, "limit": limit})
    assert len(body["results"]) == returned
    assert body["total"] == len(UNAVAILABLE)


def test_unavailable_page_keeps_search_order():
    full = _search({"available": False, "limit": 50})
    page = _search({"available": False, "limit": 3})
    assert page["results"] == full["results"][:3]


def test_available_true_counts_all_matches_before_limit():
    body = _search({"available": True, "limit": 3})
    assert len(body["results"]) == 3
    assert all(item["property"]["available"] for item in body["results"])
    assert body["total"] == len(AVAILABLE)


@pytest.mark.parametrize("payload", [{"limit": 3}, {"available": None, "limit": 3}])
def test_missing_or_null_available_keeps_current_available_only_behavior(payload):
    body = _search(payload)
    assert all(item["property"]["available"] for item in body["results"])
    assert body["total"] == len(AVAILABLE)


def test_total_matches_filtered_quantity_and_can_exceed_page():
    expected = [
        item for item in CATALOG if not item.available and item.purpose.value == "venda"
    ]
    assert len(expected) > 1
    body = _search({"available": False, "purpose": "venda", "limit": 1})
    assert len(body["results"]) == 1
    assert body["results"][0]["property"]["purpose"] == "venda"
    assert body["total"] == len(expected)
