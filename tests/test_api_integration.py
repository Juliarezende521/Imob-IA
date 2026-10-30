import sys

from fastapi.testclient import TestClient

from imob_ai.api.app import create_app


def test_real_startup_trains_baseline_once_without_transformer_or_challenge():
    before_transformer = "imob_ai.intent_transformer.model" in sys.modules
    with TestClient(create_app()) as client:
        assert client.get("/health/ready").status_code == 200
        first = client.post(
            "/v1/intents/classify",
            json={"message": "Quero conhecer o apartamento amanhã"},
        )
        second = client.post(
            "/v1/intents/classify",
            json={"message": "Quero conhecer o apartamento amanhã"},
        )
        assert first.json() == second.json()
        assert client.app.state.runtime.services.classifier.fit_count == 1
    assert ("imob_ai.intent_transformer.model" in sys.modules) == before_transformer
