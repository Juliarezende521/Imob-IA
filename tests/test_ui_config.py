import pytest

from imob_ai.ui.config import DEFAULT_API_URL, UiConfig, UiConfigError
from imob_ai.ui.schemas import PropertySearchInput


def test_config_uses_local_default(monkeypatch):
    monkeypatch.delenv("IMOB_AI_API_URL", raising=False)
    assert UiConfig.from_env().api_url == DEFAULT_API_URL


def test_config_reads_local_environment(monkeypatch):
    monkeypatch.setenv("IMOB_AI_API_URL", "http://localhost:9000/")
    assert UiConfig.from_env().api_url == "http://localhost:9000"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://example.com",
        "file:///tmp/api",
        "http://user:password@localhost:8000",
        "http://localhost:8000?token=secret",
        "http://localhost:8000/internal/path",
    ],
)
def test_config_rejects_non_local_or_sensitive_url(url):
    with pytest.raises(UiConfigError):
        UiConfig(api_url=url)


def test_search_input_rejects_invalid_price_range():
    with pytest.raises(ValueError, match="preço mínimo"):
        PropertySearchInput(min_price=2000, max_price=1000)


def test_search_input_enforces_message_and_result_limits():
    with pytest.raises(ValueError):
        PropertySearchInput(text="x" * 501)
    with pytest.raises(ValueError):
        PropertySearchInput(limit=51)


def test_streamlit_disables_telemetry_and_binds_locally():
    config = open(".streamlit/config.toml", encoding="utf-8").read()
    assert "gatherUsageStats = false" in config
    assert 'address = "127.0.0.1"' in config
