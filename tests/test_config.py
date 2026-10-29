from unittest.mock import patch

import pytest

from imob_ai.config import AppConfig


def test_config_uses_development_by_default() -> None:
    with patch.dict("os.environ", {}, clear=True):
        config = AppConfig.from_env()

    assert config.environment == "development"


def test_config_normalizes_environment_value() -> None:
    with patch.dict("os.environ", {"APP_ENV": " TEST "}, clear=True):
        config = AppConfig.from_env()

    assert config.environment == "test"


def test_config_rejects_unknown_environment() -> None:
    with patch.dict("os.environ", {"APP_ENV": "staging"}, clear=True):
        with pytest.raises(ValueError, match="APP_ENV inválido"):
            AppConfig.from_env()
