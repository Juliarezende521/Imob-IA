"""Configuração central e não sensível da API."""

from dataclasses import dataclass
from pathlib import Path

API_VERSION = "1.0.0"
CATALOG_PATH = Path("data/synthetic/properties.json")
UNCERTAINTY_POLICY_PATH = Path("docs/experiments/api-uncertainty-policy.json")
MAX_MESSAGE_LENGTH = 500
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 50
CLASSIFIER_NAME = "tfidf_logistic_regression"
SEARCH_ENGINE_NAME = "lexical"


@dataclass(frozen=True)
class ApiConfig:
    catalog_path: Path = CATALOG_PATH
    policy_path: Path = UNCERTAINTY_POLICY_PATH

    def validate(self) -> None:
        if not self.catalog_path.is_file():
            raise ValueError("Catálogo sintético indisponível.")
        if not self.policy_path.is_file():
            raise ValueError("Política operacional indisponível.")
