import os

import pytest

from imob_ai.data import load_catalog
from imob_ai.search import PropertySearchQuery
from imob_ai.semantic import SentenceTransformerEncoder
from imob_ai.semantic.service import SemanticSearchService


@pytest.mark.model_integration
@pytest.mark.skipif(
    os.getenv("RUN_MODEL_INTEGRATION") != "1",
    reason="Defina RUN_MODEL_INTEGRATION=1; o primeiro uso pode baixar o modelo.",
)
def test_real_model_loads_and_returns_expected_dimension() -> None:
    encoder = SentenceTransformerEncoder()

    vectors = encoder.encode(["casa com quintal"])
    result = SemanticSearchService(encoder).search(
        load_catalog("data/synthetic/properties.json"),
        PropertySearchQuery(text="casa com quintal", limit=3),
    )

    assert len(vectors) == 1
    assert len(vectors[0]) == encoder.dimension
    assert len(result.items) == 3
