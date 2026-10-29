import math

import pytest

from imob_ai.semantic.encoder import validate_embeddings
from imob_ai.semantic.service import cosine_similarity


def test_cosine_similarity_is_mathematically_correct() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_rejects_incompatible_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensões incompatíveis"):
        cosine_similarity([1.0, 0.0], [1.0])


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_rejects_invalid_numbers(invalid: float) -> None:
    with pytest.raises(ValueError, match="NaN ou infinito"):
        cosine_similarity([1.0, invalid], [1.0, 0.0])


def test_validates_embedding_count_and_dimension() -> None:
    with pytest.raises(ValueError, match="Quantidade"):
        validate_embeddings([[1.0, 0.0]], 2, 2)
    with pytest.raises(ValueError, match="Dimensão"):
        validate_embeddings([[1.0]], 1, 2)
