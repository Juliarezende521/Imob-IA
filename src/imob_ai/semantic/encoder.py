"""Abstração e implementação preguiçosa do encoder textual."""

from collections.abc import Sequence
from math import isfinite
from typing import Optional, Protocol

from imob_ai.semantic.config import MODEL_DIMENSION, MODEL_NAME


class TextEncoder(Protocol):
    """Contrato mínimo para encoders reais e falsos."""

    @property
    def identity(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEncoder:
    """Encoder local que carrega pesos somente no primeiro uso."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: Optional[object] = None

    @property
    def identity(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return MODEL_DIMENSION

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        try:
            vectors = model.encode(  # type: ignore[attr-defined]
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as error:
            raise RuntimeError(f"Falha ao gerar embeddings: {error}") from error
        result = vectors.tolist()
        validate_embeddings(result, len(texts), self.dimension)
        return result

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        except Exception as error:
            raise RuntimeError(
                f"Não foi possível carregar o modelo {self._model_name}: {error}"
            ) from error
        return self._model


def validate_embeddings(
    embeddings: Sequence[Sequence[float]], expected_count: int, dimension: int
) -> None:
    """Rejeita contagem, dimensão e números inválidos."""
    if len(embeddings) != expected_count:
        raise ValueError("Quantidade de embeddings diferente da quantidade de textos.")
    for vector in embeddings:
        if len(vector) != dimension:
            raise ValueError(
                f"Dimensão inválida: esperado {dimension}, recebido {len(vector)}."
            )
        if not all(isfinite(float(value)) for value in vector):
            raise ValueError("Embedding contém NaN ou infinito.")
