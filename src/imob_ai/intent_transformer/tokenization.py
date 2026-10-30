"""Análise de comprimento restrita a treino e validação."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LengthSummary:
    count: int
    minimum: int
    median: float
    p95: int
    maximum: int
    max_length: int
    truncated_count: int
    truncated_percent: float


def analyze_lengths(
    tokenizer: Any, records: Sequence[Any], max_length: int
) -> LengthSummary:
    split_values = {
        getattr(getattr(item, "split", None), "value", getattr(item, "split", None))
        for item in records
    }
    if not split_values.issubset({"train", "validation"}):
        raise ValueError("Comprimento só pode usar treino e validação.")
    lengths = sorted(
        len(
            tokenizer(item.text, add_special_tokens=True, truncation=False)["input_ids"]
        )
        for item in records
    )
    if not lengths:
        raise ValueError("Análise de comprimento exige registros.")
    middle = len(lengths) // 2
    median = (
        float(lengths[middle])
        if len(lengths) % 2
        else (lengths[middle - 1] + lengths[middle]) / 2
    )
    p95 = lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))]
    truncated = sum(length > max_length for length in lengths)
    return LengthSummary(
        count=len(lengths),
        minimum=lengths[0],
        median=median,
        p95=p95,
        maximum=lengths[-1],
        max_length=max_length,
        truncated_count=truncated,
        truncated_percent=100 * truncated / len(lengths),
    )


def tokenize_text(tokenizer: Any, text: str, max_length: int) -> dict[str, Any]:
    if not text.strip():
        raise ValueError("A mensagem não pode estar vazia.")
    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
