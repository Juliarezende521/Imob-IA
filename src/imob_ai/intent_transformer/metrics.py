"""Métricas compartilhadas pelas avaliações Transformer."""

from dataclasses import asdict

from imob_ai.intent.evaluate import evaluate_predictions
from imob_ai.intent_transformer.labels import LABELS


def calculate_metrics(expected_ids: list[int], predicted_ids: list[int]) -> dict:
    expected = [LABELS[index] for index in expected_ids]
    predicted = [LABELS[index] for index in predicted_ids]
    return asdict(evaluate_predictions(expected, predicted))


def minimum_class_f1(metrics: dict) -> float:
    return min(item["f1"] for item in metrics["per_class"].values())
