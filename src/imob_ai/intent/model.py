"""Construção e uso do classificador tradicional."""

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from imob_ai.intent.features import FeatureConfig, build_features

RANDOM_SEED = 42


@dataclass(frozen=True)
class IntentPrediction:
    intent: str
    scores: dict[str, float]


def build_pipeline(config: FeatureConfig) -> Pipeline:
    return Pipeline(
        [
            ("features", build_features(config)),
            (
                "classifier",
                LogisticRegression(
                    C=config.c,
                    class_weight=config.class_weight,
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def predict_message(pipeline: Pipeline, text: str) -> IntentPrediction:
    if not text.strip():
        raise ValueError("A mensagem não pode estar vazia.")
    predicted = str(pipeline.predict([text])[0])
    classifier = pipeline.named_steps["classifier"]
    probabilities = pipeline.predict_proba([text])[0]
    scores = {
        str(label): float(score)
        for label, score in zip(classifier.classes_, probabilities)
    }
    return IntentPrediction(intent=predicted, scores=scores)
