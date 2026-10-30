"""Classificador baseline e política operacional de incerteza."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from imob_ai.intent.dataset import load_intent_dataset
from imob_ai.intent.features import CANDIDATE_CONFIGS
from imob_ai.intent.model import build_pipeline
from imob_ai.intent.schemas import DatasetSplit
from imob_ai.intent.train import SELECTED_CONFIG_NAME


@dataclass(frozen=True)
class UncertaintyPolicy:
    minimum_model_score: float
    minimum_margin: float

    @classmethod
    def load(cls, path: Path) -> "UncertaintyPolicy":
        content = json.loads(path.read_text(encoding="utf-8"))
        if content.get("selection_split") != "validation":
            raise ValueError("Política deve ter sido selecionada em validation.")
        selected = content["selected"]
        return cls(
            minimum_model_score=float(selected["minimum_model_score"]),
            minimum_margin=float(selected["minimum_margin"]),
        )


@dataclass(frozen=True)
class Classification:
    intent: str
    model_score: float
    second_intent: str
    second_model_score: float
    margin: float
    requires_human_review: bool
    review_reason: Optional[str]


class ClassifierService:
    def __init__(self, pipeline: Any, policy: UncertaintyPolicy):
        self._pipeline = pipeline
        self.policy = policy
        self.fit_count = 1

    def classify(self, message: str) -> Classification:
        probabilities = self._pipeline.predict_proba([message])[0]
        classes = self._pipeline.named_steps["classifier"].classes_
        ranked = sorted(zip(classes, probabilities), key=lambda item: -float(item[1]))
        first_label, first_score = ranked[0]
        second_label, second_score = ranked[1]
        margin = float(first_score - second_score)
        reasons = []
        if first_score < self.policy.minimum_model_score:
            reasons.append("model_score_below_operational_threshold")
        if margin < self.policy.minimum_margin:
            reasons.append("margin_below_operational_threshold")
        return Classification(
            intent=str(first_label),
            model_score=float(first_score),
            second_intent=str(second_label),
            second_model_score=float(second_score),
            margin=margin,
            requires_human_review=bool(reasons),
            review_reason=";".join(reasons) if reasons else None,
        )


def build_classifier_service(policy: UncertaintyPolicy) -> ClassifierService:
    records = load_intent_dataset()
    development = [
        item
        for item in records
        if item.split in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}
    ]
    if len(development) != 320:
        raise ValueError("Inicialização exige exatamente train + validation.")
    config = next(
        item for item in CANDIDATE_CONFIGS if item.name == SELECTED_CONFIG_NAME
    )
    pipeline = build_pipeline(config)
    pipeline.fit(
        [item.text for item in development],
        [item.intent.value for item in development],
    )
    return ClassifierService(pipeline, policy)
