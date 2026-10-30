"""Seleção reproduzível do pipeline usando treino e validação."""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from imob_ai.intent.dataset import load_intent_dataset
from imob_ai.intent.features import CANDIDATE_CONFIGS, FeatureConfig
from imob_ai.intent.model import fit_pipeline
from imob_ai.intent.schemas import DatasetSplit, IntentMessage

# Congelado após seleção exclusiva em validation.
SELECTED_CONFIG_NAME = "word_char"


@dataclass(frozen=True)
class CandidateResult:
    name: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    min_class_f1: float
    dimensions: int
    complexity_rank: int


def select_configuration(
    records: Sequence[IntentMessage],
    configs: Sequence[FeatureConfig] = CANDIDATE_CONFIGS,
) -> tuple[list[CandidateResult], CandidateResult]:
    split_values = {
        getattr(getattr(item, "split", None), "value", getattr(item, "split", None))
        for item in records
    }
    if "challenge" in split_values:
        raise ValueError("Seleção não aceita registros do split challenge.")
    if DatasetSplit.TEST.value in split_values:
        raise ValueError("Seleção não aceita registros do split test.")
    train = [item for item in records if item.split == DatasetSplit.TRAIN]
    validation = [item for item in records if item.split == DatasetSplit.VALIDATION]
    if not train or not validation:
        raise ValueError("Seleção exige treino e validação.")

    results: list[CandidateResult] = []
    for config in configs:
        pipeline = fit_pipeline(config, train)
        predicted = pipeline.predict([item.text for item in validation])
        expected = [item.intent.value for item in validation]
        class_f1 = f1_score(expected, predicted, average=None, zero_division=0)
        dimensions = (
            pipeline.named_steps["features"].transform([train[0].text]).shape[1]
        )
        results.append(
            CandidateResult(
                name=config.name,
                accuracy=accuracy_score(expected, predicted),
                precision_macro=precision_score(
                    expected, predicted, average="macro", zero_division=0
                ),
                recall_macro=recall_score(
                    expected, predicted, average="macro", zero_division=0
                ),
                f1_macro=f1_score(
                    expected, predicted, average="macro", zero_division=0
                ),
                min_class_f1=float(min(class_f1)),
                dimensions=dimensions,
                complexity_rank=config.complexity_rank,
            )
        )
    selected = max(
        results,
        key=lambda result: (
            result.f1_macro,
            result.min_class_f1,
            result.recall_macro,
            -result.complexity_rank,
            -result.dimensions,
        ),
    )
    return results, selected


def main() -> None:
    records = load_intent_dataset()
    development = [item for item in records if item.split != DatasetSplit.TEST]
    results, selected = select_configuration(development)
    print(json.dumps([asdict(result) for result in results], indent=2))
    print(f"SELECTED={selected.name}")


if __name__ == "__main__":
    main()
