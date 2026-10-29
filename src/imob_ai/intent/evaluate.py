"""Avaliação congelada do classificador de intenções."""

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from imob_ai.intent.dataset import load_intent_dataset
from imob_ai.intent.features import CANDIDATE_CONFIGS
from imob_ai.intent.model import build_pipeline
from imob_ai.intent.schemas import DatasetSplit, Intent, IntentMessage
from imob_ai.intent.train import SELECTED_CONFIG_NAME, select_configuration

DEFAULT_REPORT_PATH = Path("docs/intent-classification-evaluation.md")
LABELS = [intent.value for intent in Intent]


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]


@dataclass(frozen=True)
class ErrorExample:
    id: str
    text: str
    expected: str
    predicted: str
    top_scores: list[tuple[str, float]]
    scenario_family: str
    difficulty: str


def evaluate_predictions(expected: list[str], predicted: list[str]) -> Metrics:
    report = classification_report(
        expected,
        predicted,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": float(report[label]["support"]),
        }
        for label in LABELS
    }
    return Metrics(
        accuracy=accuracy_score(expected, predicted),
        precision_macro=precision_score(
            expected, predicted, labels=LABELS, average="macro", zero_division=0
        ),
        recall_macro=recall_score(
            expected, predicted, labels=LABELS, average="macro", zero_division=0
        ),
        f1_macro=f1_score(
            expected, predicted, labels=LABELS, average="macro", zero_division=0
        ),
        f1_weighted=f1_score(
            expected, predicted, labels=LABELS, average="weighted", zero_division=0
        ),
        per_class=per_class,
        confusion_matrix=confusion_matrix(expected, predicted, labels=LABELS).tolist(),
    )


def run_evaluation(records: Sequence[IntentMessage]) -> dict[str, Any]:
    train = [item for item in records if item.split == DatasetSplit.TRAIN]
    validation = [item for item in records if item.split == DatasetSplit.VALIDATION]
    test = [item for item in records if item.split == DatasetSplit.TEST]
    development = [*train, *validation]

    candidate_results, selected = select_configuration(development)
    if selected.name != SELECTED_CONFIG_NAME:
        raise RuntimeError(
            f"Configuração congelada {SELECTED_CONFIG_NAME} diverge da seleção "
            f"reproduzida {selected.name}."
        )
    selected_config = next(
        config for config in CANDIDATE_CONFIGS if config.name == SELECTED_CONFIG_NAME
    )

    validation_pipeline = build_pipeline(selected_config)
    validation_pipeline.fit(
        [item.text for item in train], [item.intent.value for item in train]
    )
    validation_metrics = _evaluate_model(validation_pipeline, validation)

    validation_dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    validation_dummy.fit(
        [[item.text] for item in train], [item.intent.value for item in train]
    )
    validation_dummy_metrics = evaluate_predictions(
        [item.intent.value for item in validation],
        list(validation_dummy.predict([[item.text] for item in validation])),
    )

    final_pipeline = build_pipeline(selected_config)
    final_pipeline.fit(
        [item.text for item in development],
        [item.intent.value for item in development],
    )
    test_metrics = _evaluate_model(final_pipeline, test)

    final_dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    final_dummy.fit(
        [[item.text] for item in development],
        [item.intent.value for item in development],
    )
    test_dummy_metrics = evaluate_predictions(
        [item.intent.value for item in test],
        list(final_dummy.predict([[item.text] for item in test])),
    )
    errors = _error_analysis(final_pipeline, test)
    confusions = Counter((error.expected, error.predicted) for error in errors)
    return {
        "selected_config": asdict(selected_config),
        "candidate_results": [asdict(result) for result in candidate_results],
        "validation": asdict(validation_metrics),
        "validation_dummy": asdict(validation_dummy_metrics),
        "test": asdict(test_metrics),
        "test_dummy": asdict(test_dummy_metrics),
        "errors": [asdict(error) for error in errors],
        "confusions": [
            {"expected": pair[0], "predicted": pair[1], "count": count}
            for pair, count in confusions.most_common()
        ],
    }


def _evaluate_model(pipeline: Pipeline, records: Sequence[IntentMessage]) -> Metrics:
    expected = [item.intent.value for item in records]
    predicted = list(pipeline.predict([item.text for item in records]))
    return evaluate_predictions(expected, predicted)


def _error_analysis(
    pipeline: Pipeline, records: Sequence[IntentMessage]
) -> list[ErrorExample]:
    texts = [item.text for item in records]
    predicted = pipeline.predict(texts)
    probabilities = pipeline.predict_proba(texts)
    classes = list(pipeline.named_steps["classifier"].classes_)
    errors: list[ErrorExample] = []
    for item, prediction, scores in zip(records, predicted, probabilities):
        if prediction == item.intent.value:
            continue
        ranked = sorted(zip(classes, scores), key=lambda pair: -pair[1])[:3]
        errors.append(
            ErrorExample(
                id=item.id,
                text=item.text,
                expected=item.intent.value,
                predicted=str(prediction),
                top_scores=[(str(label), float(score)) for label, score in ranked],
                scenario_family=item.scenario_family,
                difficulty=item.difficulty.value,
            )
        )
    return errors


def main() -> None:
    from imob_ai.intent.report import render_evaluation_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = run_evaluation(load_intent_dataset())
    markdown = render_evaluation_report(results)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
