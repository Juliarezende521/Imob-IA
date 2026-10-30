"""Seleção pequena da política operacional usando somente validation."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from imob_ai.intent.dataset import load_intent_dataset
from imob_ai.intent.features import CANDIDATE_CONFIGS
from imob_ai.intent.model import fit_pipeline
from imob_ai.intent.schemas import DatasetSplit
from imob_ai.intent.train import SELECTED_CONFIG_NAME

SCORE_CANDIDATES = (0.0, 0.20, 0.25, 0.30, 0.35)
MARGIN_CANDIDATES = (0.0, 0.02, 0.05, 0.10, 0.15)
MINIMUM_COVERAGE = 0.50
POLICY_OUTPUT = Path("docs/experiments/api-uncertainty-policy.json")


@dataclass(frozen=True)
class Candidate:
    minimum_model_score: float
    minimum_margin: float
    coverage: float
    selective_accuracy: float
    accepted: int
    reviewed: int


def calibrate_policy() -> dict[str, object]:
    records = load_intent_dataset()
    train = [item for item in records if item.split == DatasetSplit.TRAIN]
    validation = [item for item in records if item.split == DatasetSplit.VALIDATION]
    config = next(
        item for item in CANDIDATE_CONFIGS if item.name == SELECTED_CONFIG_NAME
    )
    pipeline = fit_pipeline(config, train)
    probabilities = pipeline.predict_proba([item.text for item in validation])
    classes = pipeline.named_steps["classifier"].classes_
    observations = []
    for item, row in zip(validation, probabilities):
        ranked = sorted(zip(classes, row), key=lambda pair: -float(pair[1]))
        observations.append(
            (
                str(ranked[0][0]),
                float(ranked[0][1]),
                float(ranked[0][1] - ranked[1][1]),
                item.intent.value,
            )
        )
    candidates = []
    for score in SCORE_CANDIDATES:
        for margin in MARGIN_CANDIDATES:
            accepted = [
                item for item in observations if item[1] >= score and item[2] >= margin
            ]
            correct = sum(item[0] == item[3] for item in accepted)
            candidates.append(
                Candidate(
                    minimum_model_score=score,
                    minimum_margin=margin,
                    coverage=len(accepted) / len(observations),
                    selective_accuracy=(correct / len(accepted) if accepted else 0.0),
                    accepted=len(accepted),
                    reviewed=len(observations) - len(accepted),
                )
            )
    eligible = [item for item in candidates if item.coverage >= MINIMUM_COVERAGE]
    selected = max(
        eligible,
        key=lambda item: (
            item.selective_accuracy,
            item.coverage,
            item.minimum_margin,
            item.minimum_model_score,
        ),
    )
    return {
        "schema_version": "1",
        "selection_split": "validation",
        "validation_size": len(validation),
        "minimum_coverage": MINIMUM_COVERAGE,
        "score_candidates": list(SCORE_CANDIDATES),
        "margin_candidates": list(MARGIN_CANDIDATES),
        "candidates": [asdict(item) for item in candidates],
        "selected": asdict(selected),
        "score_note": "model_score não é certeza nem probabilidade calibrada.",
        "limitation": "Decisão operacional baseada em apenas 64 mensagens sintéticas.",
    }


def main() -> None:
    result = calibrate_policy()
    POLICY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    POLICY_OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
