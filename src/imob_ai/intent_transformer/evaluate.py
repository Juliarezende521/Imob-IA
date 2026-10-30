"""Avaliação preliminar no teste anteriormente observado."""

import json

from imob_ai.intent.dataset import load_intent_dataset
from imob_ai.intent.schemas import DatasetSplit
from imob_ai.intent_transformer.config import TEST_RESULTS_PATH
from imob_ai.intent_transformer.runtime import (
    baseline_predictions,
    evaluation_payload,
    load_and_verify_frozen,
    load_local_transformer,
    transformer_predictions,
)
from imob_ai.intent_transformer.train import choose_device


def main() -> None:
    if TEST_RESULTS_PATH.exists():
        raise RuntimeError("A avaliação preliminar já possui resultado.")
    config, metadata = load_and_verify_frozen()
    records = [
        item for item in load_intent_dataset() if item.split == DatasetSplit.TEST
    ]
    baseline_expected, baseline_predicted = baseline_predictions(records)
    device = choose_device()
    tokenizer, model = load_local_transformer(device)
    expected, predicted, scores = transformer_predictions(
        records, tokenizer, model, device, config["max_length"]
    )
    result = {
        "status": "preliminary_observed_test",
        "frozen_config_sha256": metadata["frozen_config_sha256"],
        "artifact_sha256": metadata["artifact_sha256"],
        "baseline": evaluation_payload(records, baseline_expected, baseline_predicted),
        "transformer": evaluation_payload(records, expected, predicted),
        "paired": {
            "baseline_only_correct": sum(
                b == y and t != y
                for y, b, t in zip(expected, baseline_predicted, predicted)
            ),
            "transformer_only_correct": sum(
                t == y and b != y
                for y, b, t in zip(expected, baseline_predicted, predicted)
            ),
            "both_correct": sum(
                b == y and t == y
                for y, b, t in zip(expected, baseline_predicted, predicted)
            ),
            "both_wrong": sum(
                b != y and t != y
                for y, b, t in zip(expected, baseline_predicted, predicted)
            ),
        },
        "score_note": "Softmax são scores estimados, não certeza calibrada.",
        "transformer_top_scores": [
            sorted(
                zip(config["id2label"].values(), row),
                key=lambda item: -item[1],
            )[:3]
            for row in scores
        ],
    }
    TEST_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_RESULTS_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
