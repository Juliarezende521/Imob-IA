"""Carregamento congelado, verificação e inferência local."""

import json
from collections.abc import Sequence
from typing import Any

from imob_ai.intent.dataset import load_intent_dataset
from imob_ai.intent.features import CANDIDATE_CONFIGS
from imob_ai.intent.model import build_pipeline
from imob_ai.intent.schemas import DatasetSplit
from imob_ai.intent.train import SELECTED_CONFIG_NAME
from imob_ai.intent_transformer.config import (
    FINAL_MODEL_PATH,
    FROZEN_CONFIG_HASH_PATH,
    FROZEN_CONFIG_PATH,
    TRAINING_METADATA_PATH,
)
from imob_ai.intent_transformer.labels import LABELS
from imob_ai.intent_transformer.metrics import calculate_metrics
from imob_ai.intent_transformer.train import directory_sha256, file_sha256, split_hash


def load_and_verify_frozen() -> tuple[dict[str, Any], dict[str, Any]]:
    if not FROZEN_CONFIG_PATH.exists() or not FROZEN_CONFIG_HASH_PATH.exists():
        raise RuntimeError("Configuração congelada ausente.")
    expected_hash = FROZEN_CONFIG_HASH_PATH.read_text(encoding="utf-8").strip()
    actual_hash = file_sha256(FROZEN_CONFIG_PATH)
    if actual_hash != expected_hash:
        raise RuntimeError("Hash da configuração congelada diverge.")
    config = json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(TRAINING_METADATA_PATH.read_text(encoding="utf-8"))
    if metadata["frozen_config_sha256"] != actual_hash:
        raise RuntimeError("Metadados não correspondem à configuração congelada.")
    if directory_sha256(FINAL_MODEL_PATH) != metadata["artifact_sha256"]:
        raise RuntimeError("Hash do artefato Transformer diverge.")
    records = load_intent_dataset()
    for split in DatasetSplit:
        selected = [item for item in records if item.split == split]
        if split_hash(selected) != config["dataset_hashes"][split.value]:
            raise RuntimeError(f"Hash do split {split.value} diverge.")
    return config, metadata


def load_local_transformer(device: str) -> tuple[Any, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        FINAL_MODEL_PATH, local_files_only=True, use_safetensors=True
    ).to(device)
    model.eval()
    return tokenizer, model


def transformer_predictions(
    records: Sequence[Any], tokenizer: Any, model: Any, device: str, max_length: int
) -> tuple[list[int], list[int], list[list[float]]]:
    import torch

    expected: list[int] = []
    predicted: list[int] = []
    scores: list[list[float]] = []
    with torch.no_grad():
        for start in range(0, len(records), 8):
            batch = records[start : start + 8]
            encoded = tokenizer(
                [item.text for item in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu()
            predicted.extend(probabilities.argmax(dim=-1).tolist())
            scores.extend(probabilities.tolist())
            expected.extend(LABELS.index(item.intent.value) for item in batch)
    return expected, predicted, scores


def baseline_predictions(records: Sequence[Any]) -> tuple[list[int], list[int]]:
    development = [
        item
        for item in load_intent_dataset()
        if item.split in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}
    ]
    feature_config = next(
        item for item in CANDIDATE_CONFIGS if item.name == SELECTED_CONFIG_NAME
    )
    pipeline = build_pipeline(feature_config)
    pipeline.fit(
        [item.text for item in development],
        [item.intent.value for item in development],
    )
    expected = [LABELS.index(item.intent.value) for item in records]
    predicted_labels = pipeline.predict([item.text for item in records])
    predicted = [LABELS.index(str(label)) for label in predicted_labels]
    return expected, predicted


def evaluation_payload(
    records: Sequence[Any], expected: list[int], predicted: list[int]
) -> dict[str, Any]:
    metrics = calculate_metrics(expected, predicted)
    errors = [
        {
            "id": item.id,
            "text": item.text,
            "expected": LABELS[truth],
            "predicted": LABELS[guess],
            "scenario_family": item.scenario_family,
            "difficulty": item.difficulty.value,
        }
        for item, truth, guess in zip(records, expected, predicted)
        if truth != guess
    ]
    return {
        "metrics": metrics,
        "correct": sum(a == b for a, b in zip(expected, predicted)),
        "errors_count": sum(a != b for a, b in zip(expected, predicted)),
        "errors": errors,
    }
