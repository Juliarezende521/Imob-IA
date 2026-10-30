import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from imob_ai.intent.dataset import load_intent_dataset
from imob_ai.intent.schemas import DatasetSplit
from imob_ai.intent_transformer.benchmark import measure, summarize_timings
from imob_ai.intent_transformer.config import MAX_LENGTH
from imob_ai.intent_transformer.dataset import IntentTorchDataset
from imob_ai.intent_transformer.evaluate_challenge import guard_challenge_access
from imob_ai.intent_transformer.labels import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    taxonomy_hash,
    validate_label_mapping,
)
from imob_ai.intent_transformer.metrics import calculate_metrics
from imob_ai.intent_transformer.tokenization import analyze_lengths, tokenize_text
from imob_ai.intent_transformer.train import file_sha256, split_hash


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        import torch

        length = len(text.split()) + 2
        if kwargs.get("truncation"):
            length = min(length, kwargs["max_length"])
        if kwargs.get("padding") == "max_length":
            length = kwargs["max_length"]
        ids = list(range(length))
        if kwargs.get("return_tensors") == "pt":
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones((1, length), dtype=torch.long),
            }
        return {"input_ids": ids}


def test_label_mapping_and_taxonomy_hash_are_deterministic():
    validate_label_mapping()
    assert LABEL2ID == {label: index for index, label in enumerate(LABELS)}
    assert ID2LABEL == {index: label for label, index in LABEL2ID.items()}
    assert taxonomy_hash() == taxonomy_hash()
    assert len(taxonomy_hash()) == 64


def test_tokenization_analysis_uses_only_development_and_truncates():
    records = load_intent_dataset()
    development = [item for item in records if item.split != DatasetSplit.TEST]
    summary = analyze_lengths(FakeTokenizer(), development, 4)
    assert summary.count == 320
    assert summary.truncated_count > 0
    encoded = tokenize_text(FakeTokenizer(), "texto com várias palavras", 4)
    assert encoded["input_ids"].shape == (1, 4)
    with pytest.raises(ValueError, match="treino e validação"):
        analyze_lengths(FakeTokenizer(), records, MAX_LENGTH)
    with pytest.raises(ValueError, match="vazia"):
        tokenize_text(FakeTokenizer(), " ", 4)


def test_torch_dataset_and_challenge_guard():
    records = load_intent_dataset()
    train = [item for item in records if item.split == DatasetSplit.TRAIN]
    dataset = IntentTorchDataset(train[:2], FakeTokenizer(), 8)
    assert len(dataset) == 2
    assert dataset[0]["input_ids"].shape[0] == 8
    challenge_record = SimpleNamespace(
        split="challenge", text="sintético", intent=SimpleNamespace(value=LABELS[0])
    )
    with pytest.raises(ValueError, match="Challenge"):
        IntentTorchDataset([challenge_record], FakeTokenizer(), 8)


def test_split_hash_is_stable_and_distinct():
    records = load_intent_dataset()
    train = [item for item in records if item.split == DatasetSplit.TRAIN]
    validation = [item for item in records if item.split == DatasetSplit.VALIDATION]
    assert split_hash(train) == split_hash(train)
    assert split_hash(train) != split_hash(validation)


def test_challenge_requires_freeze_hash_and_single_receipt():
    guard_challenge_access(True, False, "same", "same")
    with pytest.raises(RuntimeError, match="Confirmação"):
        guard_challenge_access(False, False, "same", "same")
    with pytest.raises(RuntimeError, match="receipt"):
        guard_challenge_access(True, True, "same", "same")
    with pytest.raises(RuntimeError, match="Hash"):
        guard_challenge_access(True, False, "changed", "frozen")


def test_metrics_include_eight_by_eight_confusion_matrix():
    identifiers = list(range(8))
    metrics = calculate_metrics(identifiers, identifiers)
    assert metrics["f1_macro"] == 1.0
    assert len(metrics["confusion_matrix"]) == 8
    assert all(len(row) == 8 for row in metrics["confusion_matrix"])


def test_file_hash_detects_change(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"frozen": True}), encoding="utf-8")
    first = file_sha256(path)
    path.write_text(json.dumps({"frozen": False}), encoding="utf-8")
    assert file_sha256(path) != first


def test_benchmark_clock_can_be_simulated():
    with patch("time.perf_counter", side_effect=[0.0, 0.1, 0.1, 0.3]):
        values = measure(lambda: None, 2)
    summary = summarize_timings(values)
    assert summary["repetitions"] == 2
    assert summary["median_ms"] == pytest.approx(150.0)


def test_common_import_does_not_download_or_publish():
    with patch("transformers.AutoTokenizer.from_pretrained") as tokenizer:
        import imob_ai.intent_transformer.labels as labels

        assert labels.LABELS == LABELS
        tokenizer.assert_not_called()
