import json

import pytest

from imob_ai.intent.dataset import (
    IntentDatasetError,
    find_cross_split_near_duplicates,
    load_intent_dataset,
    normalize_message,
)
from imob_ai.intent.schemas import DatasetSplit, Intent


def test_valid_dataset_is_balanced_and_complete():
    records = load_intent_dataset()
    assert len(records) == 384
    assert len({item.id for item in records}) == 384
    assert len({item.text for item in records}) == 384
    assert len({normalize_message(item.text) for item in records}) == 384
    for intent in Intent:
        selected = [item for item in records if item.intent == intent]
        assert len(selected) == 48
        assert {item.split for item in selected} == set(DatasetSplit)


def test_dataset_has_no_cross_split_near_duplicates():
    assert find_cross_split_near_duplicates(load_intent_dataset()) == []


def test_missing_and_malformed_files_are_rejected(tmp_path):
    with pytest.raises(IntentDatasetError, match="não encontrado"):
        load_intent_dataset(tmp_path / "missing.jsonl")
    malformed = tmp_path / "bad.jsonl"
    malformed.write_text("{bad", encoding="utf-8")
    with pytest.raises(IntentDatasetError, match="linha 1"):
        load_intent_dataset(malformed)


def test_duplicate_id_is_rejected(tmp_path):
    records = load_intent_dataset()
    raw = [item.model_dump(mode="json") for item in records]
    raw[1]["id"] = raw[0]["id"]
    path = tmp_path / "duplicate.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in raw), encoding="utf-8")
    with pytest.raises(IntentDatasetError, match="IDs duplicados"):
        load_intent_dataset(path)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (lambda text: text, "mensagens idênticas"),
        (lambda text: text.upper() + "!!!", "após normalização"),
    ],
)
def test_duplicate_text_is_rejected(tmp_path, replacement, message):
    records = load_intent_dataset()
    raw = [item.model_dump(mode="json") for item in records]
    raw[1]["text"] = replacement(raw[0]["text"])
    path = tmp_path / "duplicate-text.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in raw), encoding="utf-8")
    with pytest.raises(IntentDatasetError, match=message):
        load_intent_dataset(path)


@pytest.mark.parametrize(("field", "value"), [("intent", "OUTRA"), ("split", "dev")])
def test_controlled_fields_reject_unknown_values(tmp_path, field, value):
    records = load_intent_dataset()
    raw = [item.model_dump(mode="json") for item in records]
    raw[0][field] = value
    path = tmp_path / "invalid-field.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in raw), encoding="utf-8")
    with pytest.raises(IntentDatasetError, match="linha 1"):
        load_intent_dataset(path)


def test_dataset_contains_no_basic_personal_data_patterns():
    records = load_intent_dataset()
    assert all("@" not in item.text and "http" not in item.text for item in records)
