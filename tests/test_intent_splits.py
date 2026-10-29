import pytest

from imob_ai.intent.dataset import (
    IntentDatasetError,
    dataset_summary,
    load_intent_dataset,
    validate_dataset,
)
from imob_ai.intent.schemas import DatasetSplit


def test_split_sizes_and_families_are_deterministic():
    records = load_intent_dataset()
    summary = dataset_summary(records)
    assert summary["by_split"] == {"test": 64, "train": 256, "validation": 64}
    assert summary["families_by_split"] == {"test": 8, "train": 32, "validation": 8}


def test_scenario_family_cannot_cross_splits():
    records = load_intent_dataset()
    changed = records[0].model_copy(update={"split": DatasetSplit.TEST})
    with pytest.raises(IntentDatasetError, match="Famílias atravessam splits"):
        validate_dataset([changed, *records[1:]])
