import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from imob_ai.hub.dataset_export import export_dataset
from imob_ai.hub.validation import read_jsonl, validate_exported_dataset


@pytest.mark.hub
def test_dataset_export_has_frozen_distribution_and_no_leakage():
    output = Path("build/huggingface/pytest-dataset")
    export_dataset(output)
    summary = validate_exported_dataset(output)
    records = []
    for split in ("train", "validation", "test"):
        records.extend(read_jsonl(output / "data" / f"{split}.jsonl"))

    assert summary["records"] == 384
    assert summary["by_split"] == {"train": 256, "validation": 64, "test": 64}
    assert set(Counter(item["label"] for item in records).values()) == {48}
    families = defaultdict(set)
    for item in records:
        families[item["scenario_family"]].add(item["split"])
        assert set(item) == {
            "id",
            "text",
            "label",
            "split",
            "scenario_family",
            "difficulty",
        }
    assert len(families) == 48
    assert all(len(splits) == 1 for splits in families.values())
    json.loads((output / "manifest.json").read_text(encoding="utf-8"))
