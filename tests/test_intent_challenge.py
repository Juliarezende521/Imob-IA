import json
from collections import Counter

import pytest

from imob_ai.intent.challenge import (
    CHALLENGE_PATH,
    MANIFEST_PATH,
    build_manifest,
    challenge_summary,
    find_near_duplicates,
    load_challenge,
    manifest_json,
    verify_manifest,
)
from imob_ai.intent.dataset import (
    IntentDatasetError,
    load_intent_dataset,
    normalize_message,
)
from imob_ai.intent.schemas import Intent


def test_challenge_contract_and_distribution():
    records = load_challenge()
    summary = challenge_summary(records)
    assert summary["total"] == 192
    assert summary["by_intent"] == {
        intent.value: 24 for intent in sorted(Intent, key=lambda x: x.value)
    }
    assert summary["by_difficulty"] == {"easy": 64, "hard": 64, "medium": 64}
    assert summary["families"] == 48
    assert summary["families_by_intent"] == {
        intent.value: 6 for intent in sorted(Intent, key=lambda x: x.value)
    }
    assert summary["messages_per_family"] == [4]
    assert {item.split for item in records} == {"challenge"}
    assert {item.dataset_version for item in records} == {"challenge-v1"}


def test_challenge_ids_texts_and_taxonomy_are_unique():
    records = load_challenge()
    assert len({item.id for item in records}) == 192
    assert len({item.text for item in records}) == 192
    assert len({normalize_message(item.text) for item in records}) == 192
    assert {item.intent for item in records} == set(Intent)
    assert Counter(item.difficulty.value for item in records) == Counter(
        {"easy": 64, "medium": 64, "hard": 64}
    )


def test_challenge_is_isolated_and_has_no_near_duplicates():
    challenge = load_challenge()
    reference = load_intent_dataset()
    assert not (
        {x.scenario_family for x in challenge} & {x.scenario_family for x in reference}
    )
    assert not (
        {normalize_message(x.text) for x in challenge}
        & {normalize_message(x.text) for x in reference}
    )
    assert find_near_duplicates(challenge, reference) == []


def test_manifest_hash_and_generation_are_deterministic():
    records = load_challenge()
    stored = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert verify_manifest(records) == stored
    assert build_manifest(records) == build_manifest(records)
    assert manifest_json(stored) == MANIFEST_PATH.read_text(encoding="utf-8")


def test_manifest_detects_changed_challenge(tmp_path):
    changed = tmp_path / CHALLENGE_PATH.name
    changed.write_bytes(CHALLENGE_PATH.read_bytes() + b"\n")
    manifest = tmp_path / MANIFEST_PATH.name
    manifest.write_bytes(MANIFEST_PATH.read_bytes())
    with pytest.raises(IntentDatasetError, match="diverge"):
        verify_manifest(load_challenge(), manifest, changed)
