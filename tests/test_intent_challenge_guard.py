from unittest.mock import patch

import pytest

from imob_ai.intent.challenge import load_challenge, verify_manifest
from imob_ai.intent.dataset import load_intent_dataset, load_training_records
from imob_ai.intent.features import CANDIDATE_CONFIGS
from imob_ai.intent.model import fit_pipeline
from imob_ai.intent.train import select_configuration


def test_training_and_selection_reject_challenge_before_model_use():
    records = load_challenge()
    with patch("sklearn.linear_model.LogisticRegression.fit") as fit:
        with pytest.raises(ValueError, match="challenge"):
            fit_pipeline(CANDIDATE_CONFIGS[0], records)
        with pytest.raises(ValueError, match="challenge"):
            select_configuration(records)
        fit.assert_not_called()


def test_challenge_validation_calls_no_classifier_or_inference():
    with (
        patch("sklearn.pipeline.Pipeline.fit") as fit,
        patch("sklearn.pipeline.Pipeline.predict") as predict,
        patch("sklearn.pipeline.Pipeline.predict_proba") as predict_proba,
    ):
        records = load_challenge()
        verify_manifest(records)
        fit.assert_not_called()
        predict.assert_not_called()
        predict_proba.assert_not_called()


def test_default_training_loader_does_not_discover_challenge():
    records = load_intent_dataset()
    assert len(records) == 384
    assert all(item.id.startswith("INTENT-") for item in records)
    training = load_training_records()
    assert len(training) == 256
    assert {item.split.value for item in training} == {"train"}


def test_challenge_validation_needs_no_network(monkeypatch):
    def deny_network(*args, **kwargs):
        raise AssertionError("Acesso à rede não permitido")

    monkeypatch.setattr("socket.create_connection", deny_network)
    assert len(load_challenge()) == 192
