import pytest

from imob_ai.intent.dataset import load_intent_dataset
from imob_ai.intent.features import CANDIDATE_CONFIGS
from imob_ai.intent.model import build_pipeline, predict_message
from imob_ai.intent.schemas import DatasetSplit, Intent
from imob_ai.intent.train import SELECTED_CONFIG_NAME, select_configuration


def test_selection_rejects_test_split():
    with pytest.raises(ValueError, match="não aceita"):
        select_configuration(load_intent_dataset())


def test_selected_pipeline_is_reproducible_and_predicts_known_label():
    records = load_intent_dataset()
    train = [item for item in records if item.split == DatasetSplit.TRAIN]
    config = next(
        item for item in CANDIDATE_CONFIGS if item.name == SELECTED_CONFIG_NAME
    )
    first = build_pipeline(config).fit(
        [x.text for x in train], [x.intent.value for x in train]
    )
    second = build_pipeline(config).fit(
        [x.text for x in train], [x.intent.value for x in train]
    )
    message = "Quero conversar com uma pessoa"
    assert first.predict([message]).tolist() == second.predict([message]).tolist()
    prediction = predict_message(first, message)
    assert prediction.intent in {intent.value for intent in Intent}
    assert set(prediction.scores) == {intent.value for intent in Intent}


def test_blank_prediction_is_rejected():
    with pytest.raises(ValueError, match="vazia"):
        predict_message(object(), "  ")


def test_vectorizer_is_fit_only_with_training_text():
    config = CANDIDATE_CONFIGS[0]
    records = load_intent_dataset()
    train = [item for item in records if item.split == DatasetSplit.TRAIN]
    pipeline = build_pipeline(config).fit(
        [x.text for x in train], [x.intent.value for x in train]
    )
    vocabulary = pipeline.named_steps["features"].vocabulary_
    assert "orçamento" not in vocabulary
