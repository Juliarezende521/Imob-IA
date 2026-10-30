import pytest

from imob_ai.intent_transformer.config import MODEL_ID, MODEL_REVISION
from imob_ai.intent_transformer.labels import LABEL2ID


@pytest.mark.transformer_integration
def test_real_model_produces_eight_logits():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        num_labels=len(LABEL2ID),
        use_safetensors=True,
        trust_remote_code=False,
    )
    encoded = tokenizer("Mensagem sintética", return_tensors="pt")
    assert model(**encoded).logits.shape == (1, 8)
