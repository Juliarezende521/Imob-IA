"""Carregamento preguiçoso do modelo Hugging Face."""

from typing import Any

from imob_ai.intent_transformer.config import MODEL_ID, MODEL_REVISION
from imob_ai.intent_transformer.labels import ID2LABEL, LABEL2ID, validate_label_mapping


def load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )


def load_model(freeze_encoder: bool) -> Any:
    from transformers import AutoModelForSequenceClassification

    validate_label_mapping()
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        use_safetensors=True,
        trust_remote_code=False,
    )
    if freeze_encoder:
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False
    return model


def count_parameters(model: Any) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable}
