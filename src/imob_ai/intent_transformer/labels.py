"""Taxonomia canônica e determinística do Transformer."""

import hashlib
import json

LABELS = (
    "BUSCA_IMOVEL",
    "AGENDAR_VISITA",
    "DUVIDA_IMOVEL",
    "DOCUMENTACAO_LOCACAO",
    "FINANCEIRO_LOCACAO",
    "MANUTENCAO_IMOVEL",
    "ATENDIMENTO_HUMANO",
    "FORA_DE_ESCOPO",
)
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


def validate_label_mapping() -> None:
    if len(LABEL2ID) != len(ID2LABEL) or set(LABEL2ID.values()) != set(ID2LABEL):
        raise ValueError("Mapping de rótulos inconsistente.")
    if any(ID2LABEL[index] != label for label, index in LABEL2ID.items()):
        raise ValueError("label2id e id2label não são inversos.")


def taxonomy_hash() -> str:
    payload = json.dumps(LABELS, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
