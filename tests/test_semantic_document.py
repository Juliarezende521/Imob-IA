from imob_ai.data import load_catalog
from imob_ai.semantic.document import build_property_document


def test_document_is_deterministic_and_omits_operational_fields() -> None:
    property_ = load_catalog("data/synthetic/properties.json")[0]

    first = build_property_document(property_)
    second = build_property_document(property_)

    assert first == second
    assert property_.id not in first
    assert "disponibilidade" not in first
    assert "finalidade: aluguel" in first


def test_document_represents_booleans_and_numbers_in_portuguese() -> None:
    property_ = load_catalog("data/synthetic/properties.json")[0]
    document = build_property_document(property_)

    assert "valor em reais: 1600" in document
    assert "área em metros quadrados: 85" in document
    assert "aceita animais: não" in document
    assert "mobiliado: sim" in document
