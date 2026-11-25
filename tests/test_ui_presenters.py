import pytest

from imob_ai.ui.presenters import (
    SCORE_EXPLANATION,
    format_area,
    format_currency,
    format_score,
    format_unresolved_constraint,
    human_follow_up,
    uncertainty_label,
)
from imob_ai.ui.schemas import Classification, UnresolvedConstraint


def classification(requires_review: bool = False) -> Classification:
    return Classification(
        intent="BUSCA_IMOVEL",
        model_score=0.81234,
        second_intent="DUVIDA_IMOVEL",
        second_model_score=0.102,
        margin=0.71034,
        requires_human_review=requires_review,
        review_reason="score_below_threshold" if requires_review else None,
        model="tfidf_logistic_regression",
    )


def test_formats_currency_in_brazilian_portuguese():
    assert format_currency(1234567.8) == "R$ 1.234.567,80"


def test_formats_area_and_model_score():
    assert format_area(98.25) == "98,2 m²"
    assert format_score(0.81234) == "0.812"


def test_score_explanation_does_not_call_it_probability():
    assert "model_score" in SCORE_EXPLANATION
    assert "não uma probabilidade calibrada" in SCORE_EXPLANATION


def test_uncertain_result_recommends_human_review():
    label = uncertainty_label(classification(requires_review=True))
    assert "incerto" in label
    assert "humano" in label


def test_human_actions_are_identified():
    assert human_follow_up("human_review")
    assert human_follow_up("human_financial_support")
    assert not human_follow_up("search_properties")


def unresolved(field: str, value: str, reason: str) -> UnresolvedConstraint:
    return UnresolvedConstraint(field=field, value=value, reason=reason)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        (
            "bedrooms",
            "até 3 quartos",
            "Quartos: até 3 quartos — limite máximo entendido, mas ainda não suportado",
        ),
        (
            "bathrooms",
            "até 2 banheiros",
            "Banheiros: até 2 banheiros — limite máximo entendido, mas ainda não "
            "suportado",
        ),
        (
            "parking_spaces",
            "até 1 vaga",
            "Vagas de garagem: até 1 vaga — limite máximo entendido, mas ainda "
            "não suportado",
        ),
        (
            "area_m2",
            "até 100 m",
            "Área: até 100 m — limite máximo entendido, mas ainda não suportado",
        ),
    ],
)
def test_formats_unsupported_upper_bound(field, value, expected):
    text = format_unresolved_constraint(
        unresolved(field, value, "limite_maximo_nao_suportado")
    )
    assert text == expected
    assert "não encontrado" not in text


@pytest.mark.parametrize(
    ("field", "value", "reason", "expected"),
    [
        (
            "city",
            "Sorocaba",
            "nao_encontrada_no_catalogo",
            "Cidade: Sorocaba — critério não encontrado no catálogo sintético",
        ),
        (
            "neighborhood",
            "Jardim Secreto",
            "nao_encontrado_no_catalogo",
            "Bairro: Jardim Secreto — critério não encontrado no catálogo sintético",
        ),
        (
            "purpose",
            "aluguel / venda",
            "valores_contraditorios",
            "Finalidade: aluguel / venda — valores contraditórios; revise o critério",
        ),
        (
            "property_type",
            "casa / terreno",
            "conflito_com_filtro_explicito",
            "Tipo de imóvel: casa / terreno — conflito entre a mensagem e o "
            "filtro informado",
        ),
        (
            "neighborhood",
            "Centro",
            "nao_disponivel_na_cidade",
            "Bairro: Centro — critério precisa de esclarecimento",
        ),
    ],
)
def test_formats_unresolved_reasons(field, value, reason, expected):
    assert format_unresolved_constraint(unresolved(field, value, reason)) == expected


def test_unknown_field_uses_generic_label():
    text = format_unresolved_constraint(unresolved("outro", "x", "outro_motivo"))
    assert text == "Critério: x — critério precisa de esclarecimento"


def test_unresolved_text_adds_no_markdown():
    value = "**negrito** [link](https://exemplo.invalid)"
    text = format_unresolved_constraint(
        unresolved("city", value, "nao_encontrada_no_catalogo")
    )
    assert text.startswith("Cidade: ")
    assert text.count("**") == 2
    assert text.replace(value, "") == (
        "Cidade:  — critério não encontrado no catálogo sintético"
    )
    for marker in ("`", "#", "<", "\\"):
        assert marker not in text
