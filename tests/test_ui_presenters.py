from imob_ai.ui.presenters import (
    SCORE_EXPLANATION,
    format_area,
    format_currency,
    format_score,
    human_follow_up,
    uncertainty_label,
)
from imob_ai.ui.schemas import Classification


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
