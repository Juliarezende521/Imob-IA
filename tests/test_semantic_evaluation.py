from imob_ai.data import load_catalog
from imob_ai.evaluation.dataset import load_evaluation_cases
from imob_ai.evaluation.runner import evaluate_cases
from imob_ai.semantic.report import render_comparison


def test_comparison_preserves_lexical_report_and_renders_deltas() -> None:
    catalog = load_catalog("data/synthetic/properties.json")
    cases = load_evaluation_cases("data/evaluation/retrieval_queries.json", catalog)
    lexical = evaluate_cases(catalog, cases)
    semantic = evaluate_cases(catalog, cases)

    markdown = render_comparison(lexical, semantic)

    assert lexical.overall.precision_at_1 == 0.9230769230769231
    assert "| Precision@1 | 0.923 | 0.923 | +0.000 |" in markdown
    assert "Split test" in markdown
    assert "Resultados por categoria" in markdown
