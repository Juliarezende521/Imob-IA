from imob_ai.data import load_catalog
from imob_ai.evaluation.dataset import load_evaluation_cases
from imob_ai.evaluation.runner import evaluate_cases
from imob_ai.hybrid.report import render_hybrid_comparison


def test_three_way_report_contains_frozen_configuration() -> None:
    catalog = load_catalog("data/synthetic/properties.json")
    cases = load_evaluation_cases("data/evaluation/retrieval_queries.json", catalog)
    report = evaluate_cases(catalog, cases)

    markdown = render_hybrid_comparison(report, report, report)

    assert "Weighted Reciprocal Rank Fusion" in markdown
    assert "peso lexical: 0.9" in markdown
    assert "| Precision@1 | 0.923 | 0.923 | 0.923 | +0.000 |" in markdown
    assert "Split test" in markdown
    assert "Resultados por categoria" in markdown
