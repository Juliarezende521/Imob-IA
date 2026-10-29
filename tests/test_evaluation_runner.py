from pathlib import Path

from imob_ai.data import load_catalog
from imob_ai.evaluation.dataset import load_evaluation_cases
from imob_ai.evaluation.runner import evaluate_cases, render_markdown
from imob_ai.evaluation.schemas import EvaluationCase

CATALOG_PATH = Path("data/synthetic/properties.json")
EVALUATION_PATH = Path("data/evaluation/retrieval_queries.json")


def make_case(**updates: object) -> EvaluationCase:
    data: dict[str, object] = {
        "id": "QUERY-UNIT",
        "query": "casa quintal",
        "filters": {"property_type": "casa"},
        "relevant_property_ids": ["IMOB-001"],
        "split": "development",
        "category": "literal",
        "notes": "Caso controlado para teste.",
        "expected_no_results": False,
    }
    data.update(updates)
    return EvaluationCase.model_validate(data)


def test_no_result_query_is_evaluated_separately() -> None:
    catalog = load_catalog(CATALOG_PATH)
    case = make_case(
        query="piscina",
        relevant_property_ids=[],
        category="no_result",
        expected_no_results=True,
    )

    report = evaluate_cases(catalog, [case])

    assert report.overall.retrieval_cases == 0
    assert report.overall.no_result_cases == 1
    assert report.overall.no_result_accuracy == 1.0
    assert report.overall.mrr is None


def test_aggregates_metrics_by_split() -> None:
    catalog = load_catalog(CATALOG_PATH)
    cases = [
        make_case(id="DEV", split="development"),
        make_case(id="TEST", split="test"),
    ]

    report = evaluate_cases(catalog, cases)

    assert report.by_split["development"].total_cases == 1
    assert report.by_split["test"].total_cases == 1


def test_calculates_mean_over_retrieval_cases() -> None:
    catalog = load_catalog(CATALOG_PATH)
    cases = [
        make_case(id="HIT"),
        make_case(id="MISS", query="termoinexistente"),
    ]

    report = evaluate_cases(catalog, cases)

    assert report.overall.precision_at_1 == 0.5
    assert report.overall.mrr == 0.5


def test_groups_metrics_by_category() -> None:
    catalog = load_catalog(CATALOG_PATH)
    cases = [
        make_case(id="LITERAL", category="literal"),
        make_case(id="SYNONYM", category="synonym"),
    ]

    report = evaluate_cases(catalog, cases)

    assert set(report.by_category) == {"literal", "synonym"}


def test_evaluation_is_deterministic() -> None:
    catalog = load_catalog(CATALOG_PATH)
    cases = load_evaluation_cases(EVALUATION_PATH, catalog)

    first = evaluate_cases(catalog, cases)
    second = evaluate_cases(catalog, cases)

    assert first.model_dump() == second.model_dump()


def test_integrates_with_real_baseline() -> None:
    catalog = load_catalog(CATALOG_PATH)
    cases = load_evaluation_cases(EVALUATION_PATH, catalog)

    report = evaluate_cases(catalog, cases)

    assert report.overall.total_cases == 32
    assert report.overall.retrieval_cases == 26
    assert report.overall.no_result_cases == 6
    assert report.overall.mrr is not None


def test_markdown_report_contains_formulas_and_failures() -> None:
    catalog = load_catalog(CATALOG_PATH)
    cases = load_evaluation_cases(EVALUATION_PATH, catalog)
    markdown = render_markdown(evaluate_cases(catalog, cases))

    assert "Precision@K" in markdown
    assert "| Geral | 32 | 26 | 6 |" in markdown
    assert "Regra contra vazamento" in markdown
    assert "Falhas individuais" in markdown
