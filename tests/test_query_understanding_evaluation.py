import json
from pathlib import Path

from imob_ai.query_understanding.evaluate import evaluate

CASES_PATH = Path("data/evaluation/query_understanding_cases.json")
CATALOG_PATH = Path("data/synthetic/properties.json")


def test_evaluation_dataset_is_synthetic_and_pre_split():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    assert len(cases) == 40
    assert sum(case["split"] == "development" for case in cases) == 30
    assert sum(case["split"] == "test" for case in cases) == 10
    assert len({case["id"] for case in cases}) == 40
    assert all(set(case) == {"id", "split", "query", "expected"} for case in cases)


def test_deterministic_evaluation_matches_versioned_expectations():
    result = evaluate("all", CASES_PATH, CATALOG_PATH)
    assert result["cases"] == 40
    assert result["exact_cases"] == 40
    assert result["failures"] == []
