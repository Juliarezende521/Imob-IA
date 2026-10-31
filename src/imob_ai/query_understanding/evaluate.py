"""Avaliação reproduzível das regras de extração, separada dos datasets de ML."""

import argparse
import json
from pathlib import Path
from typing import Any

from imob_ai.data.catalog import load_catalog
from imob_ai.query_understanding import interpret_property_query

DEFAULT_CASES = Path("data/evaluation/query_understanding_cases.json")
DEFAULT_CATALOG = Path("data/synthetic/properties.json")
FIELDS = ("purpose", "property_type", "city", "neighborhood")


def _unresolved(result) -> list[dict[str, str]]:
    return [
        {"field": item.field, "value": item.value}
        for item in result.unresolved_constraints
    ]


def evaluate(split: str, cases_path: Path, catalog_path: Path) -> dict[str, Any]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    selected = [case for case in cases if split == "all" or case["split"] == split]
    catalog = load_catalog(catalog_path)
    correct = {field: 0 for field in FIELDS}
    correct.update({"unresolved_constraints": 0, "requires_clarification": 0})
    failures = []
    for case in selected:
        result = interpret_property_query(case["query"], catalog)
        actual_filters = result.extracted_filters.model_dump(mode="json")
        expected = case["expected"]
        case_failures = []
        for field in FIELDS:
            if actual_filters[field] == expected[field]:
                correct[field] += 1
            else:
                case_failures.append(field)
        actual_unresolved = _unresolved(result)
        if actual_unresolved == expected["unresolved_constraints"]:
            correct["unresolved_constraints"] += 1
        else:
            case_failures.append("unresolved_constraints")
        if result.requires_clarification == expected["requires_clarification"]:
            correct["requires_clarification"] += 1
        else:
            case_failures.append("requires_clarification")
        if case_failures:
            failures.append({"id": case["id"], "fields": case_failures})
    total = len(selected)
    return {
        "split": split,
        "cases": total,
        "accuracy_by_field": {
            field: correct[field] / total if total else 0.0 for field in correct
        },
        "exact_cases": total - len(failures),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=("development", "test", "all"), default="development"
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.split, args.cases, args.catalog), indent=2))


if __name__ == "__main__":
    main()
