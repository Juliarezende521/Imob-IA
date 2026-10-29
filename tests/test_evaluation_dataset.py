import json
from pathlib import Path

import pytest

from imob_ai.data import Property, load_catalog
from imob_ai.evaluation.dataset import (
    EvaluationDatasetError,
    EvaluationFileNotFoundError,
    load_evaluation_cases,
)

CATALOG_PATH = Path("data/synthetic/properties.json")
EVALUATION_PATH = Path("data/evaluation/retrieval_queries.json")


@pytest.fixture(scope="module")
def catalog() -> list[Property]:
    return load_catalog(CATALOG_PATH)


@pytest.fixture
def valid_record() -> dict[str, object]:
    return {
        "id": "QUERY-TEST",
        "query": "casa com quintal",
        "filters": {"property_type": "casa"},
        "relevant_property_ids": ["IMOB-001"],
        "split": "development",
        "category": "literal",
        "notes": "Caso mínimo para validar o loader.",
        "expected_no_results": False,
    }


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def test_loads_valid_evaluation_dataset(catalog: list[Property]) -> None:
    cases = load_evaluation_cases(EVALUATION_PATH, catalog)

    assert len(cases) == 32


def test_missing_file_is_rejected(catalog: list[Property], tmp_path: Path) -> None:
    with pytest.raises(EvaluationFileNotFoundError):
        load_evaluation_cases(tmp_path / "missing.json", catalog)


def test_malformed_json_is_rejected(catalog: list[Property], tmp_path: Path) -> None:
    path = tmp_path / "malformed.json"
    path.write_text("[{", encoding="utf-8")

    with pytest.raises(EvaluationDatasetError, match="JSON inválido"):
        load_evaluation_cases(path, catalog)


def test_empty_dataset_is_rejected(catalog: list[Property], tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(EvaluationDatasetError, match="não pode estar vazio"):
        load_evaluation_cases(path, catalog)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query", "   "),
        ("split", "training"),
        ("category", "unknown"),
    ],
)
def test_invalid_controlled_fields_are_rejected(
    catalog: list[Property],
    tmp_path: Path,
    valid_record: dict[str, object],
    field: str,
    value: object,
) -> None:
    valid_record[field] = value
    path = tmp_path / "invalid.json"
    write_records(path, [valid_record])

    with pytest.raises(EvaluationDatasetError, match="Caso inválido"):
        load_evaluation_cases(path, catalog)


def test_duplicate_query_id_is_rejected(
    catalog: list[Property], tmp_path: Path, valid_record: dict[str, object]
) -> None:
    path = tmp_path / "duplicate-query.json"
    write_records(path, [valid_record, valid_record])

    with pytest.raises(EvaluationDatasetError, match="consulta duplicados"):
        load_evaluation_cases(path, catalog)


def test_duplicate_relevant_id_is_rejected(
    catalog: list[Property], tmp_path: Path, valid_record: dict[str, object]
) -> None:
    valid_record["relevant_property_ids"] = ["IMOB-001", "IMOB-001"]
    path = tmp_path / "duplicate-property.json"
    write_records(path, [valid_record])

    with pytest.raises(EvaluationDatasetError, match="repetições"):
        load_evaluation_cases(path, catalog)


def test_unknown_property_id_is_rejected(
    catalog: list[Property], tmp_path: Path, valid_record: dict[str, object]
) -> None:
    valid_record["relevant_property_ids"] = ["IMOB-999"]
    path = tmp_path / "unknown-property.json"
    write_records(path, [valid_record])

    with pytest.raises(EvaluationDatasetError, match="inexistentes"):
        load_evaluation_cases(path, catalog)


def test_relevant_id_must_respect_case_filters(
    catalog: list[Property], tmp_path: Path, valid_record: dict[str, object]
) -> None:
    valid_record["filters"] = {"property_type": "terreno"}
    path = tmp_path / "incompatible-filter.json"
    write_records(path, [valid_record])

    with pytest.raises(EvaluationDatasetError, match="incompatíveis com os filtros"):
        load_evaluation_cases(path, catalog)


def test_no_result_case_cannot_contain_relevant_ids(
    catalog: list[Property], tmp_path: Path, valid_record: dict[str, object]
) -> None:
    valid_record["expected_no_results"] = True
    path = tmp_path / "contradictory.json"
    write_records(path, [valid_record])

    with pytest.raises(EvaluationDatasetError, match="sem resultado"):
        load_evaluation_cases(path, catalog)
