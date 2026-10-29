import json
from pathlib import Path

import pytest

from imob_ai.data.catalog import (
    CatalogFileNotFoundError,
    DuplicatePropertyIdError,
    InvalidCatalogError,
    load_catalog,
)

CATALOG_PATH = Path("data/synthetic/properties.json")


def test_loads_valid_catalog() -> None:
    properties = load_catalog(CATALOG_PATH)

    assert properties


def test_catalog_has_expected_size() -> None:
    properties = load_catalog(CATALOG_PATH)

    assert len(properties) == 48


def test_catalog_has_unique_ids() -> None:
    properties = load_catalog(CATALOG_PATH)
    ids = [property_.id for property_ in properties]

    assert len(ids) == len(set(ids))


def test_catalog_preserves_portuguese_accents() -> None:
    properties = load_catalog(CATALOG_PATH)

    assert any("área" in property_.description.lower() for property_ in properties)
    assert any(
        "área" in feature for property_ in properties for feature in property_.features
    )


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CatalogFileNotFoundError):
        load_catalog(tmp_path / "inexistente.json")


def test_malformed_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "malformado.json"
    path.write_text("[{", encoding="utf-8")

    with pytest.raises(InvalidCatalogError, match="JSON inválido"):
        load_catalog(path)


def test_empty_catalog_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "vazio.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(InvalidCatalogError, match="não pode estar vazio"):
        load_catalog(path)


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    records = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    records[1]["id"] = records[0]["id"]
    path = tmp_path / "duplicado.json"
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(DuplicatePropertyIdError, match=records[0]["id"]):
        load_catalog(path)
