"""Leitura e validação do catálogo sintético em JSON."""

import json
from pathlib import Path
from typing import Union

from pydantic import ValidationError

from imob_ai.data.schemas import Property


class CatalogError(ValueError):
    """Erro base para catálogos que não podem ser carregados."""


class CatalogFileNotFoundError(CatalogError):
    """O arquivo informado não existe."""


class InvalidCatalogError(CatalogError):
    """O conteúdo do catálogo não respeita o formato esperado."""


class DuplicatePropertyIdError(CatalogError):
    """O catálogo contém identificadores repetidos."""


def load_catalog(path: Union[str, Path]) -> list[Property]:
    """Carrega um JSON, valida cada imóvel e rejeita IDs duplicados."""
    catalog_path = Path(path)

    try:
        raw_content = catalog_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise CatalogFileNotFoundError(
            f"Arquivo de catálogo não encontrado: {catalog_path}"
        ) from error

    try:
        records = json.loads(raw_content)
    except json.JSONDecodeError as error:
        raise InvalidCatalogError(
            f"JSON inválido no catálogo {catalog_path}: linha {error.lineno}."
        ) from error

    if not isinstance(records, list):
        raise InvalidCatalogError("O catálogo deve ser uma lista JSON.")
    if not records:
        raise InvalidCatalogError("O catálogo não pode estar vazio.")

    properties: list[Property] = []
    for index, record in enumerate(records):
        try:
            properties.append(Property.model_validate(record))
        except ValidationError as error:
            raise InvalidCatalogError(
                f"Registro inválido no índice {index}: {error}"
            ) from error

    ids = [property_.id for property_ in properties]
    duplicate_ids = sorted({id_ for id_ in ids if ids.count(id_) > 1})
    if duplicate_ids:
        duplicates = ", ".join(duplicate_ids)
        raise DuplicatePropertyIdError(f"IDs duplicados: {duplicates}.")

    return properties
