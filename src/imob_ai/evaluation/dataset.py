"""Carregamento dos julgamentos manuais de relevância."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Union

from pydantic import ValidationError

from imob_ai.data.schemas import Property
from imob_ai.evaluation.schemas import EvaluationCase
from imob_ai.search import search_properties


class EvaluationDatasetError(ValueError):
    """O conjunto de avaliação não pôde ser validado."""


class EvaluationFileNotFoundError(EvaluationDatasetError):
    """O arquivo de avaliação não existe."""


def load_evaluation_cases(
    path: Union[str, Path], catalog: Sequence[Property]
) -> list[EvaluationCase]:
    """Carrega casos e valida seus IDs contra o catálogo."""
    dataset_path = Path(path)
    try:
        raw_content = dataset_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise EvaluationFileNotFoundError(
            f"Arquivo de avaliação não encontrado: {dataset_path}"
        ) from error

    try:
        records = json.loads(raw_content)
    except json.JSONDecodeError as error:
        raise EvaluationDatasetError(
            f"JSON inválido no conjunto de avaliação: linha {error.lineno}."
        ) from error

    if not isinstance(records, list):
        raise EvaluationDatasetError("O conjunto de avaliação deve ser uma lista.")
    if not records:
        raise EvaluationDatasetError("O conjunto de avaliação não pode estar vazio.")

    cases: list[EvaluationCase] = []
    for index, record in enumerate(records):
        try:
            cases.append(EvaluationCase.model_validate(record))
        except ValidationError as error:
            raise EvaluationDatasetError(
                f"Caso inválido no índice {index}: {error}"
            ) from error

    case_ids = [case.id for case in cases]
    duplicate_case_ids = sorted(
        {case_id for case_id in case_ids if case_ids.count(case_id) > 1}
    )
    if duplicate_case_ids:
        raise EvaluationDatasetError(
            f"IDs de consulta duplicados: {', '.join(duplicate_case_ids)}."
        )

    catalog_ids = {property_.id for property_ in catalog}
    unknown_ids = sorted(
        {
            property_id
            for case in cases
            for property_id in case.relevant_property_ids
            if property_id not in catalog_ids
        }
    )
    if unknown_ids:
        raise EvaluationDatasetError(
            f"IDs de imóveis inexistentes: {', '.join(unknown_ids)}."
        )

    for case in cases:
        eligible_ids = {
            item.property.id for item in search_properties(catalog, case.filters).items
        }
        incompatible_ids = sorted(set(case.relevant_property_ids) - eligible_ids)
        if incompatible_ids:
            raise EvaluationDatasetError(
                f"IDs relevantes incompatíveis com os filtros de {case.id}: "
                f"{', '.join(incompatible_ids)}."
            )
    return cases
