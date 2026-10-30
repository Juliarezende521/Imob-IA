"""Carregamento, validação e resumo do dataset sintético."""

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Union

from pydantic import ValidationError

from imob_ai.intent.schemas import DatasetSplit, Intent, IntentMessage

DEFAULT_DATASET_PATH = Path("data/intent/synthetic_messages.jsonl")
MINIMUM_PER_CLASS = 40
PERSONAL_DATA_PATTERNS = {
    "url": re.compile(r"https?://|www\.", re.IGNORECASE),
    "email": re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE),
    "phone": re.compile(r"(?:\+?55\s*)?(?:\(?\d{2}\)?\s*)?\d{4,5}[-\s]?\d{4}"),
    "cpf_cnpj": re.compile(r"\b\d{2,3}[.]?\d{3}[.]?\d{3}[/.-]?\d{2,4}[-.]?\d{2}\b"),
    "full_address": re.compile(
        r"\b(?:rua|avenida|av[.]|travessa|alameda)\s+[\wÀ-ÿ ]+[, ]+\d+",
        re.IGNORECASE,
    ),
}


class IntentDatasetError(ValueError):
    """O dataset de intenções violou seu contrato."""


def normalize_message(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    tokens = re.findall(r"[a-z0-9]+", without_accents.casefold())
    return " ".join(tokens)


def load_intent_dataset(
    path: Union[str, Path] = DEFAULT_DATASET_PATH,
) -> list[IntentMessage]:
    dataset_path = Path(path)
    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise IntentDatasetError(f"Dataset não encontrado: {dataset_path}") from error
    if not lines:
        raise IntentDatasetError("O dataset não pode estar vazio.")

    records: list[IntentMessage] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            records.append(IntentMessage.model_validate(raw))
        except (json.JSONDecodeError, ValidationError) as error:
            raise IntentDatasetError(
                f"Registro inválido na linha {line_number}: {error}"
            ) from error
    validate_dataset(records)
    return records


def load_training_records(
    path: Union[str, Path] = DEFAULT_DATASET_PATH,
) -> list[IntentMessage]:
    """Carrega exclusivamente treino; o challenge vive em arquivo separado."""
    return [
        item for item in load_intent_dataset(path) if item.split == DatasetSplit.TRAIN
    ]


def validate_dataset(records: list[IntentMessage]) -> None:
    if not records:
        raise IntentDatasetError("O dataset não pode estar vazio.")
    _reject_duplicates(records)
    _validate_family_isolation(records)
    _validate_class_coverage(records)
    _validate_privacy(records)


def dataset_summary(records: list[IntentMessage]) -> dict[str, object]:
    return {
        "total": len(records),
        "by_intent": dict(
            sorted(Counter(item.intent.value for item in records).items())
        ),
        "by_split": dict(sorted(Counter(item.split.value for item in records).items())),
        "by_difficulty": dict(
            sorted(Counter(item.difficulty.value for item in records).items())
        ),
        "families": len({item.scenario_family for item in records}),
        "families_by_split": dict(
            sorted(
                Counter(
                    next(iter(splits)) for splits in _family_splits(records).values()
                ).items()
            )
        ),
    }


def find_cross_split_near_duplicates(
    records: list[IntentMessage], threshold: float = 0.92
) -> list[tuple[str, str, float]]:
    """Sinaliza textos muito próximos em splits diferentes para revisão."""
    normalized = [(item, normalize_message(item.text)) for item in records]
    matches: list[tuple[str, str, float]] = []
    for index, (left, left_text) in enumerate(normalized):
        for right, right_text in normalized[index + 1 :]:
            if left.split == right.split:
                continue
            ratio = SequenceMatcher(None, left_text, right_text).ratio()
            if ratio >= threshold:
                matches.append((left.id, right.id, ratio))
    return matches


def _reject_duplicates(records: list[IntentMessage]) -> None:
    ids = [item.id for item in records]
    if len(ids) != len(set(ids)):
        raise IntentDatasetError("Existem IDs duplicados.")
    texts = [item.text for item in records]
    if len(texts) != len(set(texts)):
        raise IntentDatasetError("Existem mensagens idênticas.")
    normalized = [normalize_message(text) for text in texts]
    if len(normalized) != len(set(normalized)):
        raise IntentDatasetError("Existem mensagens idênticas após normalização.")


def _family_splits(records: list[IntentMessage]) -> dict[str, set[str]]:
    families: dict[str, set[str]] = defaultdict(set)
    for item in records:
        families[item.scenario_family].add(item.split.value)
    return families


def _validate_family_isolation(records: list[IntentMessage]) -> None:
    leaking = {
        family: splits
        for family, splits in _family_splits(records).items()
        if len(splits) != 1
    }
    if leaking:
        raise IntentDatasetError(f"Famílias atravessam splits: {leaking}")


def _validate_class_coverage(records: list[IntentMessage]) -> None:
    for intent in Intent:
        intent_records = [item for item in records if item.intent == intent]
        if len(intent_records) < MINIMUM_PER_CLASS:
            raise IntentDatasetError(
                f"Classe {intent.value} possui menos de 40 exemplos."
            )
        observed = {item.split for item in intent_records}
        if observed != set(DatasetSplit):
            raise IntentDatasetError(
                f"Classe {intent.value} não aparece em todos os splits."
            )


def _validate_privacy(records: list[IntentMessage]) -> None:
    for item in records:
        for label, pattern in PERSONAL_DATA_PATTERNS.items():
            if pattern.search(item.text):
                raise IntentDatasetError(
                    f"Possível dado pessoal ({label}) em {item.id}."
                )
        if not item.notes or "sintética" not in item.notes.casefold():
            raise IntentDatasetError(
                f"{item.id} não declara explicitamente origem sintética."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_DATASET_PATH)
    args = parser.parse_args()
    records = load_intent_dataset(args.path)
    print(json.dumps(dataset_summary(records), ensure_ascii=False, indent=2))
    near_duplicates = find_cross_split_near_duplicates(records)
    print(f"duplicatas próximas entre splits: {len(near_duplicates)}")


if __name__ == "__main__":
    main()
