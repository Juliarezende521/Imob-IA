"""Validação estrutural e integridade do challenge sem executar modelos."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from imob_ai.intent.dataset import (
    DEFAULT_DATASET_PATH,
    PERSONAL_DATA_PATTERNS,
    IntentDatasetError,
    load_intent_dataset,
    normalize_message,
)
from imob_ai.intent.schemas import Difficulty, Intent

CHALLENGE_PATH = Path("data/intent/challenge_messages.jsonl")
MANIFEST_PATH = Path("data/intent/challenge_manifest.json")
DATASET_VERSION = "challenge-v1"
SCHEMA_VERSION = "1"
NEAR_DUPLICATE_THRESHOLD = 0.88


class ChallengeMessage(BaseModel):
    """Registro isolado do challenge final."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    id: str = Field(pattern=r"^CHALLENGE-\d{4}$")
    text: str = Field(min_length=1)
    intent: Intent
    split: Literal["challenge"]
    scenario_family: str = Field(min_length=1)
    difficulty: Difficulty
    notes: Optional[str] = None
    dataset_version: Literal["challenge-v1"]


def sha256_file(path: Union[str, Path]) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_challenge(
    path: Union[str, Path] = CHALLENGE_PATH,
    reference_path: Union[str, Path] = DEFAULT_DATASET_PATH,
) -> list[ChallengeMessage]:
    challenge_path = Path(path)
    try:
        lines = challenge_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise IntentDatasetError(
            f"Challenge não encontrado: {challenge_path}"
        ) from error
    records: list[ChallengeMessage] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(ChallengeMessage.model_validate_json(line))
        except ValidationError as error:
            raise IntentDatasetError(
                f"Challenge inválido na linha {line_number}: {error}"
            ) from error
    reference = load_intent_dataset(reference_path)
    validate_challenge(records, reference)
    return records


def validate_challenge(
    records: list[ChallengeMessage], reference: list[object]
) -> None:
    if len(records) != 192:
        raise IntentDatasetError("O challenge deve conter exatamente 192 registros.")
    ids = [item.id for item in records]
    texts = [item.text for item in records]
    normalized = [normalize_message(text) for text in texts]
    if len(ids) != len(set(ids)):
        raise IntentDatasetError("O challenge contém IDs duplicados.")
    if len(texts) != len(set(texts)):
        raise IntentDatasetError("O challenge contém textos duplicados.")
    if len(normalized) != len(set(normalized)):
        raise IntentDatasetError("O challenge contém textos normalizados duplicados.")

    by_intent = Counter(item.intent.value for item in records)
    if by_intent != Counter({intent.value: 24 for intent in Intent}):
        raise IntentDatasetError("O challenge deve conter 24 registros por classe.")
    by_difficulty = Counter(item.difficulty.value for item in records)
    if by_difficulty != Counter({"easy": 64, "medium": 64, "hard": 64}):
        raise IntentDatasetError("As dificuldades do challenge não estão equilibradas.")

    families: dict[str, list[ChallengeMessage]] = defaultdict(list)
    for item in records:
        families[item.scenario_family].append(item)
        for label, pattern in PERSONAL_DATA_PATTERNS.items():
            if pattern.search(item.text):
                raise IntentDatasetError(
                    f"Possível dado pessoal ({label}) em {item.id}."
                )
        if not item.notes or "sintética" not in item.notes.casefold():
            raise IntentDatasetError(f"{item.id} não declara origem sintética.")
    if len(families) != 48 or any(len(items) != 4 for items in families.values()):
        raise IntentDatasetError("O challenge exige 48 famílias com 4 mensagens.")
    family_intents = Counter(items[0].intent.value for items in families.values())
    if family_intents != Counter({intent.value: 6 for intent in Intent}):
        raise IntentDatasetError("Cada classe deve possuir exatamente 6 famílias.")

    old_ids = {str(item.id) for item in reference}
    old_families = {str(item.scenario_family) for item in reference}
    old_normalized = {normalize_message(str(item.text)) for item in reference}
    if old_ids.intersection(ids):
        raise IntentDatasetError("O challenge reutiliza IDs anteriores.")
    if old_families.intersection(families):
        raise IntentDatasetError("O challenge reutiliza famílias anteriores.")
    if old_normalized.intersection(normalized):
        raise IntentDatasetError("O challenge copia mensagens anteriores.")


def find_near_duplicates(
    challenge: list[ChallengeMessage],
    reference: list[object],
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> list[dict[str, object]]:
    """Compara textos internamente e com os três splits anteriores."""
    challenge_texts = [(item.id, normalize_message(item.text)) for item in challenge]
    reference_texts = [
        (str(item.id), str(item.split.value), normalize_message(str(item.text)))
        for item in reference
    ]
    matches: list[dict[str, object]] = []
    for index, (left_id, left_text) in enumerate(challenge_texts):
        for right_id, right_text in challenge_texts[index + 1 :]:
            ratio = _similarity_at_least(left_text, right_text, threshold)
            if ratio is not None:
                matches.append(
                    {
                        "left": left_id,
                        "right": right_id,
                        "source": "challenge",
                        "ratio": ratio,
                    }
                )
        for right_id, split, right_text in reference_texts:
            ratio = _similarity_at_least(left_text, right_text, threshold)
            if ratio is not None:
                matches.append(
                    {
                        "left": left_id,
                        "right": right_id,
                        "source": split,
                        "ratio": ratio,
                    }
                )
    return matches


def _similarity_at_least(left: str, right: str, threshold: float) -> Optional[float]:
    if not left or not right:
        return None
    length_upper_bound = 2 * min(len(left), len(right)) / (len(left) + len(right))
    if length_upper_bound < threshold:
        return None
    matcher = SequenceMatcher(None, left, right)
    if matcher.quick_ratio() < threshold:
        return None
    ratio = matcher.ratio()
    return ratio if ratio >= threshold else None


def challenge_summary(records: list[ChallengeMessage]) -> dict[str, object]:
    families: dict[str, list[ChallengeMessage]] = defaultdict(list)
    for item in records:
        families[item.scenario_family].append(item)
    return {
        "total": len(records),
        "by_intent": dict(sorted(Counter(x.intent.value for x in records).items())),
        "by_difficulty": dict(
            sorted(Counter(x.difficulty.value for x in records).items())
        ),
        "families": len(families),
        "families_by_intent": dict(
            sorted(
                Counter(items[0].intent.value for items in families.values()).items()
            )
        ),
        "messages_per_family": sorted({len(items) for items in families.values()}),
    }


def build_manifest(
    records: list[ChallengeMessage], path: Union[str, Path] = CHALLENGE_PATH
) -> dict[str, object]:
    summary = challenge_summary(records)
    return {
        "dataset_version": DATASET_VERSION,
        "file": Path(path).name,
        "sha256": sha256_file(path),
        "total": summary["total"],
        "by_intent": summary["by_intent"],
        "by_difficulty": summary["by_difficulty"],
        "families": summary["families"],
        "schema_version": SCHEMA_VERSION,
        "classes": sorted(intent.value for intent in Intent),
        "synthetic": True,
        "usage_declaration": (
            "Não utilizado para treino, seleção, predição, inferência ou avaliação "
            "neste marco."
        ),
    }


def manifest_json(manifest: dict[str, object]) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def verify_manifest(
    records: list[ChallengeMessage],
    manifest_path: Union[str, Path] = MANIFEST_PATH,
    challenge_path: Union[str, Path] = CHALLENGE_PATH,
) -> dict[str, object]:
    try:
        stored = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise IntentDatasetError(f"Manifesto inválido: {error}") from error
    expected = build_manifest(records, challenge_path)
    if stored != expected:
        raise IntentDatasetError("Manifesto ou SHA-256 diverge do challenge congelado.")
    return stored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    reference = load_intent_dataset()
    records = load_challenge()
    near_duplicates = find_near_duplicates(records, reference)
    if near_duplicates:
        raise IntentDatasetError(
            f"Foram encontradas {len(near_duplicates)} duplicatas próximas."
        )
    if args.write_manifest:
        MANIFEST_PATH.write_text(
            manifest_json(build_manifest(records)), encoding="utf-8"
        )
    manifest = verify_manifest(records)
    output = {
        **challenge_summary(records),
        "sha256": manifest["sha256"],
        "near_duplicates": len(near_duplicates),
        "manifest": "válido",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
