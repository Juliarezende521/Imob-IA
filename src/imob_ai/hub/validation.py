"""Validações de integridade, privacidade e segurança dos pacotes."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from imob_ai.hub.metadata import (
    ALLOWED_OUTPUT_ROOT,
    CHALLENGE_MANIFEST,
    CHALLENGE_RECEIPT,
    CHALLENGE_SOURCE,
    DATASET_SOURCE,
    GITHUB_URL,
    LABELS,
    LICENSE_STATUS,
    SOURCE_HASHES,
)
from imob_ai.intent.dataset import PERSONAL_DATA_PATTERNS, normalize_message

SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "generic_secret": re.compile(
        r"\b(?:api[_-]?key|password|secret|token)\s*[:=]\s*[^<\s]+",
        re.IGNORECASE,
    ),
}


class HubValidationError(ValueError):
    """Um pacote local não atende ao contrato de publicação segura."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output_path(path: Path) -> Path:
    resolved = path.resolve()
    allowed = ALLOWED_OUTPUT_ROOT.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise HubValidationError(
            f"Saída deve permanecer dentro de {ALLOWED_OUTPUT_ROOT}."
        )
    return resolved


def validate_source_hashes() -> None:
    observed = {
        "dataset": sha256_file(DATASET_SOURCE),
        "challenge": sha256_file(CHALLENGE_SOURCE),
        "challenge_manifest": sha256_file(CHALLENGE_MANIFEST),
        "challenge_receipt": sha256_file(CHALLENGE_RECEIPT),
    }
    if observed != SOURCE_HASHES:
        raise HubValidationError(f"Hashes de origem divergentes: {observed}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HubValidationError(f"JSONL inválido ou não UTF-8: {path}") from error


def validate_no_sensitive_text(text: str, context: str) -> None:
    for label, pattern in {**PERSONAL_DATA_PATTERNS, **SECRET_PATTERNS}.items():
        if pattern.search(text):
            raise HubValidationError(f"Conteúdo sensível ({label}) em {context}.")


def _challenge_fingerprints() -> tuple[set[str], set[str]]:
    records = read_jsonl(CHALLENGE_SOURCE)
    return (
        {str(record["id"]) for record in records},
        {normalize_message(str(record["text"])) for record in records},
    )


def validate_exported_dataset(dataset_dir: Path) -> dict[str, Any]:
    expected_counts = {"train": 256, "validation": 64, "test": 64}
    records: list[dict[str, Any]] = []
    for split, count in expected_counts.items():
        path = dataset_dir / "data" / f"{split}.jsonl"
        split_records = read_jsonl(path)
        if len(split_records) != count:
            raise HubValidationError(
                f"Contagem incorreta em {split}: {len(split_records)}"
            )
        if any(record.get("split") != split for record in split_records):
            raise HubValidationError(f"Split inconsistente em {path}.")
        if split_records != sorted(split_records, key=lambda item: item["id"]):
            raise HubValidationError(f"Ordenação não determinística em {path}.")
        records.extend(split_records)

    allowed_fields = {"id", "text", "label", "split", "scenario_family", "difficulty"}
    if any(set(record) != allowed_fields for record in records):
        raise HubValidationError("O export contém campos não autorizados.")
    if len(records) != 384:
        raise HubValidationError("O pacote deve conter exatamente 384 registros.")
    ids = [record["id"] for record in records]
    texts = [record["text"] for record in records]
    normalized = [normalize_message(text) for text in texts]
    if len(set(ids)) != len(ids):
        raise HubValidationError("IDs duplicados no pacote.")
    if len(set(texts)) != len(texts) or len(set(normalized)) != len(normalized):
        raise HubValidationError("Textos duplicados no pacote.")
    counts = Counter(record["label"] for record in records)
    if set(counts) != set(LABELS) or set(counts.values()) != {48}:
        raise HubValidationError(f"Distribuição de classes inválida: {counts}")
    families: dict[str, set[str]] = defaultdict(set)
    for record in records:
        families[record["scenario_family"]].add(record["split"])
        validate_no_sensitive_text(record["text"], record["id"])
    if any(len(splits) != 1 for splits in families.values()):
        raise HubValidationError("Famílias atravessam splits.")
    challenge_ids, challenge_texts = _challenge_fingerprints()
    if set(ids) & challenge_ids or set(normalized) & challenge_texts:
        raise HubValidationError("O pacote contém conteúdo do challenge.")
    return {
        "records": len(records),
        "by_split": expected_counts,
        "by_label": dict(sorted(counts.items())),
        "families": len(families),
    }


def parse_card_yaml(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    if not content.startswith("---\n") or "\n---\n" not in content[4:]:
        raise HubValidationError(f"Front matter ausente em {path}.")
    front_matter = content.split("---", 2)[1]
    metadata = yaml.safe_load(front_matter)
    if not isinstance(metadata, dict):
        raise HubValidationError(f"YAML inválido em {path}.")
    return metadata


def validate_manifest(package_dir: Path) -> dict[str, Any]:
    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared_not_published":
        raise HubValidationError("Status de publicação inválido.")
    if manifest.get("license_status") != LICENSE_STATUS:
        raise HubValidationError("Revisão de licença deve permanecer pendente.")
    if not manifest.get("challenge_excluded"):
        raise HubValidationError("Manifesto não declara exclusão do challenge.")
    expected_files = {
        path.relative_to(package_dir).as_posix(): sha256_file(path)
        for path in sorted(package_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    if manifest.get("files") != expected_files:
        raise HubValidationError("Hashes do manifesto não correspondem aos arquivos.")
    return manifest


def validate_package_texts(package_dirs: Iterable[Path]) -> None:
    for package_dir in package_dirs:
        for path in package_dir.rglob("*"):
            if not path.is_file() or path.suffix == ".skops":
                continue
            content = path.read_text(encoding="utf-8")
            sanitized = content.replace(GITHUB_URL, "")
            if PERSONAL_DATA_PATTERNS["url"].search(sanitized):
                raise HubValidationError(f"URL não aprovada em {path}.")
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(sanitized):
                    raise HubValidationError(f"Possível segredo ({label}) em {path}.")
            lowered = content.casefold()
            if "challenge-" in lowered or "challenge_messages" in lowered:
                raise HubValidationError(
                    f"Referência indevida ao conteúdo do challenge: {path}"
                )


def require_no_unresolved_placeholders(package_dirs: Iterable[Path]) -> None:
    allowed = {LICENSE_STATUS}
    pattern = re.compile(r"<[A-Z][A-Z0-9_]+>")
    for package_dir in package_dirs:
        for path in package_dir.rglob("*"):
            if not path.is_file() or path.suffix == ".skops":
                continue
            matches = set(pattern.findall(path.read_text(encoding="utf-8")))
            unexpected = matches - allowed
            if unexpected:
                raise HubValidationError(
                    f"Placeholders inesperados em {path}: {unexpected}"
                )


def validate_model_package(model_dir: Path) -> dict[str, Any]:
    from skops.io import get_untrusted_types, load

    required = {
        "README.md",
        "config.json",
        "requirements.txt",
        "model.skops",
        "labels.json",
        "example_input.json",
        "example_output.json",
        "manifest.json",
    }
    observed = {path.name for path in model_dir.iterdir() if path.is_file()}
    if observed != required:
        raise HubValidationError(
            f"Arquivos inesperados no modelo: {observed ^ required}"
        )
    model_path = model_dir / "model.skops"
    untrusted = get_untrusted_types(file=model_path)
    if untrusted:
        raise HubValidationError(f"Tipos não confiáveis: {untrusted}")
    model = load(model_path, trusted=[])
    labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
    if labels.get("canonical_order") != LABELS:
        raise HubValidationError("Ordem canônica dos labels divergente.")
    if labels.get("classifier_classes") != [
        str(label) for label in model.named_steps["classifier"].classes_
    ]:
        raise HubValidationError("Classes do classificador divergentes.")
    example_input = json.loads(
        (model_dir / "example_input.json").read_text(encoding="utf-8")
    )
    example_output = json.loads(
        (model_dir / "example_output.json").read_text(encoding="utf-8")
    )
    predicted = model.predict(example_input["inputs"]).tolist()
    documented = [item["intent"] for item in example_output["predictions"]]
    if predicted != documented:
        raise HubValidationError("Exemplo de inferência diverge do modelo.")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("model_file") != "model.skops"
        or config.get("training_records") != 320
    ):
        raise HubValidationError("Configuração pública do modelo inválida.")
    if (model_dir / "model.skops").stat().st_size > 5 * 1024 * 1024:
        raise HubValidationError("Artefato do baseline excede 5 MiB inesperadamente.")
    return {"trusted_types": untrusted, "examples": len(predicted)}


def validate_all_packages(output_dir: Path) -> dict[str, Any]:
    dataset_dir = output_dir / "intent-dataset"
    model_dir = output_dir / "intent-model"
    dataset_summary = validate_exported_dataset(dataset_dir)
    model_summary = validate_model_package(model_dir)
    dataset_card = parse_card_yaml(dataset_dir / "README.md")
    model_card = parse_card_yaml(model_dir / "README.md")
    if dataset_card.get("license") != "other" or model_card.get("license") != "other":
        raise HubValidationError("Cards devem manter licença pendente como 'other'.")
    dataset_manifest = validate_manifest(dataset_dir)
    model_manifest = validate_manifest(model_dir)
    if (
        dataset_manifest["dataset_content_sha256"]
        != model_manifest["dataset_content_sha256"]
    ):
        raise HubValidationError("Modelo e dataset referenciam conteúdos diferentes.")
    validate_package_texts((dataset_dir, model_dir))
    require_no_unresolved_placeholders((dataset_dir, model_dir))
    return {
        "dataset": dataset_summary,
        "model": model_summary,
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "model_manifest_sha256": sha256_file(model_dir / "manifest.json"),
        "model_sha256": sha256_file(model_dir / "model.skops"),
    }
