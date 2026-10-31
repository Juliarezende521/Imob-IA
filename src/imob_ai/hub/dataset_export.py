"""Exportação determinística do dataset sintético autorizado."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from string import Template

from imob_ai.hub.metadata import (
    GITHUB_URL,
    LABELS,
    LICENSE_STATUS,
    PACKAGE_VERSION,
    PROJECT_ROOT,
    write_manifest,
)
from imob_ai.hub.validation import validate_exported_dataset
from imob_ai.intent.dataset import load_intent_dataset
from imob_ai.intent.schemas import DatasetSplit

SPLIT_ORDER = (DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST)


def _public_record(record) -> dict[str, object]:
    return {
        "id": record.id,
        "text": record.text,
        "label": record.intent.value,
        "split": record.split.value,
        "scenario_family": record.scenario_family,
        "difficulty": record.difficulty.value,
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    path.write_text(content, encoding="utf-8")


def _content_hash(data_dir: Path) -> str:
    digest = hashlib.sha256()
    for split in SPLIT_ORDER:
        digest.update((data_dir / f"{split.value}.jsonl").read_bytes())
    return digest.hexdigest()


def export_dataset(package_dir: Path) -> dict[str, object]:
    package_dir.mkdir(parents=True, exist_ok=True)
    data_dir = package_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    records = load_intent_dataset()
    for split in SPLIT_ORDER:
        exported = sorted(
            (_public_record(record) for record in records if record.split == split),
            key=lambda item: str(item["id"]),
        )
        _write_jsonl(data_dir / f"{split.value}.jsonl", exported)

    summary = validate_exported_dataset(package_dir)
    content_hash = _content_hash(data_dir)
    template = Template(
        (PROJECT_ROOT / "hub" / "dataset-card-template.md").read_text(encoding="utf-8")
    )
    labels = "\n".join(f"- `{label}`" for label in LABELS)
    (package_dir / "README.md").write_text(
        template.substitute(
            version=PACKAGE_VERSION,
            content_hash=content_hash,
            github_url=GITHUB_URL,
            license_status=LICENSE_STATUS,
            labels=labels,
        ),
        encoding="utf-8",
    )
    counts = Counter(record.intent.value for record in records)
    manifest = write_manifest(
        package_dir,
        {
            "name": "imob-ai-synthetic-intents-pt",
            "record_count": len(records),
            "split_counts": summary["by_split"],
            "class_counts": dict(sorted(counts.items())),
            "dataset_content_sha256": content_hash,
        },
    )
    return manifest
