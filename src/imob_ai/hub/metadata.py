"""Metadados congelados e compartilhados pelos pacotes locais."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "build" / "huggingface"
ALLOWED_OUTPUT_ROOT = DEFAULT_OUTPUT
DATASET_SOURCE = PROJECT_ROOT / "data" / "intent" / "synthetic_messages.jsonl"
CHALLENGE_SOURCE = PROJECT_ROOT / "data" / "intent" / "challenge_messages.jsonl"
CHALLENGE_MANIFEST = PROJECT_ROOT / "data" / "intent" / "challenge_manifest.json"
CHALLENGE_RECEIPT = (
    PROJECT_ROOT / "docs" / "experiments" / "intent-challenge-evaluation-receipt.json"
)

SOURCE_HASHES = {
    "dataset": "da517761e09f360344affc2ca258f00013c1edcf9e38b94c411a09ce3d4cd402",
    "challenge": "680b4e9496e1025a1e678dbacc5be17bb676124c33288d52fe2d2d3123a685b5",
    "challenge_manifest": (
        "a4506557a4303533d9c122e0728666f2448aba501de4685edd75bdb3417ea4c0"
    ),
    "challenge_receipt": (
        "d75cc43d62587d917e5df183fadf0f6968e8f73fc399f5cf05fa86098e2808ee"
    ),
}

PACKAGE_VERSION = "1.0.0"
SCHEMA_VERSION = "1"
LICENSE_STATUS = "LICENSES_DECLARED"
PUBLICATION_STATUS = "READY_FOR_REVIEW"
CODE_LICENSE_ID = "MIT"
DATASET_LICENSE_ID = "cc-by-4.0"
MODEL_LICENSE_ID = "apache-2.0"
GITHUB_URL = "https://github.com/Juliarezende521/Imob-IA"

LABELS = [
    "BUSCA_IMOVEL",
    "AGENDAR_VISITA",
    "DUVIDA_IMOVEL",
    "DOCUMENTACAO_LOCACAO",
    "FINANCEIRO_LOCACAO",
    "MANUTENCAO_IMOVEL",
    "ATENDIMENTO_HUMANO",
    "FORA_DE_ESCOPO",
]

FROZEN_METRICS = {
    "validation": {"accuracy": 0.766, "macro_f1": 0.764},
    "test": {"accuracy": 0.781, "macro_f1": 0.786, "correct": 50, "total": 64},
    "challenge": {
        "accuracy": 0.682,
        "macro_f1": 0.673,
        "correct": 131,
        "total": 192,
    },
}

MODEL_REQUIREMENTS = {
    "python": "3.9.6",
    "numpy": "1.26.4",
    "scikit-learn": "1.6.1",
    "skops": "0.16.0",
}


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(package_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    files = {
        path.relative_to(package_dir).as_posix(): file_sha256(path)
        for path in sorted(package_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "name": metadata.pop("name"),
        "version": PACKAGE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "source_commit": current_commit(),
        "files": files,
        "dependencies": MODEL_REQUIREMENTS,
        "synthetic": True,
        "challenge_excluded": True,
        "personal_data_included": False,
        "license_status": LICENSE_STATUS,
        "status": PUBLICATION_STATUS,
        **metadata,
    }
    (package_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
