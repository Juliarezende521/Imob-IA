"""Valida cópias públicas e imutáveis dos artefatos no Hugging Face Hub."""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from imob_ai.hub.metadata import (
    DATASET_LICENSE_ID,
    DEFAULT_OUTPUT,
    MODEL_LICENSE_ID,
)
from imob_ai.hub.validation import (
    HubValidationError,
    parse_card_yaml,
    validate_dataset_package_files,
    validate_exported_dataset,
    validate_manifest,
    validate_model_package,
    validate_package_texts,
)

DEFAULT_DATASET_REPO = "JuliaRezende521/imob-ai-intent-dataset-pt"
DEFAULT_MODEL_REPO = "JuliaRezende521/imob-ai-intent-classifier-pt"
DEFAULT_DATASET_REVISION = "1c1fdd14a4730a7a616a8e133b1f4f1bf6577c3c"
DEFAULT_MODEL_REVISION = "a2bc23ea970bf1c4186c6798e6359367e85a7d7f"
PUBLIC_CACHE = DEFAULT_OUTPUT / "public-cache"
REPO_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
)
REVISION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")

DATASET_FILES = {
    "LICENSE",
    "README.md",
    "manifest.json",
    "data/train.jsonl",
    "data/validation.jsonl",
    "data/test.jsonl",
}
MODEL_FILES = {
    "LICENSE",
    "README.md",
    "config.json",
    "requirements.txt",
    "model.skops",
    "labels.json",
    "example_input.json",
    "example_output.json",
    "manifest.json",
}


def validate_repo_id(repo_id: str) -> str:
    if not REPO_ID_PATTERN.fullmatch(repo_id):
        raise HubValidationError(f"ID de repositório Hugging Face inválido: {repo_id}")
    return repo_id


def validate_revision(revision: str) -> str:
    if not REVISION_PATTERN.fullmatch(revision) or ".." in revision:
        raise HubValidationError(f"Revisão Hugging Face inválida: {revision}")
    return revision


def _files_in(directory: Path) -> set[str]:
    return {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _validate_examples(model_dir: Path) -> list[dict[str, object]]:
    from skops.io import get_untrusted_types, load

    model_path = model_dir / "model.skops"
    untrusted = get_untrusted_types(file=model_path)
    if untrusted:
        raise HubValidationError(f"Tipos não confiáveis no modelo: {untrusted}")
    model = load(model_path, trusted=[])
    inputs = json.loads((model_dir / "example_input.json").read_text(encoding="utf-8"))[
        "inputs"
    ]
    expected = json.loads(
        (model_dir / "example_output.json").read_text(encoding="utf-8")
    )["predictions"]
    predicted = model.predict(inputs).tolist()
    probabilities = model.predict_proba(inputs)
    results = []
    for index, (text, intent, scores, documented) in enumerate(
        zip(inputs, predicted, probabilities, expected), start=1
    ):
        score = float(np.max(scores))
        if documented.get("input") != text or documented.get("intent") != intent:
            raise HubValidationError(
                f"Predição publicada divergente no exemplo {index}."
            )
        if not np.isclose(
            score, float(documented.get("model_score")), rtol=0, atol=1e-15
        ):
            raise HubValidationError(f"Score publicado divergente no exemplo {index}.")
        results.append(
            {
                "example": index,
                "intent": intent,
                "model_score": score,
                "score_notice": "Relative model score; not a calibrated probability.",
            }
        )
    return results


def verify_downloaded_packages(dataset_dir: Path, model_dir: Path) -> dict[str, object]:
    if _files_in(dataset_dir) != DATASET_FILES:
        raise HubValidationError("Arquivos públicos inesperados no pacote do dataset.")
    if _files_in(model_dir) != MODEL_FILES:
        raise HubValidationError("Arquivos públicos inesperados no pacote do modelo.")
    validate_dataset_package_files(dataset_dir)
    dataset_summary = validate_exported_dataset(dataset_dir)
    model_summary = validate_model_package(model_dir)
    dataset_manifest = validate_manifest(dataset_dir)
    model_manifest = validate_manifest(model_dir)
    if (
        dataset_manifest["dataset_content_sha256"]
        != model_manifest["dataset_content_sha256"]
    ):
        raise HubValidationError(
            "Dataset e modelo públicos referenciam dados diferentes."
        )
    if dataset_manifest.get("license") != DATASET_LICENSE_ID:
        raise HubValidationError("Licença pública do dataset divergente.")
    if model_manifest.get("license") != MODEL_LICENSE_ID:
        raise HubValidationError("Licença pública do modelo divergente.")
    if parse_card_yaml(dataset_dir / "README.md").get("license") != DATASET_LICENSE_ID:
        raise HubValidationError("Dataset card público possui licença divergente.")
    if parse_card_yaml(model_dir / "README.md").get("license") != MODEL_LICENSE_ID:
        raise HubValidationError("Model card público possui licença divergente.")
    validate_package_texts((dataset_dir, model_dir))
    predictions = _validate_examples(model_dir)
    return {
        "dataset": dataset_summary,
        "model": model_summary,
        "dataset_content_sha256": dataset_manifest["dataset_content_sha256"],
        "model_sha256": model_manifest["model_sha256"],
        "predictions": predictions,
    }


def _resolve_public_revision(
    repo_id: str, repo_type: str, revision: str, timeout: float
) -> tuple[str, list[str]]:
    from huggingface_hub import HfApi

    info = HfApi(token=False).repo_info(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        timeout=timeout,
        files_metadata=True,
        token=False,
    )
    if info.private is not False:
        raise HubValidationError(f"Repositório não confirmado como público: {repo_id}")
    return info.sha, sorted(item.rfilename for item in info.siblings)


def _download(
    repo_id: str,
    repo_type: str,
    revision: str,
    files: set[str],
    offline: bool,
    timeout: float,
) -> Path:
    from huggingface_hub import constants, snapshot_download

    constants.HF_HUB_DOWNLOAD_TIMEOUT = timeout
    constants.HF_HUB_ETAG_TIMEOUT = timeout
    path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        cache_dir=PUBLIC_CACHE,
        allow_patterns=sorted(files),
        token=False,
        local_files_only=offline,
        etag_timeout=timeout,
        max_workers=4,
    )
    return Path(path)


def verify_publication(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    model_repo: str = DEFAULT_MODEL_REPO,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    model_revision: str = DEFAULT_MODEL_REVISION,
    offline: bool = False,
    timeout: float = 15,
) -> dict[str, Any]:
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    dataset_repo = validate_repo_id(dataset_repo)
    model_repo = validate_repo_id(model_repo)
    dataset_revision = validate_revision(dataset_revision)
    model_revision = validate_revision(model_revision)
    public_files: dict[str, list[str]] = {}
    if not offline:
        dataset_revision, public_files["dataset"] = _resolve_public_revision(
            dataset_repo, "dataset", dataset_revision, timeout
        )
        model_revision, public_files["model"] = _resolve_public_revision(
            model_repo, "model", model_revision, timeout
        )
    dataset_dir = _download(
        dataset_repo, "dataset", dataset_revision, DATASET_FILES, offline, timeout
    )
    model_dir = _download(
        model_repo, "model", model_revision, MODEL_FILES, offline, timeout
    )
    result = verify_downloaded_packages(dataset_dir, model_dir)
    return {
        "status": "PUBLICATION_VERIFIED",
        "authenticated": False,
        "offline": offline,
        "public_access_checked": not offline,
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "model_repo": model_repo,
        "model_revision": model_revision,
        "public_files": public_files,
        **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--timeout", type=float, default=15)
    args = parser.parse_args()
    if args.timeout <= 0 or args.timeout > 60:
        parser.error("--timeout deve estar entre 0 e 60 segundos.")
    try:
        result = verify_publication(
            dataset_repo=args.dataset_repo,
            model_repo=args.model_repo,
            dataset_revision=args.dataset_revision,
            model_revision=args.model_revision,
            offline=args.offline,
            timeout=args.timeout,
        )
    except (HubValidationError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
