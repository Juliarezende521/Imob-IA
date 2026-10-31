"""Prepara localmente pacotes revisáveis; nunca autentica ou publica."""

import argparse
import json
import shutil
from pathlib import Path

from imob_ai.hub.dataset_export import export_dataset
from imob_ai.hub.metadata import DEFAULT_OUTPUT, PROJECT_ROOT
from imob_ai.hub.model_export import export_model
from imob_ai.hub.validation import (
    HubValidationError,
    validate_all_packages,
    validate_output_path,
    validate_source_hashes,
)


def _resolve_output(value: Path) -> Path:
    candidate = value if value.is_absolute() else PROJECT_ROOT / value
    return validate_output_path(candidate)


def prepare_packages(output_dir: Path) -> dict[str, object]:
    output_dir = validate_output_path(output_dir)
    validate_source_hashes()
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        dataset_manifest = export_dataset(staging / "intent-dataset")
        export_model(
            staging / "intent-model", dataset_manifest["dataset_content_sha256"]
        )
        validate_all_packages(staging)
        for package_name in ("intent-dataset", "intent-model"):
            destination = output_dir / package_name
            if destination.exists():
                shutil.rmtree(destination)
            (staging / package_name).replace(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return validate_all_packages(output_dir)


def validate_existing(output_dir: Path) -> dict[str, object]:
    output_dir = validate_output_path(output_dir)
    validate_source_hashes()
    return validate_all_packages(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    output = _resolve_output(args.output)
    try:
        summary = (
            validate_existing(output)
            if args.validate_only
            else prepare_packages(output)
        )
    except HubValidationError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "output": str(output.relative_to(PROJECT_ROOT)),
                "status": "prepared_not_published",
                "network_used": False,
                "license_status": "LICENSE_REVIEW_REQUIRED",
                **summary,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
