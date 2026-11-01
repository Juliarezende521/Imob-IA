from pathlib import Path

import pytest

from imob_ai.hub.metadata import (
    DATASET_LICENSE_ID,
    LICENSE_STATUS,
    MODEL_LICENSE_ID,
    PUBLICATION_STATUS,
)
from imob_ai.hub.prepare import prepare_packages
from imob_ai.hub.validation import (
    HubValidationError,
    parse_card_yaml,
    validate_output_path,
)


@pytest.mark.hub
def test_output_path_cannot_escape_build_directory():
    with pytest.raises(HubValidationError):
        validate_output_path(Path("/tmp/imob-ai-hub"))


@pytest.mark.hub
def test_generated_cards_and_manifests_use_declared_licenses():
    root = Path("build/huggingface/pytest-cards").resolve()
    prepare_packages(root)
    expected = {
        "intent-dataset": DATASET_LICENSE_ID,
        "intent-model": MODEL_LICENSE_ID,
    }
    for package, license_id in expected.items():
        metadata = parse_card_yaml(root / package / "README.md")
        assert metadata["license"] == license_id
        assert LICENSE_STATUS in (root / package / "README.md").read_text(
            encoding="utf-8"
        )
        manifest = __import__("json").loads(
            (root / package / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["license"] == license_id
        assert manifest["status"] == PUBLICATION_STATUS
        assert (root / package / "LICENSE").is_file()
