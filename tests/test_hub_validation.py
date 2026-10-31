from pathlib import Path

import pytest

from imob_ai.hub.metadata import LICENSE_STATUS
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
def test_generated_cards_keep_license_review_blocking_publication():
    root = Path("build/huggingface/pytest-cards").resolve()
    prepare_packages(root)
    for package in ("intent-dataset", "intent-model"):
        metadata = parse_card_yaml(root / package / "README.md")
        assert metadata["license"] == "other"
        assert LICENSE_STATUS in (root / package / "README.md").read_text(
            encoding="utf-8"
        )
