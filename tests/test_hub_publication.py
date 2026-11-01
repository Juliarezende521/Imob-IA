from pathlib import Path

import pytest

from imob_ai.hub.prepare import prepare_packages
from imob_ai.hub.validation import HubValidationError
from imob_ai.hub.verify_publication import (
    validate_repo_id,
    validate_revision,
    verify_downloaded_packages,
)


@pytest.mark.hub
def test_downloaded_package_validation_without_network():
    root = Path("build/huggingface/pytest-publication").resolve()
    prepare_packages(root)
    result = verify_downloaded_packages(root / "intent-dataset", root / "intent-model")

    assert result["dataset"]["records"] == 384
    assert result["dataset"]["by_split"] == {
        "train": 256,
        "validation": 64,
        "test": 64,
    }
    assert result["model"]["trusted_types"] == []
    assert [item["intent"] for item in result["predictions"]] == [
        "BUSCA_IMOVEL",
        "ATENDIMENTO_HUMANO",
        "MANUTENCAO_IMOVEL",
    ]


@pytest.mark.hub
@pytest.mark.parametrize(
    "repo_id", ["https://huggingface.co/user/repo", "user", "../user/repo"]
)
def test_rejects_arbitrary_or_invalid_repository_ids(repo_id):
    with pytest.raises(HubValidationError):
        validate_repo_id(repo_id)


@pytest.mark.hub
@pytest.mark.parametrize("revision", ["../main", "https://example.com", "main?x=1"])
def test_rejects_unsafe_revisions(revision):
    with pytest.raises(HubValidationError):
        validate_revision(revision)
