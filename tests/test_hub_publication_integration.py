import pytest

from imob_ai.hub.verify_publication import (
    DEFAULT_DATASET_REVISION,
    DEFAULT_MODEL_REVISION,
    verify_publication,
)


@pytest.mark.hub_publication_integration
def test_public_hugging_face_artifacts_without_authentication():
    result = verify_publication()

    assert result["status"] == "PUBLICATION_VERIFIED"
    assert result["authenticated"] is False
    assert result["public_access_checked"] is True
    assert result["dataset_revision"] == DEFAULT_DATASET_REVISION
    assert result["model_revision"] == DEFAULT_MODEL_REVISION
    assert result["dataset"]["records"] == 384
    assert result["model"]["trusted_types"] == []
