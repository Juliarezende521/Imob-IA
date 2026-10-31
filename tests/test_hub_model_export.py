import hashlib
import socket
from pathlib import Path

import pytest
from skops.io import get_untrusted_types, load

from imob_ai.hub.prepare import prepare_packages


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.hub
def test_complete_export_is_offline_safe_and_deterministic(monkeypatch):
    def block_network(*args, **kwargs):
        raise AssertionError("A preparação tentou acessar a rede.")

    monkeypatch.setattr(socket, "create_connection", block_network)
    monkeypatch.setattr(socket.socket, "connect", block_network)
    output = Path("build/huggingface/pytest-complete").resolve()
    first = prepare_packages(output)
    first_hashes = _tree_hashes(output)
    second = prepare_packages(output)

    assert first == second
    assert first_hashes == _tree_hashes(output)
    model_path = output / "intent-model" / "model.skops"
    assert get_untrusted_types(file=model_path) == []
    model = load(model_path, trusted=[])
    assert len(model.named_steps["classifier"].classes_) == 8
    assert model_path.stat().st_size < 5 * 1024 * 1024
