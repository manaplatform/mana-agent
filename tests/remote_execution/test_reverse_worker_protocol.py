from __future__ import annotations

import os
from pathlib import Path

import pytest

from mana_agent.remote_execution.credentials import CredentialStore, generate_identity
from mana_agent.remote_execution.installer import LABEL, launchagent_payload
from mana_agent.remote_execution.protocol import WorkerMessage


def test_protocol_rejects_unknown_version_and_oversized_frame() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        WorkerMessage.parse_frame('{"protocol_version":2,"type":"worker.hello"}')
    with pytest.raises(ValueError, match="size"):
        WorkerMessage.parse_frame(b"x" * 1_048_577)


def test_identity_fallback_is_owner_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "keyring", None)
    store = CredentialStore(tmp_path)
    identity = generate_identity("worker-1", "opaque-credential")
    store.save(identity)
    assert store.load("worker-1") == identity
    if os.name == "posix":
        assert store.path.stat().st_mode & 0o077 == 0


def test_launchagent_contains_no_credentials(tmp_path: Path) -> None:
    payload = launchagent_payload(executable="/usr/local/bin/mana-agent", state_dir=tmp_path, log_dir=tmp_path)
    rendered = str(payload)
    assert payload["Label"] == LABEL
    assert "token" not in rendered.lower()
    assert "credential" not in rendered.lower()
