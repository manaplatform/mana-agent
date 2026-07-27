from __future__ import annotations

from pathlib import Path

import pytest

from mana_agent.protocols.common.auth import require_bearer_token
from mana_agent.protocols.common.config import validate_protocol_configuration
from mana_agent.protocols.common.exceptions import ProtocolAuthenticationError, ProtocolPolicyError
from mana_agent.protocols.common.security import ProtocolSecurityPolicy


def test_protocol_paths_are_absolute_and_workspace_scoped(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    policy = ProtocolSecurityPolicy.for_workspace(root)

    assert policy.validate_path(root / "src") == (root / "src").resolve()
    with pytest.raises(ProtocolPolicyError, match="absolute"):
        policy.validate_path("src")
    with pytest.raises(ProtocolPolicyError, match="outside"):
        policy.validate_path(tmp_path / "other")


def test_bearer_authentication_is_fail_closed() -> None:
    identity = require_bearer_token("Bearer correct", "correct")
    assert identity.authenticated is True
    assert "correct" not in identity.caller_id
    with pytest.raises(ProtocolAuthenticationError):
        require_bearer_token(None, "correct")
    with pytest.raises(ProtocolAuthenticationError):
        require_bearer_token("Bearer wrong", "correct")
    with pytest.raises(ProtocolAuthenticationError, match="not configured"):
        require_bearer_token("Bearer any", "")


def test_protocol_configuration_rejects_public_http_and_unimplemented_push() -> None:
    baseline = {
        "MANA_A2A_SERVER_ENABLED": True,
        "MANA_A2A_SERVER_TOKEN": "secret",
        "MANA_A2A_HOST": "0.0.0.0",
        "MANA_A2A_PUBLIC_BASE_URL": "http://agent.example",
        "MANA_A2A_PORT": 8766,
        "MANA_A2A_TASK_RETENTION_DAYS": 30,
        "MANA_A2A_MAX_REQUEST_BYTES": 1024,
        "MANA_A2A_MAX_ARTIFACT_BYTES": 2048,
        "MANA_A2A_MAX_CONCURRENT_TASKS": 1,
        "MANA_A2A_MAX_DELEGATION_DEPTH": 3,
        "MANA_ACP_SESSION_RETENTION_DAYS": 30,
    }
    with pytest.raises(ValueError, match="HTTPS"):
        validate_protocol_configuration(baseline)
    baseline["MANA_A2A_PUBLIC_BASE_URL"] = "https://agent.example"
    baseline["MANA_A2A_PUSH_NOTIFICATIONS"] = True
    with pytest.raises(ValueError, match="not implemented"):
        validate_protocol_configuration(baseline)
