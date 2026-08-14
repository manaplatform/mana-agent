from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mana_agent.config.settings import Settings
from mana_agent.config.user_config import (
    load_user_identity,
    resolve_local_user_id,
    save_user_config,
)
from mana_agent.gateway import AgentChatGateway
from mana_agent.gateway.config import ChatGatewayConfig
from mana_agent.gateway.entry_routing import EntryRouteContext, EntryRoutingDecision
from mana_agent.memory import (
    CapsuleScope,
    CapsuleService,
    CapsuleTaskContext,
    MemoryPrincipal,
)


def test_gateway_config_preserves_the_authenticated_memory_user() -> None:
    config = ChatGatewayConfig(
        session_id="conversation-1",
        memory_user_id=" local-user ",
    ).normalized()

    assert config.session_id == "conversation-1"
    assert config.memory_user_id == "local-user"


def test_lane_token_budget_zero_means_unlimited() -> None:
    """Product policy: 0 is unlimited; only positive values cap the lane budget."""
    unlimited = ChatGatewayConfig(
        lane_session_token_budget=0,
        lane_global_token_budget=0,
    ).normalized()
    assert unlimited.lane_session_token_budget is None
    assert unlimited.lane_global_token_budget is None

    capped = ChatGatewayConfig(
        lane_session_token_budget=120_000,
        lane_global_token_budget=500_000,
    ).normalized()
    assert capped.lane_session_token_budget == 120_000
    assert capped.lane_global_token_budget == 500_000


def test_resolve_local_user_id_stability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Local authentication resolves to a persistent, stable application user identity."""
    home = tmp_path / "home"
    monkeypatch.setenv("MANA_HOME", str(home))

    # 1. First call generates a stable user identity in identity.json
    user_id_1 = resolve_local_user_id()
    assert user_id_1.startswith("user_")

    identity_file = home / "identity.json"
    assert identity_file.is_file()
    saved = load_user_identity()
    assert saved.get("user_id") == user_id_1

    # 2. Subsequent calls in the same MANA_HOME return the exact same user identity
    user_id_2 = resolve_local_user_id()
    assert user_id_2 == user_id_1

    # 3. Explicit MANA_USER_ID in config.toml takes precedence
    save_user_config({"MANA_USER_ID": "user-explicit-configured"}, merge=False)
    user_id_configured = resolve_local_user_id()
    assert user_id_configured == "user-explicit-configured"

    # 4. Settings mana_user_id takes precedence over config.toml
    settings = Settings(mana_user_id="user-from-settings")
    user_id_settings = resolve_local_user_id(settings)
    assert user_id_settings == "user-from-settings"


def test_terminal_and_dashboard_gateway_flows_authenticate_local_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Terminal and Dashboard gateways bind the canonical local application user identity."""
    home = tmp_path / "home"
    monkeypatch.setenv("MANA_HOME", str(home))
    settings = Settings(mana_user_id="user-mana-app-test")

    # Terminal CLI flow
    cli_user = resolve_local_user_id(settings)
    terminal_config = ChatGatewayConfig(
        session_id="term-session-1",
        memory_user_id=cli_user,
    ).normalized()
    assert terminal_config.memory_user_id == "user-mana-app-test"

    # Dashboard helper flow
    dash_user = resolve_local_user_id(settings)
    dashboard_config = ChatGatewayConfig(
        session_id="dash-session-1",
        memory_user_id=dash_user,
    ).normalized()
    assert dashboard_config.memory_user_id == "user-mana-app-test"


def test_memory_route_authenticated_local_user_private_capsule_read_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Authenticated local user reading their own private capsule succeeds."""
    home = tmp_path / "home"
    monkeypatch.setenv("MANA_HOME", str(home))
    user_id = "user-authenticated-local"
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)

    capsule_service = CapsuleService(repo_root)

    # Pre-seed a private capsule owned by user_id
    writer_principal = MemoryPrincipal(
        user_id=user_id,
        project_id="repo-test",
        task_id="task-seed",
        agent_id="gateway:chat",
        capabilities=frozenset({"memory.capsule.write.private", "memory.capsule.read.private"}),
    )
    writer_context = CapsuleTaskContext(
        user_id=user_id,
        organisation_id=None,
        project_id="repo-test",
        team_ids=frozenset(),
        task_id="task-seed",
        agent_id="gateway:chat",
        session_id="session-1",
    )
    seed_capsule = capsule_service.create_capsule(
        principal=writer_principal,
        context=writer_context,
        scope=CapsuleScope.PRIVATE,
        title="who am I user identity fact",
        summary="User is Alice, lead engineer",
        content={"name": "Alice", "role": "Lead Engineer"},
        origin_type="test",
        origin_id="msg-1",
    )
    assert seed_capsule.owner_user_id == user_id

    # Construct mock gateway with memory service
    memory_mock = SimpleNamespace(
        user_id=user_id,
        config=SimpleNamespace(capsules=SimpleNamespace(enabled=True)),
        capsules=capsule_service,
    )
    gateway = SimpleNamespace(
        _stack=SimpleNamespace(memory_service=memory_mock, repository_id="repo-test"),
        config=ChatGatewayConfig(memory_user_id=user_id),
        settings=Settings(mana_memory_capsules_default_max_tokens=4000),
    )

    # Bind _execute_memory_route from AgentChatGateway
    result = AgentChatGateway._execute_memory_route(
        gateway,  # type: ignore[arg-type]
        decision=EntryRoutingDecision(
            route="memory",
            confidence=0.99,
            reason="Read user identity memory",
            required_sources=("memory",),
            memory_task_id="task-seed",
        ),
        context=EntryRouteContext(
            session_id="session-1",
            conversation_id="session-1",
            turn_id="turn-1",
            memory_capsules_enabled=True,
            memory_task_candidates=(
                {"task_id": "task-seed", "normalized_intent": "user identity", "state": "completed"},
            ),
            authenticated_user_id=user_id,
        ),
        query="who am I user",
    )

    assert result.error is None
    assert result.mode == "route-memory"
    assert result.payload["memory_record_count"] == 1
    assert "User is Alice, lead engineer" in result.answer


def test_memory_route_missing_identity_fails_with_zero_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing authenticated user identity rejects with zero private reads."""
    home = tmp_path / "home"
    monkeypatch.setenv("MANA_HOME", str(home))

    read_calls: list[Any] = []
    capsules_mock = SimpleNamespace(
        query_capsules=lambda req, correlation_id: read_calls.append(req) or []
    )
    memory_mock = SimpleNamespace(
        user_id="",
        config=SimpleNamespace(capsules=SimpleNamespace(enabled=True)),
        capsules=capsules_mock,
    )
    gateway = SimpleNamespace(
        _stack=SimpleNamespace(memory_service=memory_mock, repository_id="repo-test"),
        config=ChatGatewayConfig(memory_user_id=""),
        settings=Settings(mana_memory_capsules_default_max_tokens=4000),
    )

    result = AgentChatGateway._execute_memory_route(
        gateway,  # type: ignore[arg-type]
        decision=EntryRoutingDecision(
            route="memory",
            confidence=0.99,
            reason="Read memory",
            required_sources=("memory",),
            memory_task_id="task-offered",
        ),
        context=EntryRouteContext(
            session_id="session-1",
            conversation_id="session-1",
            turn_id="turn-1",
            memory_capsules_enabled=True,
            memory_task_candidates=(
                {"task_id": "task-offered", "normalized_intent": "lookup", "state": "completed"},
            ),
            authenticated_user_id="",
        ),
        query="who am I",
    )

    assert result.error == "memory_principal_unavailable"
    assert "Private memory retrieval requires an authenticated user identity" in result.answer
    assert len(read_calls) == 0


def test_memory_route_different_user_cannot_access_private_capsule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different user identity cannot access another user's private memory capsule."""
    home = tmp_path / "home"
    monkeypatch.setenv("MANA_HOME", str(home))
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)

    capsule_service = CapsuleService(repo_root)

    # Capsule is owned by Alice
    alice_principal = MemoryPrincipal(
        user_id="user-alice",
        project_id="repo-test",
        task_id="task-alice-seed",
        agent_id="gateway:chat",
        capabilities=frozenset({"memory.capsule.write.private", "memory.capsule.read.private"}),
    )
    alice_context = CapsuleTaskContext(
        user_id="user-alice",
        organisation_id=None,
        project_id="repo-test",
        team_ids=frozenset(),
        task_id="task-alice-seed",
        agent_id="gateway:chat",
        session_id="session-alice",
    )
    capsule_service.create_capsule(
        principal=alice_principal,
        context=alice_context,
        scope=CapsuleScope.PRIVATE,
        title="Alice Secret Note",
        summary="Alice private confidential key",
        content={"secret": "alice-classified-data"},
        origin_type="test",
        origin_id="msg-alice-1",
    )

    # Bob queries the task
    memory_mock = SimpleNamespace(
        user_id="user-bob",
        config=SimpleNamespace(capsules=SimpleNamespace(enabled=True)),
        capsules=capsule_service,
    )
    gateway = SimpleNamespace(
        _stack=SimpleNamespace(memory_service=memory_mock, repository_id="repo-test"),
        config=ChatGatewayConfig(memory_user_id="user-bob"),
        settings=Settings(mana_memory_capsules_default_max_tokens=4000),
    )

    result = AgentChatGateway._execute_memory_route(
        gateway,  # type: ignore[arg-type]
        decision=EntryRoutingDecision(
            route="memory",
            confidence=0.99,
            reason="Read memory as bob",
            required_sources=("memory",),
            memory_task_id="task-alice-seed",
        ),
        context=EntryRouteContext(
            session_id="session-bob",
            conversation_id="session-bob",
            turn_id="turn-bob-1",
            memory_capsules_enabled=True,
            memory_task_candidates=(
                {"task_id": "task-alice-seed", "normalized_intent": "confidential lookup", "state": "completed"},
            ),
            authenticated_user_id="user-bob",
        ),
        query="Alice private confidential key",
    )

    # Denied by ACL: 0 records returned for Bob
    assert result.error is None
    assert result.payload["memory_record_count"] == 0
    assert "No authorized private memory matched" in result.answer
    assert "alice-classified-data" not in result.answer
