from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("a2a")

from a2a.auth.user import User
from a2a.server.context import ServerCallContext
from a2a.types.a2a_pb2 import Task, TaskState, TaskStatus
from google.protobuf.json_format import MessageToDict

from mana_agent.protocols.a2a.agent_card import build_agent_card
from mana_agent.protocols.a2a.delegation import DelegationPolicy
from mana_agent.protocols.a2a.task_adapter import ManaA2ATaskStore, map_internal_task_state
from mana_agent.protocols.a2a.types import DelegationEnvelope, RemoteAgentRecord
from mana_agent.protocols.common.exceptions import ProtocolPolicyError


class _User(User):
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def user_name(self) -> str:
        return self.name


def test_agent_card_advertises_only_implemented_public_capabilities() -> None:
    card = build_agent_card(public_base_url="https://agent.example", enabled_skills={"conversation", "verification"})
    body = MessageToDict(card)
    assert body["supportedInterfaces"][0]["protocolVersion"] == "1.0"
    assert body["capabilities"]["streaming"] is True
    assert body["capabilities"].get("pushNotifications", False) is False
    assert {item["id"] for item in body["skills"]} == {"conversation", "verification"}
    assert "bearer" in body["securitySchemes"]


def test_task_store_is_durable_and_caller_scoped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    store = ManaA2ATaskStore()
    owner = ServerCallContext(user=_User("owner"))
    other = ServerCallContext(user=_User("other"))
    task = Task(id="task-1", context_id="context-1", status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED))
    asyncio.run(store.save(task, owner))

    reconstructed = ManaA2ATaskStore()
    assert asyncio.run(reconstructed.get("task-1", owner)).id == "task-1"
    assert asyncio.run(reconstructed.get("task-1", other)) is None


def test_task_state_mapping_and_delegation_loop_protection() -> None:
    assert map_internal_task_state("done") == "completed"
    with pytest.raises(ValueError, match="Unknown"):
        map_internal_task_state("mystery")
    remote = RemoteAgentRecord(agent_id="remote-1", name="Remote", card_url="https://agent.example/card", allowed_skills=["review"])
    envelope = DelegationEnvelope(
        origin_agent_id="mana-agent",
        correlation_id="corr-1",
        task_fingerprint="fingerprint-1",
        delegation_chain=("remote-1",),
        visited_agents=frozenset({"remote-1"}),
        approved_context="Review this patch only.",
        selected_skill="review",
    )
    with pytest.raises(ProtocolPolicyError, match="loop"):
        DelegationPolicy(enabled=True).authorize(remote, envelope, workspace_id="workspace", authentication_available=True)
