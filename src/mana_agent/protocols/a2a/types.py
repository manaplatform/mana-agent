"""Internal A2A policy records; wire types are supplied by the A2A SDK."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass(slots=True)
class RemoteAgentRecord:
    agent_id: str
    name: str
    card_url: str
    allowed_skills: list[str] = field(default_factory=list)
    auth_reference: str = ""
    trusted: bool = False
    allowed_workspaces: list[str] = field(default_factory=list)
    timeout_seconds: int = 30
    max_concurrent_tasks: int = 1
    last_discovered_at: str = ""
    card_expires_at: str = ""
    cached_card: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DelegationEnvelope:
    origin_agent_id: str
    correlation_id: str
    task_fingerprint: str
    delegation_chain: tuple[str, ...] = ()
    visited_agents: frozenset[str] = frozenset()
    hop_count: int = 0
    approved_context: str = ""
    selected_skill: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
