"""Persistent non-secret remote A2A agent registry."""

from __future__ import annotations

import hashlib
from urllib.parse import urlsplit

from mana_agent.protocols.common.lifecycle import DurableProtocolStore

from .types import RemoteAgentRecord


class RemoteAgentRegistry:
    def __init__(self) -> None:
        self.store = DurableProtocolStore("a2a", "agents.json")

    def list(self) -> list[RemoteAgentRecord]:
        return sorted((RemoteAgentRecord(**item) for item in self.store.load().get("agents", []) if isinstance(item, dict)), key=lambda item: item.name.casefold())

    def get(self, name_or_id: str) -> RemoteAgentRecord:
        target = str(name_or_id).casefold()
        for item in self.list():
            if item.agent_id.casefold() == target or item.name.casefold() == target:
                return item
        raise KeyError("Remote A2A agent is not registered.")

    def add(self, *, name: str, card_url: str, allowed_skills: list[str] | None = None, trusted: bool = False) -> RemoteAgentRecord:
        parsed = urlsplit(str(card_url or ""))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Agent Card URL must be HTTP(S) without embedded credentials.")
        if any(item.name.casefold() == name.casefold() for item in self.list()):
            raise ValueError("Remote A2A agent name already exists.")
        record = RemoteAgentRecord(
            agent_id="remote_" + hashlib.sha256(card_url.encode("utf-8")).hexdigest()[:16],
            name=name,
            card_url=card_url,
            allowed_skills=list(allowed_skills or []),
            trusted=trusted,
        )
        rows = self.list() + [record]
        self.store.save({"agents": [item.to_dict() for item in rows]})
        return record

    def update(self, record: RemoteAgentRecord) -> None:
        rows = [record if item.agent_id == record.agent_id else item for item in self.list()]
        self.store.save({"agents": [item.to_dict() for item in rows]})

    def remove(self, name_or_id: str) -> None:
        target = self.get(name_or_id)
        self.store.save({"agents": [item.to_dict() for item in self.list() if item.agent_id != target.agent_id]})
