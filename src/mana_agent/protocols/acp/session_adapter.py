"""One-to-one durable ACP-to-Mana session adaptation."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.protocols.common.lifecycle import DurableProtocolStore
from mana_agent.protocols.common.security import ProtocolSecurityPolicy, redact_protocol_value

from .types import AcpSessionState


class AcpSessionAdapter:
    def __init__(self, gateway: AgentChatGateway, *, allowed_roots: tuple[Path, ...] = ()) -> None:
        self.gateway = gateway
        self.policy = ProtocolSecurityPolicy.for_workspace(gateway.root, allowed_roots)
        self.store = DurableProtocolStore("acp", "sessions.json")
        self.sessions: dict[str, AcpSessionState] = {}
        self._load()

    def _load(self) -> None:
        for acp_id, raw in dict(self.store.load().get("sessions") or {}).items():
            if not isinstance(raw, dict):
                continue
            self.sessions[acp_id] = AcpSessionState(
                acp_session_id=acp_id,
                mana_session_id=str(raw.get("mana_session_id") or acp_id),
                cwd=str(raw.get("cwd") or self.gateway.root),
                additional_directories=tuple(raw.get("additional_directories") or ()),
                mode=str(raw.get("mode") or "ask"),
                read_only=bool(raw.get("read_only", False)),
                closed=bool(raw.get("closed", False)),
            )

    def _save(self) -> None:
        self.store.save(
            {
                "sessions": {
                    key: {
                        "mana_session_id": value.mana_session_id,
                        "cwd": value.cwd,
                        "additional_directories": list(value.additional_directories),
                        "mode": value.mode,
                        "read_only": value.read_only,
                        "closed": value.closed,
                    }
                    for key, value in self.sessions.items()
                }
            }
        )

    def create(
        self,
        *,
        cwd: str,
        additional_directories: list[str] | None,
        mcp_servers: list[Any] | None,
    ) -> AcpSessionState:
        resolved_cwd, roots = self._validate_roots(cwd, additional_directories)
        acp_id = f"acp_{uuid.uuid4().hex[:20]}"
        mana_id = self.gateway.create_new_session(frontend="acp")
        state = AcpSessionState(
            acp_session_id=acp_id,
            mana_session_id=mana_id,
            cwd=str(resolved_cwd),
            additional_directories=tuple(str(item) for item in roots),
            mcp_overrides=self._mcp_overrides(mcp_servers or []),
        )
        self.sessions[acp_id] = state
        self._save()
        return state

    def load_session(
        self,
        session_id: str,
        *,
        cwd: str,
        additional_directories: list[str] | None,
        mcp_servers: list[Any] | None,
    ) -> AcpSessionState:
        state = self.get(session_id)
        resolved_cwd, roots = self._validate_roots(cwd, additional_directories)
        record = self.gateway._workspaces.store.get_session(state.mana_session_id)  # noqa: SLF001
        if record.status != "active":
            record.status = "active"
            record.closed_at = None
            self.gateway._workspaces.store.save_session(record)  # noqa: SLF001
        self.gateway.create_session(frontend="acp", session_id=state.mana_session_id)
        state.cwd = str(resolved_cwd)
        state.additional_directories = tuple(str(item) for item in roots)
        state.mcp_overrides = self._mcp_overrides(mcp_servers or [])
        state.closed = False
        self._save()
        return state

    def get(self, session_id: str) -> AcpSessionState:
        try:
            state = self.sessions[str(session_id)]
        except KeyError as exc:
            raise ValueError("Unknown ACP session.") from exc
        return state

    def close(self, session_id: str) -> None:
        state = self.get(session_id)
        if not state.closed:
            self.gateway.close_session(state.mana_session_id)
            state.closed = True
            self._save()

    def list_states(self, *, cwd: str | None = None) -> list[AcpSessionState]:
        resolved = self.policy.validate_path(cwd) if cwd else None
        return sorted(
            (
                item
                for item in self.sessions.values()
                if resolved is None or Path(item.cwd) == resolved
            ),
            key=lambda item: item.acp_session_id,
        )

    def history(self, session_id: str) -> list[dict[str, Any]]:
        state = self.get(session_id)
        return self.gateway.session_messages(state.mana_session_id)

    def _validate_roots(self, cwd: str, additional: list[str] | None) -> tuple[Path, list[Path]]:
        resolved_cwd = self.policy.validate_path(cwd)
        roots = [self.policy.validate_path(item) for item in (additional or [])]
        return resolved_cwd, roots

    @staticmethod
    def _mcp_overrides(servers: list[Any]) -> list[str]:
        overrides: list[str] = []
        for index, server in enumerate(servers):
            raw = server.model_dump(mode="json", by_alias=True) if hasattr(server, "model_dump") else dict(server)
            kind = str(raw.get("type") or "").lower()
            identifier = str(raw.get("name") or raw.get("id") or f"acp-{index + 1}")
            if "command" in raw:
                env_rows = raw.get("env") or []
                env = {
                    str(row.get("name")): str(row.get("value"))
                    for row in env_rows
                    if isinstance(row, dict) and row.get("name")
                }
                normalized = {
                    "id": identifier,
                    "transport": "stdio",
                    "command": raw.get("command"),
                    "args": raw.get("args") or [],
                    "env": env,
                }
            else:
                normalized = {
                    "id": identifier,
                    "transport": "sse" if kind == "sse" else "streamable_http",
                    "url": raw.get("url"),
                    "headers": raw.get("headers") or {},
                }
            # Secrets remain connection-local and are deliberately not persisted.
            overrides.append(json.dumps(normalized))
        _ = redact_protocol_value(overrides)
        return overrides
