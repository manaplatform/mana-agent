"""Atomic, non-secret registry for explicitly enrolled servers."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

from mana_agent.config.settings import mana_home

from .models import ServerDefinition


class ServerRegistry:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (mana_home() / "servers" / "registry.json")
        self._lock = threading.RLock()

    def list(self) -> list[ServerDefinition]:
        with self._lock:
            return sorted(self._read().values(), key=lambda server: (server.name.lower(), server.server_id))

    def get(self, identity: str) -> ServerDefinition:
        with self._lock:
            servers = self._read()
            if identity in servers:
                return servers[identity]
            matches = [item for item in servers.values() if item.name == identity]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise LookupError(f"Server name {identity!r} is ambiguous; use its server_id.")
            raise LookupError(f"Server {identity!r} is not enrolled.")

    def add(self, server: ServerDefinition) -> ServerDefinition:
        with self._lock:
            servers = self._read()
            if server.server_id in servers:
                raise ValueError(f"Server ID {server.server_id!r} is already enrolled.")
            if any(item.name == server.name for item in servers.values()):
                raise ValueError(f"Server name {server.name!r} is already enrolled.")
            servers[server.server_id] = server
            self._write(servers)
            return server

    def update(self, server: ServerDefinition) -> ServerDefinition:
        with self._lock:
            servers = self._read()
            if server.server_id not in servers:
                raise LookupError(f"Server {server.server_id!r} is not enrolled.")
            servers[server.server_id] = server
            self._write(servers)
            return server

    def remove(self, identity: str) -> ServerDefinition:
        with self._lock:
            server = self.get(identity)
            servers = self._read()
            del servers[server.server_id]
            self._write(servers)
            return server

    def _read(self) -> dict[str, ServerDefinition]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Server registry has an unsupported schema; no fallback registry was loaded.")
        rows = payload.get("servers")
        if not isinstance(rows, list):
            raise ValueError("Server registry is invalid; no fallback registry was loaded.")
        return {item.server_id: item for item in (ServerDefinition.model_validate(row) for row in rows)}

    def _write(self, servers: dict[str, ServerDefinition]) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "servers": [server.model_dump(mode="json") for server in servers.values()],
        }
        fd, temporary = tempfile.mkstemp(prefix="registry-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
