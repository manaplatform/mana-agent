"""Internal backend preserving Mana-Agent's local-storage default."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from mana_agent.memory.errors import MemoryNotFoundError, MemoryStorageError
from mana_agent.memory.models import (
    MemoryContent,
    MemoryHealth,
    MemoryHealthStatus,
    MemoryRecord,
    MemoryScope,
    MemorySearchRequest,
    MemoryUpdateRequest,
    MemoryWriteRequest,
    utc_now,
)
from mana_agent.memory.providers.internal.repository import InternalMemoryRepository
from mana_agent.workspaces.paths import repository_dir, repository_id_for_path


def _scope_matches(candidate: dict[str, str], requested: MemoryScope) -> bool:
    return all(candidate.get(key) == value for key, value in requested.as_dict().items())


class InternalMemoryBackend:
    def __init__(self, root: str | Path) -> None:
        resolved = Path(root).resolve()
        self.root = resolved
        self.storage_dir = repository_dir(repository_id_for_path(resolved))
        self.repository = InternalMemoryRepository(
            self.storage_dir / "provider_memory.json"
        )
        self.closed = False

    @staticmethod
    def _record(row: dict[str, Any], *, score: float | None = None) -> MemoryRecord:
        def parsed(value: Any) -> datetime | None:
            return datetime.fromisoformat(value) if value else None
        return MemoryRecord(
            id=str(row["id"]),
            content=MemoryContent(str(row["content"])),
            scope=MemoryScope(**dict(row.get("scope", {}))),
            metadata=dict(row.get("metadata", {})),
            score=score,
            provider="mana",
            created_at=parsed(row.get("created_at")),
            updated_at=parsed(row.get("updated_at")),
        )

    async def add(self, request: MemoryWriteRequest) -> MemoryRecord:
        try:
            rows = self.repository.load()
            now = utc_now().isoformat()
            row = {
                "id": uuid.uuid4().hex,
                "content": request.content.text,
                "scope": request.scope.as_dict(),
                "metadata": dict(request.metadata),
                "created_at": now,
                "updated_at": now,
            }
            rows.append(row)
            self.repository.save(rows)
            return self._record(row)
        except (OSError, ValueError) as exc:
            raise MemoryStorageError("Internal memory write failed.") from exc

    async def search(self, request: MemorySearchRequest) -> list[MemoryRecord]:
        terms = set(request.query.lower().split())
        rows = [
            row
            for row in self.repository.load()
            if _scope_matches(dict(row.get("scope", {})), request.scope)
        ]
        ranked = sorted(
            (
                (
                    len(terms & set(str(row.get("content", "")).lower().split()))
                    / max(1, len(terms)),
                    row,
                )
                for row in rows
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            self._record(row, score=score)
            for score, row in ranked[: max(1, request.limit)]
            if score > 0 or not terms
        ]

    async def get(self, memory_id: str, scope: MemoryScope) -> MemoryRecord | None:
        return next(
            (
                self._record(row)
                for row in self.repository.load()
                if row.get("id") == memory_id
                and _scope_matches(dict(row.get("scope", {})), scope)
            ),
            None,
        )

    async def update(self, memory_id: str, request: MemoryUpdateRequest) -> MemoryRecord:
        rows = self.repository.load()
        for row in rows:
            if row.get("id") == memory_id and _scope_matches(dict(row.get("scope", {})), request.scope):
                if request.content is not None:
                    row["content"] = request.content.text
                if request.metadata is not None:
                    row["metadata"] = dict(request.metadata)
                row["updated_at"] = utc_now().isoformat()
                self.repository.save(rows)
                return self._record(row)
        raise MemoryNotFoundError(f"Memory {memory_id!r} was not found in the requested scope.")

    async def delete(self, memory_id: str, scope: MemoryScope) -> None:
        rows = self.repository.load()
        kept = [
            row
            for row in rows
            if not (
                row.get("id") == memory_id
                and _scope_matches(dict(row.get("scope", {})), scope)
            )
        ]
        if len(kept) == len(rows):
            raise MemoryNotFoundError(
                f"Memory {memory_id!r} was not found in the requested scope."
            )
        self.repository.save(kept)

    async def clear(self, scope: MemoryScope) -> None:
        self.repository.save(
            [
                row
                for row in self.repository.load()
                if not _scope_matches(dict(row.get("scope", {})), scope)
            ]
        )

    async def healthcheck(self) -> MemoryHealth:
        try:
            self.repository.load()
            return MemoryHealth(MemoryHealthStatus.HEALTHY, "internal", "mana", "Locally managed by Mana-Agent")
        except (OSError, ValueError) as exc:
            return MemoryHealth(MemoryHealthStatus.STORAGE_ERROR, "internal", "mana", str(exc))

    def project_snapshot(self, *, max_chars: int) -> str:
        """Read pre-architecture text memories without migrating or rewriting them."""
        for name in ("memory.md", "project_memory.md"):
            path = self.storage_dir / name
            if path.is_file():
                return path.read_text(encoding="utf-8")[:max_chars]
        return ""

    async def close(self) -> None:
        self.closed = True
