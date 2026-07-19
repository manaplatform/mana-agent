"""Centralized Mana scope and Mem0 response mapping."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from mana_agent.memory.models import MemoryContent, MemoryRecord, MemoryScope

SCOPE_METADATA_KEYS = {
    "repository_id": "mana_repository_id",
    "conversation_id": "mana_conversation_id",
    "task_id": "mana_task_id",
}


def scope_to_mem0(scope: MemoryScope) -> tuple[dict[str, str], dict[str, str]]:
    """Map identities without combining distinct scope dimensions."""
    entities = {
        key: value
        for key, value in {
            "user_id": scope.user_id,
            "agent_id": scope.agent_id,
            "run_id": scope.session_id,
            "app_id": scope.workspace_id,
        }.items()
        if value
    }
    metadata = {
        provider_key: getattr(scope, scope_key)
        for scope_key, provider_key in SCOPE_METADATA_KEYS.items()
        if getattr(scope, scope_key)
    }
    return entities, metadata


def scope_to_filters(
    scope: MemoryScope,
    filter_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entities, scope_metadata = scope_to_mem0(scope)
    clauses = [{key: value} for key, value in entities.items()]
    combined_metadata = {**scope_metadata, **dict(filter_metadata or {})}
    if combined_metadata:
        clauses.append({"metadata": combined_metadata})
    if not clauses:
        return {}
    return clauses[0] if len(clauses) == 1 else {"AND": clauses}


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def response_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("results", "memories", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            nested = response_rows(value)
            if nested:
                return nested
    return [payload] if any(key in payload for key in ("id", "memory_id", "memory")) else []


def response_to_record(row: dict[str, Any], scope: MemoryScope) -> MemoryRecord:
    metadata = dict(row.get("metadata") or {})
    provider_metadata = {
        key: value
        for key, value in row.items()
        if key not in {"id", "memory_id", "memory", "text", "metadata", "score", "created_at", "updated_at"}
    }
    return MemoryRecord(
        id=str(row.get("id") or row.get("memory_id") or ""),
        content=MemoryContent(str(row.get("memory") or row.get("text") or "")),
        scope=scope,
        metadata={key: value for key, value in metadata.items() if key not in set(SCOPE_METADATA_KEYS.values())},
        score=float(row["score"]) if row.get("score") is not None else None,
        provider="mem0",
        provider_metadata=provider_metadata,
        created_at=_parse_time(row.get("created_at")),
        updated_at=_parse_time(row.get("updated_at")),
    )
