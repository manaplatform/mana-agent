"""Shared provider helpers for scope mapping, metadata, and record normalization."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from mana_agent.memory.models import MemoryContent, MemoryRecord, MemoryScope
from mana_agent.services.memory_service import stable_hash

SCOPE_METADATA_KEYS = {
    "repository_id": "mana_repository_id",
    "conversation_id": "mana_conversation_id",
    "task_id": "mana_task_id",
}
SUPERMEMORY_TAG_COMPONENTS = (
    ("user_id", "user"),
    ("workspace_id", "workspace"),
    ("repository_id", "repository"),
    ("agent_id", "agent"),
    ("session_id", "session"),
)
_METADATA_SCALAR_TYPES = (str, int, float, bool)


def scope_metadata(scope: MemoryScope) -> dict[str, str]:
    return {
        provider_key: getattr(scope, scope_key)
        for scope_key, provider_key in SCOPE_METADATA_KEYS.items()
        if getattr(scope, scope_key)
    }


def parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def flat_scalar_metadata(metadata: dict[str, Any] | None) -> dict[str, str | int | float | bool]:
    flattened: dict[str, str | int | float | bool] = {}
    for key, value in dict(metadata or {}).items():
        if not str(key).strip() or value is None:
            continue
        if isinstance(value, bool):
            flattened[str(key)] = value
            continue
        if isinstance(value, _METADATA_SCALAR_TYPES):
            flattened[str(key)] = value
    return flattened


def _safe_tag_component(value: str, *, max_length: int = 48) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch in {"-", "_", ":"} else "-" for ch in str(value).strip())
    cleaned = cleaned.strip("-:") or f"id-{stable_hash({'value': str(value)})[:12]}"
    if len(cleaned) <= max_length:
        return cleaned
    return f"{cleaned[: max_length - 13].rstrip('-:')}-{stable_hash({'value': cleaned})[:12]}"


def supermemory_container_tags(scope: MemoryScope) -> list[str]:
    tags = ["mana"]
    for scope_key, label in SUPERMEMORY_TAG_COMPONENTS:
        value = str(getattr(scope, scope_key) or "").strip()
        if value:
            tags.append(f"mana:{label}:{_safe_tag_component(value)}")
    return tags


def supermemory_primary_container_tag(scope: MemoryScope) -> str:
    return f"mana:scope:{stable_hash({'scope': scope.as_dict()})}"


def supermemory_metadata(
    scope: MemoryScope,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str | int | float | bool]:
    combined: dict[str, Any] = {
        "source": "mana-agent",
        **scope.as_dict(),
        **scope_metadata(scope),
        **dict(metadata or {}),
    }
    return flat_scalar_metadata(combined)


def supermemory_custom_id(
    *,
    scope: MemoryScope,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    flattened = flat_scalar_metadata(metadata)
    kind = str(flattened.get("memory_kind") or flattened.get("mana_kind") or "memory").strip() or "memory"
    stable_identity = {
        key: value
        for key, value in {
            "kind": kind,
            "task_id": scope.task_id,
            "conversation_id": scope.conversation_id,
            "repository_id": scope.repository_id,
            "workspace_id": scope.workspace_id,
            "agent_id": scope.agent_id,
            "session_id": scope.session_id,
            "fingerprint": flattened.get("fingerprint"),
            "decision_type": flattened.get("decision_type"),
            "tool_name": flattened.get("tool_name"),
            "custom_id": flattened.get("custom_id"),
            "memory_id": flattened.get("memory_id"),
        }.items()
        if value not in {"", None}
    }
    if not stable_identity:
        return None
    content_hash = stable_hash({"content": content})[:12]
    identity_hash = stable_hash(stable_identity)
    return f"mana:{_safe_tag_component(kind, max_length=20)}:{identity_hash}:{content_hash}"


def record_from_document(
    *,
    document_id: str,
    content: str,
    scope: MemoryScope,
    provider: str,
    metadata: dict[str, Any] | None = None,
    provider_metadata: dict[str, Any] | None = None,
    score: float | None = None,
    created_at: Any = None,
    updated_at: Any = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=str(document_id),
        content=MemoryContent(str(content or "")),
        scope=scope,
        metadata=flat_scalar_metadata(metadata),
        score=score,
        provider=provider,
        provider_metadata=dict(provider_metadata or {}),
        created_at=parse_timestamp(created_at),
        updated_at=parse_timestamp(updated_at),
    )
