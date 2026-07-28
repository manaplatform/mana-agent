"""Supermemory response mapping helpers."""

from __future__ import annotations

from typing import Any

from mana_agent.memory.models import MemoryRecord, MemoryScope
from mana_agent.memory.providers.shared import (
    flat_scalar_metadata,
    record_from_document,
    supermemory_container_tags,
    supermemory_metadata,
    supermemory_primary_container_tag,
)


def supermemory_filters(metadata: dict[str, Any] | None) -> dict[str, list[dict[str, str | int | float | bool]]]:
    flattened = flat_scalar_metadata(metadata)
    clauses = [{"key": key, "value": value} for key, value in flattened.items()]
    return {"AND": clauses} if clauses else {}


def search_result_to_record(result: Any, scope: MemoryScope) -> MemoryRecord:
    result_metadata = flat_scalar_metadata(getattr(result, "metadata", None))
    content = (
        getattr(result, "memory", None)
        or getattr(result, "chunk", None)
        or " ".join(chunk.content for chunk in (getattr(result, "chunks", None) or []) if getattr(chunk, "content", None))
    )
    provider_metadata = {
        "filepath": getattr(result, "filepath", None),
        "documents": [
            {
                "id": getattr(doc, "id", None),
                "title": getattr(doc, "title", None),
                "type": getattr(doc, "type", None),
            }
            for doc in (getattr(result, "documents", None) or [])
            if getattr(doc, "id", None)
        ],
        "is_aggregated": getattr(result, "is_aggregated", None),
        "version": getattr(result, "version", None),
    }
    provider_metadata = {key: value for key, value in provider_metadata.items() if value not in (None, [], {})}
    return record_from_document(
        document_id=str(getattr(result, "id", "")),
        content=str(content or ""),
        scope=scope,
        provider="supermemory",
        metadata=result_metadata,
        provider_metadata=provider_metadata,
        score=float(getattr(result, "similarity", 0.0) or 0.0),
        updated_at=getattr(result, "updated_at", None),
    )


__all__ = [
    "record_from_document",
    "search_result_to_record",
    "supermemory_container_tags",
    "supermemory_filters",
    "supermemory_metadata",
    "supermemory_primary_container_tag",
]
