"""Supermemory response mapping helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mana_agent.memory.models import MemoryRecord, MemoryScope
from mana_agent.memory.providers.shared import record_from_document, supermemory_metadata

if TYPE_CHECKING:
    from supermemory.requests import FilterExpressionParams
    from supermemory.types import Document, SearchResult


def supermemory_filter(metadata: dict[str, str | int | float | bool]) -> FilterExpressionParams | None:
    clauses: list[FilterExpressionParams] = [
        {"field": key, "operator": "eq", "value": value}
        for key, value in metadata.items()
    ]
    return {"operator": "and", "operands": clauses} if clauses else None


def document_to_record(document: Document, scope: MemoryScope, namespace: str) -> MemoryRecord:
    return record_from_document(
        document_id=document.id,
        content=document.content or "",
        scope=scope,
        provider="supermemory",
        metadata=supermemory_metadata(scope, document.metadata),
        provider_metadata={
            "namespace": namespace,
            "status": document.system.status,
            "dreaming_status": document.system.dreaming_status,
            "title": document.title,
            "type": document.type,
        },
        created_at=document.system.created_at,
        updated_at=document.system.updated_at,
    )


def search_result_to_record(result: SearchResult, scope: MemoryScope) -> MemoryRecord:
    provider_metadata: dict[str, Any] = {
        "is_latest": result.is_latest,
        "is_inference": result.is_inference,
    }
    document = result.included.document if result.included is not None else None
    if document is not None:
        provider_metadata["document"] = {
            "id": document.id,
            "title": document.title,
            "type": document.type,
        }
    return record_from_document(
        document_id=result.id,
        content=result.memory or result.chunk or "",
        scope=scope,
        provider="supermemory",
        metadata=result.metadata,
        provider_metadata=provider_metadata,
        score=result.similarity,
        created_at=result.system.created_at,
        updated_at=result.system.updated_at,
    )
