"""Supermemory implementation of the shared memory contract."""

from __future__ import annotations

from typing import Any

from mana_agent.memory.config import MemoryConfig
from mana_agent.memory.errors import (
    MemoryAuthenticationError,
    MemoryDependencyError,
    MemoryNetworkError,
    MemoryNotFoundError,
    MemoryProviderError,
)
from mana_agent.memory.models import (
    MemoryHealth,
    MemoryHealthStatus,
    MemoryRecord,
    MemoryScope,
    MemorySearchRequest,
    MemoryUpdateRequest,
    MemoryWriteRequest,
)
from mana_agent.memory.providers.shared import (
    record_from_document,
    supermemory_document_id,
    supermemory_metadata,
    supermemory_namespace,
)
from mana_agent.memory.providers.supermemory.client import SupermemoryClient
from mana_agent.memory.providers.supermemory.mapper import (
    document_to_record,
    search_result_to_record,
    supermemory_filter,
)

_DOCUMENT_BATCH_LIMIT = 100


class SupermemoryProvider:
    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.client = SupermemoryClient(config)

    async def add(self, request: MemoryWriteRequest) -> MemoryRecord:
        scope = request.scope
        namespace = supermemory_namespace(scope)
        metadata = supermemory_metadata(scope, request.metadata)
        caller_id = supermemory_document_id(scope=scope, content=request.content.text, metadata=metadata)
        response = await self.client.call(
            "add",
            namespace,
            content=request.content.text,
            id=caller_id,
            metadata=metadata,
            task_type="memory",
            dreaming="dynamic",
            operation="add",
        )
        return record_from_document(
            document_id=response.id,
            content=request.content.text,
            scope=scope,
            provider="supermemory",
            metadata=metadata,
            provider_metadata={
                "status": response.status,
                "caller_id": caller_id,
                "namespace": namespace,
            },
        )

    async def search(self, request: MemorySearchRequest) -> list[MemoryRecord]:
        metadata_filter = supermemory_filter(supermemory_metadata(request.scope, request.metadata))
        filter_arg = {"filter": metadata_filter} if metadata_filter is not None else {}
        response = await self.client.call(
            "search",
            supermemory_namespace(request.scope),
            query=request.query,
            limit=max(1, request.limit),
            search_mode="hybrid",
            threshold=0.6,
            include={"documents": True},
            operation="search",
            **filter_arg,
        )
        return [search_result_to_record(result, request.scope) for result in response.results]

    async def get(self, memory_id: str, scope: MemoryScope) -> MemoryRecord | None:
        namespace = supermemory_namespace(scope)
        try:
            document = await self.client.call("documents.get", namespace, memory_id, operation="get")
        except MemoryNotFoundError:
            return None
        return document_to_record(document, scope, namespace)

    async def update(self, memory_id: str, request: MemoryUpdateRequest) -> MemoryRecord:
        current = await self.get(memory_id, request.scope)
        if current is None:
            raise MemoryNotFoundError(f"Memory {memory_id!r} was not found.")
        changes: dict[str, Any] = {}
        if request.content is not None:
            changes.update(content=request.content.text, task_type="memory", dreaming="dynamic")
        metadata = current.metadata
        if request.metadata is not None:
            metadata = supermemory_metadata(request.scope, request.metadata)
            changes["metadata"] = metadata
        if not changes:
            return current

        namespace = supermemory_namespace(request.scope)
        response = await self.client.call(
            "documents.update",
            namespace,
            current.id,
            operation="update",
            **changes,
        )
        return record_from_document(
            document_id=response.id,
            content=(request.content or current.content).text,
            scope=request.scope,
            provider="supermemory",
            metadata=metadata,
            provider_metadata={"status": response.status, "namespace": namespace},
            created_at=current.created_at,
        )

    async def delete(self, memory_id: str, scope: MemoryScope) -> None:
        current = await self.get(memory_id, scope)
        if current is None:
            raise MemoryNotFoundError(f"Memory {memory_id!r} was not found.")
        await self._delete_documents(supermemory_namespace(scope), [current.id], operation="delete")

    async def clear(self, scope: MemoryScope) -> None:
        if not any((scope.user_id, scope.agent_id, scope.session_id, scope.workspace_id, scope.repository_id)):
            raise MemoryProviderError(
                "Refusing to clear Supermemory without a user, agent, session, workspace, or repository scope."
            )
        namespace = supermemory_namespace(scope)
        metadata_filter = supermemory_filter(supermemory_metadata(scope))
        page = 1
        ids: dict[str, None] = {}
        while True:
            response = await self.client.call(
                "list",
                namespace,
                "documents",
                filter=metadata_filter,
                limit=_DOCUMENT_BATCH_LIMIT,
                page=page,
                sort="createdAt",
                order="asc",
                include={"memory_count": False},
                operation="clear",
            )
            if response.pagination.current_page != page:
                raise MemoryProviderError("Supermemory clear received an unexpected document page.")
            ids.update(dict.fromkeys(document.id for document in response.documents))
            if page >= response.pagination.total_pages:
                break
            page += 1
        await self._delete_documents(namespace, list(ids), operation="clear")

    async def _delete_documents(self, namespace: str, ids: list[str], *, operation: str) -> None:
        deleted = errors = processing = 0
        incomplete = False
        for offset in range(0, len(ids), _DOCUMENT_BATCH_LIMIT):
            batch = ids[offset:offset + _DOCUMENT_BATCH_LIMIT]
            result = await self.client.call("documents.delete", namespace, ids=batch, operation=operation)
            deleted += result.count
            errors += len(result.errors)
            processing += sum("still processing" in item.error.casefold() for item in result.errors)
            if result.count != len(batch) or result.errors:
                incomplete = True
        if incomplete:
            detail = f"Supermemory {operation} deleted {deleted} of {len(ids)} documents ({errors} per-document errors)."
            if processing:
                detail += f" {processing} are still processing. Retry once processing finishes."
            raise MemoryProviderError(detail)

    async def healthcheck(self) -> MemoryHealth:
        try:
            await self.client.healthcheck()
            return MemoryHealth(MemoryHealthStatus.HEALTHY, "external", "supermemory", "Connected")
        except MemoryDependencyError as exc:
            return MemoryHealth(MemoryHealthStatus.DEPENDENCY_ERROR, "external", "supermemory", str(exc))
        except MemoryAuthenticationError as exc:
            return MemoryHealth(MemoryHealthStatus.AUTHENTICATION_ERROR, "external", "supermemory", str(exc))
        except MemoryNetworkError as exc:
            return MemoryHealth(MemoryHealthStatus.NETWORK_ERROR, "external", "supermemory", str(exc))
        except MemoryProviderError as exc:
            return MemoryHealth(MemoryHealthStatus.PROVIDER_ERROR, "external", "supermemory", str(exc))

    async def close(self) -> None:
        await self.client.close()
