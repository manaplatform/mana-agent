"""Supermemory implementation of the shared memory contract."""

from __future__ import annotations

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
from mana_agent.memory.providers.shared import record_from_document, supermemory_custom_id
from mana_agent.memory.providers.supermemory.client import SupermemoryClient
from mana_agent.memory.providers.supermemory.mapper import (
    search_result_to_record,
    supermemory_container_tags,
    supermemory_filters,
    supermemory_metadata,
    supermemory_primary_container_tag,
)


class SupermemoryProvider:
    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.client = SupermemoryClient(config)

    async def add(self, request: MemoryWriteRequest) -> MemoryRecord:
        scope = request.scope
        metadata = supermemory_metadata(scope, request.metadata)
        custom_id = supermemory_custom_id(scope=scope, content=request.content.text, metadata=metadata)
        response = await self.client.call(
            "add",
            content=request.content.text,
            container_tag=supermemory_primary_container_tag(scope),
            container_tags=supermemory_container_tags(scope),
            custom_id=custom_id,
            metadata=metadata,
            task_type="memory",
            operation="add",
        )
        return record_from_document(
            document_id=response.id,
            content=request.content.text,
            scope=scope,
            provider="supermemory",
            metadata=metadata,
            provider_metadata={
                "status": getattr(response, "status", ""),
                "custom_id": custom_id,
                "container_tag": supermemory_primary_container_tag(scope),
                "container_tags": supermemory_container_tags(scope),
            },
        )

    async def search(self, request: MemorySearchRequest) -> list[MemoryRecord]:
        response = await self.client.call(
            "search.memories",
            q=request.query,
            container_tag=supermemory_primary_container_tag(request.scope),
            container_tags=supermemory_container_tags(request.scope),
            filters=supermemory_filters(supermemory_metadata(request.scope, request.metadata)),
            limit=max(1, request.limit),
            search_mode="hybrid",
            include={"documents": True, "chunks": True},
            operation="search",
        )
        return [search_result_to_record(result, request.scope) for result in getattr(response, "results", [])]

    async def get(self, memory_id: str, scope: MemoryScope) -> MemoryRecord | None:
        try:
            document = await self.client.call("documents.get", memory_id, operation="get")
        except MemoryNotFoundError:
            return None
        return record_from_document(
            document_id=document.id,
            content=str(getattr(document, "content", "") or ""),
            scope=scope,
            provider="supermemory",
            metadata=supermemory_metadata(scope, getattr(document, "metadata", None)),
            provider_metadata={
                "status": getattr(document, "status", ""),
                "custom_id": getattr(document, "custom_id", None),
                "title": getattr(document, "title", None),
                "type": getattr(document, "type", None),
                "container_tags": getattr(document, "container_tags", None),
            },
            created_at=getattr(document, "created_at", None),
            updated_at=getattr(document, "updated_at", None),
        )

    async def update(self, memory_id: str, request: MemoryUpdateRequest) -> MemoryRecord:
        current = await self.get(memory_id, request.scope)
        if current is None:
            raise MemoryNotFoundError(f"Memory {memory_id!r} was not found.")
        content = (request.content or current.content).text
        metadata = supermemory_metadata(
            request.scope,
            request.metadata if request.metadata is not None else current.metadata,
        )
        await self.client.call(
            "documents.update",
            memory_id,
            content=content,
            container_tag=supermemory_primary_container_tag(request.scope),
            container_tags=supermemory_container_tags(request.scope),
            custom_id=supermemory_custom_id(scope=request.scope, content=content, metadata=metadata),
            metadata=metadata,
            task_type="memory",
            operation="update",
        )
        updated = await self.get(memory_id, request.scope)
        if updated is not None:
            return updated
        return record_from_document(
            document_id=memory_id,
            content=content,
            scope=request.scope,
            provider="supermemory",
            metadata=metadata,
        )

    async def delete(self, memory_id: str, scope: MemoryScope) -> None:
        current = await self.get(memory_id, scope)
        if current is None:
            raise MemoryNotFoundError(f"Memory {memory_id!r} was not found.")
        await self.client.call(
            "memories.forget",
            container_tag=supermemory_primary_container_tag(scope),
            id=memory_id,
            content=current.content.text,
            reason="Deleted from Mana-Agent",
            operation="delete",
        )
        await self.client.call("documents.delete", memory_id, operation="delete")

    async def clear(self, scope: MemoryScope) -> None:
        if not any((scope.user_id, scope.agent_id, scope.session_id, scope.workspace_id, scope.repository_id)):
            raise MemoryProviderError(
                "Refusing to clear Supermemory without a user, agent, session, workspace, or repository scope."
            )
        page = 1
        ids: list[str] = []
        while True:
            response = await self.client.call(
                "documents.list",
                container_tags=supermemory_container_tags(scope),
                filters=supermemory_filters(supermemory_metadata(scope)),
                limit=100,
                page=page,
                include_content=False,
                operation="clear",
            )
            memories = list(getattr(response, "memories", []) or [])
            ids.extend(str(getattr(item, "id", "") or "") for item in memories if getattr(item, "id", None))
            pagination = getattr(response, "pagination", None)
            if not memories or pagination is None or page >= int(getattr(pagination, "total_pages", 1) or 1):
                break
            page += 1
        if ids:
            await self.client.call("documents.delete_bulk", ids=ids, operation="clear")

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
