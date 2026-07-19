"""Mem0 implementation of the shared memory contract."""

from __future__ import annotations

import uuid

from mana_agent.memory.config import MemoryConfig
from mana_agent.memory.errors import (
    MemoryAuthenticationError,
    MemoryConfigurationError,
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
    utc_now,
)
from mana_agent.memory.providers.mem0.client import Mem0Client
from mana_agent.memory.providers.mem0.mapper import (
    response_rows,
    response_to_record,
    scope_to_filters,
    scope_to_mem0,
)


class Mem0MemoryBackend:
    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.client = Mem0Client(config)

    async def add(self, request: MemoryWriteRequest) -> MemoryRecord:
        entities, scope_metadata = scope_to_mem0(request.scope)
        payload = await self.client.call(
            "add",
            request.content.text,
            metadata={**request.metadata, **scope_metadata},
            **entities,
        )
        rows = response_rows(payload)
        if not rows:
            acknowledgement = payload if isinstance(payload, dict) else {}
            acknowledgement_id = str(
                acknowledgement.get("event_id")
                or acknowledgement.get("request_id")
                or acknowledgement.get("id")
                or f"pending:{uuid.uuid4().hex}"
            )
            return MemoryRecord(
                id=acknowledgement_id,
                content=request.content,
                scope=request.scope,
                metadata=dict(request.metadata),
                provider="mem0",
                provider_metadata={
                    "pending": True,
                    "acknowledgement": acknowledgement,
                },
                created_at=utc_now(),
                updated_at=utc_now(),
            )
        return response_to_record(rows[0], request.scope)

    async def search(self, request: MemorySearchRequest) -> list[MemoryRecord]:
        entities, _ = scope_to_mem0(request.scope)
        if not entities:
            raise MemoryConfigurationError(
                "Mem0 search requires at least one user, agent, session, or workspace identity."
            )
        payload = await self.client.call(
            "search",
            request.query,
            filters=scope_to_filters(request.scope, request.metadata),
            top_k=max(1, request.limit),
        )
        return [response_to_record(row, request.scope) for row in response_rows(payload)]

    async def get(self, memory_id: str, scope: MemoryScope) -> MemoryRecord | None:
        payload = await self.client.call("get", memory_id)
        rows = response_rows(payload)
        return response_to_record(rows[0], scope) if rows else None

    async def update(self, memory_id: str, request: MemoryUpdateRequest) -> MemoryRecord:
        current = await self.get(memory_id, request.scope)
        if current is None:
            raise MemoryNotFoundError(f"Memory {memory_id!r} was not found.")
        data = {
            "text": (request.content or current.content).text,
            "metadata": (
                request.metadata if request.metadata is not None else current.metadata
            ),
        }
        payload = await self.client.call("update", memory_id, data)
        rows = response_rows(payload)
        if rows:
            return response_to_record(rows[0], request.scope)
        return MemoryRecord(
            id=memory_id,
            content=request.content or current.content,
            scope=request.scope,
            metadata=dict(data["metadata"]),
            provider="mem0",
        )

    async def delete(self, memory_id: str, scope: MemoryScope) -> None:
        _ = scope
        await self.client.call("delete", memory_id)

    async def clear(self, scope: MemoryScope) -> None:
        entities, _ = scope_to_mem0(scope)
        if not entities:
            raise MemoryProviderError("Refusing to clear Mem0 without a user, agent, session, or workspace scope.")
        await self.client.call("delete_all", **entities)

    async def healthcheck(self) -> MemoryHealth:
        try:
            await self.client.healthcheck()
            return MemoryHealth(MemoryHealthStatus.HEALTHY, "external", "mem0", "Connected")
        except MemoryDependencyError as exc:
            return MemoryHealth(MemoryHealthStatus.DEPENDENCY_ERROR, "external", "mem0", str(exc))
        except MemoryAuthenticationError as exc:
            return MemoryHealth(MemoryHealthStatus.AUTHENTICATION_ERROR, "external", "mem0", str(exc))
        except MemoryNetworkError as exc:
            return MemoryHealth(MemoryHealthStatus.NETWORK_ERROR, "external", "mem0", str(exc))
        except MemoryProviderError as exc:
            return MemoryHealth(MemoryHealthStatus.PROVIDER_ERROR, "external", "mem0", str(exc))

    async def close(self) -> None:
        await self.client.close()
