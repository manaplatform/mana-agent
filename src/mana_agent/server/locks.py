"""Per-server read/mutation concurrency controls shared by every entry surface."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


@dataclass
class _ServerLocks:
    mutation: asyncio.Lock = field(default_factory=asyncio.Lock)
    concurrency: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(1))


class ServerLockManager:
    def __init__(self) -> None:
        self._locks: dict[str, _ServerLocks] = {}
        self._guard = asyncio.Lock()

    async def _get(self, server_id: str, limit: int) -> _ServerLocks:
        async with self._guard:
            current = self._locks.get(server_id)
            if current is None:
                current = _ServerLocks(concurrency=asyncio.Semaphore(limit))
                self._locks[server_id] = current
            return current

    @asynccontextmanager
    async def acquire(self, server_id: str, *, mutation: bool, concurrency_limit: int = 1):
        locks = await self._get(server_id, concurrency_limit)
        async with locks.concurrency:
            if mutation:
                async with locks.mutation:
                    yield
            else:
                yield
