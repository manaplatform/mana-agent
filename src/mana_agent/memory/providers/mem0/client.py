"""Lazy, timeout-bound wrapper around the optional Mem0 SDK."""

from __future__ import annotations

import asyncio
import inspect
from functools import partial
from typing import Any, Callable

from mana_agent.memory.config import MemoryConfig
from mana_agent.memory.errors import (
    MemoryAuthenticationError,
    MemoryDependencyError,
    MemoryNetworkError,
    MemoryProviderError,
)


def _translate(exc: Exception) -> Exception:
    text = str(exc).lower()
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status in {401, 403} or "unauthorized" in text or "authentication" in text or "invalid api key" in text:
        return MemoryAuthenticationError("Mem0 authentication failed. Check the configured API key.")
    network_words = ("timeout", "connection", "network", "dns")
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)) or any(
        word in text for word in network_words
    ):
        return MemoryNetworkError("Mem0 could not be reached before the configured timeout.")
    return MemoryProviderError("Mem0 rejected the memory operation.")


class Mem0Client:
    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from mem0 import MemoryClient
        except ImportError as exc:
            raise MemoryDependencyError(
                "Mem0 is selected but unavailable. Install it with: pip install 'mana-agent[mem0]'."
            ) from exc
        kwargs: dict[str, Any] = {"api_key": self.config.api_key}
        try:
            parameters = inspect.signature(MemoryClient).parameters
        except (TypeError, ValueError):
            parameters = {}
        optional = {
            "org_id": self.config.org_id,
            "project_id": self.config.project_id,
            "host": self.config.base_url,
            "base_url": self.config.base_url,
        }
        kwargs.update({key: value for key, value in optional.items() if value and key in parameters})
        try:
            self._client = MemoryClient(**kwargs)
        except Exception as exc:
            raise _translate(exc) from exc
        return self._client

    async def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            function: Callable[..., Any] = getattr(self._get_client(), method)
            if inspect.iscoroutinefunction(function):
                return await asyncio.wait_for(
                    function(*args, **kwargs),
                    timeout=self.config.timeout_seconds,
                )
            return await asyncio.wait_for(
                asyncio.to_thread(partial(function, *args, **kwargs)),
                timeout=self.config.timeout_seconds,
            )
        except (MemoryDependencyError, MemoryAuthenticationError, MemoryNetworkError, MemoryProviderError):
            raise
        except Exception as exc:
            raise _translate(exc) from exc

    async def healthcheck(self) -> None:
        client = self._get_client()
        project = getattr(client, "project", None)
        if project is not None and hasattr(project, "get"):
            await self.call_object(project.get)
        else:
            await self.call("get_all", filters={"user_id": "mana-agent-healthcheck"}, page_size=1)

    async def call_object(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            if inspect.iscoroutinefunction(function):
                return await asyncio.wait_for(
                    function(*args, **kwargs),
                    timeout=self.config.timeout_seconds,
                )
            return await asyncio.wait_for(
                asyncio.to_thread(partial(function, *args, **kwargs)),
                timeout=self.config.timeout_seconds,
            )
        except Exception as exc:
            raise _translate(exc) from exc

    async def close(self) -> None:
        client, self._client = self._client, None
        close = getattr(client, "close", None) if client is not None else None
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result
