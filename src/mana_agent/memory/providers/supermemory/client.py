"""Lazy wrapper around the optional Supermemory SDK."""

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
    MemoryNotFoundError,
    MemoryProviderError,
)


def _translate(exc: Exception, *, operation: str) -> Exception:
    text = str(exc).strip() or exc.__class__.__name__
    lowered = text.lower()
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(exc, MemoryProviderError):
        return exc
    if isinstance(exc, MemoryNotFoundError) or status == 404:
        return MemoryNotFoundError(f"Supermemory {operation} failed: the requested memory was not found.")
    if status == 401 or "api_key" in lowered or "api key" in lowered or "authentication" in lowered:
        return MemoryAuthenticationError(
            f"Supermemory {operation} failed due to invalid credentials. Retrying will not help until the API key is fixed."
        )
    if status == 403 or "permission" in lowered or "forbidden" in lowered:
        return MemoryProviderError(
            f"Supermemory {operation} failed because access was denied. Retrying will not help without broader provider permissions."
        )
    if status == 429 or "rate limit" in lowered or "quota" in lowered:
        return MemoryNetworkError(
            f"Supermemory {operation} was rate-limited or quota-limited. Retrying later may succeed."
        )
    if status == 400 or status == 422 or "validation" in lowered or "bad request" in lowered:
        return MemoryProviderError(
            f"Supermemory {operation} failed because the request was rejected as malformed. Retrying unchanged will not help."
        )
    if status and int(status) >= 500:
        return MemoryNetworkError(
            f"Supermemory {operation} failed because the service is temporarily unavailable. Retrying may succeed."
        )
    network_words = ("timeout", "connection", "network", "dns", "unreachable")
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)) or any(word in lowered for word in network_words):
        return MemoryNetworkError(
            f"Supermemory {operation} failed because the service could not be reached before the configured timeout. Retrying may succeed."
        )
    return MemoryProviderError(
        f"Supermemory {operation} failed unexpectedly. Retrying may help only if the provider response was transient."
    )


class SupermemoryClient:
    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from supermemory import Supermemory
        except ImportError as exc:
            raise MemoryDependencyError(
                "Supermemory is selected but unavailable. Install it with: pip install 'mana-agent[supermemory]'."
            ) from exc
        kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout_seconds,
            "max_retries": 2,
        }
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        try:
            self._client = Supermemory(**kwargs)
        except Exception as exc:
            raise _translate(exc, operation="client initialization") from exc
        return self._client

    async def call(self, method: str, *args: Any, operation: str, **kwargs: Any) -> Any:
        try:
            function = self._resolve_callable(method)
            return await self._invoke(function, *args, **kwargs)
        except (MemoryDependencyError, MemoryAuthenticationError, MemoryNetworkError, MemoryNotFoundError, MemoryProviderError):
            raise
        except Exception as exc:
            raise _translate(exc, operation=operation) from exc

    def _resolve_callable(self, method: str) -> Callable[..., Any]:
        function: Any = self._get_client()
        for part in method.split("."):
            function = getattr(function, part)
        return function

    async def _invoke(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if inspect.iscoroutinefunction(function):
            return await asyncio.wait_for(function(*args, **kwargs), timeout=self.config.timeout_seconds)
        return await asyncio.wait_for(
            asyncio.to_thread(partial(function, *args, **kwargs)),
            timeout=self.config.timeout_seconds,
        )

    async def healthcheck(self) -> None:
        await self.call("documents.list", limit=1, operation="healthcheck")

    async def close(self) -> None:
        client, self._client = self._client, None
        close = getattr(client, "close", None) if client is not None else None
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result
