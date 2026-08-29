"""Reference-counted lifecycle for local Responses bridge instances."""

from __future__ import annotations

import logging
import secrets
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

import uvicorn

from mana_agent.integrations.codex.responses_bridge.models import BridgeUpstreamConfig
from mana_agent.integrations.codex.responses_bridge.server import build_bridge_app

logger = logging.getLogger(__name__)


def _ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(slots=True)
class ResponsesBridgeHandle:
    """Live bridge endpoint used by one or more Codex runtime contexts."""

    base_url: str
    temporary_api_key: str = field(repr=False)
    upstream_provider: str
    upstream_model: str
    transport: str = "codex_responses_bridge"
    _manager: "ResponsesBridgeManager | None" = field(default=None, repr=False)
    _key: str = field(default="", repr=False)
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._manager is not None and self._key:
            self._manager.release(self._key)

    def healthcheck(self, *, timeout_seconds: float = 2.0) -> dict[str, Any]:
        # App mounts Responses under /v1; health lives at the process root.
        root = self.base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        url = root + "/health"
        request = Request(url, method="GET")
        opener = build_opener(ProxyHandler({}))
        with opener.open(request, timeout=timeout_seconds) as response:  # noqa: S310 - loopback only
            body = response.read().decode("utf-8")
        import json

        payload = json.loads(body)
        if not isinstance(payload, dict) or not payload.get("ok"):
            raise RuntimeError("Mana Responses bridge health check failed.")
        return payload


@dataclass
class _BridgeInstance:
    key: str
    upstream: BridgeUpstreamConfig
    temporary_api_key: str
    host: str
    port: int
    server: uvicorn.Server
    thread: threading.Thread
    refcount: int = 1

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


class ResponsesBridgeManager:
    """Start and stop loopback Responses bridges with reference counting."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._instances: dict[str, _BridgeInstance] = {}

    @staticmethod
    def _instance_key(upstream: BridgeUpstreamConfig) -> str:
        # Share bridges only for identical upstream endpoints/models/credentials.
        # Credential material is not logged; the key itself stays in-process.
        return "|".join(
            [
                upstream.provider,
                upstream.base_url.rstrip("/"),
                upstream.model,
                str(hash(upstream.api_key)),
                str(sorted(upstream.headers.items())),
                str(sorted((upstream.request_overrides or {}).items())),
            ]
        )

    def start(self, upstream: BridgeUpstreamConfig) -> ResponsesBridgeHandle:
        key = self._instance_key(upstream)
        with self._lock:
            existing = self._instances.get(key)
            if existing is not None and existing.thread.is_alive():
                existing.refcount += 1
                return ResponsesBridgeHandle(
                    base_url=existing.base_url,
                    temporary_api_key=existing.temporary_api_key,
                    upstream_provider=upstream.provider,
                    upstream_model=upstream.model,
                    _manager=self,
                    _key=key,
                )

            host = "127.0.0.1"
            port = _ephemeral_port()
            temporary_api_key = secrets.token_urlsafe(32)
            app = build_bridge_app(upstream=upstream, expected_token=temporary_api_key)
            config = uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level="warning",
                access_log=False,
                lifespan="on",
            )
            server = uvicorn.Server(config)
            thread = threading.Thread(
                target=server.run,
                name=f"mana-responses-bridge-{port}",
                daemon=True,
            )
            thread.start()
            instance = _BridgeInstance(
                key=key,
                upstream=upstream,
                temporary_api_key=temporary_api_key,
                host=host,
                port=port,
                server=server,
                thread=thread,
            )
            self._instances[key] = instance

        # Wait until the loopback health endpoint answers.
        handle = ResponsesBridgeHandle(
            base_url=instance.base_url,
            temporary_api_key=temporary_api_key,
            upstream_provider=upstream.provider,
            upstream_model=upstream.model,
            _manager=self,
            _key=key,
        )
        deadline = time.time() + 5.0
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                handle.healthcheck(timeout_seconds=0.5)
                logger.info(
                    "responses_bridge.started provider=%s model=%s port=%s transport=codex_responses_bridge",
                    upstream.provider,
                    upstream.model,
                    port,
                )
                return handle
            except (URLError, TimeoutError, RuntimeError, OSError, ValueError) as exc:
                last_error = exc
                time.sleep(0.05)
        self.release(key)
        raise RuntimeError(
            f"Mana Responses bridge failed to become healthy: {type(last_error).__name__}."
        )

    def release(self, key: str) -> None:
        with self._lock:
            instance = self._instances.get(key)
            if instance is None:
                return
            instance.refcount -= 1
            if instance.refcount > 0:
                return
            self._instances.pop(key, None)
            instance.server.should_exit = True
        # Join outside the lock so stop cannot deadlock with start.
        instance.thread.join(timeout=5.0)
        logger.info(
            "responses_bridge.stopped provider=%s model=%s port=%s",
            instance.upstream.provider,
            instance.upstream.model,
            instance.port,
        )

    def shutdown_all(self) -> None:
        with self._lock:
            keys = list(self._instances)
        for key in keys:
            with self._lock:
                instance = self._instances.get(key)
                if instance is None:
                    continue
                instance.refcount = 0
            self.release(key)


# Process-wide manager used by Codex runtime construction.
BRIDGE_MANAGER = ResponsesBridgeManager()


__all__ = [
    "BRIDGE_MANAGER",
    "ResponsesBridgeHandle",
    "ResponsesBridgeManager",
]
