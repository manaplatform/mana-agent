"""Versioned custom HTTP sandbox runtime provider."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mana_agent.execution.errors import ExecutionFailedError, ProviderConfigurationError
from mana_agent.execution.models import (
    ArtifactRequest,
    ArtifactResult,
    ExecutionRequest,
    ExecutionResult,
    ProviderHealth,
    SandboxCapabilities,
    SandboxHandle,
    SandboxSpec,
    SnapshotRef,
    SnapshotRequest,
)
from mana_agent.execution.providers.base import ProviderBase
from mana_agent.execution.secrets import EnvironmentSecretResolver, SecretResolver


class CustomHTTPProvider(ProviderBase):
    name = "custom-http-runtime"

    def __init__(
        self, *, base_url: str = "", credential_ref: str = "", signing_secret_ref: str = "",
        connect_timeout_seconds: int = 10, read_timeout_seconds: int = 120,
        secret_resolver: SecretResolver | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.credential_ref = credential_ref
        self.signing_secret_ref = signing_secret_ref
        self.connect_timeout_seconds = connect_timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self.secret_resolver = secret_resolver or EnvironmentSecretResolver()

    def _headers(self, body: bytes, idempotency_key: str = "") -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json", "Mana-Runtime-Version": "1"}
        if self.credential_ref:
            headers["Authorization"] = f"Bearer {self.secret_resolver.resolve(self.credential_ref)}"
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if self.signing_secret_ref:
            key = self.secret_resolver.resolve(self.signing_secret_ref).encode()
            headers["X-Mana-Signature-SHA256"] = hmac.new(key, body, hashlib.sha256).hexdigest()
        return headers

    async def _request(self, method: str, path: str, payload: dict[str, Any] | None = None, *, idempotency_key: str = "") -> dict[str, Any]:
        if not self.base_url.startswith(("http://", "https://")):
            raise ProviderConfigurationError("custom HTTP runtime requires a valid base_url", provider=self.name)
        body = json.dumps(payload or {}, sort_keys=True).encode() if payload is not None else b""
        request = urllib.request.Request(
            self.base_url + path, data=body if payload is not None else None,
            headers=self._headers(body, idempotency_key), method=method,
        )
        def send() -> dict[str, Any]:
            try:
                with urllib.request.urlopen(request, timeout=self.read_timeout_seconds) as response:
                    api_version = response.headers.get("Mana-Runtime-Version", "1")
                    if api_version != "1":
                        raise ProviderConfigurationError(f"unsupported HTTP runtime API version: {api_version}", provider=self.name)
                    raw = response.read()
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as exc:
                diagnostic = exc.read(2000).decode(errors="replace")
                raise ExecutionFailedError(f"HTTP runtime returned {exc.code}", provider=self.name, diagnostics=diagnostic) from exc
            except urllib.error.URLError as exc:
                raise ExecutionFailedError("HTTP runtime request failed", provider=self.name, diagnostics=str(exc.reason)) from exc
        result = await asyncio.to_thread(send)
        if not isinstance(result, dict):
            raise ExecutionFailedError("HTTP runtime response is not an object", provider=self.name)
        return result

    async def capabilities(self) -> SandboxCapabilities:
        health = await self._request("GET", "/v1/health")
        return SandboxCapabilities.model_validate(health.get("capabilities") or {})

    async def provision(self, spec: SandboxSpec) -> SandboxHandle:
        payload = spec.model_dump(mode="json", exclude={"secrets"})
        payload["secret_references"] = [item.model_dump(mode="json") for item in spec.secrets]
        data = await self._request("POST", "/v1/sandboxes", payload, idempotency_key=f"provision:{spec.task_id}:{spec.session_id}")
        return SandboxHandle.model_validate(data["sandbox"])

    async def start(self, handle: SandboxHandle) -> SandboxHandle:
        data = await self._request("POST", f"/v1/sandboxes/{handle.external_id or handle.sandbox_id}/start", {})
        return SandboxHandle.model_validate(data["sandbox"])

    async def execute(self, handle: SandboxHandle, request: ExecutionRequest) -> ExecutionResult:
        data = await self._request("POST", f"/v1/sandboxes/{handle.external_id or handle.sandbox_id}/execute", request.model_dump(mode="json"))
        # Runtime may return a completed result or a poll URL for long commands.
        while data.get("status") in {"queued", "running"} and data.get("poll_path"):
            await asyncio.sleep(min(float(data.get("poll_after_seconds", 1)), 5))
            data = await self._request("GET", str(data["poll_path"]))
        return ExecutionResult.model_validate(data["result"])

    async def upload(self, handle: SandboxHandle, source: Path, destination: str) -> None:
        raise ProviderConfigurationError("HTTP uploads require a configured object-storage reference; inline file values are forbidden", provider=self.name)

    async def download_artifacts(self, handle: SandboxHandle, request: ArtifactRequest) -> list[ArtifactResult]:
        data = await self._request("POST", f"/v1/sandboxes/{handle.external_id or handle.sandbox_id}/artifacts", request.model_dump(mode="json", exclude={"destination"}))
        return [ArtifactResult.model_validate(item) for item in data.get("artifacts", [])]

    async def snapshot(self, handle: SandboxHandle, request: SnapshotRequest) -> SnapshotRef:
        data = await self._request("POST", f"/v1/sandboxes/{handle.external_id or handle.sandbox_id}/snapshots", request.model_dump(mode="json"), idempotency_key=f"snapshot:{handle.sandbox_id}:{request.name}")
        return SnapshotRef.model_validate(data["snapshot"])

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec) -> SandboxHandle:
        data = await self._request("POST", f"/v1/snapshots/{snapshot.snapshot_id}/restore", {"spec": spec.model_dump(mode="json", exclude={"secrets"}), "secret_references": [item.model_dump(mode="json") for item in spec.secrets]}, idempotency_key=f"restore:{snapshot.snapshot_id}:{spec.task_id}")
        return SandboxHandle.model_validate(data["sandbox"])

    async def suspend(self, handle: SandboxHandle) -> SandboxHandle:
        data = await self._request("POST", f"/v1/sandboxes/{handle.external_id or handle.sandbox_id}/suspend", {})
        return SandboxHandle.model_validate(data["sandbox"])

    async def resume(self, handle: SandboxHandle) -> SandboxHandle:
        data = await self._request("POST", f"/v1/sandboxes/{handle.external_id or handle.sandbox_id}/resume", {})
        return SandboxHandle.model_validate(data["sandbox"])

    async def terminate(self, handle: SandboxHandle) -> None:
        await self._request("DELETE", f"/v1/sandboxes/{handle.external_id or handle.sandbox_id}", {}, idempotency_key=f"terminate:{handle.sandbox_id}")

    async def cleanup(self, handle: SandboxHandle) -> None:
        await self._request("DELETE", f"/v1/sandboxes/{handle.external_id or handle.sandbox_id}", {}, idempotency_key=f"cleanup:{handle.sandbox_id}")

    async def healthcheck(self) -> ProviderHealth:
        try:
            data = await self._request("GET", "/v1/health")
            return ProviderHealth(provider=self.name, available=bool(data.get("available", True)), message=str(data.get("message") or ""))
        except Exception as exc:
            return ProviderHealth(provider=self.name, available=False, message=str(exc))
