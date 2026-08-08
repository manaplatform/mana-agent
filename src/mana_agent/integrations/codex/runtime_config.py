"""Validated per-run provider configuration for managed Codex processes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from mana_agent.config.provider_registry import CodexTransport
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.exceptions import CodexConfigurationError
from mana_agent.integrations.codex.responses_bridge import (
    BridgeUpstreamConfig,
    ResponsesBridgeHandle,
    ResponsesBridgeManager,
)
from mana_agent.integrations.codex.responses_bridge.lifecycle import BRIDGE_MANAGER

RUNTIME_PROVIDER_ID = "mana_runtime"
RUNTIME_API_KEY_ENV = "MANA_CODEX_API_KEY"


def resolve_codex_base_url(value: str) -> str:
    """Return a normalized Responses API base URL without guessing endpoints."""

    raw = str(value or "").strip()
    if not raw:
        raise CodexConfigurationError("The selected provider has no API base URL configured.")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise CodexConfigurationError(f"The selected provider has an invalid API base URL: {raw!r}.")
    if parsed.query or parsed.fragment:
        raise CodexConfigurationError(
            "The selected provider API base URL must not contain a query string or fragment. "
            "Configure query parameters separately."
        )
    segments = [part for part in parsed.path.split("/") if part]
    if len(segments) >= 2 and segments[-2:] == ["chat", "completions"]:
        raise CodexConfigurationError(
            "The selected provider exposes a Chat Completions endpoint, not a Responses-compatible API."
        )
    if segments and segments[-1] == "responses":
        segments.pop()
    path = "/" + "/".join(segments) if segments else ""
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))


@dataclass(frozen=True, slots=True)
class CodexRuntimeConfig:
    provider: str
    provider_display_name: str
    model: str
    api_key: str = field(repr=False)
    base_url: str
    approval_policy: str
    sandbox_mode: str
    http_headers: dict[str, str] = field(default_factory=dict)
    env_http_headers: dict[str, str] = field(default_factory=dict)
    query_params: dict[str, str] = field(default_factory=dict)
    request_max_retries: int = 4
    stream_max_retries: int = 5
    stream_idle_timeout_ms: int = 300_000
    transport: CodexTransport = CodexTransport.DIRECT_RESPONSES
    bridge: ResponsesBridgeHandle | None = field(default=None, repr=False, compare=False)
    # Accounting always attributes usage to the real inference provider/model.
    accounting_provider: str = ""
    accounting_model: str = ""

    @property
    def credential_fingerprint(self) -> str:
        # PBKDF2 is used so credential material is not hashed with a general-purpose
        # digest alone. This value is a non-secret fingerprint for logs/identity only.
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            self.api_key.encode("utf-8"),
            b"mana-agent-codex-credential-v1",
            120_000,
            dklen=16,
        )
        return "pbkdf2-sha256:" + digest.hex()[:16]

    @property
    def fingerprint(self) -> str:
        material = json.dumps(
            {
                "provider": self.provider,
                "base_url": self.base_url,
                "model": self.model,
                "http_headers": self.http_headers,
                "env_http_headers": self.env_http_headers,
                "query_params": self.query_params,
                "transport": self.transport.value,
                "credential": self.credential_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            material.encode("utf-8"),
            b"mana-agent-codex-runtime-v1",
            60_000,
            dklen=16,
        )
        return "pbkdf2-sha256:" + digest.hex()[:16]

    def to_toml(self) -> str:
        lines = [
            f"model = {_toml_string(self.model)}",
            f"model_provider = {_toml_string(RUNTIME_PROVIDER_ID)}",
            'forced_login_method = "api"',
            f"approval_policy = {_toml_string(self.approval_policy)}",
            f"sandbox_mode = {_toml_string(self.sandbox_mode)}",
            "",
            f"[model_providers.{RUNTIME_PROVIDER_ID}]",
            f"name = {_toml_string('Mana-Agent runtime provider')}",
            f"base_url = {_toml_string(self.base_url)}",
            f"env_key = {_toml_string(RUNTIME_API_KEY_ENV)}",
            # Current Codex requires Responses; the bridge presents that API locally.
            'wire_api = "responses"',
            f"request_max_retries = {self.request_max_retries}",
            f"stream_max_retries = {self.stream_max_retries}",
            f"stream_idle_timeout_ms = {self.stream_idle_timeout_ms}",
        ]
        if self.http_headers:
            lines.append(f"http_headers = {_toml_inline_table(self.http_headers)}")
        if self.env_http_headers:
            lines.append(f"env_http_headers = {_toml_inline_table(self.env_http_headers)}")
        if self.query_params:
            lines.append(f"query_params = {_toml_inline_table(self.query_params)}")
        return "\n".join(lines) + "\n"


class CodexRuntimeConfigBuilder:
    def __init__(self, bridge_manager: ResponsesBridgeManager | None = None) -> None:
        self._bridge_manager = bridge_manager or BRIDGE_MANAGER

    @staticmethod
    def build(
        settings: CodexSettings,
        *,
        sandbox_mode: str,
        bridge_manager: ResponsesBridgeManager | None = None,
    ) -> CodexRuntimeConfig:
        return CodexRuntimeConfigBuilder(bridge_manager=bridge_manager)._build(
            settings, sandbox_mode=sandbox_mode
        )

    def _build(self, settings: CodexSettings, *, sandbox_mode: str) -> CodexRuntimeConfig:
        provider = str(settings.provider or "").strip()
        model = str(settings.model or "").strip()
        api_key = str(settings.api_key or "")
        if not provider:
            raise CodexConfigurationError("No Mana provider was selected for the Codex run.")
        if not api_key:
            raise CodexConfigurationError(
                f"{settings.provider_display_name or provider} authentication is not configured. "
                "No Codex process was started."
            )
        if not model:
            raise CodexConfigurationError("No model was selected for the Codex run.")

        transport = settings.codex_transport
        # Explicit transport wins. When unset, only native Responses providers
        # auto-select DIRECT_RESPONSES. Chat Completions hosts must declare
        # RESPONSES_BRIDGE via the provider registry (e.g. NVIDIA).
        if transport is CodexTransport.UNSUPPORTED and settings.supports_responses_api:
            transport = CodexTransport.DIRECT_RESPONSES
        if transport is CodexTransport.UNSUPPORTED:
            raise CodexConfigurationError(
                "The selected provider cannot be used by Codex because it does not expose a "
                "Responses-compatible API. Select a compatible provider or a Chat Completions "
                "host that Mana can serve through the Responses bridge (for example NVIDIA NIM)."
            )

        bridge: ResponsesBridgeHandle | None = None
        codex_api_key = api_key
        codex_base_url: str
        http_headers = _validated_safe_values(settings.http_headers, kind="HTTP header")
        query_params = _validated_safe_values(settings.query_params, kind="query parameter")

        if transport is CodexTransport.DIRECT_RESPONSES:
            if not settings.supports_responses_api:
                raise CodexConfigurationError(
                    "The selected provider is marked for direct Responses access but does not "
                    "declare native Responses support."
                )
            codex_base_url = resolve_codex_base_url(settings.base_url)
        else:
            # RESPONSES_BRIDGE: Codex talks only to the local bridge; the real
            # upstream credential never enters the Codex config or child argv.
            #
            # Retry ownership for the bridge path:
            # * Bridge transport: exactly one upstream attempt per Codex request
            #   (no nested HTTP retries that multiply with Codex).
            # * Codex stream reconnect: only after HTTP 200 SSE accepted.
            # * Invalid request / auth / model retired: no retry at any layer.
            # * Task-level recovery: Resilient Execution Supervisor only.
            upstream = BridgeUpstreamConfig(
                provider=provider,
                display_name=settings.provider_display_name or provider,
                api_key=api_key,
                base_url=str(settings.base_url or "").rstrip("/"),
                model=model,
                headers=dict(settings.http_headers or {}),
                request_overrides=dict(settings.model_request_overrides or {}),
                transport_max_attempts=1,
            )
            if not upstream.base_url:
                raise CodexConfigurationError(
                    f"{settings.provider_display_name or provider} has no API base URL configured."
                )
            try:
                bridge = self._bridge_manager.start(upstream)
            except Exception as exc:
                raise CodexConfigurationError(
                    "Mana Responses bridge failed to start for the selected Chat Completions "
                    f"provider ({settings.provider_display_name or provider}). "
                    f"Reason: {type(exc).__name__}."
                ) from exc
            try:
                bridge.healthcheck()
            except Exception as exc:
                bridge.release()
                raise CodexConfigurationError(
                    "Mana Responses bridge health check failed. No Codex process was started."
                ) from exc
            codex_base_url = resolve_codex_base_url(bridge.base_url)
            codex_api_key = bridge.temporary_api_key
            # Codex must not forward provider-specific attribution headers to the bridge.
            http_headers = {}
            query_params = {}

        return CodexRuntimeConfig(
            provider=provider,
            provider_display_name=settings.provider_display_name or provider,
            model=model,
            api_key=codex_api_key,
            base_url=codex_base_url,
            approval_policy=settings.approval_policy,
            sandbox_mode=sandbox_mode,
            http_headers=http_headers,
            env_http_headers=(
                dict(settings.env_http_headers)
                if transport is CodexTransport.DIRECT_RESPONSES
                else {}
            ),
            query_params=query_params,
            # For the bridge path, keep Codex request retries for genuine
            # loopback/connect failures only. Non-retryable upstream HTTP
            # statuses are returned as proper Responses errors (not stream
            # disconnects), so Codex must not reconnect on HTTP 400/401/410.
            request_max_retries=(
                min(int(settings.request_max_retries), 2)
                if transport is CodexTransport.RESPONSES_BRIDGE
                else settings.request_max_retries
            ),
            stream_max_retries=settings.stream_max_retries,
            stream_idle_timeout_ms=settings.stream_idle_timeout_ms,
            transport=transport,
            bridge=bridge,
            accounting_provider=provider,
            accounting_model=model,
        )


def _toml_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _toml_inline_table(values: dict[str, str]) -> str:
    return "{ " + ", ".join(
        f"{_toml_string(key)} = {_toml_string(value)}" for key, value in sorted(values.items())
    ) + " }"


def _validated_safe_values(values: dict[str, str], *, kind: str) -> dict[str, str]:
    blocked_names = {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "api_key",
        "apikey",
        "token",
        "access_token",
        "secret",
        "password",
        "credential",
    }
    validated: dict[str, str] = {}
    for raw_name, raw_value in values.items():
        name = str(raw_name).strip()
        normalized = name.lower().replace("-", "_")
        if not name or normalized in {item.replace("-", "_") for item in blocked_names}:
            raise CodexConfigurationError(
                f"Unsafe or empty {kind} name {name!r}; secrets must use child environment variables."
            )
        validated[name] = str(raw_value)
    return validated


__all__ = [
    "CodexRuntimeConfig",
    "CodexRuntimeConfigBuilder",
    "RUNTIME_API_KEY_ENV",
    "RUNTIME_PROVIDER_ID",
    "resolve_codex_base_url",
]
