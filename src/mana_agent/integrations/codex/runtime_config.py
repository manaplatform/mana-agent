"""Validated per-run provider configuration for managed Codex processes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.exceptions import CodexConfigurationError

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

    @property
    def credential_fingerprint(self) -> str:
        return "sha256:" + hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()[:8]

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
                "credential": self.credential_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]

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
    @staticmethod
    def build(settings: CodexSettings, *, sandbox_mode: str) -> CodexRuntimeConfig:
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
        if not settings.supports_responses_api:
            raise CodexConfigurationError(
                "The selected provider cannot be used by Codex because it does not expose a "
                "Responses-compatible API. Select a compatible provider or configure a Mana "
                "gateway endpoint that supports the Responses API."
            )
        http_headers = _validated_safe_values(settings.http_headers, kind="HTTP header")
        query_params = _validated_safe_values(settings.query_params, kind="query parameter")
        return CodexRuntimeConfig(
            provider=provider,
            provider_display_name=settings.provider_display_name or provider,
            model=model,
            api_key=api_key,
            base_url=resolve_codex_base_url(settings.base_url),
            approval_policy=settings.approval_policy,
            sandbox_mode=sandbox_mode,
            http_headers=http_headers,
            env_http_headers=dict(settings.env_http_headers),
            query_params=query_params,
            request_max_retries=settings.request_max_retries,
            stream_max_retries=settings.stream_max_retries,
            stream_idle_timeout_ms=settings.stream_idle_timeout_ms,
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
