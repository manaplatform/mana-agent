"""Memory configuration resolution and secret references."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from mana_agent.config.user_config import (
    delete_managed_user_secret,
    load_effective_settings,
    load_user_secrets,
    save_managed_user_secret,
)
from mana_agent.memory.errors import MemoryConfigurationError

KEYRING_SERVICE = "mana-agent-memory"
MANA_SECRETS_REFERENCE_PREFIX = "mana-secrets:"
_EXTERNAL_PROVIDER_ENV = {
    "mem0": "MEM0_API_KEY",
    "supermemory": "SUPERMEMORY_API_KEY",
}
_EXTERNAL_PROVIDER_SECRET_PREFIX = {
    "mem0": "mem0",
    "supermemory": "supermemory",
}


def _bool(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class CapsuleRetentionConfig:
    private_days: int = 7
    parent_child_days: int = 30
    team_days: int = 90
    project_days: int = 180
    organisation_days: int = 365

    def validate(self) -> "CapsuleRetentionConfig":
        if any(value < 1 for value in (
            self.private_days,
            self.parent_child_days,
            self.team_days,
            self.project_days,
            self.organisation_days,
        )):
            raise MemoryConfigurationError("Capsule retention periods must be positive days.")
        return self


@dataclass(frozen=True, slots=True)
class CapsuleConfig:
    enabled: bool = True
    default_max_capsules: int = 12
    default_max_tokens: int = 4000
    shared_writes_require_review: bool = True
    organisation_scope_enabled: bool = False
    user_scope_enabled: bool = True
    record_access_events: bool = True
    quarantine_prompt_injection: bool = True
    retention: CapsuleRetentionConfig = field(default_factory=CapsuleRetentionConfig)

    def validate(self) -> "CapsuleConfig":
        if self.default_max_capsules < 1 or self.default_max_capsules > 100:
            raise MemoryConfigurationError("Capsule retrieval limit must be between 1 and 100.")
        if self.default_max_tokens < 1:
            raise MemoryConfigurationError("Capsule retrieval token budget must be positive.")
        if not self.shared_writes_require_review:
            raise MemoryConfigurationError("Shared capsule writes without review are not supported.")
        self.retention.validate()
        return self

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "CapsuleConfig":
        prefix = "MANA_MEMORY_CAPSULES_"
        value = lambda name, default="": raw.get(prefix + name, os.getenv(prefix + name, default))
        retention = CapsuleRetentionConfig(
            private_days=int(value("RETENTION_PRIVATE_DAYS", 7)),
            parent_child_days=int(value("RETENTION_PARENT_CHILD_DAYS", 30)),
            team_days=int(value("RETENTION_TEAM_DAYS", 90)),
            project_days=int(value("RETENTION_PROJECT_DAYS", 180)),
            organisation_days=int(value("RETENTION_ORGANISATION_DAYS", 365)),
        )
        return cls(
            enabled=_bool(value("ENABLED", True), True),
            default_max_capsules=int(value("DEFAULT_MAX_CAPSULES", 12)),
            default_max_tokens=int(value("DEFAULT_MAX_TOKENS", 4000)),
            shared_writes_require_review=_bool(value("SHARED_WRITES_REQUIRE_REVIEW", True), True),
            organisation_scope_enabled=_bool(value("ORGANISATION_SCOPE_ENABLED", False), False),
            user_scope_enabled=_bool(value("USER_SCOPE_ENABLED", True), True),
            record_access_events=_bool(value("RECORD_ACCESS_EVENTS", True), True),
            quarantine_prompt_injection=_bool(value("QUARANTINE_PROMPT_INJECTION", True), True),
            retention=retention,
        ).validate()


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    mode: str = "internal"
    provider: str = "mana"
    fallback_to_internal: bool = False
    api_key: str = ""
    secret_ref: str = ""
    org_id: str = ""
    project_id: str = ""
    base_url: str = ""
    timeout_seconds: float = 15.0
    capsules: CapsuleConfig = field(default_factory=CapsuleConfig)

    def validate(self) -> "MemoryConfig":
        allowed = {"internal": {"mana"}, "external": {"mem0", "supermemory"}}
        if self.mode not in allowed:
            raise MemoryConfigurationError("Memory mode must be 'internal' or 'external'.")
        if self.provider not in allowed[self.mode]:
            raise MemoryConfigurationError(
                f"Memory provider {self.provider!r} is not valid for {self.mode!r} mode."
            )
        if self.mode == "external" and not (self.api_key or self.secret_ref):
            env_var = _EXTERNAL_PROVIDER_ENV.get(self.provider, "provider API key")
            raise MemoryConfigurationError(
                f"{self.provider.capitalize()} requires {env_var} or MANA_MEMORY_SECRET_REF."
            )
        if self.fallback_to_internal:
            raise MemoryConfigurationError(
                "External-to-internal fallback is not implemented; no fallback action was executed."
            )
        self.capsules.validate()
        return self

    @classmethod
    def load(cls, values: dict[str, Any] | None = None) -> "MemoryConfig":
        raw = load_effective_settings(include_env=True) if values is None else values
        mode = str(raw.get("MANA_MEMORY_MODE") or os.getenv("MANA_MEMORY_MODE", "internal")).strip().lower()
        default_provider = "mana" if mode == "internal" else "mem0"
        provider = str(raw.get("MANA_MEMORY_PROVIDER") or os.getenv("MANA_MEMORY_PROVIDER", default_provider)).strip().lower()
        fallback = str(
            raw.get("MANA_MEMORY_FALLBACK_TO_INTERNAL")
            if raw.get("MANA_MEMORY_FALLBACK_TO_INTERNAL", "") != ""
            else os.getenv("MANA_MEMORY_FALLBACK_TO_INTERNAL", False)
        ).lower() in {"1", "true", "yes", "on"}
        provider_api_env = _EXTERNAL_PROVIDER_ENV.get(provider, "")
        explicit_api_key = str(raw.get(provider_api_env, "") or "").strip() if provider_api_env else ""
        secret_ref = str(raw.get("MANA_MEMORY_SECRET_REF", "") or "").strip()
        secret_api_key = MemorySecretStore().get(secret_ref) if (mode == "external" and secret_ref) else ""
        api_key = explicit_api_key or secret_api_key or str(os.getenv(provider_api_env, "") or "").strip()
        config = cls(
            mode=mode,
            provider=provider,
            fallback_to_internal=fallback,
            api_key=api_key,
            secret_ref=secret_ref,
            org_id=str(raw.get("MEM0_ORG_ID") or os.getenv("MEM0_ORG_ID", "")).strip(),
            project_id=str(raw.get("MEM0_PROJECT_ID") or os.getenv("MEM0_PROJECT_ID", "")).strip(),
            base_url=str(
                raw.get("MEM0_BASE_URL")
                or raw.get("SUPERMEMORY_BASE_URL")
                or os.getenv("MEM0_BASE_URL", "")
                or os.getenv("SUPERMEMORY_BASE_URL", "")
            ).strip(),
            timeout_seconds=float(
                raw.get("MANA_MEMORY_TIMEOUT_SECONDS")
                or os.getenv("MANA_MEMORY_TIMEOUT_SECONDS", 15)
                or 15
            ),
            capsules=CapsuleConfig.load(raw),
        )
        if config.mode == "external" and config.secret_ref and not config.api_key:
            raise MemoryConfigurationError(
                f"The configured {config.provider.capitalize()} secret reference is empty."
            )
        return config.validate()


class MemorySecretStore:
    """Secure credential storage with explicit keyring/Mana-secret references."""

    @staticmethod
    def _recommended_keyring():
        try:
            import keyring
        except ImportError:
            return None
        try:
            backend = keyring.get_keyring()
            priority = float(getattr(backend, "priority", 0) or 0)
        except Exception:
            return None
        return keyring if priority > 0 else None

    @staticmethod
    def _secret_name(provider: str) -> str:
        try:
            return _EXTERNAL_PROVIDER_ENV[provider]
        except KeyError as exc:
            raise MemoryConfigurationError(
                f"Memory provider {provider!r} has no registered credential name."
            ) from exc

    def set(self, api_key: str, reference: str = "", *, provider: str = "mem0") -> str:
        prefix = _EXTERNAL_PROVIDER_SECRET_PREFIX.get(provider, provider or "memory")
        ref = reference or f"{prefix}:{uuid.uuid4().hex}"
        if reference.startswith(MANA_SECRETS_REFERENCE_PREFIX):
            secret_name = self._secret_name(provider)
            save_managed_user_secret(secret_name, api_key)
            return f"{MANA_SECRETS_REFERENCE_PREFIX}{secret_name}"
        keyring = self._recommended_keyring()
        if keyring is not None:
            keyring.set_password(KEYRING_SERVICE, ref, api_key)
            return ref
        secret_name = self._secret_name(provider)
        save_managed_user_secret(secret_name, api_key)
        return f"{MANA_SECRETS_REFERENCE_PREFIX}{secret_name}"

    def get(self, reference: str) -> str:
        if reference.startswith(MANA_SECRETS_REFERENCE_PREFIX):
            secret_name = reference.removeprefix(MANA_SECRETS_REFERENCE_PREFIX)
            if secret_name not in set(_EXTERNAL_PROVIDER_ENV.values()):
                raise MemoryConfigurationError("The configured Mana memory secret reference is invalid.")
            return str(load_user_secrets().get(secret_name) or "")
        try:
            keyring = self._recommended_keyring()
            if keyring is None:
                raise MemoryConfigurationError(
                    "The configured memory credential uses an OS keyring, but no recommended keyring backend is available."
                )
            return str(keyring.get_password(KEYRING_SERVICE, reference) or "")
        except Exception as exc:
            if isinstance(exc, MemoryConfigurationError):
                raise
            raise MemoryConfigurationError("The configured Mem0 secret reference could not be read.") from exc

    def delete(self, reference: str) -> None:
        if reference.startswith(MANA_SECRETS_REFERENCE_PREFIX):
            delete_managed_user_secret(reference.removeprefix(MANA_SECRETS_REFERENCE_PREFIX))
            return
        keyring = self._recommended_keyring()
        if keyring is None:
            raise MemoryConfigurationError(
                "The configured memory credential uses an OS keyring, but no recommended keyring backend is available."
            )
        keyring.delete_password(KEYRING_SERVICE, reference)
