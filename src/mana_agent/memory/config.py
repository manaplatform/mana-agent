"""Memory configuration resolution and secret references."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, replace
from typing import Any

from mana_agent.config.user_config import load_effective_settings
from mana_agent.memory.errors import MemoryConfigurationError, MemoryDependencyError

KEYRING_SERVICE = "mana-agent-memory"
_EXTERNAL_PROVIDER_ENV = {
    "mem0": "MEM0_API_KEY",
    "supermemory": "SUPERMEMORY_API_KEY",
}
_EXTERNAL_PROVIDER_SECRET_PREFIX = {
    "mem0": "mem0",
    "supermemory": "supermemory",
}


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
        api_key = secret_api_key or explicit_api_key or str(os.getenv(provider_api_env, "") or "").strip()
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
        )
        if config.mode == "external" and config.secret_ref and not config.api_key:
            raise MemoryConfigurationError(
                f"The configured {config.provider.capitalize()} secret reference is empty."
            )
        return config.validate()


class MemorySecretStore:
    """OS-keyring storage; normal configuration contains only the reference."""

    @staticmethod
    def _keyring():
        try:
            import keyring
        except ImportError as exc:
            raise MemoryDependencyError(
                "External memory credentials require the optional keyring dependency: pip install 'mana-agent[mem0]' or 'mana-agent[supermemory]'."
            ) from exc
        return keyring

    def set(self, api_key: str, reference: str = "", *, provider: str = "mem0") -> str:
        prefix = _EXTERNAL_PROVIDER_SECRET_PREFIX.get(provider, provider or "memory")
        ref = reference or f"{prefix}:{uuid.uuid4().hex}"
        self._keyring().set_password(KEYRING_SERVICE, ref, api_key)
        return ref

    def get(self, reference: str) -> str:
        try:
            return str(self._keyring().get_password(KEYRING_SERVICE, reference) or "")
        except MemoryDependencyError:
            raise
        except Exception as exc:
            raise MemoryConfigurationError("The configured Mem0 secret reference could not be read.") from exc

    def delete(self, reference: str) -> None:
        self._keyring().delete_password(KEYRING_SERVICE, reference)
