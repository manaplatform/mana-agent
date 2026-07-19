"""Memory configuration resolution and secret references."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, replace
from typing import Any

from mana_agent.config.user_config import load_effective_settings
from mana_agent.memory.errors import MemoryConfigurationError, MemoryDependencyError

KEYRING_SERVICE = "mana-agent-memory"


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
        allowed = {"internal": {"mana"}, "external": {"mem0"}}
        if self.mode not in allowed:
            raise MemoryConfigurationError("Memory mode must be 'internal' or 'external'.")
        if self.provider not in allowed[self.mode]:
            raise MemoryConfigurationError(
                f"Memory provider {self.provider!r} is not valid for {self.mode!r} mode."
            )
        if self.mode == "external" and not (self.api_key or self.secret_ref):
            raise MemoryConfigurationError("Mem0 requires MEM0_API_KEY or MANA_MEMORY_SECRET_REF.")
        if self.fallback_to_internal:
            raise MemoryConfigurationError(
                "External-to-internal fallback is not implemented; no fallback action was executed."
            )
        return self

    @classmethod
    def load(cls, values: dict[str, Any] | None = None) -> "MemoryConfig":
        raw = load_effective_settings(include_env=True) if values is None else values
        mode = str(os.getenv("MANA_MEMORY_MODE", raw.get("MANA_MEMORY_MODE", "internal"))).strip().lower()
        default_provider = "mana" if mode == "internal" else "mem0"
        provider = str(
            os.getenv("MANA_MEMORY_PROVIDER", raw.get("MANA_MEMORY_PROVIDER", default_provider))
        ).strip().lower()
        fallback = str(
            os.getenv(
                "MANA_MEMORY_FALLBACK_TO_INTERNAL",
                raw.get("MANA_MEMORY_FALLBACK_TO_INTERNAL", False),
            )
        ).lower() in {"1", "true", "yes", "on"}
        config = cls(
            mode=mode,
            provider=provider,
            fallback_to_internal=fallback,
            api_key=str(os.getenv("MEM0_API_KEY", raw.get("MEM0_API_KEY", "")) or "").strip(),
            secret_ref=str(raw.get("MANA_MEMORY_SECRET_REF", "") or "").strip(),
            org_id=str(os.getenv("MEM0_ORG_ID", raw.get("MEM0_ORG_ID", "")) or "").strip(),
            project_id=str(os.getenv("MEM0_PROJECT_ID", raw.get("MEM0_PROJECT_ID", "")) or "").strip(),
            base_url=str(os.getenv("MEM0_BASE_URL", raw.get("MEM0_BASE_URL", "")) or "").strip(),
            timeout_seconds=float(
                os.getenv(
                    "MANA_MEMORY_TIMEOUT_SECONDS",
                    raw.get("MANA_MEMORY_TIMEOUT_SECONDS", 15),
                )
                or 15
            ),
        )
        if config.mode == "external" and config.secret_ref and not config.api_key:
            config = replace(config, api_key=MemorySecretStore().get(config.secret_ref))
            if not config.api_key:
                raise MemoryConfigurationError("The configured Mem0 secret reference is empty.")
        return config.validate()


class MemorySecretStore:
    """OS-keyring storage; normal configuration contains only the reference."""

    @staticmethod
    def _keyring():
        try:
            import keyring
        except ImportError as exc:
            raise MemoryDependencyError(
                "Mem0 credentials require the optional memory dependency: pip install 'mana-agent[mem0]'."
            ) from exc
        return keyring

    def set(self, api_key: str, reference: str = "") -> str:
        ref = reference or f"mem0:{uuid.uuid4().hex}"
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
