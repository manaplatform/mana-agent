from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mana_agent.config.provider_registry import qualify_model_id, split_qualified_model_id
from mana_agent.config.user_config import (
    SECRET_KEYS,
    invalidate_model_cache,
    load_effective_settings,
    save_effective_user_config,
    validate_config_values,
)


UNCHANGED_SECRET = "__MANA_SECRET_UNCHANGED__"


@dataclass(slots=True)
class ConfigurationDraft:
    """Mutable, non-persistent settings used by the configuration TUI."""

    original: dict[str, Any]
    values: dict[str, Any] = field(default_factory=dict)
    removed_secrets: set[str] = field(default_factory=set)

    @classmethod
    def load(cls) -> "ConfigurationDraft":
        current = load_effective_settings(include_env=False)
        return cls(original=dict(current), values=dict(current))

    @property
    def dirty(self) -> bool:
        return self.values != self.original or bool(self.removed_secrets)

    def set_secret(self, name: str, value: str) -> None:
        if name not in SECRET_KEYS:
            raise KeyError(name)
        if value == UNCHANGED_SECRET or not value:
            return
        self.values[name] = value
        self.removed_secrets.discard(name)

    def remove_secret(self, name: str) -> None:
        if name not in SECRET_KEYS:
            raise KeyError(name)
        self.values[name] = ""
        self.removed_secrets.add(name)

    def set_models(self, *, provider: str, high: str, coding: str, fast: str, embedding: str) -> None:
        configured = self.values.get("MANA_CONFIGURED_PROVIDERS", [])
        if not isinstance(configured, list):
            configured = [str(configured)] if str(configured).strip() else []
        self.values.update(
            {
                "MANA_AI_PROVIDER": provider,
                "MANA_CONFIGURED_PROVIDERS": sorted(set([provider, *configured])),
                "MANA_PRIMARY_MODEL": qualify_model_id(provider, high),
                "MANA_EMBEDDING_MODEL": qualify_model_id(provider, embedding),
                "OPENAI_CHAT_MODEL": high,
                "LLM_MODEL": high,
                "OPENAI_CODING_PLANNER_MODEL": coding,
                "OPENAI_TOOL_WORKER_MODEL": fast,
                "OPENAI_EMBED_MODEL": embedding,
                "MODEL_LEVEL_3_HIGH_REASONING": high,
                "MODEL_LEVEL_2_CODING": coding,
                "MODEL_LEVEL_1_FAST_TOOL": fast,
            }
        )

    def save(self) -> None:
        def provider_identity(values: dict[str, Any]) -> tuple[Any, Any, Any]:
            provider = str(values.get("MANA_AI_PROVIDER") or "openai")
            if provider == "openrouter":
                return provider, values.get("OPENROUTER_BASE_URL"), values.get("OPENROUTER_API_KEY")
            return provider, values.get("OPENAI_BASE_URL"), values.get("OPENAI_API_KEY")
        old_identity = (
            *provider_identity(self.original),
        )
        transient_mem0_key = str(self.values.get("MEM0_API_KEY") or "")
        values = {key: value for key, value in self.values.items() if key != "MEM0_API_KEY"}
        if transient_mem0_key:
            from mana_agent.memory.config import MemorySecretStore

            values["MANA_MEMORY_SECRET_REF"] = MemorySecretStore().set(
                transient_mem0_key,
                str(values.get("MANA_MEMORY_SECRET_REF") or ""),
            )
        if str(values.get("MANA_MEMORY_MODE") or "internal") == "internal":
            values["MANA_MEMORY_PROVIDER"] = "mana"
        cleaned = validate_config_values(values)
        save_effective_user_config(cleaned, merge=False)
        new_identity = provider_identity(cleaned)
        if new_identity != old_identity:
            invalidate_model_cache()
        self.original = dict(cleaned)
        self.values = dict(cleaned)
        self.removed_secrets.clear()


def runtime_model_id(value: str, *, provider: str = "openai") -> str:
    return split_qualified_model_id(value, default_provider=provider)[1]
