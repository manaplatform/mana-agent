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

    def set_models(self, *, provider: str, high: str, coding: str, fast: str) -> None:
        configured = self.values.get("MANA_CONFIGURED_PROVIDERS", [])
        if not isinstance(configured, list):
            configured = [str(configured)] if str(configured).strip() else []
        updates: dict[str, Any] = {
            "MANA_AI_PROVIDER": provider,
            "MANA_CONFIGURED_PROVIDERS": sorted(set([provider, *configured])),
        }
        # Model selectors may be empty until a provider catalog has been tested or selected.
        # Keep the last valid model configuration untouched in that transient
        # state instead of passing empty IDs to qualify_model_id().
        high_clean = str(high or "").strip()
        coding_clean = str(coding or "").strip() or high_clean
        fast_clean = str(fast or "").strip() or high_clean

        has_valid_high = bool(high_clean and high_clean != "<Select.BLANK>")

        if has_valid_high:
            updates.update(
                {
                    "MANA_PRIMARY_MODEL": qualify_model_id(provider, high_clean),
                    "OPENAI_CHAT_MODEL": high_clean,
                    "LLM_MODEL": high_clean,
                    "OPENAI_CODING_PLANNER_MODEL": coding_clean,
                    "OPENAI_TOOL_WORKER_MODEL": fast_clean,
                    "MODEL_LEVEL_3_HIGH_REASONING": high_clean,
                    "MODEL_LEVEL_2_CODING": coding_clean,
                    "MODEL_LEVEL_1_FAST_TOOL": fast_clean,
                }
            )
        self.values.update(updates)

    def save(self) -> None:
        def provider_identity(values: dict[str, Any]) -> tuple[Any, Any, Any]:
            provider = str(values.get("MANA_AI_PROVIDER") or "openai")
            if provider == "openrouter":
                return (
                    provider,
                    values.get("OPENROUTER_BASE_URL"),
                    values.get("OPENROUTER_API_KEY"),
                )
            if provider == "nvidia":
                return (
                    provider,
                    values.get("NVIDIA_BASE_URL"),
                    values.get("NVIDIA_API_KEY"),
                )
            if provider == "anthropic":
                return (
                    provider,
                    values.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com",
                    values.get("ANTHROPIC_API_KEY"),
                )
            if provider in {"xai", "x-ai", "grok"}:
                return (
                    "xai",
                    values.get("XAI_BASE_URL") or "https://api.x.ai/v1",
                    values.get("XAI_API_KEY"),
                )
            return provider, values.get("OPENAI_BASE_URL"), values.get("OPENAI_API_KEY")
        old_identity = (
            *provider_identity(self.original),
        )
        provider = str(self.values.get("MANA_MEMORY_PROVIDER") or "mana").strip().lower()
        transient_memory_key = str(
            self.values.get("SUPERMEMORY_API_KEY")
            or self.values.get("MEM0_API_KEY")
            or ""
        )
        values = {
            key: value
            for key, value in self.values.items()
            if key not in {"MEM0_API_KEY", "SUPERMEMORY_API_KEY"}
        }
        if transient_memory_key:
            from mana_agent.memory.config import MemorySecretStore

            values["MANA_MEMORY_SECRET_REF"] = MemorySecretStore().set(
                transient_memory_key,
                str(values.get("MANA_MEMORY_SECRET_REF") or ""),
                provider=provider,
            )
        if str(values.get("MANA_MEMORY_MODE") or "internal") == "internal":
            values["MANA_MEMORY_PROVIDER"] = "mana"
        values.pop("MANA_EMBEDDING_MODEL", None)
        values.pop("OPENAI_EMBED_MODEL", None)
        cleaned = validate_config_values(values)
        save_effective_user_config(cleaned, merge=False)
        from mana_agent.config.model_capabilities import clear_capability_cache
        clear_capability_cache()
        new_identity = provider_identity(cleaned)
        if new_identity != old_identity:
            invalidate_model_cache()
        self.original = dict(cleaned)
        self.values = dict(cleaned)
        self.removed_secrets.clear()


def runtime_model_id(value: str, *, provider: str = "openai") -> str:
    return split_qualified_model_id(value, default_provider=provider)[1]
