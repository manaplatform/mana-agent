"""Tool compatibility resolver and model configuration validator for Mana-Agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from mana_agent.config.model_discovery import (
    GLOBAL_CAPABILITY_REGISTRY,
    ModelCapabilityProfile,
    ModelCapabilityRegistry,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolCompatibilityReport:
    """Detailed report on tool compatibility for a given model profile."""

    model_id: str
    provider: str
    compatible_tools: list[str] = field(default_factory=list)
    incompatible_tools: list[tuple[str, str]] = field(default_factory=list)  # (tool_name, reason)
    all_compatible: bool = True
    requires_batch: bool = False
    requires_parallel: bool = False
    batch_restricted_tools: list[str] = field(default_factory=list)

    @property
    def has_incompatibilities(self) -> bool:
        return bool(self.incompatible_tools)


@dataclass
class ValidationResult:
    """Validation outcome for a model configuration and execution intent."""

    valid: bool
    model_id: str
    provider: str
    mode: str = "auto"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    resolved_profile: ModelCapabilityProfile | None = None

    @property
    def ok(self) -> bool:
        return self.valid and not self.errors


class ToolCompatibilityResolver:
    """Resolves whether a model candidate supports requested agent and coding tools."""

    def __init__(self, registry: ModelCapabilityRegistry | None = None) -> None:
        self.registry = registry or GLOBAL_CAPABILITY_REGISTRY

    def resolve(
        self,
        model_id: str,
        tools: Sequence[str] | Sequence[dict[str, Any]],
        *,
        provider: str = "openai",
        execution_mode: str = "realtime",
    ) -> ToolCompatibilityReport:
        profile = self.registry.get(provider, model_id)
        if profile is None:
            from mana_agent.config.model_discovery import ModelMetadataResolver

            resolver = ModelMetadataResolver(self.registry)
            profile = resolver.resolve(model_id, provider=provider)

        tool_names: list[str] = []
        for t in tools:
            if isinstance(t, str):
                tool_names.append(t)
            elif isinstance(t, dict):
                name = t.get("name") or (t.get("function", {}) if isinstance(t.get("function"), dict) else {}).get("name")
                if name:
                    tool_names.append(str(name))

        compatible: list[str] = []
        incompatible: list[tuple[str, str]] = []
        batch_restricted: list[str] = []
        requires_batch = False
        requires_parallel = False

        for tool in tool_names:
            t_lower = tool.lower().replace("-", "_")

            # 1. Batch API tools
            if t_lower in {"batch_create_job", "batch_submit_requests", "batch_get_status", "batch_check_status",
                           "batch_list_jobs", "batch_cancel_job", "batch_download_results", "batch_retrieve_results"}:
                requires_batch = True
                # This is a client-side function. The target request's model
                # and transport are validated when the selected tool runs.
                # Native batch file tools never imply official Batch access.

            # 3. Computer use
            if "computer" in t_lower:
                if not profile.computer_use:
                    incompatible.append(
                        (tool, f"Model '{profile.model_id}' does not have computer-use capability enabled.")
                    )
                else:
                    compatible.append(tool)
                continue

            # 4. Image generation
            if "image_gen" in t_lower or "dall_e" in t_lower:
                if not profile.image_generation:
                    incompatible.append(
                        (tool, f"Model '{profile.model_id}' does not support direct image generation.")
                    )
                else:
                    compatible.append(tool)
                continue

            # 5. Programmatic tool calling
            if t_lower in {"programmatic_tool_calling", "program"}:
                if not profile.programmatic_tool_calling:
                    incompatible.append(
                        (tool, f"Model '{profile.model_id}' does not support programmatic tool calling.")
                    )
                else:
                    compatible.append(tool)
                continue

            # 6. Standard function calling tools
            if not profile.function_calling:
                incompatible.append(
                    (tool, f"Model '{profile.model_id}' does not support tool calling / function calling.")
                )
            else:
                compatible.append(tool)

        all_ok = len(incompatible) == 0

        return ToolCompatibilityReport(
            model_id=profile.model_id,
            provider=profile.provider,
            compatible_tools=compatible,
            incompatible_tools=incompatible,
            all_compatible=all_ok,
            requires_batch=requires_batch,
            requires_parallel=requires_parallel,
            batch_restricted_tools=batch_restricted,
        )


class ModelConfigurationValidator:
    """Validates model configuration against requested execution mode, tokens, and credentials."""

    def __init__(self, registry: ModelCapabilityRegistry | None = None) -> None:
        self.registry = registry or GLOBAL_CAPABILITY_REGISTRY

    def validate(
        self,
        model_id: str,
        *,
        provider: str = "openai",
        mode: str = "auto",
        api_key: str = "",
        requested_tokens: int | None = None,
        reasoning_effort: str | None = None,
        endpoint: str = "/v1/responses",
    ) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        clean_id = str(model_id or "").strip()
        clean_provider = str(provider or "openai").strip().lower()

        if not clean_id:
            return ValidationResult(
                valid=False,
                model_id="",
                provider=clean_provider,
                mode=mode,
                errors=["Model ID cannot be empty."],
            )

        profile = self.registry.get(clean_provider, clean_id)
        if profile is None:
            from mana_agent.config.model_discovery import ModelMetadataResolver

            resolver = ModelMetadataResolver(self.registry)
            profile = resolver.resolve(clean_id, provider=clean_provider)

        # 1. Mode and batch compatibility
        if mode.lower() == "batch":
            if profile.supports_batch is False:
                errors.append(
                    f"Model '{profile.model_id}' is explicitly unsupported for OpenAI Batch API."
                )
            elif profile.supports_batch is None:
                warnings.append(
                    f"Model '{profile.model_id}' has unknown Batch API compatibility status."
                )

            if endpoint not in profile.supported_batch_endpoints:
                errors.append(
                    f"Endpoint '{endpoint}' is not supported for Batch API by model '{profile.model_id}'. "
                    f"Supported endpoints: {profile.supported_batch_endpoints}"
                )

        # 2. Context window limits
        if requested_tokens and profile.context_window:
            if requested_tokens > profile.context_window:
                errors.append(
                    f"Requested tokens ({requested_tokens}) exceed model context window ({profile.context_window})."
                )

        # 3. Reasoning effort
        if reasoning_effort and profile.reasoning_effort_levels:
            if reasoning_effort.lower() not in profile.reasoning_effort_levels:
                warnings.append(
                    f"Reasoning effort '{reasoning_effort}' is not in known levels {profile.reasoning_effort_levels} for model '{clean_id}'."
                )

        return ValidationResult(
            valid=len(errors) == 0,
            model_id=clean_id,
            provider=clean_provider,
            mode=mode,
            errors=errors,
            warnings=warnings,
            resolved_profile=profile,
        )


__all__ = [
    "ModelConfigurationValidator",
    "ToolCompatibilityReport",
    "ToolCompatibilityResolver",
    "ValidationResult",
]
