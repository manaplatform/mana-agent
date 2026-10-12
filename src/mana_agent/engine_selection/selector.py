"""Automatic capability detection and engine selection for Mana-Agent operations."""

from __future__ import annotations

import logging
from typing import Any
from mana_agent.config.user_config import get_setting
from mana_agent.engine_selection.models import (
    DecisionsChoiceItem,
    DecisionsScoreLevel,
    EngineSelectionResult,
    EngineType,
    Operation,
    OperationKind,
)

logger = logging.getLogger(__name__)


class EngineSelector:
    """Deterministic capability detector and engine selection authority.

    Evaluates operations using deterministic application logic, tool schemas,
    and output contracts. Never calls an LLM to decide which engine to use.
    """

    def __init__(
        self,
        *,
        decisions_enabled: bool | None = None,
        provider: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if decisions_enabled is None:
            raw_setting = get_setting("MANA_DECISIONS_ENABLED", True)
            if isinstance(raw_setting, str):
                self.decisions_enabled = raw_setting.strip().lower() not in {"0", "false", "no", "off"}
            else:
                self.decisions_enabled = bool(raw_setting)
        else:
            self.decisions_enabled = bool(decisions_enabled)

        self.provider = str(provider or get_setting("MANA_AI_PROVIDER", "openai") or "openai").strip().lower()
        self.base_url = str(base_url or get_setting("OPENAI_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1").rstrip("/")

    def select_engine(self, operation: Operation, context: Any = None) -> EngineSelectionResult:
        """Select the appropriate execution engine for an operation."""
        # 1. Deterministic execution
        if operation.is_deterministic():
            return EngineSelectionResult(
                selected_engine=EngineType.DETERMINISTIC,
                is_decisions_eligible=False,
                reason="Operation is deterministic or has an explicit native handler; bypassing AI.",
                metadata={"phase": "deterministic_bypass"},
            )

        # 2. Check kind
        kind = operation.kind

        # Arbitrary JSON object generation
        if kind == OperationKind.STRUCTURED_OUTPUT:
            return EngineSelectionResult(
                selected_engine=EngineType.RESPONSES_STRUCTURED,
                is_decisions_eligible=False,
                reason="Arbitrary JSON structured object generation requires Responses API Structured Outputs.",
                metadata={"schema": str(operation.output_schema)},
            )

        # Tool calling with generated arguments
        if kind == OperationKind.FUNCTION_CALL:
            return EngineSelectionResult(
                selected_engine=EngineType.FUNCTION_CALLING,
                is_decisions_eligible=False,
                reason="Tool invocation requiring dynamic generated arguments requires function calling.",
                metadata={"tool_name": operation.tool_name},
            )

        # Complex reasoning & coding
        if kind == OperationKind.REASONING:
            return EngineSelectionResult(
                selected_engine=EngineType.REASONING_ENGINE,
                is_decisions_eligible=False,
                reason="Coding and complex multi-step reasoning requires the reasoning engine.",
            )

        # Text generation & summarization
        if kind == OperationKind.GENERATION:
            return EngineSelectionResult(
                selected_engine=EngineType.GENERATION_ENGINE,
                is_decisions_eligible=False,
                reason="Text generation or summarization requires the generation engine.",
            )

        # Open-ended conversation
        if kind == OperationKind.CONVERSATION:
            return EngineSelectionResult(
                selected_engine=EngineType.CONVERSATION_LLM,
                is_decisions_eligible=False,
                reason="Open-ended conversation requires the conversation LLM engine.",
            )

        # Semantic evaluation: predicate, choice, score, batch
        if kind in {OperationKind.PREDICATE, OperationKind.CHOICE, OperationKind.SCORE, OperationKind.BATCH}:
            eligible, reason = self.check_decisions_eligibility(operation, context)
            if eligible:
                if kind == OperationKind.PREDICATE:
                    engine = EngineType.DECISIONS_PREDICATE
                elif kind == OperationKind.CHOICE:
                    engine = EngineType.DECISIONS_CHOICE
                elif kind == OperationKind.SCORE:
                    engine = EngineType.DECISIONS_SCORE
                else:
                    engine = EngineType.DECISIONS_BATCH

                return EngineSelectionResult(
                    selected_engine=engine,
                    is_decisions_eligible=True,
                    reason=f"Operation natively supported by Decisions API ({engine.value}): {reason}",
                    metadata={"decisions_kind": kind.value},
                )

            # Not eligible for Decisions -> use compatible fallback engine
            return EngineSelectionResult(
                selected_engine=EngineType.COMPATIBLE_FALLBACK,
                is_decisions_eligible=False,
                reason=f"Decisions not eligible ({reason}); selecting existing compatible engine.",
                metadata={"ineligibility_reason": reason},
            )

        # Default fallback
        return EngineSelectionResult(
            selected_engine=EngineType.COMPATIBLE_FALLBACK,
            is_decisions_eligible=False,
            reason=f"Unknown or unmapped operation kind '{kind}'; selecting existing compatible engine.",
        )

    def check_decisions_eligibility(self, operation: Operation, context: Any = None) -> tuple[bool, str]:
        """Check whether all strict criteria for Decisions API execution are met.

        Returns:
            (eligible: bool, reason: str)
        """
        # 1. Decisions must be enabled
        if not self.decisions_enabled:
            return False, "Decisions API is disabled by configuration"

        # 2. Endpoint/provider support check
        # Decisions is an OpenAI API capability (/v1/decisions)
        is_openai_compatible = (
            self.provider == "openai"
            or "api.openai.com" in self.base_url
            or bool(get_setting("MANA_LLM_SUPPORTS_DECISIONS_API", False))
        )
        if not is_openai_compatible:
            return False, f"Provider '{self.provider}' with base URL '{self.base_url}' does not support Decisions API"

        # 3. Input format & image check
        # Decisions supports text, and images ONLY as base64 data URLs: data:image/...;base64,...
        # Hosted URLs (http/https) and OpenAI file IDs (file-...) are unsupported directly.
        unsupported_image_error = self._validate_evidence_formats(operation.input)
        if unsupported_image_error:
            return False, unsupported_image_error

        # 4. Kind-specific requirements
        if operation.kind == OperationKind.PREDICATE:
            if not operation.instructions and not operation.name:
                return False, "Predicate operation requires instructions or question name"
            return True, "Valid boolean semantic evaluation"

        if operation.kind == OperationKind.CHOICE:
            normalized_choices = self._normalize_choices(operation.choices)
            if len(normalized_choices) < 2:
                return False, "Choice operation requires at least two fixed choices"
            return True, f"Valid choice classification among {len(normalized_choices)} options"

        if operation.kind == OperationKind.SCORE:
            normalized_levels = self._normalize_levels(operation.levels)
            if len(normalized_levels) < 2:
                return False, "Score operation requires an explicit ordered rubric/levels with at least 2 levels"
            return True, f"Valid ordered scoring against {len(normalized_levels)} levels"

        if operation.kind == OperationKind.BATCH:
            if not operation.sub_operations:
                return False, "Batch operation requires non-empty sub_operations"
            for idx, sub_op in enumerate(operation.sub_operations):
                sub_eligible, sub_reason = self.check_decisions_eligibility(sub_op, context)
                if not sub_eligible:
                    return False, f"Batch sub-operation #{idx} ineligible: {sub_reason}"
            return True, f"Valid batched evaluations ({len(operation.sub_operations)} operations)"

        return False, f"Operation kind '{operation.kind}' does not map to Decisions API"

    def _validate_evidence_formats(self, evidence: Any) -> str | None:
        """Verify evidence formats. Hosted URLs and file IDs for images are unsupported."""
        if evidence is None:
            return None

        # Check dictionary / list structures
        if isinstance(evidence, dict):
            # Check for type: input_image or image_url
            t = evidence.get("type")
            if t in {"input_image", "image"}:
                url_val = evidence.get("image_url") or evidence.get("url") or ""
                if isinstance(url_val, dict):
                    url_val = url_val.get("url", "")
                url_str = str(url_val).strip()
                if url_str.startswith("http://") or url_str.startswith("https://"):
                    return "Decisions API does not support hosted external image URLs directly; base64 data URLs required"
                if url_str.startswith("file-"):
                    return "Decisions API does not support file IDs directly; base64 data URLs required"
                if url_str and not url_str.startswith("data:image/"):
                    return "Decisions API only supports images as base64 data:image/ URIs"

            # Check nested items
            for v in evidence.values():
                err = self._validate_evidence_formats(v)
                if err:
                    return err

        elif isinstance(evidence, list):
            for item in evidence:
                err = self._validate_evidence_formats(item)
                if err:
                    return err

        return None

    def _normalize_choices(self, raw_choices: list[Any]) -> list[DecisionsChoiceItem]:
        items: list[DecisionsChoiceItem] = []
        for c in raw_choices:
            if isinstance(c, DecisionsChoiceItem):
                items.append(c)
            elif isinstance(c, str):
                s = c.strip()
                if s:
                    items.append(DecisionsChoiceItem(value=s))
            elif isinstance(c, dict):
                val = str(c.get("value") or c.get("name") or c.get("label") or "").strip()
                if val:
                    items.append(
                        DecisionsChoiceItem(
                            value=val,
                            label=str(c.get("label") or ""),
                            description=str(c.get("description") or ""),
                        )
                    )
        return items

    def _normalize_levels(self, raw_levels: list[Any]) -> list[DecisionsScoreLevel]:
        items: list[DecisionsScoreLevel] = []
        for l in raw_levels:
            if isinstance(l, DecisionsScoreLevel):
                items.append(l)
            elif isinstance(l, str):
                s = l.strip()
                if s:
                    items.append(DecisionsScoreLevel(label=s))
            elif isinstance(l, dict):
                label = str(l.get("label") or l.get("name") or "").strip()
                if label:
                    items.append(
                        DecisionsScoreLevel(
                            label=label,
                            description=str(l.get("description") or ""),
                            value=l.get("value"),
                        )
                    )
        return items
