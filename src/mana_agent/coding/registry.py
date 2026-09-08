"""Validated coding backend registry."""

from __future__ import annotations

from mana_agent.coding.backend import CodingAgentBackend
from mana_agent.coding.models import CodingBackendDecision


class CodingBackendDecisionError(RuntimeError):
    """Raised when a model-selected backend cannot safely execute."""


class CodingBackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, CodingAgentBackend] = {}

    def register(self, backend: CodingAgentBackend) -> None:
        name = str(getattr(backend, "name", "") or "").strip()
        if not name:
            raise ValueError("coding backend name is required")
        if name != "codex":
            raise ValueError(
                f"Cannot register coding backend '{name}'. "
                "Codex is the single authoritative coding engine."
            )
        if name in self._backends:
            raise ValueError(f"coding backend is already registered: {name}")
        self._backends[name] = backend

    def resolve(self, decision: CodingBackendDecision) -> CodingAgentBackend:
        if not decision.safe_to_continue:
            raise CodingBackendDecisionError(
                f"Model decision failed: coding_backend ({decision.decision_id}). No backend was executed."
            )
        if not decision.coding_required:
            raise CodingBackendDecisionError("Coding backend execution was requested for a non-coding decision.")
        selected = str(decision.selected_backend or "").strip()
        if selected != "codex":
            raise CodingBackendDecisionError(
                f"Model-selected coding backend is invalid: {selected or '<missing>'}. "
                "Codex is the single authoritative coding engine; no internal coding engine can be selected or invoked."
            )
        backend = self._backends.get(selected)
        if backend is None:
            raise CodingBackendDecisionError(
                f"Model-selected coding backend is unavailable: {selected or '<missing>'}. "
                "No fallback backend was executed."
            )
        return backend

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._backends))


__all__ = ["CodingBackendDecisionError", "CodingBackendRegistry"]
