from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .adapters import ActionAdapter
from .models import ActionIntent, CompensationEvidence, Reversibility


Eligibility = Callable[[ActionIntent], tuple[bool, str]]
Compensator = Callable[[ActionIntent, ActionAdapter], CompensationEvidence]


@dataclass(frozen=True)
class CompensationDefinition:
    tool_name: str
    operation_name: str
    eligibility: Eligibility
    required_pre_execution_evidence: tuple[str, ...]
    compensate: Compensator
    verification_requirements: tuple[str, ...]
    unsafe_cases: tuple[str, ...]


class CompensationRegistry:
    def __init__(self) -> None:
        self._definitions: dict[tuple[str, str], CompensationDefinition] = {}

    def register(self, definition: CompensationDefinition) -> None:
        key = (definition.tool_name, definition.operation_name)
        if key in self._definitions:
            raise ValueError(f"compensation is already registered for {key}")
        self._definitions[key] = definition

    def get(self, action: ActionIntent) -> CompensationDefinition | None:
        return self._definitions.get((action.tool_name, action.operation_name))

    def assert_eligible(self, action: ActionIntent) -> CompensationDefinition:
        if action.reversibility in {Reversibility.IRREVERSIBLE, Reversibility.UNKNOWN}:
            raise ValueError("irreversible or unknown actions cannot be compensated automatically")
        definition = self.get(action)
        if definition is None:
            raise ValueError("no compensation is registered for this action")
        eligible, reason = definition.eligibility(action)
        if not eligible:
            raise ValueError(reason or "compensation preconditions are not satisfied")
        return definition


def file_compensation_registry() -> CompensationRegistry:
    registry = CompensationRegistry()

    def eligible(action: ActionIntent) -> tuple[bool, str]:
        reference = str(action.execution_result.get("snapshot_reference") or "")
        if action.operation_name == "create" or reference:
            return True, ""
        return False, "verified pre-execution snapshot evidence is missing"

    def compensate(action: ActionIntent, adapter: ActionAdapter) -> CompensationEvidence:
        return adapter.compensate(action)

    for operation in ("create", "edit", "move", "delete"):
        registry.register(CompensationDefinition(
            tool_name="file",
            operation_name=operation,
            eligibility=eligible,
            required_pre_execution_evidence=("existence", "content snapshot", "file mode"),
            compensate=compensate,
            verification_requirements=("restored existence", "restored content hash", "restored mode"),
            unsafe_cases=("target changed after execution", "snapshot missing", "destination occupied"),
        ))
    return registry
