"""Data contracts and schemas for the automatic AI engine selection layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
from pydantic import BaseModel


class EngineType(str, Enum):
    """Execution engine selected for an operation."""

    DECISIONS_PREDICATE = "decisions_predicate"
    DECISIONS_CHOICE = "decisions_choice"
    DECISIONS_SCORE = "decisions_score"
    DECISIONS_BATCH = "decisions_batch"
    RESPONSES_STRUCTURED = "responses_structured"
    FUNCTION_CALLING = "function_calling"
    CONVERSATION_LLM = "conversation_llm"
    REASONING_ENGINE = "reasoning_engine"
    GENERATION_ENGINE = "generation_engine"
    DETERMINISTIC = "deterministic"
    COMPATIBLE_FALLBACK = "compatible_fallback"


class OperationKind(str, Enum):
    """Semantic kind and expected result type of an operation."""

    PREDICATE = "predicate"
    CHOICE = "choice"
    SCORE = "score"
    BATCH = "batch"
    STRUCTURED_OUTPUT = "structured_output"
    FUNCTION_CALL = "function_call"
    CONVERSATION = "conversation"
    REASONING = "reasoning"
    GENERATION = "generation"
    DETERMINISTIC = "deterministic"


@dataclass(frozen=True, slots=True)
class DecisionsChoiceItem:
    """A valid option for a Decisions choice question."""

    value: str
    label: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, str]:
        res: dict[str, str] = {"value": self.value}
        if self.label:
            res["label"] = self.label
        if self.description:
            res["description"] = self.description
        return res


@dataclass(frozen=True, slots=True)
class DecisionsScoreLevel:
    """An ordered level in a Decisions score rubric."""

    label: str
    description: str = ""
    value: float | int | None = None

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {"label": self.label}
        if self.description:
            res["description"] = self.description
        if self.value is not None:
            res["value"] = self.value
        return res


@dataclass(slots=True)
class Question:
    """A question specification for Decisions API."""

    type: str  # predicate | choice | score
    name: str
    instructions: str
    choices: list[DecisionsChoiceItem] = field(default_factory=list)
    levels: list[DecisionsScoreLevel] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.type,
            "name": self.name,
            "instructions": self.instructions,
        }
        if self.type == "choice" and self.choices:
            d["choices"] = [c.to_dict() for c in self.choices]
        if self.type == "score" and self.levels:
            d["levels"] = [l.to_dict() for l in self.levels]
        return d


@dataclass(slots=True)
class Operation:
    """Canonical representation of a unit of work submitted for engine execution."""

    kind: OperationKind
    name: str = ""
    instructions: str = ""
    input: Any = None
    choices: list[str | dict[str, Any] | DecisionsChoiceItem] = field(default_factory=list)
    levels: list[dict[str, Any] | DecisionsScoreLevel] = field(default_factory=list)
    output_schema: type[BaseModel] | dict[str, Any] | None = None
    tool_name: str = ""
    tool_schema: dict[str, Any] | None = None
    handler: Callable[..., Any] | None = None
    sub_operations: list[Operation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_deterministic(self) -> bool:
        return self.kind == OperationKind.DETERMINISTIC or self.handler is not None


@dataclass(frozen=True, slots=True)
class DecisionsAnswer:
    """Normalized response item from the Decisions API."""

    type: str  # predicate | choice | score | refusal
    name: str
    result: bool | None = None
    choice: str | None = None
    score: float | int | None = None
    confidence: float = 1.0
    probability: float | None = None
    probabilities: list[dict[str, Any]] = field(default_factory=list)
    refusal: str | None = None

    @property
    def is_refusal(self) -> bool:
        return self.type == "refusal" or bool(self.refusal)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "result": self.result,
            "choice": self.choice,
            "score": self.score,
            "confidence": self.confidence,
            "probability": self.probability,
            "probabilities": self.probabilities,
            "refusal": self.refusal,
        }


@dataclass(frozen=True, slots=True)
class DecisionsResult:
    """Complete structured response from the Decisions API."""

    model: str
    answers: list[DecisionsAnswer] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0

    @property
    def primary_answer(self) -> DecisionsAnswer | None:
        return self.answers[0] if self.answers else None


@dataclass(frozen=True, slots=True)
class EngineSelectionResult:
    """Result of automatic engine capability detection."""

    selected_engine: EngineType
    is_decisions_eligible: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionResult:
    """Normalized result returned from SmartDispatcher.execute_auto."""

    success: bool
    engine_used: EngineType
    result: Any
    confidence: float = 1.0
    reasoning: str = ""
    fallback_occurred: bool = False
    fallback_reason: str = ""
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
