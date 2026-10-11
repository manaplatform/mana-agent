"""Convenience operation builders for automatic engine selection."""

from __future__ import annotations

from typing import Any, Callable
from pydantic import BaseModel

from mana_agent.engine_selection.models import (
    DecisionsChoiceItem,
    DecisionsScoreLevel,
    Operation,
    OperationKind,
)


def predicate_op(
    instructions: str,
    evidence: Any,
    *,
    name: str = "predicate_eval",
    metadata: dict[str, Any] | None = None,
) -> Operation:
    """Create a boolean semantic evaluation operation."""
    return Operation(
        kind=OperationKind.PREDICATE,
        name=name,
        instructions=instructions,
        input=evidence,
        metadata=metadata or {},
    )


def choice_op(
    instructions: str,
    choices: list[str | dict[str, Any] | DecisionsChoiceItem],
    evidence: Any,
    *,
    name: str = "choice_eval",
    metadata: dict[str, Any] | None = None,
) -> Operation:
    """Create a choice classification operation among fixed options."""
    return Operation(
        kind=OperationKind.CHOICE,
        name=name,
        instructions=instructions,
        choices=choices,
        input=evidence,
        metadata=metadata or {},
    )


def score_op(
    instructions: str,
    levels: list[dict[str, Any] | DecisionsScoreLevel],
    evidence: Any,
    *,
    name: str = "score_eval",
    metadata: dict[str, Any] | None = None,
) -> Operation:
    """Create an ordered scoring evaluation operation against levels / rubric."""
    return Operation(
        kind=OperationKind.SCORE,
        name=name,
        instructions=instructions,
        levels=levels,
        input=evidence,
        metadata=metadata or {},
    )


def batch_op(
    sub_operations: list[Operation],
    evidence: Any = None,
    *,
    name: str = "batch_eval",
    metadata: dict[str, Any] | None = None,
) -> Operation:
    """Create a batched evaluation operation with multiple independent checks."""
    return Operation(
        kind=OperationKind.BATCH,
        name=name,
        input=evidence,
        sub_operations=sub_operations,
        metadata=metadata or {},
    )


def structured_op(
    instructions: str,
    output_schema: type[BaseModel] | dict[str, Any],
    evidence: Any,
    *,
    name: str = "structured_output",
    metadata: dict[str, Any] | None = None,
) -> Operation:
    """Create an arbitrary JSON object generation/extraction operation."""
    return Operation(
        kind=OperationKind.STRUCTURED_OUTPUT,
        name=name,
        instructions=instructions,
        output_schema=output_schema,
        input=evidence,
        metadata=metadata or {},
    )


def function_call_op(
    tool_name: str,
    tool_schema: dict[str, Any],
    input_prompt: Any,
    *,
    instructions: str = "",
    name: str = "function_call",
    metadata: dict[str, Any] | None = None,
) -> Operation:
    """Create a tool calling operation with dynamic generated arguments."""
    return Operation(
        kind=OperationKind.FUNCTION_CALL,
        name=name,
        instructions=instructions,
        tool_name=tool_name,
        tool_schema=tool_schema,
        input=input_prompt,
        metadata=metadata or {},
    )


def reasoning_op(
    instructions: str,
    input_data: Any,
    *,
    name: str = "reasoning_task",
    metadata: dict[str, Any] | None = None,
) -> Operation:
    """Create a complex reasoning or coding task operation."""
    return Operation(
        kind=OperationKind.REASONING,
        name=name,
        instructions=instructions,
        input=input_data,
        metadata=metadata or {},
    )


def generation_op(
    instructions: str,
    input_data: Any,
    *,
    name: str = "generation_task",
    metadata: dict[str, Any] | None = None,
) -> Operation:
    """Create a text generation or summarization operation."""
    return Operation(
        kind=OperationKind.GENERATION,
        name=name,
        instructions=instructions,
        input=input_data,
        metadata=metadata or {},
    )


def conversation_op(
    prompt: str,
    *,
    name: str = "conversation_task",
    metadata: dict[str, Any] | None = None,
) -> Operation:
    """Create an open-ended conversation operation."""
    return Operation(
        kind=OperationKind.CONVERSATION,
        name=name,
        input=prompt,
        metadata=metadata or {},
    )


def deterministic_op(
    handler: Callable[..., Any],
    input_data: Any = None,
    *,
    name: str = "deterministic_task",
    metadata: dict[str, Any] | None = None,
) -> Operation:
    """Create an explicit deterministic operation that runs native code directly."""
    return Operation(
        kind=OperationKind.DETERMINISTIC,
        name=name,
        handler=handler,
        input=input_data,
        metadata=metadata or {},
    )
