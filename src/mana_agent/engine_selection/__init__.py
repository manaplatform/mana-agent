"""Automatic AI Engine Selection layer for Mana-Agent."""

from __future__ import annotations

from mana_agent.engine_selection.decisions_client import (
    DEFAULT_DECISIONS_MODEL,
    DecisionsAPIError,
    DecisionsClient,
    DecisionsError,
    DecisionsMalformedResponseError,
    DecisionsRefusalError,
    DecisionsTimeoutError,
)
from mana_agent.engine_selection.dispatcher import (
    SmartDispatcher,
    execute_auto,
    executeAuto,
)
from mana_agent.engine_selection.models import (
    DecisionsAnswer,
    DecisionsChoiceItem,
    DecisionsResult,
    DecisionsScoreLevel,
    EngineSelectionResult,
    EngineType,
    ExecutionResult,
    Operation,
    OperationKind,
    Question,
)
from mana_agent.engine_selection.operations import (
    batch_op,
    choice_op,
    conversation_op,
    deterministic_op,
    function_call_op,
    generation_op,
    predicate_op,
    reasoning_op,
    score_op,
    structured_op,
)
from mana_agent.engine_selection.selector import EngineSelector

__all__ = [
    "DEFAULT_DECISIONS_MODEL",
    "DecisionsAPIError",
    "DecisionsAnswer",
    "DecisionsChoiceItem",
    "DecisionsClient",
    "DecisionsError",
    "DecisionsMalformedResponseError",
    "DecisionsRefusalError",
    "DecisionsResult",
    "DecisionsScoreLevel",
    "DecisionsTimeoutError",
    "EngineSelectionResult",
    "EngineSelector",
    "EngineType",
    "ExecutionResult",
    "Operation",
    "OperationKind",
    "Question",
    "SmartDispatcher",
    "batch_op",
    "choice_op",
    "conversation_op",
    "deterministic_op",
    "executeAuto",
    "execute_auto",
    "function_call_op",
    "generation_op",
    "predicate_op",
    "reasoning_op",
    "score_op",
    "structured_op",
]
