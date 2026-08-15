"""Context retrieval tools for episodic conversation and durable memory capsules.

Execution models retrieve previous conversation or memory only through these
explicit, bounded, host-scoped tools. Model-controlled arguments are restricted
to semantic parameters. Identity, session, and authorization contexts are supplied
by the trusted host and cannot be overridden by model inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field
from langchain_core.tools import BaseTool, StructuredTool

from mana_agent.context_cost.estimator import estimate_value_tokens
from mana_agent.context_cost.governor import ContextCostGovernor
from mana_agent.memory.capsules.models import (
    CapsuleReadRequest,
    CapsuleScope,
    CapsuleTaskContext,
    MemoryPrincipal,
)
from mana_agent.memory.capsules.service import CapsuleService


@dataclass
class TurnRetrievalLedger:
    """Host-owned turn ledger tracking cumulative retrieval token allowances."""

    retrieval_budget_tokens: int = 12000
    conversation_retrieval_tokens: int = 0
    memory_retrieval_tokens: int = 0

    @property
    def retrieval_used_tokens(self) -> int:
        return self.conversation_retrieval_tokens + self.memory_retrieval_tokens

    @property
    def retrieval_remaining_tokens(self) -> int:
        return max(0, self.retrieval_budget_tokens - self.retrieval_used_tokens)

    def to_dict(self) -> dict[str, int]:
        return {
            "retrieval_budget_tokens": self.retrieval_budget_tokens,
            "conversation_retrieval_tokens": self.conversation_retrieval_tokens,
            "memory_retrieval_tokens": self.memory_retrieval_tokens,
            "retrieval_used_tokens": self.retrieval_used_tokens,
            "retrieval_remaining_tokens": self.retrieval_remaining_tokens,
        }


class MemoryTaskBinding:
    """Trusted mutable binding allowing runtime task authorization after routing."""

    def __init__(self, selected_memory_task_id: str = "") -> None:
        self.selected_memory_task_id = str(selected_memory_task_id or "").strip()

    def bind(self, task_id: str) -> None:
        self.selected_memory_task_id = str(task_id or "").strip()


class ConversationContextReadInput(BaseModel):
    """Input parameters for reading episodic conversation history."""

    model_config = ConfigDict(extra="ignore")

    query: str | None = Field(
        default=None,
        description="Optional semantic query to filter relevant turns in conversation history.",
    )
    max_turns: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of previous turns to retrieve (1-20).",
    )
    max_tokens: int = Field(
        default=2000,
        ge=1,
        le=12000,
        description="Maximum tokens to return.",
    )
    before_turn_id: str | None = Field(
        default=None,
        description="Optional turn ID to retrieve turns prior to.",
    )


class MemoryReadInput(BaseModel):
    """Input parameters for reading authorized durable memory capsules."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(
        default="",
        description="Semantic query to search authorized memory capsules.",
    )
    max_capsules: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of memory capsules to retrieve (1-10).",
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        le=4000,
        description="Maximum tokens to return.",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Optional tag filters for memory capsules.",
    )


def execute_conversation_context_read(
    *,
    session_id: str,
    conversation_id: str,
    authenticated_user_id: str,
    history_store: Any,
    current_turn_id: str = "",
    query: str | None = None,
    max_turns: int = 5,
    max_tokens: int = 2000,
    before_turn_id: str | None = None,
    governor: ContextCostGovernor | None = None,
    turn_retrieval_cache: dict[str, Any] | None = None,
    event_sink: Callable[..., Any] | None = None,
    retrieval_ledger: TurnRetrievalLedger | None = None,
    retrieval_budget: int = 12000,
) -> str:
    """Read bounded episodic conversation context for the active authorized session."""
    bounded_max_turns = max(1, min(int(max_turns or 5), 20))
    remaining_allowance = (
        retrieval_ledger.retrieval_remaining_tokens
        if retrieval_ledger is not None
        else retrieval_budget
    )
    effective_token_limit = min(int(max_tokens or 2000), remaining_allowance, retrieval_budget)
    bounded_max_tokens = max(0, effective_token_limit)
    norm_query = str(query or "").strip().lower()
    norm_before = str(before_turn_id or "").strip()

    cache_key = (
        "conversation_context_read",
        session_id,
        norm_query,
        bounded_max_turns,
        int(max_tokens or 2000),
        norm_before,
    )
    if turn_retrieval_cache is not None and cache_key in turn_retrieval_cache:
        cached = turn_retrieval_cache[cache_key]
        if callable(event_sink):
            event_sink(
                "context.retrieval_deduplicated",
                "Conversation retrieval deduplicated in turn",
                metadata={
                    "tool": "conversation_context_read",
                    "session_id": session_id,
                    "turn_id": current_turn_id,
                    "tokens_charged": 0,
                    "deduplicated": True,
                },
            )
        return cached

    if bounded_max_tokens <= 0:
        empty_payload = {
            "source": "conversation_context",
            "session_id": session_id,
            "conversation_id": conversation_id,
            "turns_returned": 0,
            "tokens": 0,
            "empty": True,
            "truncated": False,
            "turns": [],
        }
        return json.dumps(empty_payload, ensure_ascii=False)

    messages: list[Any] = []
    if history_store is not None:
        if hasattr(history_store, "list") and callable(history_store.list):
            try:
                messages = history_store.list(session_id)
            except Exception:
                messages = []
        elif isinstance(history_store, list):
            messages = list(history_store)

    # Group messages into turns
    turns_by_id: dict[str, list[dict[str, Any]]] = {}
    turn_order: list[str] = []
    for msg in messages:
        msg_dict = msg.to_dict() if hasattr(msg, "to_dict") else dict(msg)
        t_id = str(msg_dict.get("turn_id") or "")
        if not t_id or t_id == current_turn_id:
            continue
        if norm_before and t_id == norm_before:
            break
        if t_id not in turns_by_id:
            turns_by_id[t_id] = []
            turn_order.append(t_id)
        turns_by_id[t_id].append(
            {
                "turn_id": t_id,
                "role": str(msg_dict.get("role") or ""),
                "content": str(msg_dict.get("content") or ""),
                "timestamp": str(msg_dict.get("timestamp") or ""),
            }
        )

    # Filter turns by query if present
    matching_turn_ids: list[str] = []
    if norm_query:
        query_words = set(norm_query.split())
        for t_id in turn_order:
            turn_text = " ".join(m["content"].lower() for m in turns_by_id[t_id])
            if any(w in turn_text for w in query_words):
                matching_turn_ids.append(t_id)
    else:
        matching_turn_ids = list(turn_order)

    # Select the most recent max_turns matching turns
    selected_turn_ids = matching_turn_ids[-bounded_max_turns:]
    projected_turns: list[dict[str, Any]] = []
    for t_id in selected_turn_ids:
        projected_turns.extend(turns_by_id[t_id])

    # Bounded token truncation
    used_tokens = estimate_value_tokens(projected_turns)
    truncated = False
    if used_tokens > bounded_max_tokens and projected_turns:
        truncated = True
        while projected_turns and estimate_value_tokens(projected_turns) > bounded_max_tokens:
            projected_turns.pop(0)
        used_tokens = estimate_value_tokens(projected_turns)

    payload = {
        "source": "conversation_context",
        "session_id": session_id,
        "conversation_id": conversation_id,
        "turns_returned": len(set(m["turn_id"] for m in projected_turns)),
        "tokens": used_tokens,
        "empty": len(projected_turns) == 0,
        "truncated": truncated,
        "turns": projected_turns,
    }
    encoded = json.dumps(payload, ensure_ascii=False)

    if turn_retrieval_cache is not None:
        turn_retrieval_cache[cache_key] = encoded

    if retrieval_ledger is not None and used_tokens > 0:
        retrieval_ledger.conversation_retrieval_tokens += used_tokens

    if callable(event_sink):
        event_sink(
            "context.conversation_read",
            "Conversation context retrieved",
            metadata={
                "session_id": session_id,
                "conversation_id": conversation_id,
                "turn_id": current_turn_id,
                "query": norm_query,
                "turns_count": payload["turns_returned"],
                "tokens": used_tokens,
                "empty_result": payload["empty"],
                "truncated": truncated,
                "history_injected": False,
            },
        )

    return encoded


def execute_memory_read(
    *,
    capsule_service: CapsuleService | None,
    authenticated_user_id: str,
    session_id: str,
    repository_id: str,
    current_turn_id: str = "",
    current_task_id: str = "",
    selected_memory_task_id: str = "",
    parent_task_id: str | None = None,
    memory_task_candidates: tuple[dict[str, str], ...] = (),
    query: str = "",
    max_capsules: int = 3,
    max_tokens: int = 1000,
    task_id: str | None = None,
    tags: list[str] | None = None,
    governor: ContextCostGovernor | None = None,
    turn_retrieval_cache: dict[str, Any] | None = None,
    event_sink: Callable[..., Any] | None = None,
    retrieval_ledger: TurnRetrievalLedger | None = None,
    retrieval_budget: int = 4000,
) -> str:
    """Read authorized durable memory capsules for authenticated principal and validated task."""
    effective_turn_id = current_turn_id or current_task_id
    effective_selected_task = str(selected_memory_task_id or task_id or "").strip()
    bounded_max_capsules = max(1, min(int(max_capsules or 3), 10))
    remaining_allowance = (
        retrieval_ledger.retrieval_remaining_tokens
        if retrieval_ledger is not None
        else retrieval_budget
    )
    effective_token_limit = min(
        int(max_tokens if max_tokens is not None else 1000),
        remaining_allowance,
        retrieval_budget,
    )
    bounded_max_tokens = max(0, effective_token_limit)
    norm_query = str(query or "").strip()
    norm_tags = tuple(sorted(str(t).strip().lower() for t in (tags or []) if str(t).strip()))

    cache_key = (
        "memory_read",
        authenticated_user_id,
        session_id,
        repository_id,
        norm_query,
        bounded_max_capsules,
        int(max_tokens or 1000),
        effective_selected_task,
        norm_tags,
    )
    if turn_retrieval_cache is not None and cache_key in turn_retrieval_cache:
        cached = turn_retrieval_cache[cache_key]
        if callable(event_sink):
            event_sink(
                "context.retrieval_deduplicated",
                "Memory retrieval deduplicated in turn",
                metadata={
                    "tool": "memory_read",
                    "session_id": session_id,
                    "turn_id": effective_turn_id,
                    "tokens_charged": 0,
                    "deduplicated": True,
                },
            )
        return cached

    # Deny-by-default if unauthenticated
    if not authenticated_user_id:
        error_payload = {
            "source": "memory_capsules",
            "status": "unauthorized",
            "selected_memory_task_id": effective_selected_task,
            "capsules_returned": 0,
            "tokens": 0,
            "empty": True,
            "goal_satisfied": False,
            "error": "Private memory retrieval requires an authenticated user identity. No memory was read.",
            "capsules": [],
        }
        encoded = json.dumps(error_payload, ensure_ascii=False)
        if callable(event_sink):
            event_sink(
                "context.memory_read",
                "Memory retrieval rejected: unauthenticated",
                metadata={
                    "session_id": session_id,
                    "turn_id": effective_turn_id,
                    "empty_result": True,
                    "error": "unauthenticated",
                    "history_injected": False,
                },
            )
        return encoded

    # Validate candidate task_id against router-offered candidates ONLY.
    # Turn IDs and parent execution IDs are NEVER authorized implicitly.
    offered_tasks = {
        str(item.get("task_id") or "").strip()
        for item in memory_task_candidates
        if str(item.get("task_id") or "").strip()
    }

    if not effective_selected_task or effective_selected_task not in offered_tasks:
        error_payload = {
            "source": "memory_capsules",
            "status": "unauthorized",
            "selected_memory_task_id": effective_selected_task,
            "capsules_returned": 0,
            "tokens": 0,
            "empty": True,
            "goal_satisfied": False,
            "error": (
                f"Selected memory task {effective_selected_task!r} was not offered to the router. Access denied."
                if effective_selected_task
                else "No memory task was authorized by entry routing. Access denied."
            ),
            "capsules": [],
        }
        encoded = json.dumps(error_payload, ensure_ascii=False)
        if callable(event_sink):
            event_sink(
                "context.memory_read",
                "Memory retrieval rejected: task not offered",
                metadata={
                    "session_id": session_id,
                    "turn_id": effective_turn_id,
                    "task_id": effective_selected_task,
                    "empty_result": True,
                    "error": "task_not_offered",
                    "history_injected": False,
                },
            )
        return encoded

    if bounded_max_tokens <= 0:
        empty_payload = {
            "source": "memory_capsules",
            "status": "retrieval_budget_exhausted",
            "selected_memory_task_id": effective_selected_task,
            "capsules_returned": 0,
            "tokens": 0,
            "empty": True,
            "goal_satisfied": False,
            "error": "Turn retrieval budget exhausted.",
            "capsules": [],
        }
        return json.dumps(empty_payload, ensure_ascii=False)

    if capsule_service is None or not getattr(
        getattr(capsule_service, "config", None), "enabled", True
    ):
        empty_payload = {
            "source": "memory_capsules",
            "status": "not_configured",
            "selected_memory_task_id": effective_selected_task,
            "capsules_returned": 0,
            "tokens": 0,
            "empty": True,
            "goal_satisfied": False,
            "error": "Memory capsule service is not configured or disabled.",
            "capsules": [],
        }
        return json.dumps(empty_payload, ensure_ascii=False)

    principal = MemoryPrincipal(
        user_id=authenticated_user_id,
        project_id=repository_id or None,
        task_id=effective_selected_task,
        agent_id="gateway:chat",
        capabilities=frozenset({"memory.capsule.read.private"}),
    )
    task_context = CapsuleTaskContext(
        user_id=authenticated_user_id,
        organisation_id=None,
        project_id=repository_id or None,
        team_ids=frozenset(),
        task_id=effective_selected_task,
        agent_id="gateway:chat",
        session_id=session_id,
    )

    try:
        projections = capsule_service.query_capsules(
            CapsuleReadRequest(
                principal=principal,
                task_context=task_context,
                query=norm_query,
                allowed_scopes=frozenset({CapsuleScope.PRIVATE, CapsuleScope.PROJECT}),
                max_capsules=bounded_max_capsules,
                max_tokens=bounded_max_tokens,
            ),
            correlation_id=effective_turn_id,
        )
    except Exception as exc:
        error_payload = {
            "source": "memory_capsules",
            "status": "query_failed",
            "selected_memory_task_id": effective_selected_task,
            "capsules_returned": 0,
            "tokens": 0,
            "empty": True,
            "goal_satisfied": False,
            "error": f"Memory query error: {exc}",
            "capsules": [],
        }
        encoded = json.dumps(error_payload, ensure_ascii=False)
        return encoded

    if norm_tags:
        target_tags = set(norm_tags)
        projections = [
            p
            for p in projections
            if target_tags.intersection(
                t.lower() for t in getattr(p, "tags", ()) or ()
            )
        ]

    capsule_rows = [
        {
            "capsule_id": getattr(p, "capsule_id", ""),
            "revision": getattr(p, "revision", 0),
            "title": getattr(p, "title", ""),
            "summary": getattr(p, "summary", ""),
            "content": getattr(p, "content", ""),
            "tags": list(getattr(p, "tags", ()) or ()),
        }
        for p in projections[:bounded_max_capsules]
    ]

    used_tokens = estimate_value_tokens(capsule_rows)
    truncated = False
    if used_tokens > bounded_max_tokens and capsule_rows:
        truncated = True
        while capsule_rows and estimate_value_tokens(capsule_rows) > bounded_max_tokens:
            capsule_rows.pop()
        used_tokens = estimate_value_tokens(capsule_rows)

    matched = len(capsule_rows) > 0
    payload = {
        "source": "memory_capsules",
        "status": "matched" if matched else "no_match",
        "selected_memory_task_id": effective_selected_task,
        "capsules_returned": len(capsule_rows),
        "tokens": used_tokens,
        "empty": len(capsule_rows) == 0,
        "goal_satisfied": matched,
        "truncated": truncated,
        "capsules": capsule_rows,
    }
    encoded = json.dumps(payload, ensure_ascii=False)

    if turn_retrieval_cache is not None:
        turn_retrieval_cache[cache_key] = encoded

    if retrieval_ledger is not None and used_tokens > 0:
        retrieval_ledger.memory_retrieval_tokens += used_tokens

    if callable(event_sink):
        event_sink(
            "context.memory_read",
            "Memory capsules retrieved",
            metadata={
                "session_id": session_id,
                "turn_id": effective_turn_id,
                "query": norm_query,
                "capsules_count": payload["capsules_returned"],
                "tokens": used_tokens,
                "empty_result": payload["empty"],
                "goal_satisfied": matched,
                "truncated": truncated,
                "history_injected": False,
            },
        )

    return encoded


def build_context_retrieval_tools(
    *,
    session_id: str,
    conversation_id: str,
    authenticated_user_id: str,
    history_store: Any,
    capsule_service: CapsuleService | None = None,
    repository_id: str = "",
    current_turn_id: str = "",
    selected_memory_task_id: str | MemoryTaskBinding = "",
    parent_task_id: str | None = None,
    memory_task_candidates: tuple[dict[str, str], ...] = (),
    governor: ContextCostGovernor | None = None,
    turn_retrieval_cache: dict[str, Any] | None = None,
    event_sink: Callable[..., Any] | None = None,
    retrieval_ledger: TurnRetrievalLedger | None = None,
    conversation_budget: int = 12000,
    memory_budget: int = 4000,
) -> list[BaseTool]:
    """Build LangChain StructuredTools bound to trusted runtime context."""

    def conversation_context_read(
        query: str | None = None,
        max_turns: int = 5,
        max_tokens: int = 2000,
        before_turn_id: str | None = None,
        **_extra: Any,
    ) -> str:
        return execute_conversation_context_read(
            session_id=session_id,
            conversation_id=conversation_id,
            authenticated_user_id=authenticated_user_id,
            history_store=history_store,
            current_turn_id=current_turn_id,
            query=query,
            max_turns=max_turns,
            max_tokens=max_tokens,
            before_turn_id=before_turn_id,
            governor=governor,
            turn_retrieval_cache=turn_retrieval_cache,
            event_sink=event_sink,
            retrieval_ledger=retrieval_ledger,
            retrieval_budget=conversation_budget,
        )

    def memory_read(
        query: str = "",
        max_capsules: int = 3,
        max_tokens: int = 1000,
        tags: list[str] | None = None,
        **_extra: Any,
    ) -> str:
        effective_task = (
            selected_memory_task_id.selected_memory_task_id
            if isinstance(selected_memory_task_id, MemoryTaskBinding)
            else str(selected_memory_task_id or "")
        )
        return execute_memory_read(
            capsule_service=capsule_service,
            authenticated_user_id=authenticated_user_id,
            session_id=session_id,
            repository_id=repository_id,
            current_turn_id=current_turn_id,
            selected_memory_task_id=effective_task,
            parent_task_id=parent_task_id,
            memory_task_candidates=memory_task_candidates,
            query=query,
            max_capsules=max_capsules,
            max_tokens=max_tokens,
            tags=tags,
            governor=governor,
            turn_retrieval_cache=turn_retrieval_cache,
            event_sink=event_sink,
            retrieval_ledger=retrieval_ledger,
            retrieval_budget=memory_budget,
        )

    return [
        StructuredTool.from_function(
            func=conversation_context_read,
            name="conversation_context_read",
            description=(
                "Read bounded episodic conversation history for this active authorized session. "
                "Specify query, max_turns, and max_tokens to retrieve relevant past turns."
            ),
            args_schema=ConversationContextReadInput,
        ),
        StructuredTool.from_function(
            func=memory_read,
            name="memory_read",
            description=(
                "Read authorized durable memory capsules for the authenticated user and task context. "
                "Specify semantic query, max_capsules, and max_tokens."
            ),
            args_schema=MemoryReadInput,
        ),
    ]


__all__ = [
    "ConversationContextReadInput",
    "MemoryReadInput",
    "MemoryTaskBinding",
    "TurnRetrievalLedger",
    "build_context_retrieval_tools",
    "execute_conversation_context_read",
    "execute_memory_read",
]
