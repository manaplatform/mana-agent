"""Structured classification of a new chat turn relative to durable tasks."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mana_agent.context_cost.accounting import ModelContextLimitError
from mana_agent.context_cost.models import ContextBudgetExceeded


class FollowupClassificationError(RuntimeError):
    def __init__(self, message: str, *, code: str = "") -> None:
        super().__init__(message)
        self.code = code


class FollowupRetrievalAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(default="", description="Semantic query to filter relevant turns.")
    max_turns: int = Field(default=1, ge=1, le=5, description="Maximum number of previous turns to retrieve (1-5).")


class FollowupClassificationOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    action: Literal["classify", "retrieve_context"] = "classify"
    retrieval: FollowupRetrievalAction | None = None
    category: Literal[
        "new_task",
        "followup_task",
        "task_expansion",
        "task_correction",
        "retry_request",
        "resume_request",
        "status_request",
        "clarification_answer",
        "conversation_only",
        "duplicate_message",
    ] = "new_task"
    related_task_id: str = ""
    safe_to_continue: bool = Field(
        default=True,
        description=(
            "Whether this classification is unambiguous enough to proceed to the next "
            "validated decision boundary; this does not authorize tools, mutations, or "
            "other consequential actions."
        ),
    )
    reason: str = Field(default="", min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if not data.get("reason"):
                for alias in (
                    "rationale",
                    "explanation",
                    "justification",
                    "thought",
                    "summary",
                    "details",
                    "why",
                ):
                    if data.get(alias):
                        data["reason"] = str(data.pop(alias)).strip()
                        break
            for key in (
                "rationale",
                "explanation",
                "justification",
                "thought",
                "summary",
                "details",
                "why",
                "thinking",
                "reasoning",
            ):
                data.pop(key, None)
            if not data.get("reason") or not str(data.get("reason")).strip():
                category = data.get("category", "new_task")
                data["reason"] = f"Model classified turn as {category}"
            else:
                data["reason"] = str(data["reason"]).strip()
        return data


@dataclass(frozen=True, slots=True)
class FollowupClassification:
    decision_id: str
    category: str
    related_task_id: str
    safe_to_continue: bool
    reason: str
    related_turn_ids: tuple[str, ...] = ()
    retrieval_refs: tuple[str, ...] = ()


_PROMPT = """Classify this newly received chat turn. A completed task does not complete its conversation.
Use new_task for independent actionable work; use followup_task, task_expansion, or task_correction only
when one offered task is the intended parent; use retry_request/resume_request only for the same offered
unfinished task; use status_request or conversation_only when no execution should be created. A new
downstream action after a completed task, especially one that changes live external state, is a
task_expansion, not a resume, retry, duplicate, or status request. Do not use keyword matching. If no
offered task is unambiguously applicable, select new_task or conversation_only. Return strict JSON
matching the schema and select only an offered task ID.

Set related_task_id only for followup_task, task_expansion, task_correction, retry_request,
resume_request, status_request, or duplicate_message. For new_task, clarification_answer, and
conversation_only, related_task_id must be the empty string.

If prior conversation context is required to safely classify this turn (for example, if the input is a terse
follow-up such as "why?", "continue", "do that", "what did you mean?", or "and memory?" and pointers indicate
prior context exists), you may request a bounded context retrieval by setting action="retrieve_context"
and providing retrieval query/max_turns. Only the current turn is provided automatically; do not guess
unobserved context. Once sufficient context is available, set action="classify".

safe_to_continue authorizes only use of this classification to reach the next independently validated
decision boundary. It does not approve tools, mutations, retries, or consequential actions. Set it true
when the selected category and optional offered task are unambiguous. In particular, select new_task or
conversation_only with safe_to_continue true when no offered task applies. Set it false only when the
turn cannot be safely classified at all, and explain the concrete ambiguity in reason. Do not set it
false merely because downstream work may be consequential or require separate approval.

CRITICAL: When you select new_task, conversation_only, or clarification_answer with an empty
related_task_id, you MUST set safe_to_continue to true. A classification that says "this input is
independent of all offered tasks" is unambiguous by definition. Terse, bare, or context-free inputs
(single words, email addresses, URLs, numbers, short phrases) are valid new_task or conversation_only
inputs. Empty recent_history does not make a classification ambiguous when no offered task applies.
Do not block the user because their input is short or lacks conversational context."""


def _coerce_followup_output(raw: Any) -> FollowupClassificationOutput:
    if isinstance(raw, FollowupClassificationOutput):
        return raw
    if isinstance(raw, dict):
        return FollowupClassificationOutput.model_validate(raw)
    tool_calls = getattr(raw, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        first_call = tool_calls[0]
        args = first_call.get("args") or {}
        return FollowupClassificationOutput(
            action="retrieve_context",
            retrieval=FollowupRetrievalAction(
                query=str(args.get("query") or ""),
                max_turns=int(args.get("max_turns") or 1),
            ),
            reason="Tool call requested context retrieval",
        )
    from mana_agent.utils.text import extract_model_text

    content = getattr(raw, "content", raw)
    text = extract_model_text(content)
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    return FollowupClassificationOutput.model_validate_json(text)


class FollowupClassifier:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def decide(
        self,
        *,
        message: str,
        recent_history: list[tuple[str, str]] | None = None,
        candidates: list[dict[str, Any]],
        pointers: Any | None = None,
        retrieval_hints: list[str] | None = None,
        conversation_tool: Any | None = None,
        turn_retrieval_cache: dict[str, Any] | None = None,
        retrieval_ledger: Any | None = None,
    ) -> FollowupClassification:
        if self.llm is None or not callable(getattr(self.llm, "invoke", None)):
            raise FollowupClassificationError(
                "Model decision failed: followup_classification. No fallback action was executed. "
                "Reason: decision model is unavailable."
            )
        safe_history = list(recent_history or [])
        retrieved_contexts: list[dict[str, Any]] = []
        max_retrieval_iterations = 2
        iteration = 0

        while True:
            payload: dict[str, Any] = {
                "message": message,
                "recent_history": safe_history,
                "candidates": candidates,
            }
            if pointers is not None and hasattr(pointers, "to_dict"):
                payload["pointers"] = pointers.to_dict()
            if retrieval_hints:
                payload["retrieval_hints"] = list(retrieval_hints)
            if retrieved_contexts:
                payload["retrieved_context"] = retrieved_contexts

            messages = [
                SystemMessage(content=_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ]
            try:
                structured = getattr(self.llm, "with_structured_output", None)
                raw = None
                structured_error = None
                if callable(structured):
                    try:
                        raw = structured(
                            FollowupClassificationOutput, method="json_schema", strict=True
                        ).invoke(messages)
                    except (ContextBudgetExceeded, ModelContextLimitError):
                        raise
                    except Exception as exc:
                        structured_error = exc
                if raw is None:
                    raw = self.llm.invoke(messages)
                try:
                    output = _coerce_followup_output(raw)
                except Exception as coerce_exc:
                    if callable(structured) and structured_error is None:
                        direct_raw = self.llm.invoke(messages)
                        output = _coerce_followup_output(direct_raw)
                    else:
                        raise coerce_exc
            except (ContextBudgetExceeded, ModelContextLimitError) as exc:
                raise FollowupClassificationError(
                    "Model decision failed: followup_classification. No fallback action was executed. "
                    f"Reason: {exc}",
                    code="context_budget_blocked",
                ) from exc
            except Exception as exc:
                raise FollowupClassificationError(
                    f"Model decision failed: followup_classification. No fallback action was executed. Reason: {exc}"
                ) from exc

            if (
                output.action == "retrieve_context"
                and conversation_tool is not None
                and iteration < max_retrieval_iterations
            ):
                iteration += 1
                query = output.retrieval.query if output.retrieval else ""
                max_turns = output.retrieval.max_turns if output.retrieval else 1
                try:
                    if hasattr(conversation_tool, "invoke") and callable(conversation_tool.invoke):
                        tool_res = conversation_tool.invoke({"query": query, "max_turns": max_turns})
                    elif callable(conversation_tool):
                        tool_res = conversation_tool(query=query, max_turns=max_turns)
                    else:
                        tool_res = "{}"
                    if isinstance(tool_res, str):
                        res_data = json.loads(tool_res)
                    elif isinstance(tool_res, dict):
                        res_data = tool_res
                    else:
                        res_data = {"result": str(tool_res)}
                    retrieved_contexts.append(res_data)
                except Exception as exc:
                    retrieved_contexts.append({
                        "status": "query_failed",
                        "error": str(exc),
                        "query": query,
                    })
                continue
            break

        offered = {str(item.get("task_id") or "") for item in candidates}
        needs_task = output.category in {
            "followup_task",
            "task_expansion",
            "task_correction",
            "retry_request",
            "resume_request",
            "status_request",
            "duplicate_message",
        }
        if needs_task and output.related_task_id not in offered:
            raise FollowupClassificationError(
                "Model decision failed: followup_classification. No fallback action was executed. "
                "Reason: selected task was not offered."
            )
        selected = next(
            (
                item
                for item in candidates
                if str(item.get("task_id") or "") == output.related_task_id
            ),
            None,
        )
        if (
            output.category in {"retry_request", "resume_request"}
            and str((selected or {}).get("state") or "") == "completed"
        ):
            raise FollowupClassificationError(
                "Model decision failed: followup_classification. No fallback action was "
                "executed. Reason: a completed task cannot be resumed or retried; select "
                "task_expansion for a downstream action."
            )
        if not needs_task and output.related_task_id:
            raise FollowupClassificationError(
                "Model decision failed: followup_classification. No fallback action was executed. "
                "Reason: non-task category selected a task."
            )
        # Structural invariant: an independent classification (no related task,
        # non-task category) is unambiguous by construction. The model already
        # decided the input does not relate to any offered candidate, so
        # safe_to_continue=false is logically contradictory. Override it.
        # This is NOT fallback logic — the model decision (category + related_task_id)
        # is preserved exactly; only the internally inconsistent flag is corrected.
        independent_categories = {"new_task", "conversation_only", "clarification_answer"}
        if (
            not output.safe_to_continue
            and output.category in independent_categories
            and not output.related_task_id
        ):
            output = output.model_copy(update={"safe_to_continue": True})
        if not output.safe_to_continue:
            raise FollowupClassificationError(
                "Model decision failed: followup_classification. No fallback action was "
                "executed. Reason: decision did not authorize continuation: "
                f"{output.reason}"
            )
        related_turn_ids: list[str] = []
        retrieval_refs: list[str] = []
        for ctx in retrieved_contexts:
            if isinstance(ctx, dict):
                turns = ctx.get("turns") or []
                if isinstance(turns, list):
                    for turn in turns:
                        if isinstance(turn, dict) and turn.get("turn_id"):
                            related_turn_ids.append(str(turn["turn_id"]))
                if ctx.get("artifact_ref"):
                    retrieval_refs.append(str(ctx["artifact_ref"]))
        if pointers is not None:
            prev = getattr(pointers, "previous_turn_id", "")
            if prev and prev not in related_turn_ids:
                related_turn_ids.append(str(prev))

        fields = {
            "category": output.category,
            "related_task_id": output.related_task_id,
            "safe_to_continue": output.safe_to_continue,
            "reason": output.reason or "Followup classified",
            "decision_id": f"followup:{uuid.uuid4().hex[:12]}",
            "related_turn_ids": tuple(dict.fromkeys(related_turn_ids)),
            "retrieval_refs": tuple(dict.fromkeys(retrieval_refs)),
        }
        return FollowupClassification(**fields)
