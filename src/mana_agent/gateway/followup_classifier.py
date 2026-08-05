"""Structured classification of a new chat turn relative to durable tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field


class FollowupClassificationError(RuntimeError):
    pass


class FollowupClassificationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision_id: str = Field(min_length=1)
    category: Literal["new_task", "followup_task", "task_expansion", "task_correction", "retry_request", "resume_request", "status_request", "clarification_answer", "conversation_only", "duplicate_message"]
    related_task_id: str = ""
    safe_to_continue: bool = Field(
        description=(
            "Whether this classification is unambiguous enough to proceed to the next "
            "validated decision boundary; this does not authorize tools, mutations, or "
            "other consequential actions."
        )
    )
    reason: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class FollowupClassification:
    decision_id: str
    category: str
    related_task_id: str
    safe_to_continue: bool
    reason: str


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

safe_to_continue authorizes only use of this classification to reach the next independently validated
decision boundary. It does not approve tools, mutations, retries, or consequential actions. Set it true
when the selected category and optional offered task are unambiguous. In particular, select new_task or
conversation_only with safe_to_continue true when no offered task applies. Set it false only when the
turn cannot be safely classified at all, and explain the concrete ambiguity in reason. Do not set it
false merely because downstream work may be consequential or require separate approval."""


class FollowupClassifier:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def decide(self, *, message: str, recent_history: list[tuple[str, str]], candidates: list[dict[str, Any]]) -> FollowupClassification:
        if self.llm is None or not callable(getattr(self.llm, "invoke", None)):
            raise FollowupClassificationError("Model decision failed: followup_classification. No fallback action was executed. Reason: decision model is unavailable.")
        payload = {"message": message, "recent_history": recent_history[-8:], "candidates": candidates[-20:]}
        try:
            structured = getattr(self.llm, "with_structured_output", None)
            if callable(structured):
                raw = structured(FollowupClassificationOutput, method="json_schema", strict=True).invoke([SystemMessage(content=_PROMPT), HumanMessage(content=json.dumps(payload, ensure_ascii=False))])
                output = FollowupClassificationOutput.model_validate(raw)
            else:
                raw = self.llm.invoke([SystemMessage(content=_PROMPT), HumanMessage(content=json.dumps(payload, ensure_ascii=False))])
                output = FollowupClassificationOutput.model_validate_json(str(getattr(raw, "content", raw)))
        except Exception as exc:
            raise FollowupClassificationError(f"Model decision failed: followup_classification. No fallback action was executed. Reason: {exc}") from exc
        offered = {str(item.get("task_id") or "") for item in candidates}
        needs_task = output.category in {"followup_task", "task_expansion", "task_correction", "retry_request", "resume_request", "status_request", "duplicate_message"}
        if needs_task and output.related_task_id not in offered:
            raise FollowupClassificationError("Model decision failed: followup_classification. No fallback action was executed. Reason: selected task was not offered.")
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
            raise FollowupClassificationError("Model decision failed: followup_classification. No fallback action was executed. Reason: non-task category selected a task.")
        if not output.safe_to_continue:
            raise FollowupClassificationError(
                "Model decision failed: followup_classification. No fallback action was "
                "executed. Reason: decision did not authorize continuation: "
                f"{output.reason}"
            )
        return FollowupClassification(**output.model_dump())
