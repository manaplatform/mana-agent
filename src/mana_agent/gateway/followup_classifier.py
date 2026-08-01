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
    safe_to_continue: bool
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
task; use status_request or conversation_only when no execution should be created. Do not use keyword
matching. If no offered task is unambiguously applicable, select new_task or conversation_only. Return
strict JSON matching the schema and select only an offered task ID."""


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
        if not needs_task and output.related_task_id:
            raise FollowupClassificationError("Model decision failed: followup_classification. No fallback action was executed. Reason: non-task category selected a task.")
        if not output.safe_to_continue:
            raise FollowupClassificationError("Model decision failed: followup_classification. No fallback action was executed. Reason: decision did not authorize continuation.")
        return FollowupClassification(**output.model_dump())
