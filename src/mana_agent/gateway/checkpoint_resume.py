"""Model decision boundary for reusing a stopped gateway task."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from mana_agent.context_cost.models import ContextBudgetExceeded
from mana_agent.evals.ids import stable_hash
from mana_agent.evals.recorder import record_current


class CheckpointResumeError(RuntimeError):
    """Raised when no valid stopped-task recovery decision is available."""

    def __init__(self, message: str, *, code: str = "checkpoint_resume_invalid") -> None:
        super().__init__(message)
        self.code = code


class CheckpointResumeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(min_length=1)
    action: Literal["resume_checkpoint", "retry_task", "replan_task", "start_fresh", "stop"]
    task_id: str = ""
    checkpoint_id: str = ""
    same_work: bool
    fresh_data_required: bool
    checkpoint_still_valid: bool
    side_effects_safe_to_repeat: bool
    safe_to_continue: bool
    reason: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class CheckpointResumeDecision:
    decision_id: str
    action: Literal["resume_checkpoint", "retry_task", "replan_task", "start_fresh", "stop"]
    task_id: str
    checkpoint_id: str
    same_work: bool
    fresh_data_required: bool
    checkpoint_still_valid: bool
    side_effects_safe_to_repeat: bool
    safe_to_continue: bool
    reason: str


CHECKPOINT_RESUME_PROMPT = """You decide whether a new user request may resume one durable checkpoint.
Return a decision only, never answer the user.

Compare the complete current request with every candidate's original normalized intent and progress.
Select resume_checkpoint only when it is the same work, the saved progress is still applicable, and
continuing will not substitute stale state for information that must be fetched again. Prices,
mailboxes and email checks, calendars, news, weather, availability, account state, remote system
state, search results, and similarly time-sensitive facts normally require start_fresh. Decide this
semantically from the supplied request and route evidence; do not use keyword matching. Repository
editing or analysis may resume only when the candidate evidence shows that its checkpoint remains
valid for the current request. Select retry_task when it is the same stable work and repeating its
unfinished actions is safe, but the candidate has no reusable checkpoint. Select replan_task when
the task identity and goal remain the same but its incomplete plan needs a model-selected revision
before it can safely continue. When uncertain, select
stop rather than guessing.

When incomplete work is the same, does not require fresh data, and is safe to continue, you must
select resume_checkpoint, retry_task, or replan_task for the applicable candidate; do not select start_fresh. Select
start_fresh only when the work is different or fresh data is required. If the work is the same but
cannot safely resume or repeat, select stop.

Completed results are returned only by the caller after it has already classified the turn as a
duplicate or status request; this decision boundary must never select or reuse a completed result.
Do not resume or retry a completed task. A requested downstream action, including an operation on a
live external provider, must select start_fresh so it receives its own task identity and approval.

For resume_checkpoint, copy one exact candidate task_id and checkpoint_id and set same_work,
checkpoint_still_valid, side_effects_safe_to_repeat, and safe_to_continue true and
fresh_data_required false. For start_fresh or stop, leave task_id and checkpoint_id empty. For
retry_task or replan_task, copy an exact candidate task_id, leave checkpoint_id empty, set same_work,
side_effects_safe_to_repeat, and safe_to_continue true, and set fresh_data_required false. Set
safe_to_continue true for start_fresh and false for stop. Return strict JSON matching the supplied
schema.
"""


class CheckpointResumeDecider:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def decide(
        self,
        *,
        current_request: str,
        route: str,
        requires_live_data: bool,
        candidates: list[dict[str, Any]],
    ) -> CheckpointResumeDecision:
        if self.llm is None or not callable(getattr(self.llm, "invoke", None)):
            raise CheckpointResumeError(
                "Model decision failed: checkpoint_resume. No task was resumed or started. "
                "Reason: decision model is unavailable."
            )
        payload = {
            "decision_time": datetime.now(timezone.utc).isoformat(),
            "current_request": current_request,
            "route": route,
            "entry_route_requires_live_data": requires_live_data,
            "recovery_candidates": candidates,
        }
        messages = [
            SystemMessage(content=CHECKPOINT_RESUME_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        ]
        try:
            structured = getattr(self.llm, "with_structured_output", None)
            if callable(structured):
                response = structured(
                    CheckpointResumeOutput, method="json_schema", strict=True
                ).invoke(messages)
                output = CheckpointResumeOutput.model_validate(response)
            else:
                response = self.llm.invoke(messages)
                content = getattr(response, "content", response)
                output = CheckpointResumeOutput.model_validate_json(str(content))
        except ContextBudgetExceeded as exc:
            raise CheckpointResumeError(
                "Model decision failed: checkpoint_resume. No task was resumed or started. "
                f"Reason: {exc}",
                code="context_budget_blocked",
            ) from exc
        except Exception as exc:
            raise CheckpointResumeError(
                "Model decision failed: checkpoint_resume. No task was resumed or started. "
                f"Reason: {exc}"
            ) from exc
        candidate_pairs = {
            (str(item["task_id"]), str(item["checkpoint_id"]))
            for item in candidates
            if str(item.get("checkpoint_id") or "")
            and str(item.get("state") or "") != "completed"
        }
        retryable_task_ids = {
            str(item["task_id"])
            for item in candidates
            if not str(item.get("checkpoint_id") or "")
            and str(item.get("state") or "") != "completed"
        }
        if output.action == "resume_checkpoint":
            if (output.task_id, output.checkpoint_id) not in candidate_pairs:
                raise CheckpointResumeError(
                    "Model decision failed: checkpoint_resume. No task was resumed or started. "
                    "Reason: selected checkpoint is not an offered durable candidate."
                )
            if not (
                output.same_work
                and output.checkpoint_still_valid
                and output.safe_to_continue
                and output.side_effects_safe_to_repeat
                and not output.fresh_data_required
                and not requires_live_data
            ):
                raise CheckpointResumeError(
                    "Model decision failed: checkpoint_resume. No task was resumed or started. "
                    "Reason: checkpoint reuse safety fields are inconsistent."
                )
        elif output.action in {"retry_task", "replan_task"}:
            if output.task_id not in retryable_task_ids or output.checkpoint_id:
                raise CheckpointResumeError(
                    "Model decision failed: checkpoint_resume. No task was resumed or started. "
                    "Reason: retry_task or replan_task must select one offered task without a checkpoint ID."
                )
            if not (
                output.same_work
                and output.side_effects_safe_to_repeat
                and output.safe_to_continue
                and not output.fresh_data_required
                and not requires_live_data
            ):
                raise CheckpointResumeError(
                    "Model decision failed: checkpoint_resume. No task was resumed or started. "
                    "Reason: same-task retry or replan safety fields are inconsistent."
                )
        else:
            if output.task_id or output.checkpoint_id:
                raise CheckpointResumeError(
                    "Model decision failed: checkpoint_resume. No task was resumed or started. "
                    "Reason: a non-resume decision must not select a task or checkpoint."
                )
            if output.action == "start_fresh" and not output.safe_to_continue:
                raise CheckpointResumeError(
                    "Model decision failed: checkpoint_resume. No task was resumed or started. "
                    "Reason: start_fresh was not authorized as safe to continue."
                )
            if (
                output.action == "start_fresh"
                and output.same_work
                and not output.fresh_data_required
            ):
                raise CheckpointResumeError(
                    "Model decision failed: checkpoint_resume. No task was resumed or started. "
                    "Reason: same stable work must reuse its stopped task identity."
                )
            if (
                output.action == "start_fresh"
                and requires_live_data
                and not output.fresh_data_required
            ):
                raise CheckpointResumeError(
                    "Model decision failed: checkpoint_resume. No task was resumed or started. "
                    "Reason: the live-data route was not marked as requiring fresh execution."
                )
            if output.action == "stop" and output.safe_to_continue:
                raise CheckpointResumeError(
                    "Model decision failed: checkpoint_resume. No task was resumed or started. "
                    "Reason: stop cannot also authorize continued execution."
                )
        decision = CheckpointResumeDecision(**output.model_dump())
        record_current(
            "model.decision",
            {
                "boundary": "checkpoint_resume",
                "prompt_hash": stable_hash(CHECKPOINT_RESUME_PROMPT),
                "request_hash": stable_hash(payload),
                "response": output.model_dump(),
            },
        )
        return decision
