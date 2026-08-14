"""Model decision boundary for reusing a stopped gateway task."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mana_agent.context_cost.models import ContextBudgetExceeded
from mana_agent.evals.ids import stable_hash
from mana_agent.evals.recorder import record_current
from mana_agent.execution_supervisor.models import TERMINAL_STATES

# Compact structured JSON is small (reason ≤ 480 chars), but some providers
# (notably NVIDIA DeepSeek V4 with default thinking/reasoning_effort=high)
# count chain-of-thought tokens against max_tokens. A 512-token ceiling was
# exhausted by thinking alone, producing LengthFinishReasonError before the
# schema JSON completed. Keep explicit headroom for that class of models.
CHECKPOINT_RESUME_MAX_OUTPUT_TOKENS = 4_096


class CheckpointResumeError(RuntimeError):
    """Raised when no valid stopped-task recovery decision is available."""

    def __init__(self, message: str, *, code: str = "checkpoint_resume_invalid") -> None:
        super().__init__(message)
        self.code = code


def _checkpoint_resume_failure_reason(exc: BaseException) -> str:
    """Format a provider/model failure without inventing a recovery action."""
    detail = str(exc).strip() or type(exc).__name__
    lowered = detail.casefold()
    if (
        type(exc).__name__ == "LengthFinishReasonError"
        or "length limit was reached" in lowered
        or "lengthfinishreason" in lowered
    ):
        return (
            f"{detail}. Structured checkpoint_resume output was truncated at "
            f"max_tokens={CHECKPOINT_RESUME_MAX_OUTPUT_TOKENS}; the model did "
            "not return a complete decision schema."
        )
    return detail


class CheckpointResumeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["resume_checkpoint", "retry_task", "replan_task", "start_fresh", "stop"]
    task_id: str = ""
    checkpoint_id: str = ""
    same_work: bool
    fresh_data_required: bool
    checkpoint_still_valid: bool
    side_effects_safe_to_repeat: bool
    safe_to_continue: bool
    reason: str = Field(min_length=1)

    @field_validator("reason", mode="before")
    @classmethod
    def _normalize_reason(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()



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

Chat turns auto-select durable work without requiring /tasks. Use this decision matrix:
1) same incomplete work with a still-valid, eligible checkpoint → resume_checkpoint (continue saved progress).
   Terminal tasks (completed, failed, cancelled, budget_exhausted) cannot be resumed via resume_checkpoint.
2) same work that failed or was interrupted, side effects safe to repeat → retry_task (same task id,
   creates a new attempt, leave checkpoint_id empty)
3) same work whose multi-task/job plan was blocked, reverted, or must restart incomplete steps →
   replan_task (same task id, leave checkpoint_id empty so the job restarts from its first
   incomplete step; completed children may remain complete)
4) different work, live/fresh data required, or recovery_candidates empty → start_fresh (new task)
5) same work but no listed candidate is safe → stop (never invent a task id)

Compare the complete current request with every candidate's original normalized intent and progress.
Select resume_checkpoint only when it is the same work, the saved progress is still applicable, and
continuing will not substitute stale state for information that must be fetched again. Prices,
mailboxes and email checks, calendars, news, weather, availability, account state, remote system
state, search results, and similarly time-sensitive facts normally require start_fresh. Decide this
semantically from the supplied request and route evidence; do not use keyword matching. Repository
editing or analysis may resume only when the candidate evidence shows that its checkpoint remains
valid for the current request. Prefer resume_checkpoint when checkpoint_available is true and the
saved progress should continue. Select retry_task when it is the same stable work and repeating its
unfinished actions is safe under the existing task identity—even if a checkpoint is listed—because a
full restart under that identity is safer or more appropriate than continuing partial progress.
Select replan_task when the task identity and goal remain the same but its incomplete plan needs a
model-selected revision before it can safely continue, including blocked multi-task roots and
reverted compound jobs that should start again from the first incomplete step. Candidates may
report lane_state blocked/paused/waiting for such jobs. When uncertain, select stop rather than
guessing.

When incomplete work is the same, does not require fresh data, and is safe to continue, and a
recoverable candidate is listed, you must select resume_checkpoint, retry_task, or replan_task for
the applicable candidate; do not select start_fresh. Select start_fresh when the work is different,
fresh data is required, or recovery_candidates is empty. An empty candidate list means prior
attempts for this work are not recoverable (including wall-clock deadline-dead tasks); start_fresh
then creates a new task identity with a fresh deadline. If the work is the same but a listed
candidate cannot safely resume or repeat, select stop rather than inventing a non-listed task id.

Completed results are returned only by the caller after it has already classified the turn as a
duplicate or status request; this decision boundary must never select or reuse a completed result.
Do not resume or retry a completed task. Never resume or retry a task whose wall-clock deadline has
elapsed; those tasks are excluded from recovery_candidates and require start_fresh. A requested
downstream action, including an operation on a live external provider, must select start_fresh so it
receives its own task identity and approval.

For resume_checkpoint, copy one exact candidate task_id and checkpoint_id and set same_work,
checkpoint_still_valid, side_effects_safe_to_repeat, and safe_to_continue true and
fresh_data_required false. For start_fresh or stop, leave task_id and checkpoint_id empty. For
retry_task or replan_task, copy an exact non-completed candidate task_id, leave checkpoint_id empty
(even when the candidate lists a checkpoint—these actions intentionally do not resume that
checkpoint), set same_work, side_effects_safe_to_repeat, and safe_to_continue true, and set
fresh_data_required false. Set safe_to_continue true for start_fresh and false for stop. When
recovery_candidates is empty and the work is the same, start_fresh with same_work true is valid.
Return strict JSON matching the supplied schema.
"""


def _coerce_checkpoint_output(response: Any) -> CheckpointResumeOutput:
    if isinstance(response, CheckpointResumeOutput):
        return response
    if isinstance(response, dict):
        return CheckpointResumeOutput.model_validate(response)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = " ".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in content
        )
    text = str(content).strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    return CheckpointResumeOutput.model_validate_json(text)


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
        invoke_kwargs = {"max_tokens": CHECKPOINT_RESUME_MAX_OUTPUT_TOKENS}
        try:
            structured = getattr(self.llm, "with_structured_output", None)
            if callable(structured):
                response = structured(
                    CheckpointResumeOutput, method="json_schema", strict=True
                ).invoke(messages, **invoke_kwargs)
            else:
                response = self.llm.invoke(messages, **invoke_kwargs)
            output = _coerce_checkpoint_output(response)
        except ContextBudgetExceeded as exc:
            raise CheckpointResumeError(
                "Model decision failed: checkpoint_resume. No task was resumed or started. "
                f"Reason: {exc}",
                code="context_budget_blocked",
            ) from exc
        except Exception as exc:
            raise CheckpointResumeError(
                "Model decision failed: checkpoint_resume. No task was resumed or started. "
                f"Reason: {_checkpoint_resume_failure_reason(exc)}"
            ) from exc
        candidate_pairs = {
            (str(item["task_id"]), str(item["checkpoint_id"]))
            for item in candidates
            if str(item.get("checkpoint_id") or "")
            and str(item.get("state") or "") not in {s.value for s in TERMINAL_STATES}
            and not bool(item.get("is_terminal"))
            and bool(item.get("resume_eligible", True))
        }
        # Same-task restart/replan may target any non-completed offered task,
        # including ones that still list a checkpoint. Those actions intentionally
        # leave checkpoint_id empty so partial progress is not resumed.
        retryable_task_ids = {
            str(item["task_id"])
            for item in candidates
            if str(item.get("state") or "") != "completed"
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
            if output.task_id not in retryable_task_ids:
                raise CheckpointResumeError(
                    "Model decision failed: checkpoint_resume. No task was resumed or started. "
                    "Reason: retry_task or replan_task must select one offered non-completed task."
                )
            if output.checkpoint_id:
                raise CheckpointResumeError(
                    "Model decision failed: checkpoint_resume. No task was resumed or started. "
                    "Reason: retry_task or replan_task must leave checkpoint_id empty; "
                    "use resume_checkpoint to continue saved progress."
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
                and candidates
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
        fields = output.model_dump()
        fields["decision_id"] = f"checkpoint:{uuid.uuid4().hex[:12]}"
        decision = CheckpointResumeDecision(**fields)
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
