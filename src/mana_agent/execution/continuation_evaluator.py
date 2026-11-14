"""Model-directed evaluator for semantic continuation decisions."""

from __future__ import annotations

import json
from typing import Any
from mana_agent.llm.messages import UserMessage, SystemMessage

from mana_agent.execution.continuation_decision import (
    ContinuationAction,
    ContinuationDecision,
    ContinuationDecisionError,
    CodingContinuationDecision,
)
from mana_agent.utils.text import extract_model_text
from mana_agent.batch.policy import BatchPending


_CONTINUATION_PROMPT = """You are Mana-Agent's authoritative semantic continuation evaluator.
Your role is to evaluate execution progress against the user's objective and decide whether to
CONTINUE, FINALIZE, REPLAN, WAIT_FOR_USER, or mark the task BLOCKED.

You must return strictly one valid JSON object conforming to ContinuationDecision schema:
{
  "task_id": "exact task id",
  "action": "CONTINUE" | "FINALIZE" | "REPLAN" | "WAIT_FOR_USER" | "BLOCKED",
  "goal_satisfied": true | false,
  "progress_detected": true | false,
  "evidence_summary": "concrete evidence observed in this step or cumulative progress",
  "remaining_work": "exact description of remaining steps needed (empty if FINALIZE)",
  "user_message": "message to present to user if FINALIZE, WAIT_FOR_USER, or BLOCKED",
  "verification_passed": true | false,
  "safe_to_continue": true | false,
  "renewal_requested": true | false,
  "renewal_reason": "reason if requesting budget extension, else empty",
  "public_summary": "Optional concise user-facing progress, finding, or decision; no private reasoning, instructions, or secrets",
  "reason": "justification for this continuation decision"
}

Critical Rules:
1. NEVER choose FINALIZE simply because no tool call was returned, the assistant output was empty,
   or an iteration cycle ended. FINALIZE requires that concrete evidence proves the user's goal
   is completely satisfied.
2. If the user's goal is satisfied on turn 1 (e.g. simple conversational greeting, informational query,
   or quick lookup that is fully answered), select FINALIZE immediately without unnecessary looping.
3. If more work, tools, verification, or modifications are needed, select CONTINUE or REPLAN.
4. If soft execution limits are approached or reached and the task is progressing, set
   renewal_requested=true and provide renewal_reason.
5. If user approval, clarification, or credentials are required before proceeding, select WAIT_FOR_USER.
6. If the task is blocked by an unrecoverable failure, policy denial, or missing capability, select BLOCKED.
7. progress_detected must be false if the step produced identical output, repeated failing errors,
   or made no observable advancement toward the goal.
8. safe_to_continue must be true for CONTINUE, REPLAN, and FINALIZE; false for BLOCKED.
9. Return JSON only, with no markdown fences, no conversational prose, and no fallback defaults.
10. Read taskboard_summary if present to inspect authoritative task progress. A checked-off task is not
    necessarily independently verified; ensure verification requirements are met before FINALIZE. If all
    current taskboard tasks are done but the user's overall goal is not yet satisfied, select CONTINUE or REPLAN.
"""


def _coerce_continuation_decision(response: Any, task_id: str = "", *, strict: bool = False) -> ContinuationDecision:
    if isinstance(response, ContinuationDecision):
        if strict:
            _validate_explicit_coding_decision(response.model_dump(), response.model_fields_set, task_id)
            if response.task_id in {"task_id", ""}:
                object.__setattr__(response, "task_id", task_id)
        return response
    if isinstance(response, dict):
        payload = dict(response)
    else:
        raw = getattr(response, "content", response)
        text = extract_model_text(raw)
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").strip().removesuffix("```").strip()
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
        data = json.loads(text)
        payload = dict(data) if isinstance(data, dict) else {}

    if strict:
        _validate_explicit_coding_decision(payload, set(payload), task_id)
        return ContinuationDecision.model_validate(payload)

    if not payload.get("task_id") and task_id:
        payload["task_id"] = task_id

    # If the response was an intent/routing payload from a routing model
    if not payload.get("action") and payload.get("intent"):
        intent = str(payload.get("intent") or "").lower()
        if intent in {"answer", "conversation"}:
            payload["action"] = ContinuationAction.FINALIZE
            payload["goal_satisfied"] = True
            payload["safe_to_continue"] = True
        elif intent in {"web_research", "research", "coding", "code", "analyze", "planning", "plan"}:
            payload["action"] = ContinuationAction.FINALIZE
            payload["goal_satisfied"] = True
            payload["safe_to_continue"] = True

    if payload.get("action"):
        if "goal_satisfied" not in payload:
            payload["goal_satisfied"] = payload.get("action") == ContinuationAction.FINALIZE
        if "safe_to_continue" not in payload:
            payload["safe_to_continue"] = payload.get("action") != ContinuationAction.BLOCKED
        if not payload.get("evidence_summary"):
            payload["evidence_summary"] = str(payload.get("reasoning_summary") or payload.get("reason") or "Progress evaluated")
        if not payload.get("reason"):
            payload["reason"] = str(payload.get("reasoning_summary") or payload.get("evidence_summary") or "Continuation evaluated")
        if "progress_detected" not in payload:
            payload["progress_detected"] = True

    return ContinuationDecision.model_validate(payload)


def _validate_explicit_coding_decision(payload: dict[str, Any], fields: set[str], task_id: str) -> None:
    required = {"task_id", "action", "goal_satisfied", "progress_detected", "evidence_summary", "safe_to_continue", "reason"}
    missing = required - fields
    if missing or fields - set(ContinuationDecision.model_fields):
        raise ValueError(f"Coding continuation requires explicit schema fields; missing={sorted(missing)}")
    if payload.get("task_id") in {"task_id", ""}:
        payload["task_id"] = task_id
    elif payload.get("task_id") != task_id:
        raise ValueError("Coding continuation selected a different task")
    for key in ("goal_satisfied", "progress_detected", "safe_to_continue"):
        if type(payload[key]) is not bool:
            raise ValueError(f"Coding continuation {key} must be a boolean")


class ContinuationEvaluator:
    """Evaluates task progress and produces validated ContinuationDecision."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def evaluate(
        self,
        *,
        task_id: str,
        user_goal: str,
        step_output: Any,
        completed_steps: list[dict[str, Any]],
        metrics: dict[str, Any],
        soft_limit_reached: bool = False,
        consecutive_no_progress_cycles: int = 0,
        independent_verification_status: str | None = None,
        taskboard_progress: dict[str, Any] | None = None,
        coding_context: dict[str, Any] | None = None,
    ) -> ContinuationDecision:
        if self.llm is None or not callable(getattr(self.llm, "generate_sync", None)):
            raise ContinuationDecisionError(
                "Model decision failed: continuation_decision. "
                "Reason: decision model is unavailable."
            )

        payload: dict[str, Any] = {
            "task_id": task_id,
            "user_goal": user_goal,
            "user_request": user_goal,
            "step_output": step_output,
            "completed_steps_count": len(completed_steps),
            "completed_steps_summary": [
                {
                    "step": s.get("step_index"),
                    "tool": s.get("tool_name"),
                    "status": s.get("status"),
                    "summary": str(s.get("summary") or "")[:200],
                }
                for s in completed_steps[-5:]
            ],
            "metrics": metrics,
            "soft_limit_reached": soft_limit_reached,
            "consecutive_no_progress_cycles": consecutive_no_progress_cycles,
            "independent_verification_status": independent_verification_status,
        }
        if taskboard_progress:
            payload["taskboard_summary"] = {
                "total_tasks": taskboard_progress.get("total_tasks", 0),
                "completed_tasks": taskboard_progress.get("completed_tasks", 0),
                "verified_tasks": taskboard_progress.get("verified_tasks", 0),
                "percent_complete": taskboard_progress.get("percent_complete", 0.0),
                "tasks": [
                    {
                        "task_id": t.get("task_id"),
                        "title": t.get("title"),
                        "status": t.get("status"),
                        "crossed_off": t.get("crossed_off"),
                        "verification_status": t.get("verification_status"),
                        "blockers": t.get("blockers"),
                    }
                    for t in (taskboard_progress.get("tasks") or [])
                ],
            }

        system_prompt = _CONTINUATION_PROMPT
        if coding_context is not None:
            from mana_agent.coding_research_contracts import CODING_RESEARCH_DECISION_INSTRUCTION
            system_prompt += "\n" + CODING_RESEARCH_DECISION_INSTRUCTION
            payload["coding_context"] = coding_context

        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)),
        ]

        try:
            response = self.llm.decide_sync(messages, CodingContinuationDecision if coding_context is not None else ContinuationDecision)
            decision = _coerce_continuation_decision(response, task_id=task_id, strict=coding_context is not None)

            if decision.task_id != task_id:
                reconciled = False
                if decision.task_id in {"task_id", ""}:
                    reconciled = True
                elif taskboard_progress:
                    tasks = taskboard_progress.get("tasks") or []
                    known = {str(t.get("task_id")) for t in tasks} | {str(t.get("title")) for t in tasks}
                    known.add(str(taskboard_progress.get("root_task_id")))
                    if decision.task_id in known or "task-tb" in decision.task_id:
                        reconciled = True
                if reconciled:
                    object.__setattr__(decision, "task_id", task_id)
                else:
                    raise ValueError("continuation decision selected a different task")
            return decision

        except BatchPending:
            raise
        except Exception as exc:
            if coding_context is not None:
                # Validation exceptions can embed the rejected raw model payload.
                # Keep actionable field/type diagnostics without echoing secrets.
                details = type(exc).__name__
                if isinstance(exc, ValueError) and str(exc).startswith(("Coding continuation", "Main Model returned")):
                    details = str(exc)
                if callable(getattr(exc, "errors", None)):
                    details = "; ".join(f"{'.'.join(map(str, error['loc']))}: {error['type']}" for error in exc.errors(include_input=False, include_context=False))
                raise ContinuationDecisionError("Main Model coding continuation decision is invalid: " + details) from None
            raise ContinuationDecisionError(
                f"Model decision failed: continuation_decision. Reason: {exc}"
            ) from exc
