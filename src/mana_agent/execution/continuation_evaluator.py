"""Model-directed evaluator for semantic continuation decisions."""

from __future__ import annotations

import json
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage

from mana_agent.execution.continuation_decision import (
    ContinuationAction,
    ContinuationDecision,
    ContinuationDecisionError,
)
from mana_agent.utils.text import extract_model_text


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
"""


def _coerce_continuation_decision(response: Any, task_id: str = "") -> ContinuationDecision:
    if isinstance(response, ContinuationDecision):
        return response
    if isinstance(response, dict):
        payload = dict(response)
        if not payload.get("task_id") and task_id:
            payload["task_id"] = task_id
        return ContinuationDecision.model_validate(payload)
    raw = getattr(response, "content", response)
    text = extract_model_text(raw)
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip().removesuffix("```").strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    data = json.loads(text)
    if isinstance(data, dict) and not data.get("task_id") and task_id:
        data["task_id"] = task_id
    return ContinuationDecision.model_validate(data)


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
    ) -> ContinuationDecision:
        if self.llm is None or not callable(getattr(self.llm, "invoke", None)):
            raise ContinuationDecisionError(
                "Model decision failed: continuation_decision. "
                "Reason: decision model is unavailable."
            )

        payload = {
            "task_id": task_id,
            "user_goal": user_goal,
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

        messages = [
            SystemMessage(content=_CONTINUATION_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)),
        ]

        try:
            structured = getattr(self.llm, "with_structured_output", None)
            response = None
            structured_error = None
            if callable(structured):
                try:
                    response = structured(
                        ContinuationDecision, method="json_schema", strict=True
                    ).invoke(messages)
                except Exception as exc:
                    structured_error = exc

            if response is None:
                response = self.llm.invoke(messages)

            try:
                decision = _coerce_continuation_decision(response, task_id=task_id)
            except Exception as coerce_exc:
                if callable(structured) and structured_error is None:
                    direct_response = self.llm.invoke(messages)
                    decision = _coerce_continuation_decision(direct_response, task_id=task_id)
                else:
                    raise coerce_exc

            if decision.task_id != task_id:
                object.__setattr__(decision, "task_id", task_id)
            return decision

        except Exception as exc:
            raise ContinuationDecisionError(
                f"Model decision failed: continuation_decision. Reason: {exc}"
            ) from exc
