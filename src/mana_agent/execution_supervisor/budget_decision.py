"""Fresh model decision boundary for durable execution-budget overruns."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mana_agent.execution_supervisor.models import (
    BudgetOverrunFinalizationDecision,
    ExecutionState,
    TaskRecord,
)


class BudgetOverrunDecisionError(RuntimeError):
    """Raised when the required finalization decision is absent or invalid."""


_PROMPT = """You are Mana-Agent's execution-budget finalization decision maker.
Return only one strict JSON BudgetOverrunFinalizationDecision.

The task already has a durable result but exceeded an immutable budget. Do not
change budgets, select a new provider, invoke tools, or invent a fallback. Choose
accept_with_overrun only when the supplied result evidence says completion
verification passed. Choose require_review when evidence is incomplete, uncertain,
or needs an operator. Choose retry_or_replan only with a complete nested recovery
decision and only when the supplied recovery allocation is explicitly available.
Every identifier and evidence hash must be copied exactly from the input.

safe_to_continue must be true for every valid decision object. It means this
finalization decision itself is authorized to apply — not that the original task
should keep spending budget automatically. require_review is a valid finalization
(hand off to human review) and still requires safe_to_continue=true.
accept_with_overrun also requires safe_to_continue=true. retry_or_replan requires
safe_to_continue=true on both the outer decision and the nested recovery decision.
"""


class BudgetOverrunDecider:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def decide(self, task: TaskRecord, *, result_payload: dict[str, Any]) -> BudgetOverrunFinalizationDecision:
        if task.state is not ExecutionState.PENDING_BUDGET_DECISION:
            raise BudgetOverrunDecisionError("task is not awaiting a budget-overrun decision")
        if self.llm is None or not callable(getattr(self.llm, "invoke", None)):
            raise BudgetOverrunDecisionError(
                "Model decision failed: budget_overrun_finalization. No result was finalized. "
                "Reason: decision model is unavailable."
            )
        payload = {
            "task_id": task.task_id,
            "attempt_id": task.attempt_id,
            "result_id": task.result_id,
            "budget_overrun": task.budget_overrun,
            "token_budget": task.token_budget,
            "token_usage": task.token_usage,
            "monetary_budget": task.monetary_budget,
            "actual_cost": task.actual_cost,
            "verification_status": task.verification_status.value,
            "result_payload": result_payload,
            "available_recovery_tokens": (
                None if task.token_budget is None else max(0, task.token_budget - task.token_usage)
            ),
            "available_recovery_cost": (
                None if task.monetary_budget is None else max(0.0, task.monetary_budget - task.actual_cost)
            ),
        }
        messages = [
            SystemMessage(content=_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)),
        ]
        try:
            structured = getattr(self.llm, "with_structured_output", None)
            if callable(structured):
                response = structured(
                    BudgetOverrunFinalizationDecision, method="json_schema", strict=True
                ).invoke(messages)
                return BudgetOverrunFinalizationDecision.model_validate(response)
            response = self.llm.invoke(messages)
            return BudgetOverrunFinalizationDecision.model_validate_json(
                str(getattr(response, "content", response))
            )
        except Exception as exc:
            raise BudgetOverrunDecisionError(
                "Model decision failed: budget_overrun_finalization. No result was finalized. "
                f"Reason: {exc}"
            ) from exc

