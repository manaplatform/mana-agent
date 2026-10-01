"""Model-driven approval request and wait tools for durable human-in-the-loop decisions."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from pydantic import BaseModel, Field

try:
    from langchain_core.tools import StructuredTool
except ImportError:
    from langchain.tools import StructuredTool  # type: ignore[no-redef]

from mana_agent.human_inbox.models import (
    InboxRequest,
    InboxRequestType,
    InboxStatus,
    ResponseOperation,
    ReviewerAssignment,
    ReviewerType,
    RiskLevel,
    TERMINAL_STATUSES,
    UNRESOLVED_STATUSES,
)
from mana_agent.human_inbox.service import HumanInboxService


class ApprovalWaitResult(dict):
    """Result returned by wait_for_approval supporting both dict and attribute access."""

    def __init__(
        self,
        *,
        verdict: str,
        may_continue: bool,
        inbox_item_id: str,
        status: str = "",
        response: Any = None,
        reason: str = "",
    ) -> None:
        super().__init__(
            verdict=verdict,
            may_continue=may_continue,
            inbox_item_id=inbox_item_id,
            status=status,
            response=response,
            reason=reason,
        )

    @property
    def verdict(self) -> str:
        return self["verdict"]

    @property
    def may_continue(self) -> bool:
        return self["may_continue"]

    @property
    def inbox_item_id(self) -> str:
        return self["inbox_item_id"]

    @property
    def status(self) -> str:
        return self["status"]

    @property
    def response(self) -> Any:
        return self["response"]

    @property
    def reason(self) -> str:
        return self["reason"]


def _validate_decision_and_identity(source_decision_id: str, task_id: str, agent_id: str) -> None:
    if not source_decision_id or not str(source_decision_id).strip():
        raise ValueError(
            "The model did not return a valid source_decision_id. No fallback decision was executed."
        )
    if not task_id or not str(task_id).strip():
        raise ValueError("A valid task_id is required; no fallback decision was executed.")
    if not agent_id or not str(agent_id).strip():
        raise ValueError("A valid agent_id is required; no fallback decision was executed.")


def _resolve_inbox_service(inbox_service: HumanInboxService | None = None) -> HumanInboxService:
    if inbox_service is not None:
        return inbox_service
    from mana_agent.human_inbox import default_human_inbox_service

    return default_human_inbox_service()


def request_user_approval(
    source_decision_id: str,
    task_id: str,
    agent_id: str,
    *,
    title: str = "",
    summary: str = "",
    risk_level: RiskLevel | str = RiskLevel.HIGH,
    reviewer_id: str = "",
    reviewer_type: ReviewerType | str = ReviewerType.PERSON,
    branch_id: str = "",
    checkpoint_id: str = "",
    minimal_context: dict[str, Any] | None = None,
    protected_context: dict[str, Any] | None = None,
    expires_at: datetime | None = None,
    inbox_service: HumanInboxService | None = None,
) -> str:
    """Create a pending human approval request in the durable inbox and return inbox_item_id."""
    _validate_decision_and_identity(source_decision_id, task_id, agent_id)
    service = _resolve_inbox_service(inbox_service)

    resolved_reviewer_id = str(reviewer_id or "").strip()
    if not resolved_reviewer_id and hasattr(service, "identities"):
        identities_dict = getattr(service.identities, "_identities", {})
        for ident_id, ident in identities_dict.items():
            if getattr(ident, "active", True) and ident_id != agent_id:
                resolved_reviewer_id = ident_id
                break
    if not resolved_reviewer_id:
        resolved_reviewer_id = "reviewer-1"

    r_type = (
        ReviewerType(reviewer_type) if isinstance(reviewer_type, str) else reviewer_type
    )
    reviewer = ReviewerAssignment(reviewer_type=r_type, reviewer_id=resolved_reviewer_id)

    if isinstance(risk_level, str):
        try:
            r_risk = RiskLevel(risk_level.lower())
        except ValueError:
            r_risk = RiskLevel.HIGH
    elif isinstance(risk_level, RiskLevel):
        r_risk = risk_level
    else:
        r_risk = RiskLevel.HIGH

    idempotency_key = f"approval_{task_id}_{source_decision_id}"
    deduplication_key = f"approval_{task_id}_{source_decision_id}"

    context = dict(minimal_context or {})
    context.setdefault("source_decision_id", source_decision_id)
    context.setdefault("task_id", task_id)
    context.setdefault("agent_id", agent_id)

    if expires_at is None:
        clock_fn = getattr(service, "clock", None)
        base_now = clock_fn() if callable(clock_fn) else datetime.now(timezone.utc)
        expires_at = base_now + timedelta(days=7)

    request = InboxRequest(
        request_type=InboxRequestType.APPROVAL,
        task_id=task_id,
        branch_id=branch_id or task_id,
        checkpoint_id=checkpoint_id or "",
        policy_decision_id=source_decision_id,
        requested_by_agent_id=agent_id,
        reviewer=reviewer,
        title=title or f"Approval required for task {task_id}",
        summary=summary or f"Approval requested by agent {agent_id} for decision {source_decision_id}.",
        risk_level=r_risk,
        allowed_responses=[ResponseOperation.APPROVE, ResponseOperation.DENY],
        minimal_context=context,
        protected_context=dict(protected_context or {}),
        idempotency_key=idempotency_key,
        deduplication_key=deduplication_key,
        expires_at=expires_at,
    )
    item = service.create(request)
    return item.inbox_item_id


def wait_for_approval(
    inbox_item_id: str,
    source_decision_id: str,
    task_id: str,
    agent_id: str,
    *,
    timeout_seconds: float = 60.0,
    poll_interval_seconds: float = 0.5,
    on_timeout_suspend: Any = None,
    execution_supervisor: Any = None,
    checkpoint_id: str = "",
    inbox_service: HumanInboxService | None = None,
) -> ApprovalWaitResult:
    """Poll the durable inbox until terminal verdict or timeout (max 600s).

    Only approved verdict returns may_continue=True.
    """
    _validate_decision_and_identity(source_decision_id, task_id, agent_id)
    if not inbox_item_id or not str(inbox_item_id).strip():
        raise ValueError("A valid inbox_item_id is required; no fallback decision was executed.")

    service = _resolve_inbox_service(inbox_service)
    max_timeout = min(max(0.0, float(timeout_seconds)), 600.0)
    poll_interval = max(0.005, float(poll_interval_seconds))
    start_time = time.monotonic()
    last_observation = None

    while True:
        obs = service.observe_for_agent(
            inbox_item_id,
            requesting_agent_id=agent_id,
            task_id=task_id,
        )
        last_observation = obs

        if obs.status is InboxStatus.APPROVED:
            return ApprovalWaitResult(
                verdict="approved",
                may_continue=True,
                inbox_item_id=inbox_item_id,
                status=obs.status.value,
                response=obs.response.model_dump(mode="json") if obs.response else None,
            )
        if obs.status is InboxStatus.DENIED:
            return ApprovalWaitResult(
                verdict="denied",
                may_continue=False,
                inbox_item_id=inbox_item_id,
                status=obs.status.value,
                response=obs.response.model_dump(mode="json") if obs.response else None,
            )
        if obs.status is InboxStatus.EXPIRED:
            return ApprovalWaitResult(
                verdict="expired",
                may_continue=False,
                inbox_item_id=inbox_item_id,
                status=obs.status.value,
                response=obs.response.model_dump(mode="json") if obs.response else None,
            )
        if obs.status in {InboxStatus.CANCELLED, InboxStatus.SUPERSEDED}:
            return ApprovalWaitResult(
                verdict="cancelled",
                may_continue=False,
                inbox_item_id=inbox_item_id,
                status=obs.status.value,
                response=obs.response.model_dump(mode="json") if obs.response else None,
            )
        if obs.status not in UNRESOLVED_STATUSES:
            return ApprovalWaitResult(
                verdict=obs.status.value,
                may_continue=False,
                inbox_item_id=inbox_item_id,
                status=obs.status.value,
                response=obs.response.model_dump(mode="json") if obs.response else None,
            )

        elapsed = time.monotonic() - start_time
        if elapsed >= max_timeout:
            break
        remaining = max_timeout - elapsed
        time.sleep(min(poll_interval, remaining))

    # Timeout reached without terminal approval
    _trigger_timeout_suspend(
        on_timeout_suspend=on_timeout_suspend,
        execution_supervisor=execution_supervisor,
        task_id=task_id,
        inbox_item_id=inbox_item_id,
        checkpoint_id=checkpoint_id,
    )

    return ApprovalWaitResult(
        verdict="still_pending",
        may_continue=False,
        inbox_item_id=inbox_item_id,
        status=last_observation.status.value if last_observation else InboxStatus.PENDING.value,
        reason="timeout elapsed before approval was granted",
    )


def _trigger_timeout_suspend(
    *,
    on_timeout_suspend: Any,
    execution_supervisor: Any,
    task_id: str,
    inbox_item_id: str,
    checkpoint_id: str,
) -> None:
    if on_timeout_suspend is not None:
        if callable(on_timeout_suspend):
            try:
                on_timeout_suspend(
                    task_id,
                    inbox_item_id=inbox_item_id,
                    checkpoint_id=checkpoint_id,
                    request_type="approval",
                )
            except TypeError:
                try:
                    on_timeout_suspend(task_id, inbox_item_id=inbox_item_id)
                except TypeError:
                    try:
                        on_timeout_suspend(task_id)
                    except TypeError:
                        on_timeout_suspend()
        elif hasattr(on_timeout_suspend, "suspend_for_human_input"):
            on_timeout_suspend.suspend_for_human_input(
                task_id,
                inbox_item_id=inbox_item_id,
                checkpoint_id=checkpoint_id,
                request_type="approval",
            )
    elif execution_supervisor is not None and hasattr(execution_supervisor, "suspend_for_human_input"):
        execution_supervisor.suspend_for_human_input(
            task_id,
            inbox_item_id=inbox_item_id,
            checkpoint_id=checkpoint_id,
            request_type="approval",
        )


class RequestUserApprovalInput(BaseModel):
    source_decision_id: str = Field(description="ID of the validated model decision requiring approval.")
    task_id: str = Field(description="Task ID for this approval request.")
    agent_id: str = Field(description="Agent ID requesting the approval.")
    title: str = Field(default="User approval required", description="Title for the approval request.")
    summary: str = Field(default="", description="Summary explaining why approval is needed.")
    risk_level: str = Field(default="high", description="Risk level (low, medium, high, critical).")
    reviewer_id: str = Field(default="", description="Optional reviewer ID.")


class WaitForApprovalInput(BaseModel):
    inbox_item_id: str = Field(description="The inbox_item_id to wait for.")
    source_decision_id: str = Field(description="ID of the validated model decision.")
    task_id: str = Field(description="Task ID.")
    agent_id: str = Field(description="Agent ID.")
    timeout_seconds: float = Field(default=60.0, description="Timeout in seconds (max 600s).")


def build_approval_tools(
    inbox_service: HumanInboxService | None = None,
    execution_supervisor: Any = None,
    on_timeout_suspend: Any = None,
) -> list[StructuredTool]:
    """Return LangChain StructuredTools for model-driven human approval requests and waiting."""
    metadata = {"read_only": True, "inbox_only": True}

    def _req_tool(
        source_decision_id: str,
        task_id: str,
        agent_id: str,
        title: str = "User approval required",
        summary: str = "",
        risk_level: str = "high",
        reviewer_id: str = "",
    ) -> str:
        return request_user_approval(
            source_decision_id=source_decision_id,
            task_id=task_id,
            agent_id=agent_id,
            title=title,
            summary=summary,
            risk_level=risk_level,
            reviewer_id=reviewer_id,
            inbox_service=inbox_service,
        )

    def _wait_tool(
        inbox_item_id: str,
        source_decision_id: str,
        task_id: str,
        agent_id: str,
        timeout_seconds: float = 60.0,
    ) -> dict[str, Any]:
        result = wait_for_approval(
            inbox_item_id=inbox_item_id,
            source_decision_id=source_decision_id,
            task_id=task_id,
            agent_id=agent_id,
            timeout_seconds=timeout_seconds,
            on_timeout_suspend=on_timeout_suspend,
            execution_supervisor=execution_supervisor,
            inbox_service=inbox_service,
        )
        return dict(result)

    return [
        StructuredTool.from_function(
            func=_req_tool,
            name="request_user_approval",
            description=(
                "Create a pending human approval request in the durable inbox for a model-decided action. "
                "Runs nothing and returns the inbox_item_id."
            ),
            args_schema=RequestUserApprovalInput,
            metadata=metadata,
        ),
        StructuredTool.from_function(
            func=_wait_tool,
            name="wait_for_approval",
            description=(
                "Wait for a human approval decision on a pending inbox item. "
                "Returns verdict and may_continue (True only if approved)."
            ),
            args_schema=WaitForApprovalInput,
            metadata=metadata,
        ),
    ]


__all__ = [
    "ApprovalWaitResult",
    "RequestUserApprovalInput",
    "WaitForApprovalInput",
    "build_approval_tools",
    "request_user_approval",
    "wait_for_approval",
]
