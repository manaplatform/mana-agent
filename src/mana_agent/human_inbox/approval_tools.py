"""Model-driven approval request and wait tools for durable human-in-the-loop decisions."""

from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
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


class ApprovalRequestInput(BaseModel):
    command: str = Field(
        default="",
        description="Shell command or operation that requires approval to run.",
    )
    title: str = Field(
        default="User approval required",
        description="Title for the approval request describing what needs approval.",
    )
    reason: str = Field(
        default="",
        description="Explanation of why this action or command needs to run and its intended effects.",
    )
    inbox_item_id: str = Field(
        default="",
        description="Existing inbox item ID if awaiting an already-created approval request.",
    )
    action_id: str = Field(
        default="",
        description="Existing action intent ID if awaiting an already-proposed action.",
    )
    risk_level: str = Field(
        default="medium",
        description="Risk level ('low', 'medium', 'high', 'critical').",
    )
    timeout_seconds: float = Field(
        default=60.0,
        description="Seconds to wait for user approval (1 to 300s).",
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


def approval_request(
    *,
    command: str = "",
    title: str = "User approval required",
    reason: str = "",
    inbox_item_id: str = "",
    action_id: str = "",
    risk_level: str = "medium",
    timeout_seconds: float = 60.0,
    workspace_root: Any = None,
    inbox_service: HumanInboxService | None = None,
    source_decision_id: str = "",
    task_id: str = "",
    agent_id: str = "",
) -> dict[str, Any]:
    """Request human approval for a command or action, emit live approval events, and wait for verdict.

    Returns a dictionary with approved status, approval_id (grant), inbox_item_id, and explanatory message.
    """
    service = _resolve_inbox_service(inbox_service)
    root = Path(workspace_root) if workspace_root is not None else Path.cwd()
    cmd = str(command or "").strip()
    req_inbox_id = str(inbox_item_id or "").strip()
    req_action_id = str(action_id or "").strip()

    if not cmd and not req_inbox_id and not req_action_id:
        raise ValueError("Either 'command', 'inbox_item_id', or 'action_id' must be provided.")

    item = None
    if req_inbox_id:
        try:
            item = service.repository.get(req_inbox_id)
        except Exception:
            item = None

    if item is None and req_action_id:
        try:
            matches = service.repository.find_for_action(req_action_id)
            if matches:
                item = matches[0]
                req_inbox_id = item.inbox_item_id
        except Exception:
            item = None

    from mana_agent.transactional_actions.runtime import default_action_gateway

    gateway = default_action_gateway(root)

    if item is None and cmd:
        from mana_agent.tools.shell_exec import split_shell_command
        from mana_agent.transactional_actions.adapters import ShellActionAdapter
        from mana_agent.transactional_actions.models import PolicyOutcome

        try:
            argv = split_shell_command(cmd)
        except Exception:
            argv = cmd.split()

        cmd_digest = hashlib.sha256(cmd.encode("utf-8")).hexdigest()
        idempotency_key = f"shell_approval:{cmd_digest}:{time.time()}"
        adapter = ShellActionAdapter(
            argv=argv,
            cwd=root,
            environment={},
            expected_outputs=[],
            parent_task_id=task_id or "approval_request",
            actor="user",
            originating_agent=agent_id or "ask_agent",
            idempotency_key=idempotency_key,
            allow_command_result_verification=True,
        )
        action_intent = gateway.propose(adapter)
        if action_intent.policy_decision and action_intent.policy_decision.outcome is PolicyOutcome.DENY:
            return {
                "approved": False,
                "inbox_item_id": "",
                "status": "denied",
                "message": f"Command denied by security policy: {action_intent.policy_decision.explanation}",
            }
        if action_intent.policy_decision and action_intent.policy_decision.outcome is PolicyOutcome.ALLOW:
            grant = gateway.approvals.find_valid(action_intent)
            if grant is None:
                grant = gateway.approvals.issue(action_intent, approved_by="policy_allow", ttl_seconds=300)
            return {
                "approved": True,
                "approval_id": grant.approval_id,
                "inbox_item_id": "",
                "status": "approved",
                "command": cmd,
                "message": "Command is allowed by security policy.",
            }

        req_inbox_id = action_intent.inbox_item_id
        if not req_inbox_id and gateway.inbox_service is not None:
            matches = gateway.inbox_service.repository.find_for_action(action_intent.action_id)
            if matches:
                req_inbox_id = matches[0].inbox_item_id
        if req_inbox_id:
            try:
                item = service.repository.get(req_inbox_id)
            except Exception:
                item = None

    if item is None:
        # Create general inbox request
        decision_id = source_decision_id or f"decision_{uuid.uuid4().hex[:8]}"
        t_id = task_id or f"task_{uuid.uuid4().hex[:8]}"
        a_id = agent_id or "ask_agent"
        req_inbox_id = request_user_approval(
            source_decision_id=decision_id,
            task_id=t_id,
            agent_id=a_id,
            title=title or (f"Approval required for command: {cmd}" if cmd else "User approval required"),
            summary=reason or "Action requires approval before execution.",
            risk_level=risk_level,
            inbox_service=service,
        )
        try:
            item = service.repository.get(req_inbox_id)
        except Exception:
            item = None

    target_inbox_id = item.inbox_item_id if item else req_inbox_id
    target_action_id = (
        getattr(item, "action_intent_id", "")
        or getattr(item, "action_id", "")
        or req_action_id
    )

    # Post live activity events so connected TUI displays the approval modal immediately
    approval_metadata = {
        "permission_request_id": target_inbox_id,
        "inbox_item_id": target_inbox_id,
        "action_id": target_action_id,
        "permission_scope": "transactional_action.once",
        "preview": item.card() if (item and hasattr(item, "card")) else {"command": cmd, "reason": reason},
        "transactional_action_approval": True,
        "title": title or (getattr(item, "title", "User approval required") if item else "User approval required"),
    }
    try:
        from mana_agent.chat.history import get_history
        from mana_agent.chat.models import CodingActivityEvent

        get_history().add(
            CodingActivityEvent(
                activity={
                    "event_type": "action.approval.required",
                    "title": title or f"Approval required: {cmd or target_inbox_id}",
                    "metadata": approval_metadata,
                }
            )
        )
    except Exception:
        pass

    try:
        from mana_agent.services.execution_event_hub import get_execution_event_hub

        get_execution_event_hub().publish(
            {
                "type": "action.approval.required",
                "event_type": "action.approval.required",
                "kind": "transactional_action",
                "title": title or f"Approval required: {cmd or target_inbox_id}",
                "metadata": approval_metadata,
            },
            persist=False,
        )
    except Exception:
        pass

    # Await approval resolution
    max_timeout = min(max(0.01, float(timeout_seconds)), 300.0)
    poll_interval = min(0.25, max(0.005, max_timeout / 4.0))
    start_time = time.monotonic()
    approved = False
    grant_id = ""

    while time.monotonic() - start_time < max_timeout:
        # Check action grant
        if target_action_id:
            action = gateway.store.get_action(target_action_id)
            if action is not None:
                grant = gateway.approvals.find_valid(action)
                if grant is not None:
                    approved = True
                    grant_id = grant.approval_id
                    break

        if target_inbox_id:
            try:
                cur_item = service.repository.get(target_inbox_id)
            except Exception:
                cur_item = None
            if cur_item is not None:
                if cur_item.status == InboxStatus.APPROVED:
                    approved = True
                    if cur_item.action_intent_id:
                        action = gateway.store.get_action(cur_item.action_intent_id)
                        if action is not None:
                            grant = gateway.approvals.find_valid(action)
                            if grant is None:
                                grant = gateway.approvals.issue(
                                    action,
                                    approved_by=cur_item.response_actor_id or "user",
                                    ttl_seconds=300,
                                )
                            grant_id = grant.approval_id
                    break
                if cur_item.status in {
                    InboxStatus.DENIED,
                    InboxStatus.CANCELLED,
                    InboxStatus.SUPERSEDED,
                    InboxStatus.EXPIRED,
                }:
                    return {
                        "approved": False,
                        "inbox_item_id": target_inbox_id,
                        "status": cur_item.status.value,
                        "message": f"Approval request was {cur_item.status.value} by the reviewer.",
                    }

        time.sleep(poll_interval)

    if approved:
        return {
            "approved": True,
            "approval_id": grant_id,
            "inbox_item_id": target_inbox_id,
            "status": "approved",
            "command": cmd,
            "message": (
                f"Approval granted. You may now execute the command with action_approval_id={grant_id!r}."
                if grant_id
                else "Approval granted."
            ),
        }

    return {
        "approved": False,
        "inbox_item_id": target_inbox_id,
        "status": "pending",
        "message": f"Timed out waiting for approval after {int(max_timeout)}s. The request remains pending in inbox.",
    }


def build_approval_tools(
    inbox_service: HumanInboxService | None = None,
    execution_supervisor: Any = None,
    on_timeout_suspend: Any = None,
) -> list[StructuredTool]:
    """Return LangChain StructuredTools for model-driven human approval requests and waiting."""
    metadata = {"read_only": True, "inbox_only": True}

    def _approval_req_tool(
        command: str = "",
        title: str = "User approval required",
        reason: str = "",
        inbox_item_id: str = "",
        action_id: str = "",
        risk_level: str = "medium",
        timeout_seconds: float = 60.0,
    ) -> dict[str, Any]:
        return approval_request(
            command=command,
            title=title,
            reason=reason,
            inbox_item_id=inbox_item_id,
            action_id=action_id,
            risk_level=risk_level,
            timeout_seconds=timeout_seconds,
            inbox_service=inbox_service,
        )

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
            func=_approval_req_tool,
            name="approval_request",
            description=(
                "Request human approval and wait for user decision when a command or transactional action requires approval to run. "
                "Displays the approval modal in the user's interface in real time and returns the approved approval_id grant."
            ),
            args_schema=ApprovalRequestInput,
            metadata=metadata,
        ),
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
    "ApprovalRequestInput",
    "ApprovalWaitResult",
    "RequestUserApprovalInput",
    "WaitForApprovalInput",
    "approval_request",
    "build_approval_tools",
    "request_user_approval",
    "wait_for_approval",
]
