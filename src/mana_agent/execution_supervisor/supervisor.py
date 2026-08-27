"""Durable orchestration for root tasks, child tasks, attempts, and results."""

from __future__ import annotations

import hashlib
import hmac
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, runtime_checkable
from uuid import uuid4

from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.errors import (
    BudgetExceededError,
    CompletionVerificationError,
    ConcurrentUpdateError,
    EscrowConflictError,
    EscrowCorruptError,
    EscrowIncompatibleVersionError,
    EscrowNotFoundError,
    ExecutionSupervisorError,
    InvalidTransitionError,
    LeaseConflictError,
    RetrySafetyError,
    StaleLeaseError,
)
from mana_agent.execution_supervisor.models import (
    ActionEffectScope,
    AttemptRecord,
    ActionRecord,
    ActionRequestState,
    BudgetOverrunAction,
    BudgetOverrunFinalizationDecision,
    BudgetRevision,
    CancellationStatus,
    CheckpointRecord,
    CheckpointResumeEligibility,
    CompletionContract,
    CompletionContractType,
    EscrowLookupStatus,
    EscrowResult,
    EscrowStatus,
    ExecutionEvent,
    ExecutionState,
    HumanRecoveryDecisionAction,
    LOCAL_REPOSITORY_TOOLS,
    LostLeaseOutcome,
    ParentProgress,
    ReconciliationOutcome,
    RecoveryAction,
    RecoveryDecision,
    RecoveryInterventionReason,
    RecoveryInterventionRecord,
    RecoverySummary,
    ResultAcknowledgement,
    RetryBudget,
    RetryCategory,
    SideEffectClassification,
    TaskRecord,
    TERMINAL_STATES,
    VerificationStatus,
    VerificationReport,
    VerifiedExecutionResultLookup,
    WaitPolicy,
    utc_now,
)
from mana_agent.execution_supervisor.retry import RetryPolicy
from mana_agent.execution_supervisor.state_machine import validate_transition
from mana_agent.execution_supervisor.store import ExecutionStore, LocalExecutionStore
from mana_agent.execution_supervisor.verifier import ArtifactVerifier

EventSink = Callable[[str, dict[str, Any]], None]
Clock = Callable[[], datetime]


@runtime_checkable
class RecoveryReviewPublisher(Protocol):
    def create_recovery_review(
        self,
        *,
        intervention: RecoveryInterventionRecord,
        task: TaskRecord,
        action: ActionRecord | None = None,
    ) -> str:
        """Create a real durable Human Inbox item and return its inbox_item_id."""
        ...


class DefaultRecoveryReviewPublisher:
    """Default publisher creating real durable Human Inbox items for ambiguous lost leases."""

    def __init__(self, inbox_service: Any = None, inbox_root: Path | None = None) -> None:
        self._inbox_service = inbox_service
        self._inbox_root = inbox_root
        self._supervisor: Any = None

    def _get_service(self) -> Any:
        if self._inbox_service is not None:
            return self._inbox_service
        from mana_agent.human_inbox import (
            HumanInboxService,
            LocalInboxRepository,
            StaticIdentityDirectory,
            ReviewerIdentity,
            ResponseTokenSigner,
        )
        import getpass

        supervisor = self._supervisor
        if self._inbox_root is not None:
            root = self._inbox_root
        elif supervisor is not None and getattr(supervisor, "config", None) is not None and supervisor.config.root:
            root = supervisor.config.root.parent / "human_inbox"
        else:
            from mana_agent.config.settings import mana_home

            root = mana_home() / "inbox"

        username = getpass.getuser()
        identities = StaticIdentityDirectory([
            ReviewerIdentity(identity_id=username, tenant_ids={"local"}),
            ReviewerIdentity(identity_id="operator", tenant_ids={"local"}),
            ReviewerIdentity(identity_id="admin", tenant_ids={"local"}),
        ])
        self._inbox_service = HumanInboxService(
            repository=LocalInboxRepository(root),
            identities=identities,
            token_signer=ResponseTokenSigner(root / "response-signing.key"),
            branch_controller=supervisor,
            clock=supervisor.clock if supervisor is not None else utc_now,
        )
        return self._inbox_service

    def create_recovery_review(
        self,
        *,
        intervention: RecoveryInterventionRecord,
        task: TaskRecord,
        action: ActionRecord | None = None,
    ) -> str:
        import getpass
        from mana_agent.human_inbox.models import (
            InboxRequest,
            InboxRequestType,
            ResponseOperation,
            ReviewerAssignment,
            ReviewerType,
            RiskLevel as InboxRiskLevel,
        )

        service = self._get_service()
        username = getpass.getuser()
        tool_name = action.tool_name if action else "external consequential operation"

        dedup_key = f"recovery_review:{task.task_id}:{task.attempt_id}:{action.action_id if action else ''}:{intervention.intervention_id}"
        idemp_key = dedup_key
        for existing_item in service.repository.list():
            if existing_item.deduplication_key == dedup_key:
                return existing_item.inbox_item_id

        item = service.create(
            InboxRequest(
                request_type=InboxRequestType.APPROVAL,
                task_id=task.task_id,
                root_task_id=task.root_task_id or task.task_id,
                execution_id=intervention.execution_id,
                attempt_id=intervention.attempt_id,
                action_id=intervention.action_id,
                intervention_id=intervention.intervention_id,
                integration_stage=intervention.integration_stage,
                recovery_reason=intervention.reason.value,
                branch_id=task.task_id,
                checkpoint_id=task.checkpoint_id,
                execution_attempt_id=task.attempt_id,
                policy_decision_id=task.routing_decision_id or f"policy:{task.task_id}",
                action_intent_id=f"recovery:{intervention.intervention_id}",
                action_digest=intervention.intervention_id,
                requested_by_agent_id="execution_supervisor",
                reviewer=ReviewerAssignment(
                    reviewer_type=ReviewerType.PERSON,
                    reviewer_id=username,
                ),
                title=f"Approve recovery for task {task.task_id}",
                summary=(
                    f"A lost lease occurred during execution of task '{task.task_id}' while executing "
                    f"consequential action '{tool_name}'. Automatic replay is unsafe without human confirmation."
                ),
                risk_level=InboxRiskLevel.CRITICAL,
                allowed_responses=[ResponseOperation.APPROVE, ResponseOperation.DENY],
                recovery_intervention_id=intervention.intervention_id,
                minimal_context={
                    "task_id": task.task_id,
                    "attempt_id": task.attempt_id,
                    "action_id": action.action_id if action else "",
                    "tool_name": tool_name,
                    "intervention_id": intervention.intervention_id,
                    "reason": intervention.reason.value,
                    "receipt_state": intervention.receipt_lookup_state,
                    "integration_stage": intervention.integration_stage,
                },
                expires_at=intervention.created_at + timedelta(days=7),
                idempotency_key=idemp_key,
                deduplication_key=dedup_key,
            )
        )
        return item.inbox_item_id


def _token_hash(token: str) -> str:
    return "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()


def _provider_metadata(value: Any) -> dict[str, Any]:
    """Convert provider evidence to a JSON-safe envelope."""
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        value = asdict(value)
    if not isinstance(value, dict):
        raise TypeError("provider_metadata must be a mapping or serializable model")

    def normalize(item: Any) -> Any:
        if hasattr(item, "value") and not isinstance(item, (str, bytes, dict, list, tuple)):
            return normalize(item.value)
        if isinstance(item, dict):
            return {str(key): normalize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        return item

    return normalize(value)


class ExecutionSupervisor:
    """Coordinates durable task state without selecting models, tools, or agents.

    Routing and escalation choices enter through typed ``RecoveryDecision``
    objects. The supervisor validates and executes them; it never substitutes a
    default worker, model, tool, or workflow when the decision is invalid.
    """

    def __init__(
        self,
        config: ExecutionSupervisorConfig | None = None,
        *,
        store: ExecutionStore | None = None,
        verifier: ArtifactVerifier | None = None,
        event_sink: EventSink | None = None,
        clock: Clock = utc_now,
        startup_recovery: bool | None = None,
        recovery_review_publisher: RecoveryReviewPublisher | None = None,
    ) -> None:
        self.config = config or ExecutionSupervisorConfig()
        self.store = store or LocalExecutionStore(self.config.root)
        self.verifier = verifier or ArtifactVerifier()
        self.retry_policy = RetryPolicy(self.config)
        self.event_sink = event_sink
        self.clock = clock
        self.recovery_review_publisher = recovery_review_publisher or DefaultRecoveryReviewPublisher()
        if hasattr(self.recovery_review_publisher, "_supervisor"):
            self.recovery_review_publisher._supervisor = self
        self._last_live_heartbeat: dict[str, datetime] = {}
        self.startup_recovery_summary: RecoverySummary | None = None
        if (
            self.config.startup_recovery
            if startup_recovery is None
            else startup_recovery
        ):
            self.reconnect_tree()
            self.startup_recovery_summary = self.recover()

    def _emit(self, event_type: str, task: TaskRecord, **details: Any) -> ExecutionEvent:
        last_cp_id = task.checkpoint_id or None
        resume_cp_id: str | None = None
        if last_cp_id and task.state not in TERMINAL_STATES and task.state != ExecutionState.CANCELLING:
            eligibility = self.validate_checkpoint_resume(task, allow_explicit_retry_seed=False)
            if eligibility.resumable and eligibility.checkpoint:
                resume_cp_id = eligibility.checkpoint.checkpoint_id

        event_details = {
            "assigned_agent": task.assigned_agent,
            "assigned_model": task.assigned_model,
            "assigned_worker": task.assigned_worker,
            "runtime_provider": task.runtime_provider,
            "lease_owner": task.lease_owner,
            "checkpoint_id": task.checkpoint_id,
            "last_checkpoint_id": last_cp_id,
            "resume_checkpoint_id": resume_cp_id,
            "attempt_generation": task.attempt_generation,
            "session_id": task.session_id,
            "workspace_id": task.workspace_id,
            "repository_id": task.repository_id,
            "retry_count": task.retry_count,
            "side_effect_classification": task.side_effect_classification.value,
            "verification_status": task.verification_status.value,
            "token_usage": task.token_usage,
            "estimated_cost": task.estimated_cost,
            "actual_cost": task.actual_cost,
            **details,
        }
        event = ExecutionEvent(
            event_type=event_type,
            task_id=task.task_id,
            parent_task_id=task.parent_task_id,
            root_task_id=task.root_task_id,
            attempt_id=task.attempt_id,
            state=task.state,
            created_at=self.clock(),
            details=event_details,
        )
        self.store.append_event(event)
        if event_type == "heartbeat":
            previous = self._last_live_heartbeat.get(task.task_id)
            if previous is not None and (event.created_at - previous).total_seconds() < 60:
                return event
            self._last_live_heartbeat[task.task_id] = event.created_at
        if self.event_sink is not None:
            self.event_sink(event_type, event.model_dump(mode="json"))
        else:
            # The shared event hub is the existing transport for TUI,
            # dashboard, API, and connector observers. Import lazily so the
            # domain layer remains usable without the optional UI services.
            from mana_agent.services.execution_event_hub import get_execution_event_hub

            get_execution_event_hub().publish(
                {
                    "type": event_type,
                    "kind": event_type,
                    "status": "success" if event_type == "task_completed" else "running",
                    "message": event_type.replace("_", " "),
                    "execution_id": task.task_id,
                    "metadata": {
                        "execution_supervisor": True,
                        **event.model_dump(mode="json"),
                    },
                },
                persist=False,
            )
        return event

    def create_task(
        self,
        *,
        task_id: str | None = None,
        parent_task_id: str | None = None,
        task_type: str = "task",
        assigned_agent: str = "",
        assigned_model: str = "",
        runtime_provider: str = "",
        workspace_path: str | Path = "",
        routing_decision_id: str,
        side_effect_classification: SideEffectClassification = SideEffectClassification.IDEMPOTENT,
        completion_contract: Iterable[CompletionContract] = (),
        dependency_task_ids: Iterable[str] = (),
        idempotency_key: str = "",
        compensation_strategy: str = "",
        token_budget: int | None = None,
        estimated_cost: float | None = None,
        model_context_window: int = 0,
        model_max_output_tokens: int = 0,
        estimate_confidence: str = "",
        estimate_source: str = "",
        monetary_budget: float | None = None,
        deadline_at: datetime | None = None,
        wait_policy: WaitPolicy = WaitPolicy.WAIT_ALL,
        minimum_success_count: int | None = None,
        execution_fingerprint: str = "",
        session_id: str = "",
        workspace_id: str = "",
        repository_id: str = "",
        normalized_intent: str = "",
        requested_operation: str = "",
        target_resources: Iterable[str] = (),
        expected_output: str = "",
        important_constraints: Iterable[str] = (),
        field_provenance: dict[str, str] | None = None,
        supervision_contract_decision_id: str = "",
        supersedes_execution_id: str = "",
        derived_from_execution_id: str = "",
        previous_execution_id: str = "",
        trigger_turn_id: str = "",
        relation_type: str = "independent",
        previous_task_id: str = "",
        delegated_capsule_revisions: dict[str, int] | None = None,
    ) -> TaskRecord:
        if not self.config.enabled:
            raise ExecutionSupervisorError(
                "execution supervisor is disabled; unsupervised execution was not started"
            )
        identifier = task_id or f"task_{uuid4()}"
        contracts = list(completion_contract)
        if not contracts:
            # Every supervised execution has a durable, non-empty completion
            # boundary from creation. Route-specific callers replace this
            # minimal contract with their validated model-selected contract.
            contracts = [
                CompletionContract(
                    contract_type=CompletionContractType.STRUCTURED_RESULT_VALID,
                    metadata={"required_keys": []},
                )
            ]
        dependencies = list(dependency_task_ids)
        targets = list(target_resources)
        constraints = list(important_constraints)
        effective_contract_decision_id = (
            supervision_contract_decision_id or routing_decision_id
        )
        provenance = dict(field_provenance or {})
        provenance.setdefault("side_effect_classification", "model_selected")
        provenance.setdefault(
            "completion_contract",
            "model_selected" if contracts else "pending_runtime_evidence",
        )
        provenance.setdefault(
            "target_resources",
            "model_selected" if targets else "not_applicable_or_not_selected",
        )
        provenance.setdefault(
            "important_constraints",
            "model_selected" if constraints else "not_applicable_or_not_selected",
        )
        provenance.setdefault(
            "estimated_cost",
            "provider_estimate" if estimated_cost is not None else "unknown_provider_pricing",
        )
        provenance.setdefault("actual_cost", "pending_runtime_accounting")
        provenance.setdefault("completion_artefacts", "pending_completion_verification")
        idempotency_hash = (
            _token_hash(f"idempotency:{idempotency_key}") if idempotency_key else ""
        )
        existing = self.store.get_task_or_none(identifier)
        if existing is not None:
            if (
                existing.parent_task_id != parent_task_id
                or existing.task_type != task_type
                or existing.assigned_model != assigned_model
                or existing.runtime_provider != runtime_provider
                or existing.routing_decision_id != routing_decision_id
                or existing.side_effect_classification != side_effect_classification
                or existing.idempotency_key != idempotency_hash
                or existing.compensation_strategy != compensation_strategy
                or existing.completion_contract != contracts
                or existing.dependency_task_ids != dependencies
                or existing.execution_fingerprint != execution_fingerprint
                or existing.session_id != session_id
                or existing.workspace_id != workspace_id
                or existing.repository_id != repository_id
                or existing.normalized_intent != normalized_intent
                or existing.requested_operation != requested_operation
                or existing.target_resources != targets
                or existing.expected_output != expected_output
                or existing.important_constraints != constraints
                or existing.field_provenance != provenance
                or existing.supervision_contract_decision_id != effective_contract_decision_id
                or existing.supersedes_execution_id != supersedes_execution_id
                or existing.derived_from_execution_id != derived_from_execution_id
                or existing.previous_execution_id != previous_execution_id
                or existing.trigger_turn_id != trigger_turn_id
                or existing.relation_type != relation_type
                or existing.previous_task_id != previous_task_id
                or existing.delegated_capsule_revisions != dict(delegated_capsule_revisions or {})
                or existing.token_budget != token_budget
                or existing.estimated_cost_known != (estimated_cost is not None)
                or (
                    estimated_cost is not None
                    and existing.estimated_cost != max(0.0, float(estimated_cost))
                )
                or existing.monetary_budget != monetary_budget
                or existing.model_context_window != max(0, model_context_window)
                or existing.model_max_output_tokens != max(0, model_max_output_tokens)
                or existing.token_estimate_confidence != estimate_confidence
                or existing.token_estimate_source != estimate_source
            ):
                raise ConcurrentUpdateError(
                    f"task identity {identifier} already exists with a different immutable contract"
                )
            if existing.parent_task_id:
                durable_parent = self.store.get_task(existing.parent_task_id)
                if existing.task_id not in durable_parent.child_task_ids:
                    def relink(current: TaskRecord) -> None:
                        if existing.task_id not in current.child_task_ids:
                            current.child_task_ids.append(existing.task_id)
                            current.updated_at = self.clock()
                    self.store.update_task(durable_parent.task_id, relink)
                    self._emit(
                        "child_reconnected",
                        existing,
                        parent_task_id=durable_parent.task_id,
                    )
            return existing
        if not routing_decision_id.strip():
            raise ValueError("a validated routing decision ID is required")
        if monetary_budget is not None and estimated_cost is None:
            raise BudgetExceededError(
                "model pricing is unknown for a task with an explicit monetary budget"
            )
        if monetary_budget is not None and estimated_cost is not None and estimated_cost > monetary_budget:
            raise BudgetExceededError(
                "model-selected estimated task cost exceeds the task monetary budget"
            )
        parent = self.store.get_task(parent_task_id) if parent_task_id else None
        depth = 0
        cursor = parent
        while cursor is not None:
            depth += 1
            cursor = self.store.get_task(cursor.parent_task_id) if cursor.parent_task_id else None
        if depth > self.config.max_child_depth:
            raise ValueError("maximum supervised child depth exceeded")
        if parent and len(parent.child_task_ids) >= self.config.max_children_per_task:
            raise ValueError("maximum supervised children per task exceeded")
        if parent:
            root_subtasks = sum(
                item.root_task_id == parent.root_task_id and item.parent_task_id is not None
                for item in self.store.list_tasks()
            )
            if root_subtasks >= self.config.max_total_subtasks:
                raise ValueError("maximum total supervised subtasks exceeded")
        budget = RetryBudget(
            infrastructure=self.config.default_retry_budget,
            model=self.config.default_retry_budget,
            tool=self.config.default_retry_budget,
            verification=self.config.default_retry_budget,
            lease_loss=self.config.default_retry_budget,
            replan=self.config.max_replans,
        )
        selected_deadline = deadline_at or (
            self.clock() + timedelta(seconds=self.config.default_task_deadline_seconds)
        )
        if parent is not None and parent.wall_clock_deadline_exceeded(self.clock()):
            raise BudgetExceededError(
                "parent task wall-clock deadline exceeded; create a new root task "
                "instead of attaching a child under the dead parent"
            )
        if (
            parent is not None
            and parent.state is not ExecutionState.COMPLETED
            and parent.deadline_at is not None
        ):
            selected_deadline = min(selected_deadline, parent.deadline_at)
        task = TaskRecord(
            task_id=identifier,
            parent_task_id=parent_task_id,
            root_task_id=parent.root_task_id if parent else identifier,
            task_type=task_type,
            assigned_agent=assigned_agent,
            assigned_model=assigned_model,
            runtime_provider=runtime_provider,
            retry_budget=budget,
            idempotency_key=idempotency_hash,
            compensation_strategy=compensation_strategy,
            side_effect_classification=side_effect_classification,
            completion_contract=contracts,
            dependency_task_ids=dependencies,
            token_budget=token_budget,
            estimated_cost=max(0.0, float(estimated_cost or 0.0)),
            estimated_cost_known=estimated_cost is not None,
            model_context_window=max(0, model_context_window),
            model_max_output_tokens=max(0, model_max_output_tokens),
            token_estimate_confidence=estimate_confidence,
            token_estimate_source=estimate_source,
            monetary_budget=monetary_budget,
            deadline_at=selected_deadline,
            wait_policy=wait_policy,
            minimum_success_count=minimum_success_count,
            max_child_depth=self.config.max_child_depth,
            max_children=self.config.max_children_per_task,
            max_total_subtasks=self.config.max_total_subtasks,
            max_concurrent_children=self.config.max_concurrent_children,
            routing_decision_id=routing_decision_id,
            workspace_path=str(Path(workspace_path).expanduser().resolve()) if workspace_path else "",
            execution_fingerprint=execution_fingerprint,
            session_id=session_id,
            workspace_id=workspace_id,
            repository_id=repository_id,
            normalized_intent=normalized_intent,
            requested_operation=requested_operation,
            target_resources=targets,
            expected_output=expected_output,
            important_constraints=constraints,
            field_provenance=provenance,
            supervision_contract_decision_id=effective_contract_decision_id,
            supersedes_execution_id=supersedes_execution_id,
            derived_from_execution_id=derived_from_execution_id,
            previous_execution_id=previous_execution_id,
            trigger_turn_id=trigger_turn_id,
            relation_type=relation_type,
            previous_task_id=previous_task_id,
            delegated_capsule_revisions=dict(delegated_capsule_revisions or {}),
        )
        self.store.create_task(task)
        if parent is not None:
            def link(current: TaskRecord) -> None:
                if identifier not in current.child_task_ids:
                    current.child_task_ids.append(identifier)
                    current.updated_at = self.clock()
            parent, _ = self.store.update_task(parent.task_id, link)
            self._emit("child_created", task, parent_task_id=parent.task_id)
        self._emit("task_created", task)
        return task

    def transition(
        self,
        task_id: str,
        target: ExecutionState,
        *,
        reason: str = "",
        recovery_reason: str = "",
    ) -> TaskRecord:
        prior: ExecutionState | None = None

        def update(task: TaskRecord) -> None:
            nonlocal prior
            prior = task.state
            if task.waiting_inbox_item_id and target in {
                ExecutionState.QUEUED,
                ExecutionState.LEASED,
                ExecutionState.RUNNING,
                ExecutionState.CHECKPOINTING,
                ExecutionState.RETRY_SCHEDULED,
                ExecutionState.COMPLETED_PENDING_VERIFICATION,
            }:
                raise LeaseConflictError(
                    "a human-waiting branch may resume only through its durable inbox claim"
                )
            validate_transition(task.state, target)
            task.state = target
            task.updated_at = self.clock()
            if target == ExecutionState.RUNNING and task.started_at is None:
                task.started_at = task.updated_at
            if target == ExecutionState.CANCELLING:
                task.cancellation_status = CancellationStatus.REQUESTED
                task.cancellation_reason = reason
            if target in TERMINAL_STATES:
                task.finished_at = task.updated_at
                task.lease_owner = ""
                task.lease_token = ""
                task.lease_expires_at = None
                task.retry_not_before = None
                if target is not ExecutionState.RECOVERY_REVIEW_REQUIRED:
                    task.waiting_inbox_item_id = ""
                    task.waiting_reason = ""
                    task.human_wait_started_at = None
                existing_res = (
                    self.store.get_result(task.result_id)
                    if task.result_id
                    else self.store.get_result_by_execution_id(task.task_id)
                )
                if existing_res is not None:
                    if not task.result_id:
                        task.result_id = existing_res.result_id
                elif target != ExecutionState.COMPLETED and not task.result_id:
                    result_kind = (
                        "terminal_failure"
                        if target in {
                            ExecutionState.FAILED,
                            ExecutionState.CANCELLED,
                            ExecutionState.BUDGET_EXHAUSTED,
                            ExecutionState.RECOVERY_REVIEW_REQUIRED,
                        }
                        else "chat_result"
                    )
                    status = EscrowStatus.AVAILABLE
                    verification_status = (
                        VerificationStatus.FAILED
                        if target in {
                            ExecutionState.FAILED,
                            ExecutionState.RECOVERY_REVIEW_REQUIRED,
                        }
                        else VerificationStatus.NOT_SUPPORTED
                    )
                    result_payload = {
                        "status": target.value,
                        "reason": reason or task.failure_reason,
                        "recovery_reason": task.recovery_reason,
                        "is_resumable": False,
                        "chat_result": {
                            "answer": reason or f"Execution ended as {target.value}.",
                            "error": reason or target.value,
                            "mode": f"lane-{target.value.replace('_', '-')}",
                            "payload": {
                                "execution_id": task.task_id,
                                "lane_task_id": task.task_id,
                                "status": target.value,
                                "terminal_failure": True,
                                "is_resumable": False,
                            },
                        },
                    }
                    err_meta = {
                        "state": target.value,
                        "reason": reason or task.failure_reason,
                        "recovery_reason": task.recovery_reason,
                        "is_resumable": False,
                    }
                    term_result = EscrowResult(
                        task_id=task.task_id,
                        execution_id=task.task_id,
                        root_task_id=task.root_task_id or task.task_id,
                        parent_task_id=task.parent_task_id,
                        trigger_turn_id=task.trigger_turn_id,
                        session_id=task.session_id,
                        lane_id=(
                            task.assigned_agent.removeprefix("lane:")
                            if task.assigned_agent.startswith("lane:")
                            else task.assigned_agent
                        ),
                        owning_lane=(
                            task.assigned_agent.removeprefix("lane:")
                            if task.assigned_agent.startswith("lane:")
                            else task.assigned_agent
                        ),
                        attempt_id=task.attempt_id,
                        attempt_generation=task.attempt_generation,
                        lease_token_hash=task.lease_token,
                        status=status,
                        supervisor_state=target.value,
                        verification_status=verification_status,
                        result_kind=result_kind,
                        payload=result_payload,
                        error_metadata=err_meta,
                        provider_metadata=_provider_metadata(task.provider_metadata),
                        created_at=self.clock(),
                        completed_at=self.clock(),
                    )
                    task.result_id = term_result.result_id
                    self.store.save_result(term_result)
            elif target == ExecutionState.QUEUED:
                task.lease_owner = ""
                task.lease_token = ""
                task.lease_expires_at = None
                task.retry_not_before = None
                task.waiting_inbox_item_id = ""
                task.waiting_reason = ""
                task.human_wait_started_at = None
            if reason and target in {
                ExecutionState.FAILED,
                ExecutionState.BUDGET_EXHAUSTED,
                ExecutionState.RECOVERY_REVIEW_REQUIRED,
            }:
                task.failure_reason = reason
            if recovery_reason:
                task.recovery_reason = recovery_reason

        try:
            task, _ = self.store.update_task(task_id, update)
        except InvalidTransitionError as exc:
            current = self.store.get_task(task_id)
            self._emit(
                "invalid_transition",
                current,
                source=current.state.value,
                target=target.value,
                reason=str(exc),
            )
            raise
        event_name = {
            ExecutionState.QUEUED: "task_queued",
            ExecutionState.RUNNING: "task_started",
            ExecutionState.WAITING: "child_waiting",
            ExecutionState.RETRY_SCHEDULED: "retry_scheduled",
            ExecutionState.REPLANNING: "replan_started",
            ExecutionState.CANCELLING: "cancellation_requested",
            ExecutionState.CANCELLED: "task_cancelled",
            ExecutionState.FAILED: "task_failed",
            ExecutionState.BUDGET_EXHAUSTED: "task_budget_exhausted",
            ExecutionState.RECOVERY_REVIEW_REQUIRED: "recovery_review_required",
            ExecutionState.COMPLETED: "task_completed",
        }.get(target, f"task_{target.value}")
        if target in TERMINAL_STATES and task.attempt_id:
            attempt = self.store.get_attempt(task.attempt_id)
            if attempt is not None:
                attempt.state = target.value
                attempt.finished_at = task.finished_at
                if target in {ExecutionState.FAILED, ExecutionState.RECOVERY_REVIEW_REQUIRED}:
                    attempt.failure_reason = reason
                self.store.save_attempt(attempt)
        self._emit(event_name, task, previous_state=prior.value if prior else "", reason=reason or recovery_reason)
        return task

    def record_terminal_result(
        self,
        task_id: str,
        *,
        state: ExecutionState,
        reason: str = "",
        payload: dict[str, Any] | None = None,
        is_resumable: bool = False,
        error_metadata: dict[str, Any] | None = None,
        provider_metadata: dict[str, Any] | Any | None = None,
    ) -> EscrowResult:
        """Persist a terminal outcome or resumable wait in durable result escrow."""
        task = self.store.get_task(task_id)
        existing_result = (
            self.store.get_result(task.result_id)
            if task.result_id
            else self.store.get_result_by_execution_id(task.task_id)
        )
        if existing_result is not None:
            changed = False
            if (
                existing_result.supervisor_state != state.value
                and existing_result.status != EscrowStatus.ACKNOWLEDGED
            ):
                existing_result.supervisor_state = state.value
                if state in TERMINAL_STATES:
                    existing_result.completed_at = existing_result.completed_at or self.clock()
                changed = True
            metadata_to_merge = provider_metadata if provider_metadata is not None else task.provider_metadata
            if metadata_to_merge:
                merged_provider_metadata = {
                    **existing_result.provider_metadata,
                    **_provider_metadata(metadata_to_merge),
                }
                if merged_provider_metadata != existing_result.provider_metadata:
                    existing_result.provider_metadata = merged_provider_metadata
                    changed = True
            effective_error_metadata = error_metadata
            if effective_error_metadata is None and task.provider_metadata.get("state") == "AUTH_REQUIRED":
                effective_error_metadata = {"state": "AUTH_REQUIRED", "action": "reauthenticate"}
            if effective_error_metadata is not None:
                merged_error_metadata = {
                    **existing_result.error_metadata,
                    **effective_error_metadata,
                }
                if merged_error_metadata != existing_result.error_metadata:
                    existing_result.error_metadata = merged_error_metadata
                    changed = True
            if changed:
                self.store.save_result(existing_result)
            if not task.result_id or task.result_id != existing_result.result_id:
                def link_existing(curr: TaskRecord) -> None:
                    curr.result_id = existing_result.result_id
                    curr.updated_at = self.clock()
                self.store.update_task(task.task_id, link_existing)
            return existing_result

        result_kind = (
            "terminal_failure"
            if state in {
                ExecutionState.FAILED,
                ExecutionState.CANCELLED,
                ExecutionState.BUDGET_EXHAUSTED,
                ExecutionState.RECOVERY_REVIEW_REQUIRED,
            }
            else ("resumable_wait" if is_resumable else "chat_result")
        )
        status = EscrowStatus.AVAILABLE
        verification_status = (
            VerificationStatus.PASSED
            if state == ExecutionState.COMPLETED
            else (
                VerificationStatus.FAILED
                if state in {ExecutionState.FAILED, ExecutionState.RECOVERY_REVIEW_REQUIRED}
                else VerificationStatus.NOT_SUPPORTED
            )
        )
        result_payload = payload or {
            "status": state.value,
            "reason": reason or task.failure_reason,
            "recovery_reason": task.recovery_reason,
            "is_resumable": is_resumable,
            "chat_result": {
                "answer": reason or f"Execution ended as {state.value}.",
                "error": reason or state.value,
                "mode": f"lane-{state.value.replace('_', '-')}",
                "payload": {
                    "execution_id": task.task_id,
                    "lane_task_id": task.task_id,
                    "status": state.value,
                    "terminal_failure": not is_resumable,
                    "is_resumable": is_resumable,
                },
            },
        }
        err_meta = error_metadata or {
            "state": state.value,
            "reason": reason or task.failure_reason,
            "recovery_reason": task.recovery_reason,
            "is_resumable": is_resumable,
        }
        normalized_provider_metadata = _provider_metadata(provider_metadata or task.provider_metadata)
        if normalized_provider_metadata.get("state") == "AUTH_REQUIRED":
            err_meta = {"state": "AUTH_REQUIRED", "action": "reauthenticate", **err_meta}
        result = EscrowResult(
            task_id=task.task_id,
            execution_id=task.task_id,
            root_task_id=task.root_task_id or task.task_id,
            parent_task_id=task.parent_task_id,
            trigger_turn_id=task.trigger_turn_id,
            session_id=task.session_id,
            lane_id=(
                task.assigned_agent.removeprefix("lane:")
                if task.assigned_agent.startswith("lane:")
                else task.assigned_agent
            ),
            owning_lane=(
                task.assigned_agent.removeprefix("lane:")
                if task.assigned_agent.startswith("lane:")
                else task.assigned_agent
            ),
            attempt_id=task.attempt_id,
            attempt_generation=task.attempt_generation,
            lease_token_hash=task.lease_token,
            status=status,
            supervisor_state=state.value,
            verification_status=verification_status,
            result_kind=result_kind,
            payload=result_payload,
            error_metadata=err_meta,
            provider_metadata=normalized_provider_metadata,
            created_at=self.clock(),
            completed_at=self.clock() if state in TERMINAL_STATES else None,
        )
        if not task.result_id:
            def link_res(curr: TaskRecord) -> None:
                curr.result_id = result.result_id
                curr.updated_at = self.clock()

            self.store.update_task(task.task_id, link_res)
        self.store.save_result(result)
        self._emit("result_stored", task, result_id=result.result_id, status=status.value)
        return result

    def persist_provider_metadata(
        self,
        task_id: str,
        provider_metadata: dict[str, Any] | Any,
    ) -> TaskRecord:
        """Persist provider lifecycle evidence before terminal publication.

        Providers remain unaware of supervisor storage. This method lets an
        execution adapter record failure metadata while the task is still
        running, so a later failure or restart cannot discard the evidence.
        """
        incoming = _provider_metadata(provider_metadata)

        def update(task: TaskRecord) -> None:
            task.provider_metadata = {**task.provider_metadata, **incoming}
            task.updated_at = self.clock()

        task, _ = self.store.update_task(task_id, update)
        return task


    def queue(self, task_id: str) -> TaskRecord:
        task = self.store.get_task(task_id)
        if task.state == ExecutionState.QUEUED:
            return task
        return self.transition(task_id, ExecutionState.QUEUED)

    def record_accounting_reservations(
        self, task_id: str, reservation_ids: Iterable[str]
    ) -> TaskRecord:
        """Link privacy-safe accounting records to durable task and attempt state."""
        identifiers = tuple(dict.fromkeys(str(item) for item in reservation_ids if str(item)))
        if any(not item.startswith("reservation_") for item in identifiers):
            raise ValueError("invalid accounting reservation identifier")

        def update(task: TaskRecord) -> None:
            for identifier in identifiers:
                if identifier not in task.accounting_reservation_ids:
                    task.accounting_reservation_ids.append(identifier)
            task.updated_at = self.clock()

        task, _ = self.store.update_task(task_id, update)
        if task.attempt_id:
            attempt = self.store.get_attempt(task.attempt_id)
            if attempt is not None:
                for identifier in identifiers:
                    if identifier not in attempt.accounting_reservation_ids:
                        attempt.accounting_reservation_ids.append(identifier)
                self.store.save_attempt(attempt)
        return task

    def acquire_lease(self, task_id: str, *, owner: str, worker: str = "") -> tuple[TaskRecord, str]:
        if not owner.strip():
            raise ValueError("lease owner is required")
        lease_token = uuid4().hex
        now = self.clock()
        current = self.store.get_task(task_id)
        if current.state not in {ExecutionState.QUEUED, ExecutionState.CREATED}:
            raise LeaseConflictError(f"task is not leaseable from state {current.state.value}")
        if current.assigned_worker and worker != current.assigned_worker:
            raise LeaseConflictError(
                f"lease claimant {worker or '<unset>'} does not match the model-selected "
                f"worker {current.assigned_worker}"
            )
        if current.deadline_at is not None and current.deadline_at <= now:
            self.transition(task_id, ExecutionState.FAILED, reason="task wall-clock deadline exceeded")
            raise BudgetExceededError("task wall-clock deadline exceeded")

        def claim(task: TaskRecord) -> AttemptRecord:
            if task.state not in {ExecutionState.QUEUED, ExecutionState.CREATED}:
                raise LeaseConflictError(f"task is not leaseable from state {task.state.value}")
            if task.lease_token and task.lease_expires_at and task.lease_expires_at > now:
                raise LeaseConflictError("task already holds an active lease")
            if task.parent_task_id:
                active_siblings = sum(
                    item.parent_task_id == task.parent_task_id
                    and item.task_id != task.task_id
                    and (
                        item.state in {
                            ExecutionState.LEASED,
                            ExecutionState.RUNNING,
                            ExecutionState.CHECKPOINTING,
                        }
                        or (
                            item.state is ExecutionState.WAITING
                            and not item.waiting_inbox_item_id
                        )
                    )
                    for item in self.store.list_tasks()
                )
                if active_siblings >= self.config.max_concurrent_children:
                    raise LeaseConflictError("maximum concurrent supervised children reached")
            attempt = AttemptRecord(
                task_id=task.task_id,
                number=len(task.attempt_ids) + 1,
                generation=task.attempt_generation + 1,
                state="leased",
                lease_owner=owner,
                lease_token=_token_hash(lease_token),
                lease_expires_at=now + timedelta(seconds=self.config.lease_seconds),
                estimated_cost=task.estimated_cost,
                estimated_cost_known=task.estimated_cost_known,
            )
            task.state = ExecutionState.LEASED
            task.attempt_id = attempt.attempt_id
            task.attempt_generation = attempt.generation
            task.attempt_ids.append(attempt.attempt_id)
            task.assigned_worker = worker
            task.lease_owner = owner
            task.lease_token = _token_hash(lease_token)
            task.lease_expires_at = attempt.lease_expires_at
            task.heartbeat_at = now
            task.updated_at = now
            return attempt

        task, attempt = self.store.update_task_and_attempt(task_id, claim)
        self._emit("lease_acquired", task, lease_owner=owner, lease_expires_at=task.lease_expires_at)
        return task, lease_token

    def _validate_lease(self, task: TaskRecord, *, attempt_id: str, lease_token: str) -> None:
        if task.attempt_id != attempt_id:
            raise StaleLeaseError("attempt no longer owns the task")
        if not hmac.compare_digest(task.lease_token, _token_hash(lease_token)):
            raise StaleLeaseError("lease token is stale or invalid")
        if task.lease_expires_at is None or task.lease_expires_at <= self.clock():
            raise StaleLeaseError("lease has expired")
        if task.deadline_at is not None and task.deadline_at <= self.clock():
            raise BudgetExceededError("task wall-clock deadline exceeded")

    def start(self, task_id: str, *, attempt_id: str, lease_token: str) -> TaskRecord:
        def update(task: TaskRecord) -> None:
            self._validate_lease(task, attempt_id=attempt_id, lease_token=lease_token)
            validate_transition(task.state, ExecutionState.RUNNING)
            task.state = ExecutionState.RUNNING
            task.started_at = task.started_at or self.clock()
            task.updated_at = self.clock()
        task, _ = self.store.update_task(task_id, update)
        attempt = self.store.get_attempt(attempt_id)
        if attempt:
            attempt.state = "running"
            attempt.started_at = task.started_at
            self.store.save_attempt(attempt)
        self._emit("task_started", task)
        return task

    def release_lease(
        self,
        task_id: str,
        *,
        attempt_id: str,
        lease_token: str,
        reason: str,
    ) -> TaskRecord:
        """Release a claimed-but-not-started attempt without retrying executed work."""

        current = self.store.get_task(task_id)
        self._validate_lease(current, attempt_id=attempt_id, lease_token=lease_token)
        if current.state != ExecutionState.LEASED:
            raise LeaseConflictError(
                "only a leased attempt that has not started may be released directly"
            )
        task = self.transition(
            task_id,
            ExecutionState.QUEUED,
            recovery_reason=reason or "lease released before execution",
        )
        attempt = self.store.get_attempt(attempt_id)
        if attempt is not None:
            attempt.state = "released"
            attempt.finished_at = self.clock()
            attempt.recovery_reason = reason
            self.store.save_attempt(attempt)
        self._emit("lease_released", task, released_attempt_id=attempt_id, reason=reason)
        return task

    def heartbeat(self, task_id: str, *, attempt_id: str, lease_token: str) -> TaskRecord:
        now = self.clock()
        def renew(task: TaskRecord) -> None:
            self._validate_lease(task, attempt_id=attempt_id, lease_token=lease_token)
            if task.state not in {ExecutionState.LEASED, ExecutionState.RUNNING, ExecutionState.CHECKPOINTING, ExecutionState.WAITING}:
                raise LeaseConflictError(f"task state does not accept heartbeats: {task.state.value}")
            task.heartbeat_at = now
            task.lease_expires_at = now + timedelta(seconds=self.config.lease_seconds)
            task.updated_at = now
        task, _ = self.store.update_task(task_id, renew)
        attempt = self.store.get_attempt(attempt_id)
        if attempt:
            attempt.lease_expires_at = task.lease_expires_at
            self.store.save_attempt(attempt)
        self._emit("heartbeat", task, lease_expires_at=task.lease_expires_at)
        return task

    @contextmanager
    def lease_renewal(self, task_id: str, *, attempt_id: str, lease_token: str):
        """Renew a live attempt independently of provider/tool boundaries.

        Callers wrap the complete model, integration, verifier, or reviewer
        operation in this context. Renewal stops on normal completion,
        cancellation, a deliberate WAITING transition, or an expired lease.
        It never extends the immutable execution deadline.
        """
        stopped = threading.Event()
        failure: list[BaseException] = []

        def renew() -> None:
            while not stopped.is_set():
                try:
                    current = self.store.get_task_or_none(task_id)
                    if (
                        current is None
                        or stopped.is_set()
                        or current.state in TERMINAL_STATES
                        or current.state is ExecutionState.WAITING
                        or current.wall_clock_deadline_exceeded(self.clock())
                    ):
                        return
                    self.heartbeat(task_id, attempt_id=attempt_id, lease_token=lease_token)
                except (StaleLeaseError, LeaseConflictError, BudgetExceededError) as exc:
                    current = self.store.get_task_or_none(task_id)
                    if current is not None and current.state in {ExecutionState.RUNNING, ExecutionState.LEASED}:
                        failure.append(exc)
                    return
                if stopped.wait(min(0.02, max(0.005, float(self.config.heartbeat_seconds) / 10))):
                    break

        worker = threading.Thread(target=renew, name=f"lease-renewal-{task_id}", daemon=True)
        worker.start()
        try:
            yield
        finally:
            stopped.set()
            worker.join(timeout=max(1.0, float(self.config.heartbeat_seconds) + 0.5))
        if failure:
            raise failure[0]

    def resume_running(
        self,
        task_id: str,
        *,
        attempt_id: str,
        lease_token: str,
    ) -> TaskRecord:
        """Resume an existing, still-valid leased attempt without creating a rival attempt."""

        def resume(task: TaskRecord) -> None:
            self._validate_lease(task, attempt_id=attempt_id, lease_token=lease_token)
            if task.state == ExecutionState.RUNNING:
                task.heartbeat_at = task.updated_at = self.clock()
                return
            if task.state not in {ExecutionState.LEASED, ExecutionState.WAITING}:
                raise LeaseConflictError(
                    f"task cannot resume its active attempt from state {task.state.value}"
                )
            validate_transition(task.state, ExecutionState.RUNNING)
            task.state = ExecutionState.RUNNING
            task.heartbeat_at = task.updated_at = self.clock()
            task.started_at = task.started_at or task.updated_at

        task, _ = self.store.update_task(task_id, resume)
        attempt = self.store.get_attempt(attempt_id)
        if attempt is not None:
            attempt.state = "running"
            attempt.started_at = attempt.started_at or task.started_at
            self.store.save_attempt(attempt)
        self._emit("task_resumed", task, reused_attempt=True)
        return task

    def checkpoint(
        self,
        task_id: str,
        *,
        attempt_id: str,
        lease_token: str,
        resume_payload: dict[str, Any],
        completed_steps: Iterable[str] = (),
        pending_steps: Iterable[str] = (),
        tool_results: Iterable[dict[str, Any]] = (),
        workspace_reference: str = "",
        git_reference: str = "",
        generated_files: Iterable[str] = (),
        plan_version: int = 0,
        child_execution_ids: Iterable[str] = (),
        result_escrow_references: Iterable[str] = (),
        artifact_references: Iterable[str] = (),
        context_manifest_id: str = "",
        budget_snapshot: dict[str, Any] | None = None,
        retry_state: dict[str, Any] | None = None,
        idempotency_records: Iterable[str] = (),
        external_action_receipts: Iterable[str] = (),
        resume_cursor: str = "",
        capsule_revisions: dict[str, int] | None = None,
        provider_metadata: dict[str, Any] | Any | None = None,
    ) -> CheckpointRecord:
        original_state = self.store.get_task(task_id).state
        if original_state not in {ExecutionState.RUNNING, ExecutionState.WAITING}:
            raise LeaseConflictError(f"task cannot checkpoint from state {original_state.value}")

        def begin(task: TaskRecord) -> None:
            self._validate_lease(task, attempt_id=attempt_id, lease_token=lease_token)
            validate_transition(task.state, ExecutionState.CHECKPOINTING)
            task.state = ExecutionState.CHECKPOINTING
            task.updated_at = self.clock()

        checkpointing, _ = self.store.update_task(task_id, begin)
        self._emit("task_checkpointing", checkpointing)

        def persist(task: TaskRecord) -> CheckpointRecord:
            self._validate_lease(task, attempt_id=attempt_id, lease_token=lease_token)
            if task.state != ExecutionState.CHECKPOINTING:
                raise LeaseConflictError(f"task cannot checkpoint from state {task.state.value}")
            checkpoint = CheckpointRecord(
                task_id=task.task_id,
                attempt_id=attempt_id,
                state_version=task.state_version,
                resume_payload=resume_payload,
                completed_steps=list(completed_steps),
                pending_steps=list(pending_steps),
                tool_results=list(tool_results),
                workspace_reference=workspace_reference,
                git_reference=git_reference,
                generated_files=list(generated_files),
                plan_version=max(0, int(plan_version)),
                child_execution_ids=list(child_execution_ids),
                result_escrow_references=list(result_escrow_references),
                artifact_references=list(artifact_references),
                context_manifest_id=context_manifest_id,
                capsule_revisions=dict(
                    task.delegated_capsule_revisions
                    if capsule_revisions is None
                    else capsule_revisions
                ),
                budget_snapshot=dict(budget_snapshot or {}),
                retry_state=dict(retry_state or {}),
                idempotency_records=list(idempotency_records),
                external_action_receipts=list(external_action_receipts),
                resume_cursor=resume_cursor,
                provider_metadata=_provider_metadata(provider_metadata or task.provider_metadata),
            )
            task.provider_metadata = _provider_metadata(provider_metadata or task.provider_metadata)
            task.checkpoint_id = checkpoint.checkpoint_id
            task.checkpoint_count += 1
            validate_transition(task.state, original_state)
            task.state = original_state
            task.updated_at = self.clock()
            return checkpoint
        task, checkpoint = self.store.update_task_and_checkpoint(task_id, persist)
        attempt = self.store.get_attempt(attempt_id)
        if attempt:
            attempt.checkpoint_id = checkpoint.checkpoint_id
            self.store.save_attempt(attempt)
        self._emit("checkpoint_saved", task, checkpoint_id=checkpoint.checkpoint_id)
        return checkpoint

    def suspend_for_human_input(
        self,
        task_id: str,
        *,
        inbox_item_id: str,
        checkpoint_id: str,
        request_type: str,
    ) -> TaskRecord:
        """Durably suspend exactly one task branch and release its worker lease."""
        if request_type not in {"approval", "clarification"}:
            raise ValueError("unsupported human input request type")
        checkpoint = self.store.get_checkpoint(checkpoint_id) if checkpoint_id else None
        if checkpoint is None or checkpoint.task_id != task_id:
            raise RetrySafetyError("human-input suspension requires this branch's durable checkpoint")

        def suspend(task: TaskRecord) -> None:
            if task.waiting_inbox_item_id == inbox_item_id and task.state is ExecutionState.WAITING:
                return
            if task.state not in {
                ExecutionState.QUEUED,
                ExecutionState.LEASED,
                ExecutionState.RUNNING,
                ExecutionState.CHECKPOINTING,
                ExecutionState.WAITING,
                ExecutionState.RETRY_SCHEDULED,
            }:
                raise LeaseConflictError(
                    f"task cannot wait for human input from state {task.state.value}"
                )
            if task.checkpoint_id != checkpoint_id:
                raise RetrySafetyError("human-input checkpoint is no longer the active branch checkpoint")
            validate_transition(task.state, ExecutionState.WAITING)
            task.state = ExecutionState.WAITING
            task.waiting_inbox_item_id = inbox_item_id
            task.waiting_kind = "human_review" if request_type == "approval" else "human_input"
            task.wake_up_source = "human_inbox"
            task.wake_up_reference = inbox_item_id
            task.resume_checkpoint_id = checkpoint_id
            task.resume_operation = "resume_from_human_input"
            task.waiting_reason = f"waiting_for_{request_type}"
            task.human_wait_started_at = task.human_wait_started_at or self.clock()
            task.lease_owner = ""
            task.lease_token = ""
            task.lease_expires_at = None
            task.assigned_worker = ""
            task.updated_at = self.clock()

        task, _ = self.store.update_task(task_id, suspend)
        if task.attempt_id:
            attempt = self.store.get_attempt(task.attempt_id)
            if attempt is not None:
                attempt.state = task.waiting_reason
                attempt.lease_owner = ""
                attempt.lease_token = ""
                attempt.lease_expires_at = None
                self.store.save_attempt(attempt)
        self._emit(
            "branch_suspended",
            task,
            inbox_item_id=inbox_item_id,
            waiting_reason=task.waiting_reason,
            checkpoint_id=checkpoint_id,
        )
        return task

    def resume_from_human_input(
        self,
        task_id: str,
        *,
        inbox_item_id: str,
        checkpoint_id: str,
        resume_claim_id: str,
        structured_response: dict[str, Any],
    ) -> TaskRecord:
        """Queue one checkpointed branch once for one durable human response."""
        checkpoint = self.store.get_checkpoint(checkpoint_id) if checkpoint_id else None
        if checkpoint is None and checkpoint_id:
            raise RetrySafetyError("human response references a missing checkpoint")
        if checkpoint is not None and checkpoint.task_id != task_id:
            raise RetrySafetyError("human response references a foreign checkpoint")
        branch_snapshot = self.store.get_task_or_none(task_id)
        if branch_snapshot is None:
            return None
        ancestor_id = branch_snapshot.parent_task_id
        while ancestor_id:
            ancestor = self.store.get_task(ancestor_id)
            if (
                ancestor.state in TERMINAL_STATES
                and ancestor.state is not ExecutionState.RECOVERY_REVIEW_REQUIRED
            ):
                raise RetrySafetyError(
                    f"human response cannot resume under non-runnable ancestor {ancestor.task_id}"
                )
            ancestor_id = ancestor.parent_task_id

        if branch_snapshot.recovery_intervention_id:
            op = str(structured_response.get("operation") or "").lower()
            comment = str(structured_response.get("comment") or "")
            ans = structured_response.get("answer") or {}
            actor_id = str(structured_response.get("actor_id") or "operator")
            requested_action = str(ans.get("action") or "")
            if op == "deny" or requested_action == "ABORT_EXECUTION":
                return self.resolve_recovery_intervention(
                    branch_snapshot.recovery_intervention_id,
                    action=HumanRecoveryDecisionAction.ABORT_EXECUTION,
                    actor_id=actor_id,
                    comment=comment,
                    response_data=structured_response,
                )
            elif requested_action == "RETRY_ACTION":
                return self.resolve_recovery_intervention(
                    branch_snapshot.recovery_intervention_id,
                    action=HumanRecoveryDecisionAction.RETRY_ACTION,
                    actor_id=actor_id,
                    comment=comment,
                    response_data=structured_response,
                )
            elif requested_action == "MARK_ACTION_ALREADY_COMPLETED" or (op == "approve" and not requested_action):
                return self.resolve_recovery_intervention(
                    branch_snapshot.recovery_intervention_id,
                    action=HumanRecoveryDecisionAction.MARK_ACTION_ALREADY_COMPLETED,
                    actor_id=actor_id,
                    comment=comment,
                    response_data=structured_response,
                )
            else:
                intervention = self.store.get_recovery_intervention(branch_snapshot.recovery_intervention_id)
                action_to_resolve = (
                    HumanRecoveryDecisionAction.MARK_ACTION_ALREADY_COMPLETED
                    if intervention and intervention.action_id
                    else HumanRecoveryDecisionAction.RESUME_WITHOUT_REPLAY
                )
                return self.resolve_recovery_intervention(
                    branch_snapshot.recovery_intervention_id,
                    action=action_to_resolve,
                    actor_id=actor_id,
                    comment=comment,
                    response_data=structured_response,
                )

        wait_duration = timedelta(0)

        def resume(task: TaskRecord) -> None:
            nonlocal wait_duration
            if resume_claim_id in task.human_resume_claim_ids:
                return
            if task.state is not ExecutionState.WAITING or task.waiting_inbox_item_id != inbox_item_id:
                raise LeaseConflictError("only the branch waiting for this inbox item may resume")
            if task.checkpoint_id != checkpoint_id:
                raise RetrySafetyError("human response checkpoint no longer matches the branch")
            validate_transition(task.state, ExecutionState.QUEUED)
            task.human_inputs.append({
                "inbox_item_id": inbox_item_id,
                "checkpoint_id": checkpoint_id,
                "resume_claim_id": resume_claim_id,
                "response": structured_response,
                "received_at": self.clock().isoformat(),
            })
            task.human_resume_claim_ids.append(resume_claim_id)
            if task.deadline_at is not None and task.human_wait_started_at is not None:
                wait_duration = self.clock() - task.human_wait_started_at
                task.deadline_at += wait_duration
            task.state = ExecutionState.QUEUED
            task.waiting_inbox_item_id = ""
            task.waiting_kind = ""
            task.wake_up_source = ""
            task.wake_up_reference = ""
            task.resume_checkpoint_id = ""
            task.resume_operation = ""
            task.waiting_reason = ""
            task.human_wait_started_at = None
            task.lease_owner = ""
            task.lease_token = ""
            task.lease_expires_at = None
            task.retry_not_before = None
            task.updated_at = self.clock()

        task, _ = self.store.update_task(task_id, resume)
        parent_id = task.parent_task_id
        while parent_id and wait_duration > timedelta(0):
            def extend_parent(parent: TaskRecord) -> None:
                if parent.deadline_at is not None:
                    parent.deadline_at += wait_duration
                parent.updated_at = self.clock()
            parent, _ = self.store.update_task(parent_id, extend_parent)
            parent_id = parent.parent_task_id
        self._emit(
            "branch_resumed",
            task,
            inbox_item_id=inbox_item_id,
            checkpoint_id=checkpoint_id,
            resume_claim_id=resume_claim_id,
        )
        return task

    def restore_human_wait(
        self,
        task_id: str,
        *,
        inbox_item_id: str,
        checkpoint_id: str,
        request_type: str,
    ) -> TaskRecord:
        """Repair a branch projection from an authoritative unresolved inbox item."""
        return self.suspend_for_human_input(
            task_id,
            inbox_item_id=inbox_item_id,
            checkpoint_id=checkpoint_id,
            request_type=request_type,
        )

    def record_external_wait(
        self,
        task_id: str,
        *,
        waiting_kind: str,
        wake_up_source: str,
        wake_up_reference: str,
        resume_checkpoint_id: str = "",
        resume_operation: str = "",
    ) -> TaskRecord:
        """Persist the wake contract required for a non-human external wait."""
        if not waiting_kind.strip() or not wake_up_source.strip() or not wake_up_reference.strip():
            raise ValueError("external waits require kind, wake-up source, and wake-up reference")

        def update(task: TaskRecord) -> None:
            if task.state is not ExecutionState.WAITING:
                raise InvalidTransitionError("external wait metadata requires a waiting supervisor task")
            task.waiting_kind = waiting_kind
            task.wake_up_source = wake_up_source
            task.wake_up_reference = wake_up_reference
            task.resume_checkpoint_id = resume_checkpoint_id
            task.resume_operation = resume_operation
            task.updated_at = self.clock()

        task, _ = self.store.update_task(task_id, update)
        self._emit("external_wait_recorded", task, wake_up_source=wake_up_source)
        return task

    def suspend_for_connector(
        self,
        task_id: str,
        *,
        connector_id: str,
        checkpoint_id: str = "",
        reason: str = "connector_unavailable",
    ) -> TaskRecord:
        """Durably pause a branch that depends on an unavailable connector.

        Does not retry into a known outage. Resume only via
        :meth:`resume_from_connector` when the connector is healthy again.
        """
        if not connector_id.strip():
            raise ValueError("connector_id is required")
        active_checkpoint = checkpoint_id
        if not active_checkpoint:
            task = self.store.get_task(task_id)
            active_checkpoint = task.checkpoint_id
        if active_checkpoint:
            checkpoint = self.store.get_checkpoint(active_checkpoint)
            if checkpoint is None or checkpoint.task_id != task_id:
                raise RetrySafetyError("connector suspension requires this branch's durable checkpoint")

        def suspend(task: TaskRecord) -> None:
            if (
                task.waiting_connector_id == connector_id
                and task.state is ExecutionState.WAITING
                and task.waiting_reason == "waiting_for_connector"
            ):
                return
            if task.state not in {
                ExecutionState.QUEUED,
                ExecutionState.LEASED,
                ExecutionState.RUNNING,
                ExecutionState.CHECKPOINTING,
                ExecutionState.WAITING,
                ExecutionState.RETRY_SCHEDULED,
            }:
                raise LeaseConflictError(
                    f"task cannot wait for connector from state {task.state.value}"
                )
            if active_checkpoint and task.checkpoint_id and task.checkpoint_id != active_checkpoint:
                raise RetrySafetyError("connector checkpoint is no longer the active branch checkpoint")
            if task.state is not ExecutionState.WAITING:
                validate_transition(task.state, ExecutionState.WAITING)
            task.state = ExecutionState.WAITING
            task.waiting_reason = "waiting_for_connector"
            task.waiting_kind = "external_dependency"
            task.wake_up_source = "connector_health"
            task.wake_up_reference = connector_id
            task.resume_checkpoint_id = active_checkpoint
            task.resume_operation = "resume_from_connector"
            task.waiting_connector_id = connector_id
            task.human_wait_started_at = task.human_wait_started_at or self.clock()
            task.lease_owner = ""
            task.lease_token = ""
            task.lease_expires_at = None
            task.assigned_worker = ""
            task.recovery_reason = reason
            task.updated_at = self.clock()

        task, _ = self.store.update_task(task_id, suspend)
        if task.attempt_id:
            attempt = self.store.get_attempt(task.attempt_id)
            if attempt is not None:
                attempt.state = "waiting_for_connector"
                attempt.lease_owner = ""
                attempt.lease_token = ""
                attempt.lease_expires_at = None
                self.store.save_attempt(attempt)
        self._emit(
            "connector_branch_suspended",
            task,
            connector_id=connector_id,
            waiting_reason=task.waiting_reason,
            checkpoint_id=active_checkpoint,
            reason=reason,
        )
        return task

    def resume_from_connector(
        self,
        task_id: str,
        *,
        connector_id: str,
        resume_claim_id: str,
    ) -> TaskRecord:
        """Resume a connector-paused branch exactly once after health recovery."""
        if not resume_claim_id.strip():
            raise ValueError("resume_claim_id is required")

        def resume(task: TaskRecord) -> None:
            if resume_claim_id in task.human_resume_claim_ids:
                return
            if (
                task.state is not ExecutionState.WAITING
                or task.waiting_reason != "waiting_for_connector"
                or task.waiting_connector_id != connector_id
            ):
                raise LeaseConflictError("only the branch waiting for this connector may resume")
            validate_transition(task.state, ExecutionState.QUEUED)
            task.human_resume_claim_ids.append(resume_claim_id)
            task.state = ExecutionState.QUEUED
            task.waiting_reason = ""
            task.waiting_kind = ""
            task.wake_up_source = ""
            task.wake_up_reference = ""
            task.resume_checkpoint_id = ""
            task.resume_operation = ""
            task.waiting_connector_id = ""
            task.human_wait_started_at = None
            task.recovery_reason = "connector_healthy"
            task.lease_owner = ""
            task.lease_token = ""
            task.lease_expires_at = None
            task.retry_not_before = None
            task.updated_at = self.clock()

        task, _ = self.store.update_task(task_id, resume)
        self._emit(
            "connector_branch_resumed",
            task,
            connector_id=connector_id,
            resume_claim_id=resume_claim_id,
        )
        return task

    def submit_result(
        self,
        task_id: str,
        *,
        attempt_id: str,
        lease_token: str,
        payload: dict[str, Any],
        token_usage: int = 0,
        actual_cost: float | None = None,
        capsule_revisions: dict[str, int] | None = None,
        provider_metadata: dict[str, Any] | Any | None = None,
    ) -> TaskRecord:
        current = self.store.get_task(task_id)
        projected_tokens = current.token_usage + max(0, token_usage)
        projected_cost = current.actual_cost + max(0.0, float(actual_cost or 0.0))
        ancestors: list[TaskRecord] = []
        parent_id = current.parent_task_id
        while parent_id:
            ancestor = self.store.get_task(parent_id)
            ancestors.append(ancestor)
            parent_id = ancestor.parent_task_id
        overrun_scopes: list[dict[str, Any]] = []
        if current.token_budget is not None and projected_tokens > current.token_budget:
            overrun_scopes.append({"scope": "task", "task_id": current.task_id, "kind": "tokens", "limit": current.token_budget, "actual": projected_tokens})
        if current.monetary_budget is not None and projected_cost > current.monetary_budget:
            overrun_scopes.append({"scope": "task", "task_id": current.task_id, "kind": "cost", "limit": current.monetary_budget, "actual": projected_cost})
        for ancestor in ancestors:
            ancestor_tokens = ancestor.token_usage + max(0, token_usage)
            ancestor_cost = ancestor.actual_cost + max(0.0, float(actual_cost or 0.0))
            if (
                ancestor.token_budget is not None
                and ancestor_tokens > ancestor.token_budget
            ) or (
                ancestor.monetary_budget is not None
                and ancestor_cost > ancestor.monetary_budget
            ):
                overrun_scopes.append({
                    "scope": "ancestor", "task_id": ancestor.task_id,
                    "kind": "tokens" if ancestor.token_budget is not None and ancestor_tokens > ancestor.token_budget else "cost",
                    "limit": ancestor.token_budget if ancestor.token_budget is not None and ancestor_tokens > ancestor.token_budget else ancestor.monetary_budget,
                    "actual": ancestor_tokens if ancestor.token_budget is not None and ancestor_tokens > ancestor.token_budget else ancestor_cost,
                })

        def escrow(task: TaskRecord) -> EscrowResult:
            self._validate_lease(task, attempt_id=attempt_id, lease_token=lease_token)
            if task.state not in {ExecutionState.RUNNING, ExecutionState.WAITING, ExecutionState.COMPLETED_PENDING_VERIFICATION}:
                raise LeaseConflictError(f"task cannot publish a result from state {task.state.value}")
            target_state = (
                ExecutionState.PENDING_BUDGET_DECISION
                if overrun_scopes else ExecutionState.COMPLETED_PENDING_VERIFICATION
            )
            validate_transition(task.state, target_state)
            result = EscrowResult(
                task_id=task.task_id,
                execution_id=task.task_id,
                root_task_id=task.root_task_id or task.task_id,
                parent_task_id=task.parent_task_id,
                trigger_turn_id=task.trigger_turn_id,
                session_id=task.session_id,
                lane_id=(
                    task.assigned_agent.removeprefix("lane:")
                    if task.assigned_agent.startswith("lane:")
                    else task.assigned_agent
                ),
                owning_lane=(
                    task.assigned_agent.removeprefix("lane:")
                    if task.assigned_agent.startswith("lane:")
                    else task.assigned_agent
                ),
                attempt_id=attempt_id,
                attempt_generation=task.attempt_generation,
                lease_token_hash=_token_hash(lease_token),
                payload=payload,
                capsule_revisions=dict(capsule_revisions or {}),
                status=EscrowStatus.PRODUCED,
                supervisor_state=target_state.value,
                verification_status=VerificationStatus.PENDING,
                result_kind="chat_result",
                provider_metadata=_provider_metadata(provider_metadata or task.provider_metadata),
                created_at=self.clock(),
            )
            task.provider_metadata = _provider_metadata(provider_metadata or task.provider_metadata)
            task.result_id = result.result_id
            task_had_usage = task.token_usage > 0
            task.token_usage += max(0, token_usage)
            task.actual_cost += max(0.0, float(actual_cost or 0.0))
            task.actual_cost_known = (
                (not task_had_usage or task.actual_cost_known)
                and actual_cost is not None
            )
            task.result_capsule_revisions = dict(capsule_revisions or {})
            if overrun_scopes:
                evidence_payload = {
                    "task_id": task.task_id, "attempt_id": attempt_id,
                    "result_id": result.result_id, "token_usage": projected_tokens,
                    "actual_cost": projected_cost, "overruns": overrun_scopes,
                }
                task.budget_overrun = {
                    **evidence_payload,
                    "evidence_hash": "sha256:" + hashlib.sha256(
                        repr(evidence_payload).encode("utf-8")
                    ).hexdigest(),
                    "status": "pending_model_decision",
                }
            task.state = target_state
            task.updated_at = self.clock()
            return result
        try:
            task, result = self.store.update_task_and_result(task_id, escrow)
        except StaleLeaseError as exc:
            rejected = self.store.get_task(task_id)
            self._emit(
                "result_rejected",
                rejected,
                rejected_attempt_id=attempt_id,
                active_attempt_id=rejected.attempt_id,
                active_generation=rejected.attempt_generation,
                reason=str(exc),
            )
            raise
        self._emit("result_produced", task, result_id=result.result_id)
        result.status = EscrowStatus.STORED
        self.store.save_result(result)
        self._emit("result_stored", task, result_id=result.result_id)
        result.status = EscrowStatus.AVAILABLE
        self.store.save_result(result)
        attempt = self.store.get_attempt(attempt_id)
        if attempt is not None:
            attempt.token_usage += max(0, token_usage)
            attempt.actual_cost += max(0.0, float(actual_cost or 0.0))
            attempt.actual_cost_known = actual_cost is not None
            self.store.save_attempt(attempt)
        for ancestor in ancestors:
            def aggregate(parent: TaskRecord) -> None:
                parent_had_usage = parent.token_usage > 0
                parent.token_usage += max(0, token_usage)
                parent.actual_cost += max(0.0, float(actual_cost or 0.0))
                parent.actual_cost_known = (
                    (not parent_had_usage or parent.actual_cost_known)
                    and actual_cost is not None
                )
                parent.updated_at = self.clock()
            self.store.update_task(ancestor.task_id, aggregate)
        self._emit("result_escrowed", task, result_id=result.result_id)
        if overrun_scopes:
            self._emit(
                "budget_overrun_decision_required", task,
                result_id=result.result_id,
                evidence_hash=task.budget_overrun["evidence_hash"],
                overruns=overrun_scopes,
            )
            return task
        self._emit("verification_started", task)
        return self.verify_completion(task_id)

    def revise_budget(
        self,
        task_id: str,
        *,
        token_budget: int,
        estimated_cost: float | None,
        reason: str,
        evidence: dict[str, Any],
    ) -> TaskRecord:
        """Persist a policy-validated reservation revision before more work runs."""
        requested = max(0, int(token_budget))

        def revise(task: TaskRecord) -> None:
            if task.state not in {
                ExecutionState.QUEUED, ExecutionState.LEASED, ExecutionState.RUNNING,
                ExecutionState.WAITING, ExecutionState.RETRY_SCHEDULED, ExecutionState.REPLANNING,
            }:
                raise BudgetExceededError("task cannot revise its budget from the current state")
            if requested < task.token_usage:
                raise BudgetExceededError("revised token budget is below recorded usage")
            next_cost = max(task.estimated_cost, float(estimated_cost or 0.0))
            if task.monetary_budget is not None and (
                estimated_cost is None or next_cost > task.monetary_budget
            ):
                raise BudgetExceededError("revised estimated cost exceeds the immutable monetary budget")
            previous = task.token_budget
            previous_cost = task.estimated_cost
            if previous is not None and requested <= previous and next_cost <= previous_cost:
                return
            task.token_budget = max(previous or 0, requested) or None
            task.estimated_cost = next_cost
            task.estimated_cost_known = estimated_cost is not None
            task.budget_revisions.append(BudgetRevision(
                reason=reason,
                previous_token_budget=previous,
                revised_token_budget=task.token_budget,
                previous_estimated_cost=previous_cost,
                revised_estimated_cost=next_cost,
                evidence=dict(evidence),
            ))
            task.updated_at = self.clock()

        task, _ = self.store.update_task(task_id, revise)
        self._emit("budget_revised", task, reason=reason, evidence=evidence)
        return task

    def finalize_budget_overrun(
        self,
        decision: BudgetOverrunFinalizationDecision,
    ) -> TaskRecord:
        """Apply only a fresh model decision tied to the immutable overrun evidence."""
        task = self.store.get_task(decision.task_id)
        overrun = dict(task.budget_overrun or {})
        if task.state is not ExecutionState.PENDING_BUDGET_DECISION:
            raise BudgetExceededError("task is not awaiting a budget-overrun decision")
        if (
            task.attempt_id != decision.attempt_id or task.result_id != decision.result_id
            or overrun.get("evidence_hash") != decision.result_evidence_hash
        ):
            raise BudgetExceededError("budget-overrun decision does not match durable result evidence")
        if decision.action is BudgetOverrunAction.RETRY_OR_REPLAN:
            recovery = decision.recovery_decision
            if recovery is None:
                raise BudgetExceededError("budget-overrun recovery decision is missing")
            if task.token_budget is not None and task.token_usage >= task.token_budget:
                raise BudgetExceededError("no separately funded token allocation is available for recovery")
            if task.monetary_budget is not None and task.actual_cost >= task.monetary_budget:
                raise BudgetExceededError("no separately funded cost allocation is available for recovery")
            self.retry(task.task_id, recovery)
            return self.store.get_task(task.task_id)

        def finalize(current: TaskRecord) -> None:
            if current.state is not ExecutionState.PENDING_BUDGET_DECISION:
                raise BudgetExceededError("budget-overrun decision is stale")
            current.budget_finalization_decision_id = decision.decision_id
            current.budget_overrun["decision"] = decision.action.value
            current.budget_overrun["decision_reason"] = decision.reason
            current.updated_at = self.clock()
            if decision.action is BudgetOverrunAction.ACCEPT_WITH_OVERRUN:
                if current.verification_status is not VerificationStatus.PASSED:
                    raise CompletionVerificationError("budget-overrun result must pass completion verification before acceptance")
                validate_transition(current.state, ExecutionState.COMPLETED)
                current.state = ExecutionState.COMPLETED
                current.finished_at = current.updated_at
                current.budget_overrun["status"] = "accepted_with_overrun"
            else:
                current.budget_overrun["status"] = "requires_human_review"

        if decision.action is BudgetOverrunAction.ACCEPT_WITH_OVERRUN:
            self.verify_completion(task.task_id)
        task, _ = self.store.update_task(task.task_id, finalize)
        self._emit("budget_overrun_finalized", task, decision_id=decision.decision_id, action=decision.action.value)
        return task

    def set_completion_contract(
        self,
        task_id: str,
        *,
        attempt_id: str,
        lease_token: str,
        contracts: Iterable[CompletionContract],
    ) -> TaskRecord:
        selected = list(contracts)
        if not selected:
            raise ValueError("at least one explicit completion contract is required")
        def update(task: TaskRecord) -> None:
            self._validate_lease(task, attempt_id=attempt_id, lease_token=lease_token)
            task.completion_contract = selected
            task.updated_at = self.clock()
        task, _ = self.store.update_task(task_id, update)
        self._emit("completion_contract_updated", task, contract_count=len(selected))
        return task

    def verify_completion(self, task_id: str) -> TaskRecord:
        task = self.store.get_task(task_id)
        if task.state not in {
            ExecutionState.COMPLETED_PENDING_VERIFICATION,
            ExecutionState.PENDING_BUDGET_DECISION,
        }:
            raise CompletionVerificationError("task is not awaiting completion verification")
        result = self.store.get_result(task.result_id)
        if result is None:
            self._emit("verification_failed", task, reason="result escrow is missing")
            raise CompletionVerificationError("durable result escrow is missing")
        if not self.config.verify_completion_artifacts:
            self._emit(
                "verification_failed",
                task,
                reason=(
                    "completion verification is disabled; the claimed result remains in escrow "
                    "and cannot be advertised as completed"
                ),
            )
            raise CompletionVerificationError(
                "completion verification is disabled; claimed result remains in escrow"
            )
        attempt = self.store.get_attempt(result.attempt_id)
        workspace = Path(task.workspace_path) if task.workspace_path else Path.cwd()
        blocking_children = []
        for child_id in task.child_task_ids:
            child = self.store.get_task(child_id)
            if child.state is ExecutionState.COMPLETED_PENDING_VERIFICATION:
                try:
                    child = self.verify_completion(child_id)
                except Exception:
                    pass
            child_result = self.store.get_result(child.result_id) if child.result_id else None
            if child.state != ExecutionState.COMPLETED or child_result is None:
                blocking_children.append(child_id)
            elif child_result.acknowledged_at is None:
                try:
                    self.acknowledge_result(child_result.result_id, parent_task_id=task.task_id)
                except Exception:
                    pass
        unresolved_actions = [
            action.action_id
            for action in self.store.actions_for_task(task_id)
            if action.request_state not in {
                ActionRequestState.SUCCEEDED,
                ActionRequestState.RECONCILED,
            }
        ]
        if blocking_children or unresolved_actions:
            report = VerificationReport(
                status=VerificationStatus.FAILED,
                checks=[{
                    "verifier_type": "supervisor_completion_gate",
                    "expected_condition": {
                        "children": "completed_and_acknowledged",
                        "actions": "succeeded_or_reconciled",
                    },
                    "observed_condition": {
                        "blocking_child_execution_ids": blocking_children,
                        "unresolved_action_ids": unresolved_actions,
                    },
                    "status": "failed",
                    "passed": False,
                    "failure_reason": "child results or consequential actions remain unresolved",
                    "timestamp": self.clock().isoformat(),
                }],
            )
        else:
            try:
                report = self.verifier.verify(
                    task.completion_contract,
                    workspace=workspace,
                    result_payload=result.payload,
                    attempt_started_at=attempt.started_at if attempt else None,
                )
            except Exception as exc:
                self._emit(
                    "verification_failed",
                    task,
                    reason=f"completion verifier failed safely: {type(exc).__name__}: {exc}",
                )
                raise CompletionVerificationError(
                    "completion verifier failed; claimed result remains in escrow"
                ) from exc
        result.artifacts = list(report.artifacts)
        result.status = EscrowStatus.AVAILABLE
        result.verification_status = report.status
        result.supervisor_state = (
            ExecutionState.COMPLETED.value
            if report.status == VerificationStatus.PASSED
            else ExecutionState.FAILED.value
        )
        result.verified_at = self.clock()
        result.completed_at = result.completed_at or self.clock()
        self.store.save_result(result)
        def apply_report(current: TaskRecord) -> None:
            if current.result_id != result.result_id:
                raise CompletionVerificationError("task result changed during verification")
            current.completion_artefacts = list(report.artifacts)
            current.verification_status = report.status
            current.updated_at = self.clock()
            if (
                report.status == VerificationStatus.PASSED
                and current.state is ExecutionState.COMPLETED_PENDING_VERIFICATION
            ):
                validate_transition(current.state, ExecutionState.COMPLETED)
                current.state = ExecutionState.COMPLETED
                current.finished_at = current.updated_at
                current.lease_owner = ""
                current.lease_token = ""
                current.lease_expires_at = None
        task, _ = self.store.update_task(task_id, apply_report)
        self.store.save_artifact_manifest(
            task_id,
            {
                "task_id": task_id,
                "attempt_id": result.attempt_id,
                "artefacts": [item.model_dump(mode="json") for item in report.artifacts],
                "verification": report.model_dump(mode="json"),
            },
        )
        if report.status == VerificationStatus.PASSED and task.state is ExecutionState.COMPLETED:
            for artifact in report.artifacts:
                self._emit("artefact_verified", task, artifact=artifact.model_dump(mode="json"))
            self._emit("task_completed", task, result_id=result.result_id)
            attempt = self.store.get_attempt(result.attempt_id)
            if attempt:
                attempt.state = "completed"
                attempt.finished_at = task.finished_at
                self.store.save_attempt(attempt)
        else:
            self._emit("verification_failed", task, checks=report.checks)
        return task

    def acknowledge_result(
        self,
        result_id: str,
        *,
        parent_task_id: str | None = None,
        consumer_execution_id: str = "",
        consumer_turn_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ResultAcknowledgement:
        result = self.store.get_result(result_id)
        if result is None:
            raise CompletionVerificationError(f"unknown escrow result: {result_id}")
        if parent_task_id and result.parent_task_id and result.parent_task_id != parent_task_id:
            raise CompletionVerificationError("result does not belong to the acknowledging parent")
        existing_ack = self.store.get_acknowledgement(result_id)
        if existing_ack is not None:
            return existing_ack
        now = self.clock()
        ack = ResultAcknowledgement(
            result_id=result_id,
            execution_id=result.execution_id or result.task_id,
            consumer_execution_id=consumer_execution_id or parent_task_id or "",
            consumer_turn_id=consumer_turn_id,
            acknowledged_at=now,
            acknowledged_by=parent_task_id or consumer_execution_id or consumer_turn_id or "caller",
            metadata=dict(metadata or {}),
        )
        self.store.save_acknowledgement(ack)
        result.acknowledged_at = now
        result.acknowledged_by = ack.acknowledged_by
        self.store.save_result(result)
        task = self.store.get_task_or_none(result.task_id)
        if task is not None:
            self._emit(
                "result_acknowledged",
                task,
                result_id=result_id,
                parent_task_id=parent_task_id or "",
                consumer_turn_id=consumer_turn_id,
                consumer_execution_id=consumer_execution_id,
            )
        return ack

    def get_verified_execution_result(
        self, execution_id: str
    ) -> VerifiedExecutionResultLookup:
        if not str(execution_id).strip():
            return VerifiedExecutionResultLookup(
                status=EscrowLookupStatus.NOT_FOUND,
                execution_id=str(execution_id),
                error_code="RESULT_NOT_FOUND",
                error_message="execution_id is empty or missing",
            )
        exec_id = str(execution_id).strip()
        task = self.store.get_task_or_none(exec_id)

        try:
            result = self.store.get_result_by_execution_id(exec_id)
            if result is None and task is not None and task.result_id:
                result = self.store.get_result(task.result_id)
        except EscrowCorruptError as exc:
            return VerifiedExecutionResultLookup(
                status=EscrowLookupStatus.CORRUPT,
                execution_id=exec_id,
                task=task,
                error_code="RESULT_CORRUPT",
                error_message=str(exc),
            )
        except EscrowIncompatibleVersionError as exc:
            return VerifiedExecutionResultLookup(
                status=EscrowLookupStatus.INCOMPATIBLE_VERSION,
                execution_id=exec_id,
                task=task,
                error_code="RESULT_SCHEMA_INCOMPATIBLE",
                error_message=str(exc),
            )

        if result is not None:
            is_term = (
                result.supervisor_state
                in {
                    ExecutionState.COMPLETED.value,
                    ExecutionState.FAILED.value,
                    ExecutionState.CANCELLED.value,
                    ExecutionState.BUDGET_EXHAUSTED.value,
                    ExecutionState.RECOVERY_REVIEW_REQUIRED.value,
                }
                or (task is not None and task.state in TERMINAL_STATES)
            )
            is_ver = (
                result.verification_status == VerificationStatus.PASSED
                or (task is not None and task.state == ExecutionState.COMPLETED)
            )
            is_resumable = bool(
                result.result_kind == "resumable_wait"
                or result.error_metadata.get("is_resumable")
                or (task is not None and task.state is ExecutionState.WAITING)
            )

            # Repair stale task record if authoritative result is terminal or verified
            if task is not None and (is_term or is_ver):
                if (
                    task.state not in TERMINAL_STATES
                    or task.result_id != result.result_id
                    or task.verification_status != result.verification_status
                ):
                    target_state = (
                        ExecutionState.COMPLETED
                        if (is_ver or result.supervisor_state == ExecutionState.COMPLETED.value)
                        else (
                            ExecutionState(result.supervisor_state)
                            if result.supervisor_state in {s.value for s in TERMINAL_STATES}
                            else ExecutionState.COMPLETED
                        )
                    )
                    def repair_task(current: TaskRecord) -> None:
                        current.result_id = result.result_id
                        current.state = target_state
                        current.verification_status = (
                            VerificationStatus.PASSED
                            if target_state == ExecutionState.COMPLETED
                            else result.verification_status
                        )
                        if result.artifacts:
                            current.completion_artefacts = list(result.artifacts)
                        if target_state in TERMINAL_STATES:
                            current.finished_at = result.completed_at or current.finished_at or self.clock()
                            current.lease_owner = ""
                            current.lease_token = ""
                            current.lease_expires_at = None
                            current.retry_not_before = None
                        if result.error_metadata:
                            if result.error_metadata.get("reason"):
                                current.failure_reason = str(result.error_metadata["reason"])
                            if result.error_metadata.get("recovery_reason"):
                                current.recovery_reason = str(result.error_metadata["recovery_reason"])
                        if result.provider_metadata:
                            current.provider_metadata = {**current.provider_metadata, **result.provider_metadata}
                        current.updated_at = self.clock()
                    try:
                        task, _ = self.store.update_task(task.task_id, repair_task)
                        if task.attempt_id:
                            attempt = self.store.get_attempt(task.attempt_id)
                            if attempt is not None:
                                attempt.state = target_state.value
                                attempt.finished_at = task.finished_at
                                self.store.save_attempt(attempt)
                    except Exception:
                        pass

            ack = self.store.get_acknowledgement(result.result_id)
            if is_term or is_ver or is_resumable:
                return VerifiedExecutionResultLookup(
                    status=EscrowLookupStatus.FOUND,
                    execution_id=exec_id,
                    result=result,
                    task=task,
                    acknowledgement=ack,
                    is_terminal=is_term,
                    is_resumable=is_resumable,
                    is_verified=is_ver,
                    requires_action=is_resumable,
                )

            if (
                task is not None
                and task.state is ExecutionState.COMPLETED_PENDING_VERIFICATION
            ):
                return VerifiedExecutionResultLookup(
                    status=EscrowLookupStatus.UNVERIFIED,
                    execution_id=exec_id,
                    task=task,
                    result=result,
                    acknowledgement=ack,
                    error_code="RESULT_NOT_VERIFIED",
                    error_message=f"Execution {exec_id} is pending completion verification",
                )

            return VerifiedExecutionResultLookup(
                status=EscrowLookupStatus.EXECUTION_STILL_RUNNING,
                execution_id=exec_id,
                task=task,
                result=result,
                error_code="EXECUTION_STILL_RUNNING",
                error_message=f"Execution {exec_id} is active",
            )

        if task is not None and task.state in {
            ExecutionState.QUEUED,
            ExecutionState.LEASED,
            ExecutionState.RUNNING,
            ExecutionState.CHECKPOINTING,
            ExecutionState.RETRY_SCHEDULED,
            ExecutionState.REPLANNING,
            ExecutionState.CANCELLING,
        }:
            return VerifiedExecutionResultLookup(
                status=EscrowLookupStatus.EXECUTION_STILL_RUNNING,
                execution_id=exec_id,
                task=task,
                error_code="EXECUTION_STILL_RUNNING",
                error_message=f"Execution {exec_id} is active ({task.state.value})",
            )

        if task is not None and task.state is ExecutionState.WAITING:
            return VerifiedExecutionResultLookup(
                status=EscrowLookupStatus.EXECUTION_STILL_RUNNING,
                execution_id=exec_id,
                task=task,
                is_resumable=True,
                requires_action=True,
                error_code="ACTION_REQUIRED",
                error_message=task.waiting_reason
                or "Execution is waiting for approval/input",
            )

        if (
            task is not None
            and task.state is ExecutionState.COMPLETED_PENDING_VERIFICATION
        ):
            return VerifiedExecutionResultLookup(
                status=EscrowLookupStatus.UNVERIFIED,
                execution_id=exec_id,
                task=task,
                error_code="RESULT_NOT_VERIFIED",
                error_message=f"Execution {exec_id} is pending completion verification",
            )

        if task is not None and task.state in TERMINAL_STATES:
            return VerifiedExecutionResultLookup(
                status=EscrowLookupStatus.NOT_FOUND,
                execution_id=exec_id,
                task=task,
                is_terminal=True,
                error_code="RESULT_NOT_FOUND",
                error_message=f"No escrow result found for terminal execution {exec_id}",
            )

        return VerifiedExecutionResultLookup(
            status=EscrowLookupStatus.NOT_FOUND,
            execution_id=exec_id,
            error_code="RESULT_NOT_FOUND",
            error_message=f"Unknown execution identity: {exec_id}",
        )


    def reverify_completed(self, task_id: str) -> TaskRecord:
        """Invalidate the completion projection until durable evidence passes again."""
        current = self.store.get_task(task_id)
        if current.state != ExecutionState.COMPLETED:
            raise CompletionVerificationError("only a completed task can be reverified")
        if current.verification_status != VerificationStatus.PASSED or not current.result_id:
            raise CompletionVerificationError("completed task has no reusable verified result")
        recorded_hashes = {
            artifact.path: artifact.sha256
            for artifact in current.completion_artefacts
            if artifact.path and artifact.sha256
        }

        def reopen(task: TaskRecord) -> None:
            validate_transition(task.state, ExecutionState.COMPLETED_PENDING_VERIFICATION)
            task.state = ExecutionState.COMPLETED_PENDING_VERIFICATION
            task.verification_status = VerificationStatus.PENDING
            task.finished_at = None
            task.completion_contract = [
                contract.model_copy(
                    update={"expected_sha256": recorded_hashes[contract.path]}
                )
                if contract.path in recorded_hashes
                else contract
                for contract in task.completion_contract
            ]
            task.updated_at = self.clock()

        task, _ = self.store.update_task(task_id, reopen)
        self._emit("reverification_started", task, result_id=task.result_id)
        verified = self.verify_completion(task_id)
        if verified.state != ExecutionState.COMPLETED:
            raise CompletionVerificationError(
                "stored completion evidence no longer satisfies the contract"
            )
        return verified

    def mark_irreversible_side_effect(
        self, task_id: str, *, attempt_id: str, lease_token: str
    ) -> TaskRecord:
        def update(task: TaskRecord) -> None:
            self._validate_lease(task, attempt_id=attempt_id, lease_token=lease_token)
            task.irreversible_side_effect_started = True
            task.updated_at = self.clock()
        task, _ = self.store.update_task(task_id, update)
        self._emit("side_effect_phase_started", task, irreversible=True)
        return task

    def prepare_action(
        self,
        task_id: str,
        *,
        attempt_id: str,
        lease_token: str,
        tool_name: str,
        action_fingerprint: str,
        classification: SideEffectClassification,
        idempotency_key: str = "",
    ) -> ActionRecord:
        task = self.store.get_task(task_id)
        self._validate_lease(task, attempt_id=attempt_id, lease_token=lease_token)
        for existing in self.store.actions_for_task(task_id):
            if existing.action_fingerprint != action_fingerprint:
                continue
            if existing.request_state in {
                ActionRequestState.STARTED,
                ActionRequestState.SUCCEEDED,
                ActionRequestState.OUTCOME_UNKNOWN,
            }:
                raise RetrySafetyError(
                    "matching consequential action already exists; reconcile its durable receipt before retry"
                )
        if classification in {
            SideEffectClassification.IDEMPOTENT,
            SideEffectClassification.CONDITIONALLY_IDEMPOTENT,
            SideEffectClassification.DEDUPLICATED,
        } and not idempotency_key:
            raise RetrySafetyError(
                f"{classification.value} action requires a durable idempotency key"
            )
        action = ActionRecord(
            execution_id=task_id,
            attempt_id=attempt_id,
            attempt_generation=task.attempt_generation,
            tool_name=tool_name,
            action_fingerprint=action_fingerprint,
            idempotency_key=_token_hash(f"action:{idempotency_key}") if idempotency_key else "",
            classification=classification,
            request_state=ActionRequestState.PREPARED,
        )
        self.store.save_action(action)
        self._emit("action_prepared", task, action_id=action.action_id, tool_name=tool_name)
        return action

    def update_action(
        self,
        action_id: str,
        *,
        request_state: ActionRequestState,
        external_receipt: str = "",
        result_reference: str = "",
        verification_state: dict[str, Any] | None = None,
    ) -> ActionRecord:
        action = self.store.get_action(action_id)
        if action is None:
            raise RetrySafetyError(f"unknown durable action: {action_id}")
        task = self.store.get_task(action.execution_id)
        if action.attempt_generation != task.attempt_generation:
            raise StaleLeaseError("stale attempt generation cannot update an action record")
        action.request_state = request_state
        action.external_receipt = external_receipt or action.external_receipt
        action.result_reference = result_reference or action.result_reference
        action.verification_state.update(dict(verification_state or {}))
        action.updated_at = self.clock()
        self.store.save_action(action)
        self._emit("action_updated", task, action_id=action_id, request_state=request_state.value)
        return action

    def cancellation_requested(self, task_id: str) -> bool:
        return self.store.get_task(task_id).state in {ExecutionState.CANCELLING, ExecutionState.CANCELLED}

    def cancel_attempt(self, task_id: str, *, attempt_id: str, reason: str) -> list[str]:
        task = self.store.get_task(task_id)
        if task.attempt_id != attempt_id:
            raise StaleLeaseError(
                f"attempt {attempt_id} is not the active attempt for task {task_id}"
            )
        return self.cancel(task_id, reason=reason, propagate=False)

    def cancel(self, task_id: str, *, reason: str, propagate: bool = True) -> list[str]:
        root = self.store.get_task(task_id)
        targets: list[TaskRecord] = []
        pending = [root]
        while pending:
            current = pending.pop()
            targets.append(current)
            if propagate:
                pending.extend(self.store.get_task(child) for child in current.child_task_ids)
        changed: list[str] = []
        blocked: list[str] = []
        for current in reversed(targets):
            if current.state in TERMINAL_STATES:
                continue
            if current.irreversible_side_effect_started:
                def block(task: TaskRecord) -> None:
                    task.cancellation_status = CancellationStatus.BLOCKED_BY_SIDE_EFFECT
                    task.cancellation_reason = reason
                    task.updated_at = self.clock()
                task, _ = self.store.update_task(current.task_id, block)
                self._emit("cancellation_blocked", task, reason=reason)
                blocked.append(current.task_id)
                continue
            if current.state != ExecutionState.CANCELLING:
                self.transition(current.task_id, ExecutionState.CANCELLING, reason=reason)
            def finish(task: TaskRecord) -> None:
                validate_transition(task.state, ExecutionState.CANCELLED)
                task.state = ExecutionState.CANCELLED
                task.cancellation_status = CancellationStatus.COMPLETED
                task.cancellation_reason = reason
                task.finished_at = task.updated_at = self.clock()
                task.lease_owner = ""
                task.lease_token = ""
                task.lease_expires_at = None
                task.retry_not_before = None
                task.waiting_inbox_item_id = ""
                task.waiting_reason = ""
                task.human_wait_started_at = None
                if not task.result_id:
                    term_result = EscrowResult(
                        task_id=task.task_id,
                        execution_id=task.task_id,
                        root_task_id=task.root_task_id or task.task_id,
                        parent_task_id=task.parent_task_id,
                        trigger_turn_id=task.trigger_turn_id,
                        session_id=task.session_id,
                        lane_id=(
                            task.assigned_agent.removeprefix("lane:")
                            if task.assigned_agent.startswith("lane:")
                            else task.assigned_agent
                        ),
                        owning_lane=(
                            task.assigned_agent.removeprefix("lane:")
                            if task.assigned_agent.startswith("lane:")
                            else task.assigned_agent
                        ),
                        attempt_id=task.attempt_id,
                        attempt_generation=task.attempt_generation,
                        lease_token_hash=task.lease_token,
                        status=EscrowStatus.AVAILABLE,
                        supervisor_state=ExecutionState.CANCELLED.value,
                        verification_status=VerificationStatus.NOT_SUPPORTED,
                        result_kind="terminal_failure",
                        payload={
                            "status": "cancelled",
                            "reason": reason or "cancelled",
                            "recovery_reason": task.recovery_reason,
                            "is_resumable": False,
                            "chat_result": {
                                "answer": reason or "Execution was cancelled.",
                                "error": reason or "cancelled",
                                "mode": "lane-cancelled",
                                "payload": {
                                    "execution_id": task.task_id,
                                    "lane_task_id": task.task_id,
                                    "status": "cancelled",
                                    "terminal_failure": True,
                                    "is_resumable": False,
                                },
                            },
                        },
                        error_metadata={
                            "state": "cancelled",
                            "reason": reason or "cancelled",
                            "recovery_reason": task.recovery_reason,
                            "is_resumable": False,
                        },
                        created_at=self.clock(),
                        completed_at=self.clock(),
                    )
                    task.result_id = term_result.result_id
                    self.store.save_result(term_result)
            task, _ = self.store.update_task(current.task_id, finish)
            if task.attempt_id:
                attempt = self.store.get_attempt(task.attempt_id)
                if attempt is not None:
                    attempt.state = "cancelled"
                    attempt.finished_at = task.finished_at
                    attempt.failure_reason = reason
                    self.store.save_attempt(attempt)
            self._emit("task_cancelled", task, reason=reason)
            changed.append(current.task_id)
        if blocked and root.task_id in changed:
            def mark_partial(task: TaskRecord) -> None:
                task.cancellation_status = CancellationStatus.PARTIALLY_COMPLETED
                task.updated_at = self.clock()
            partial, _ = self.store.update_task(root.task_id, mark_partial)
            self._emit(
                "cancellation_partially_completed",
                partial,
                reason=reason,
                blocked_descendants=blocked,
            )
        return changed

    def retry(self, task_id: str, decision: RecoveryDecision) -> TaskRecord:
        task = self._ensure_recovery_metadata(task_id)
        self.retry_policy.validate(
            task,
            decision,
            actions=self.store.actions_for_task(task_id),
            now=self.clock(),
        )
        checkpoint = None
        if decision.action == RecoveryAction.RESUME_CHECKPOINT:
            eligibility = self.validate_checkpoint_resume(
                task,
                decision.resume_checkpoint_id or task.checkpoint_id,
                allow_explicit_retry_seed=True,
            )
            if not eligibility.resumable or eligibility.checkpoint is None:
                raise RetrySafetyError(
                    f"selected checkpoint is missing or corrupt ({eligibility.reason}): {eligibility.error_message}"
                )
            checkpoint = eligibility.checkpoint
        backoff = self.retry_policy.backoff_seconds(task, decision.retry_category)
        target_state = (
            ExecutionState.REPLANNING
            if decision.action == RecoveryAction.REPLAN
            else ExecutionState.RETRY_SCHEDULED
        )
        def schedule(current: TaskRecord) -> None:
            validate_transition(current.state, target_state)
            current.state = target_state
            current.retry_count += 1
            current.retry_usage[decision.retry_category.value] = int(
                current.retry_usage.get(decision.retry_category.value, 0)
            ) + 1
            current.recovery_reason = decision.reason
            current.retry_not_before = (
                self.clock()
                if target_state == ExecutionState.REPLANNING
                else self.clock() + timedelta(seconds=backoff)
            )
            current.lease_owner = ""
            current.lease_token = ""
            current.lease_expires_at = None
            current.assigned_agent = decision.selected_agent or current.assigned_agent
            current.assigned_model = decision.selected_model or current.assigned_model
            current.assigned_worker = decision.selected_worker
            current.updated_at = self.clock()
        task, _ = self.store.update_task(task_id, schedule)
        self._emit(
            "replan_started" if target_state == ExecutionState.REPLANNING else "retry_scheduled",
            task,
            decision_id=decision.decision_id,
            category=decision.retry_category.value,
            backoff_seconds=backoff,
            retry_not_before=task.retry_not_before,
            checkpoint_id=checkpoint.checkpoint_id if checkpoint else "",
        )
        if decision.action == RecoveryAction.REASSIGN or any(
            (decision.selected_agent, decision.selected_worker, decision.selected_model)
        ):
            self._emit(
                "worker_reassigned",
                task,
                decision_id=decision.decision_id,
                selected_agent=decision.selected_agent,
                selected_worker=decision.selected_worker,
                selected_model=decision.selected_model,
            )
        return task

    def _ensure_recovery_metadata(self, task_id: str) -> TaskRecord:
        """Backfill only safe, generic evidence for legacy stopped tasks.

        A recovery decision can reuse an old task that predates the creation-time
        contract.  The method never invents outputs or receipts; it merely gives
        that task the same minimal completion boundary and explicit provenance
        used for new supervised work.
        """
        def enrich(task: TaskRecord) -> None:
            if not task.completion_contract:
                task.completion_contract = [
                    CompletionContract(
                        contract_type=CompletionContractType.STRUCTURED_RESULT_VALID,
                        metadata={"required_keys": []},
                    )
                ]
                task.field_provenance["completion_contract"] = "recovery_backfill"
            task.field_provenance.setdefault(
                "actual_cost", "pending_runtime_accounting"
            )
            task.field_provenance.setdefault(
                "completion_artefacts", "pending_completion_verification"
            )
            task.updated_at = self.clock()

        task, _ = self.store.update_task(task_id, enrich)
        return task

    def validate_checkpoint_resume(
        self,
        task: TaskRecord | str,
        checkpoint: CheckpointRecord | str | None = None,
        *,
        workspace_id: str = "",
        repository_id: str = "",
        allow_explicit_retry_seed: bool = False,
    ) -> CheckpointResumeEligibility:
        """Validate whether a checkpoint is eligible for execution continuation.

        Recovery precedence:
            terminal durable result > terminal task state > resumable checkpoint > generic recovery
        """
        task_rec: TaskRecord | None
        if isinstance(task, str):
            task_rec = self.store.get_task_or_none(task)
        else:
            task_rec = task

        if task_rec is None:
            return CheckpointResumeEligibility(
                resumable=False,
                reason="task_not_found",
                error_code="TASK_NOT_FOUND",
                error_message="Task record not found",
            )

        task_id = task_rec.task_id
        is_term = task_rec.state in TERMINAL_STATES or task_rec.state == ExecutionState.CANCELLING

        # Terminal state check for implicit resume
        if not allow_explicit_retry_seed:
            if is_term:
                return CheckpointResumeEligibility(
                    resumable=False,
                    reason="terminal_execution",
                    error_code="TERMINAL_EXECUTION",
                    error_message=f"Task {task_id} is in terminal state {task_rec.state.value}",
                    task_id=task_id,
                    checkpoint_id=task_rec.checkpoint_id,
                    state=task_rec.state.value,
                    is_terminal=True,
                )

            # Check if a durable terminal result exists in escrow
            escrow_res = (
                self.store.get_result(task_rec.result_id)
                if task_rec.result_id
                else self.store.get_result_by_execution_id(task_id)
            )
            if escrow_res is not None and (
                escrow_res.supervisor_state in {s.value for s in TERMINAL_STATES}
                or escrow_res.status == EscrowStatus.ACKNOWLEDGED
                or is_term
            ):
                return CheckpointResumeEligibility(
                    resumable=False,
                    reason="terminal_result_exists",
                    error_code="TERMINAL_RESULT_EXISTS",
                    error_message=f"Durable terminal result exists for execution {task_id}",
                    task_id=task_id,
                    checkpoint_id=task_rec.checkpoint_id,
                    state=task_rec.state.value,
                    is_terminal=True,
                )

        # Human inbox wait check
        if bool(getattr(task_rec, "waiting_inbox_item_id", "") or "") and task_rec.state is ExecutionState.WAITING:
            return CheckpointResumeEligibility(
                resumable=False,
                reason="human_wait_required",
                error_code="HUMAN_WAIT_REQUIRED",
                error_message="Human inbox waits resume only through the durable inbox claim path",
                task_id=task_id,
                checkpoint_id=task_rec.checkpoint_id,
                state=task_rec.state.value,
                is_terminal=is_term,
            )

        # Deadline check
        if task_rec.wall_clock_deadline_exceeded(self.clock()):
            return CheckpointResumeEligibility(
                resumable=False,
                reason="deadline_exceeded",
                error_code="DEADLINE_EXCEEDED",
                error_message=f"Task {task_id} wall-clock deadline has exceeded",
                task_id=task_id,
                checkpoint_id=task_rec.checkpoint_id,
                state=task_rec.state.value,
                is_terminal=is_term,
            )

        # Workspace / repository match
        if workspace_id and task_rec.workspace_id and task_rec.workspace_id != workspace_id:
            return CheckpointResumeEligibility(
                resumable=False,
                reason="workspace_mismatch",
                error_code="WORKSPACE_MISMATCH",
                error_message=f"Task workspace {task_rec.workspace_id} does not match {workspace_id}",
                task_id=task_id,
                checkpoint_id=task_rec.checkpoint_id,
                state=task_rec.state.value,
                is_terminal=is_term,
            )
        if repository_id and task_rec.repository_id and task_rec.repository_id != repository_id:
            return CheckpointResumeEligibility(
                resumable=False,
                reason="repository_mismatch",
                error_code="REPOSITORY_MISMATCH",
                error_message=f"Task repository {task_rec.repository_id} does not match {repository_id}",
                task_id=task_id,
                checkpoint_id=task_rec.checkpoint_id,
                state=task_rec.state.value,
                is_terminal=is_term,
            )

        # Checkpoint lookup
        target_cp_id = ""
        if isinstance(checkpoint, str):
            target_cp_id = checkpoint
        elif checkpoint is not None:
            target_cp_id = checkpoint.checkpoint_id
        else:
            target_cp_id = task_rec.checkpoint_id

        if not target_cp_id:
            return CheckpointResumeEligibility(
                resumable=False,
                reason="missing_checkpoint",
                error_code="CHECKPOINT_NOT_FOUND",
                error_message="Task has no checkpoint ID",
                task_id=task_id,
                checkpoint_id="",
                state=task_rec.state.value,
                is_terminal=is_term,
            )

        cp_rec: CheckpointRecord | None = None
        if isinstance(checkpoint, CheckpointRecord):
            cp_rec = checkpoint
        else:
            try:
                cp_root = getattr(self.store, "root", None)
                if cp_root is not None:
                    cp_file = Path(cp_root) / "checkpoints" / f"{target_cp_id}.json"
                    if cp_file.is_file():
                        try:
                            raw_data = cp_file.read_text(encoding="utf-8")
                            cp_rec = CheckpointRecord.model_validate_json(raw_data)
                        except Exception as exc:
                            return CheckpointResumeEligibility(
                                resumable=False,
                                reason="checkpoint_corrupt",
                                error_code="CHECKPOINT_CORRUPT",
                                error_message=f"Checkpoint {target_cp_id} is corrupt or unreadable: {exc}",
                                task_id=task_id,
                                checkpoint_id=target_cp_id,
                                state=task_rec.state.value,
                                is_terminal=is_term,
                            )
                if cp_rec is None:
                    cp_rec = self.store.get_checkpoint(target_cp_id)
            except (ExecutionSupervisorError, ValueError, OSError, Exception) as exc:
                return CheckpointResumeEligibility(
                    resumable=False,
                    reason="checkpoint_corrupt",
                    error_code="CHECKPOINT_CORRUPT",
                    error_message=f"Checkpoint {target_cp_id} is corrupt or unreadable: {exc}",
                    task_id=task_id,
                    checkpoint_id=target_cp_id,
                    state=task_rec.state.value,
                    is_terminal=is_term,
                )

        if cp_rec is None:
            return CheckpointResumeEligibility(
                resumable=False,
                reason="checkpoint_not_found",
                error_code="CHECKPOINT_NOT_FOUND",
                error_message=f"Checkpoint record {target_cp_id} not found in store",
                task_id=task_id,
                checkpoint_id=target_cp_id,
                state=task_rec.state.value,
                is_terminal=is_term,
            )

        if cp_rec.task_id != task_id:
            return CheckpointResumeEligibility(
                resumable=False,
                reason="task_id_mismatch",
                error_code="CHECKPOINT_TASK_MISMATCH",
                error_message=f"Checkpoint task {cp_rec.task_id} does not match {task_id}",
                task_id=task_id,
                checkpoint_id=target_cp_id,
                state=task_rec.state.value,
                is_terminal=is_term,
            )

        if task_rec.attempt_id and cp_rec.attempt_id != task_rec.attempt_id and not allow_explicit_retry_seed:
            return CheckpointResumeEligibility(
                resumable=False,
                reason="attempt_superseded",
                error_code="ATTEMPT_SUPERSEDED",
                error_message=f"Checkpoint attempt {cp_rec.attempt_id} superseded by {task_rec.attempt_id}",
                task_id=task_id,
                checkpoint_id=target_cp_id,
                state=task_rec.state.value,
                is_terminal=is_term,
            )

        # Boundary check: before_verification requires a candidate result or valid verification subject
        boundary = str(cp_rec.resume_cursor or cp_rec.resume_payload.get("boundary") or "")
        if boundary == "before_verification":
            has_files = bool(cp_rec.generated_files)
            has_artifacts = bool(cp_rec.artifact_references or cp_rec.result_escrow_references)
            has_changed = bool(cp_rec.resume_payload.get("changed_files"))
            has_intermediate = bool(cp_rec.resume_payload.get("intermediate_results"))
            has_task_artefacts = bool(task_rec.completion_artefacts)
            has_mode = bool(cp_rec.resume_payload.get("mode"))
            has_payload_error = bool(cp_rec.resume_payload.get("error"))

            if has_payload_error or not (has_files or has_artifacts or has_changed or has_intermediate or has_task_artefacts or (has_mode and not has_payload_error and cp_rec.resume_payload.get("mode") not in {"route-media-error", "lane-verification-failed"})):
                return CheckpointResumeEligibility(
                    resumable=False,
                    reason="missing_execution_result",
                    error_code="INVALID_VERIFICATION_BOUNDARY",
                    error_message="Verification-boundary checkpoint has no candidate result or valid subject to verify",
                    task_id=task_id,
                    checkpoint_id=target_cp_id,
                    boundary=boundary,
                    state=task_rec.state.value,
                    is_terminal=is_term,
                )

        return CheckpointResumeEligibility(
            resumable=True,
            reason="resumable",
            task_id=task_id,
            checkpoint_id=target_cp_id,
            boundary=boundary,
            state=task_rec.state.value,
            is_terminal=False,
            checkpoint=cp_rec,
        )

    def get_resumable_checkpoint(self, task_id: str) -> CheckpointRecord | None:
        """Return the checkpoint record only if it is proven eligible and safe for resume."""
        eligibility = self.validate_checkpoint_resume(task_id, allow_explicit_retry_seed=False)
        return eligibility.checkpoint if eligibility.resumable else None

    def resume_checkpoint(self, task_id: str) -> CheckpointRecord:
        eligibility = self.validate_checkpoint_resume(task_id, allow_explicit_retry_seed=False)
        if not eligibility.resumable or eligibility.checkpoint is None:
            raise RetrySafetyError(
                f"checkpoint resume invalid ({eligibility.reason}): {eligibility.error_message}"
            )
        return eligibility.checkpoint

    def release_retry(self, task_id: str) -> TaskRecord:
        current = self.store.get_task(task_id)
        if current.state not in {ExecutionState.RETRY_SCHEDULED, ExecutionState.REPLANNING}:
            raise RetrySafetyError(
                f"task cannot resume from recovery state {current.state.value}"
            )
        if (
            current.state == ExecutionState.RETRY_SCHEDULED
            and current.retry_not_before is not None
            and self.clock() < current.retry_not_before
        ):
            raise RetrySafetyError(
                f"retry backoff is active until {current.retry_not_before.isoformat()}"
            )
        reason = (
            "validated replan completed"
            if current.state == ExecutionState.REPLANNING
            else "retry backoff elapsed"
        )
        task = self.transition(task_id, ExecutionState.QUEUED, recovery_reason=reason)
        self._emit("replan_completed" if current.state == ExecutionState.REPLANNING else "retry_started", task)
        return task

    def _reconcile_local_mutation(self, task: TaskRecord, action: ActionRecord) -> ReconciliationOutcome:
        """Inspect durable workspace and checkpoint evidence for a local repository mutation."""
        # Every accepted receipt must be attributable to this exact action and
        # attempt.  A path or a generic success flag is never sufficient.
        evidence = action.verification_state
        def matching_identity(candidate: dict[str, Any]) -> bool:
            return (
                candidate.get("action_id") == action.action_id
                and candidate.get("attempt_id") == action.attempt_id
                and candidate.get("action_fingerprint", candidate.get("patch_fingerprint"))
                == action.action_fingerprint
            )

        if evidence.get("partially_applied") is True and matching_identity(evidence):
            return ReconciliationOutcome.PARTIALLY_APPLIED

        patch_result = action.verification_state.get("patch_result") or action.verification_state.get("apply_result")
        if isinstance(patch_result, dict) and matching_identity(patch_result):
            if patch_result.get("success") is True or patch_result.get("applied") is True:
                return ReconciliationOutcome.ALREADY_APPLIED
            if patch_result.get("partially_applied") is True:
                return ReconciliationOutcome.PARTIALLY_APPLIED

        mutation_receipt = evidence.get("mutation_receipt")
        if isinstance(mutation_receipt, dict) and matching_identity(mutation_receipt):
            if mutation_receipt.get("completed") is True:
                return ReconciliationOutcome.ALREADY_APPLIED
            if mutation_receipt.get("started") is True:
                return ReconciliationOutcome.PARTIALLY_APPLIED

        # A pre-existing path is not proof that this attempt produced it.  The
        # action must carry a fingerprint (or trusted patch result) tied to it.
        expected = action.verification_state.get("artifact_hashes") or {}
        artifacts = action.verification_state.get("artifacts") or []
        if isinstance(artifacts, list):
            expected = {
                **expected,
                **{
                    str(item["path"]): str(item.get("sha256") or item.get("content_fingerprint"))
                    for item in artifacts
                    if isinstance(item, dict) and item.get("path")
                    and (item.get("sha256") or item.get("content_fingerprint"))
                },
            }
        if isinstance(expected, dict) and expected:
            workspace = Path(task.workspace_path) if task.workspace_path else None
            observed = 0
            matched = 0
            if workspace is not None:
                for relative, fingerprint in expected.items():
                    path = workspace / str(relative)
                    if not path.is_file():
                        continue
                    observed += 1
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    matched += digest == str(fingerprint).removeprefix("sha256:")
            if matched == len(expected) and matched:
                return ReconciliationOutcome.ALREADY_APPLIED
            if observed:
                return ReconciliationOutcome.PARTIALLY_APPLIED

        checkpoints = self.store.checkpoints_for_task(task.task_id)
        if checkpoints:
            latest = checkpoints[-1]
            for result in latest.tool_results:
                if not isinstance(result, dict) or result.get("action_id") != action.action_id:
                    continue
                metadata = result.get("metadata") or result
                if (
                    metadata.get("attempt_id") == action.attempt_id
                    and metadata.get("action_fingerprint") == action.action_fingerprint
                    and (metadata.get("patch_applied") or metadata.get("applied") or metadata.get("success"))
                ):
                    return ReconciliationOutcome.ALREADY_APPLIED
                if (
                    metadata.get("attempt_id") == action.attempt_id
                    and metadata.get("action_fingerprint", metadata.get("patch_fingerprint")) == action.action_fingerprint
                    and metadata.get("partially_applied")
                ):
                    return ReconciliationOutcome.PARTIALLY_APPLIED

        if action.request_state == ActionRequestState.PREPARED or action.verification_state.get("execution_started") is False:
            return ReconciliationOutcome.NOT_STARTED
        return ReconciliationOutcome.UNKNOWN

    def classify_lost_lease(
        self,
        task: TaskRecord,
        *,
        now: datetime,
        actions: Iterable[ActionRecord] = (),
    ) -> tuple[LostLeaseOutcome, dict[str, Any]]:
        """Classify lost lease into explicit typed outcome based on durable evidence."""
        human_wait_in_tree = any(
            candidate.root_task_id == task.root_task_id
            and candidate.state is ExecutionState.WAITING
            and bool(candidate.waiting_inbox_item_id)
            for candidate in self.store.list_tasks(incomplete_only=True)
        )
        if task.wall_clock_deadline_exceeded(now) and not human_wait_in_tree:
            return LostLeaseOutcome.DEADLINE_EXPIRED, {"reason": "task wall-clock deadline exceeded during recovery"}

        results = self.store.results_for_task(task.task_id)
        if results and any(
            r.status in {EscrowStatus.AVAILABLE, EscrowStatus.DELIVERED, EscrowStatus.ACKNOWLEDGED}
            or r.supervisor_state == ExecutionState.COMPLETED.value
            for r in results
        ):
            return LostLeaseOutcome.DURABLE_RESULT_AVAILABLE, {"result": results[-1]}

        if task.retry_budget.remaining(RetryCategory.LEASE_LOSS, task.retry_usage) <= 0:
            return LostLeaseOutcome.RETRY_BUDGET_EXHAUSTED, {"reason": "lease_loss retry budget is exhausted"}

        action_list = list(actions)
        local_reconciliation_required = False
        local_reconciliation_details: dict[str, Any] = {}
        has_unknown_external = False
        ambiguous_external_action = None

        for action in action_list:
            if action.request_state == ActionRequestState.SUCCEEDED and action.external_receipt:
                continue

            if action.request_state in {ActionRequestState.STARTED, ActionRequestState.OUTCOME_UNKNOWN}:
                if action.effect_scope == ActionEffectScope.UNKNOWN:
                    has_unknown_external = True
                    ambiguous_external_action = action
                    break
                provider_state = action.verification_state.get("provider_state") or task.provider_metadata.get("state")
                if provider_state == "SUCCEEDED" or action.verification_state.get("succeeded"):
                    action.request_state = ActionRequestState.SUCCEEDED
                    action.external_receipt = str(action.verification_state.get("receipt") or "provider_confirmed_success")
                    action.updated_at = now
                    self.store.save_action(action)
                    continue
                if provider_state == "FAILED" or action.verification_state.get("failed"):
                    action.request_state = ActionRequestState.FAILED
                    action.updated_at = now
                    self.store.save_action(action)
                    continue

                is_local = (
                    action.effect_scope == ActionEffectScope.LOCAL_REPOSITORY
                    or action.tool_name in LOCAL_REPOSITORY_TOOLS
                )
                if is_local:
                    recon = self._reconcile_local_mutation(task, action)
                    if recon == ReconciliationOutcome.ALREADY_APPLIED:
                        action.request_state = ActionRequestState.SUCCEEDED
                        action.external_receipt = "reconciled_from_local_workspace"
                        action.verification_state["reconciliation_receipt"] = {
                            "execution_id": action.execution_id,
                            "attempt_id": action.attempt_id,
                            "action_id": action.action_id,
                            "checkpoint_id": task.checkpoint_id,
                            "evidence": "artifact_fingerprint_or_trusted_patch_result",
                        }
                        action.updated_at = now
                        self.store.save_action(action)
                    elif recon in {ReconciliationOutcome.PARTIALLY_APPLIED, ReconciliationOutcome.UNKNOWN}:
                        action.request_state = ActionRequestState.OUTCOME_UNKNOWN
                        action.verification_state["reconciliation_required"] = True
                        action.updated_at = now
                        self.store.save_action(action)
                        local_reconciliation_required = True
                        local_reconciliation_details[action.action_id] = recon
                    elif recon == ReconciliationOutcome.NOT_STARTED:
                        action.request_state = ActionRequestState.PREPARED
                        action.updated_at = now
                        self.store.save_action(action)
                    continue

                if action.classification in {SideEffectClassification.READ_ONLY, SideEffectClassification.IDEMPOTENT}:
                    continue

                if not action.external_receipt:
                    has_unknown_external = True
                    ambiguous_external_action = action
                    break

        if has_unknown_external:
            return LostLeaseOutcome.UNKNOWN_EXTERNAL_OUTCOME, {"action": ambiguous_external_action}

        if local_reconciliation_required:
            return LostLeaseOutcome.LOCAL_RECONCILIATION_REQUIRED, {"reconciliation": local_reconciliation_details}

        has_external_receipt = any(
            a.request_state == ActionRequestState.SUCCEEDED
            and a.external_receipt
            and a.external_receipt != "reconciled_from_local_workspace"
            for a in action_list
        )
        if has_external_receipt:
            return LostLeaseOutcome.DURABLE_RESULT_AVAILABLE, {"actions": action_list}

        decision = self.retry_policy.automatic_recovery_decision(
            task,
            category=RetryCategory.LEASE_LOSS,
            reason="active lease expired during execution",
            actions=action_list,
            now=now,
        )
        if decision is not None:
            return LostLeaseOutcome.SAFE_AUTOMATIC_RECOVERY, {"decision": decision}

        return LostLeaseOutcome.POLICY_BLOCKED, {"reason": "automatic recovery policy blocked continuation"}

    def _require_ambiguous_lease_review(
        self,
        task: TaskRecord,
        *,
        now: datetime,
        actions: Iterable[ActionRecord] = (),
    ) -> RecoveryInterventionRecord:
        """Persist lost-lease evidence before terminalizing uncertain work."""
        action_list = list(actions)
        ambiguous_action = next(
            (
                a for a in action_list
                if a.request_state in {ActionRequestState.STARTED, ActionRequestState.OUTCOME_UNKNOWN}
                and a.classification not in {
                    SideEffectClassification.READ_ONLY,
                    SideEffectClassification.IDEMPOTENT,
                }
                and a.effect_scope in {ActionEffectScope.EXTERNAL_CONSEQUENTIAL, ActionEffectScope.UNKNOWN}
                and not a.external_receipt
            ),
            None,
        )
        existing = next(
            (
                item
                for item in self.store.recovery_interventions_for_task(task.task_id)
                if item.attempt_id == task.attempt_id
                and item.reason is RecoveryInterventionReason.AMBIGUOUS_LOST_LEASE
            ),
            None,
        )
        external_possible = (
            ambiguous_action is not None
            or task.irreversible_side_effect_started
            or task.side_effect_classification not in {
                SideEffectClassification.READ_ONLY,
                SideEffectClassification.IDEMPOTENT,
                SideEffectClassification.DEDUPLICATED,
            }
        )
        if existing and existing.inbox_item_id:
            intervention = existing
        else:
            intervention = existing or RecoveryInterventionRecord(
                task_id=task.task_id,
                execution_id=task.task_id,
                attempt_id=task.attempt_id,
                action_id=ambiguous_action.action_id if ambiguous_action else "",
                checkpoint_id=task.checkpoint_id,
                integration_stage=getattr(task, "integration_stage", ""),
                target_resources=[ambiguous_action.tool_name] if ambiguous_action else [],
                receipt_lookup_state="missing_receipt" if ambiguous_action else "",
                reason_details=(
                    "consequential action started with unknown outcome"
                    if ambiguous_action
                    else "uncertain side-effect outcome after lease expiry"
                ),
                inbox_item_id="",
                side_effect_classification=str(
                    task.side_effect_classification.value
                    if hasattr(task.side_effect_classification, "value")
                    else task.side_effect_classification
                ),
                reason=RecoveryInterventionReason.AMBIGUOUS_LOST_LEASE,
                last_lease_owner=task.lease_owner,
                lease_expiry=task.lease_expires_at,
                external_side_effects_possible=external_possible,
                created_at=now,
            )
            inbox_item_id = ""
            if self.recovery_review_publisher is not None:
                if hasattr(self.recovery_review_publisher, "_supervisor"):
                    self.recovery_review_publisher._supervisor = self
                try:
                    inbox_item_id = self.recovery_review_publisher.create_recovery_review(
                        intervention=intervention,
                        task=task,
                        action=ambiguous_action,
                    )
                except Exception as exc:
                    self._emit(
                        "recovery_review_publication_error",
                        task,
                        intervention_id=intervention.intervention_id,
                        error=str(exc),
                    )
                    raise RetrySafetyError(
                        "RECOVERY_REVIEW_PUBLISH_FAILED: durable Human Inbox creation failed; "
                        "recovery stopped safely"
                    ) from exc
            if not inbox_item_id:
                self._emit(
                    "recovery_review_publication_error",
                    task,
                    intervention_id=intervention.intervention_id,
                    error="Human Inbox returned no durable item reference",
                )
                raise RetrySafetyError(
                    "RECOVERY_REVIEW_PUBLISH_FAILED: Human Inbox returned no durable item reference; "
                    "no fallback inbox reference was created"
                )
            intervention.inbox_item_id = inbox_item_id
            if existing is None:
                self.store.save_recovery_intervention(intervention)

        if (
            task.recovery_intervention_id != intervention.intervention_id
            or task.waiting_inbox_item_id != intervention.inbox_item_id
        ):
            def link(current: TaskRecord) -> None:
                current.recovery_intervention_id = intervention.intervention_id
                current.waiting_inbox_item_id = intervention.inbox_item_id
                current.waiting_kind = "human_review"
                current.waiting_reason = "ambiguous_lost_lease"
                current.wake_up_source = "human_inbox"
                current.wake_up_reference = intervention.inbox_item_id
                current.resume_checkpoint_id = current.checkpoint_id
                current.resume_operation = "resolve_recovery_intervention"
                current.updated_at = now

            task, _ = self.store.update_task(task.task_id, link)

        if task.state is not ExecutionState.RECOVERY_REVIEW_REQUIRED:
            task = self.transition(
                task.task_id,
                ExecutionState.RECOVERY_REVIEW_REQUIRED,
                reason=(
                    "ambiguous lost lease; external side effects may have occurred; "
                    "human review is required"
                ),
                recovery_reason="automatic recovery refused after ambiguous lost lease",
            )
        self._emit(
            "recovery_intervention_required",
            task,
            intervention_id=intervention.intervention_id,
            status=intervention.status,
            reason=intervention.reason.value,
            action=intervention.action,
            execution_id=intervention.execution_id,
            execution_state=intervention.execution_state,
            last_lease_owner=intervention.last_lease_owner,
            lease_expiry=intervention.lease_expiry,
            terminal_state=intervention.terminal_state.value,
            external_side_effects_possible=intervention.external_side_effects_possible,
        )
        return intervention

    def resolve_recovery_intervention(
        self,
        intervention_id: str,
        *,
        action: HumanRecoveryDecisionAction | str,
        actor_id: str = "operator",
        comment: str = "",
        response_data: dict[str, Any] | None = None,
    ) -> TaskRecord:
        """Resolve a durable recovery intervention and resume the original execution lineage."""
        intervention = self.store.get_recovery_intervention(intervention_id)
        if intervention is None:
            raise RetrySafetyError(f"recovery intervention not found: {intervention_id}")
        task = self.store.get_task(intervention.task_id)
        if intervention.execution_id != task.task_id:
            raise RetrySafetyError("recovery intervention execution lineage does not match its task")
        if intervention.attempt_id and intervention.attempt_id != task.attempt_id:
            raise RetrySafetyError("recovery intervention belongs to a different attempt")
        if (
            task.state is not ExecutionState.RECOVERY_REVIEW_REQUIRED
            and task.state is not ExecutionState.WAITING
        ):
            raise RetrySafetyError(
                f"task is not awaiting recovery review (current state: {task.state.value})"
            )

        act = action.value if isinstance(action, HumanRecoveryDecisionAction) else str(action)
        now = self.clock()

        if act in {HumanRecoveryDecisionAction.ABORT_EXECUTION.value, "ABORT_EXECUTION"}:
            task = self.transition(
                task.task_id,
                ExecutionState.CANCELLED,
                reason=comment or f"aborted by human reviewer {actor_id}",
                recovery_reason="human review resolution: abort",
            )
            self._emit("recovery_intervention_resolved", task, intervention_id=intervention_id, action=act)
            return task

        if act in {
            HumanRecoveryDecisionAction.MARK_ACTION_ALREADY_COMPLETED.value,
            "MARK_ACTION_ALREADY_COMPLETED",
        }:
            if intervention.action_id:
                action_rec = self.store.get_action(intervention.action_id)
                if action_rec is not None:
                    if action_rec.execution_id != intervention.execution_id or (
                        intervention.attempt_id and action_rec.attempt_id != intervention.attempt_id
                    ):
                        raise RetrySafetyError("recovery action does not belong to the original execution lineage")
                    confirmation = str(
                        (response_data or {}).get("receipt_reference")
                        or (response_data or {}).get("answer", {}).get("receipt_reference")
                        or comment
                    )
                    if not confirmation:
                        raise RetrySafetyError("human completion confirmation must include a receipt or reference")
                    action_rec.request_state = ActionRequestState.SUCCEEDED
                    action_rec.external_receipt = confirmation
                    action_rec.verification_state.update({
                        "human_confirmation": True,
                        "confirmed_by": actor_id,
                        "confirmation_reference": confirmation,
                    })
                    action_rec.updated_at = now
                    self.store.save_action(action_rec)

        elif act in {HumanRecoveryDecisionAction.RETRY_ACTION.value, "RETRY_ACTION"}:
            if intervention.action_id:
                action_rec = self.store.get_action(intervention.action_id)
                if action_rec is not None:
                    action_rec.verification_state["retry_authorized_by"] = actor_id
                    action_rec.verification_state["prior_state"] = action_rec.request_state.value
                    action_rec.request_state = ActionRequestState.PREPARED
                    action_rec.updated_at = now
                    self.store.save_action(action_rec)

        elif act in {
            HumanRecoveryDecisionAction.RESUME_WITHOUT_REPLAY.value,
            "RESUME_WITHOUT_REPLAY",
        }:
            pass
        else:
            raise RetrySafetyError(f"unsupported human recovery action: {act}")

        def clear_wait(current: TaskRecord) -> None:
            validate_transition(current.state, ExecutionState.QUEUED)
            current.state = ExecutionState.QUEUED
            current.waiting_kind = ""
            current.waiting_reason = ""
            current.wake_up_source = ""
            current.wake_up_reference = ""
            current.waiting_inbox_item_id = ""
            current.resume_checkpoint_id = intervention.checkpoint_id or current.checkpoint_id
            current.resume_operation = intervention.integration_stage or "resume_after_recovery_intervention"
            current.lease_owner = ""
            current.lease_token = ""
            current.lease_expires_at = None
            current.recovery_reason = f"human review resolved: {act}"
            current.updated_at = now

        task, _ = self.store.update_task(task.task_id, clear_wait)
        self._emit("recovery_intervention_resolved", task, intervention_id=intervention_id, action=act)
        return task

    def recover(self) -> RecoverySummary:
        """Recover expired work deterministically; safe to invoke repeatedly."""
        summary = RecoverySummary()
        now = self.clock()
        incomplete_tasks = self.store.list_tasks(incomplete_only=True)
        review_required_tasks = [
            task
            for task in self.store.list_tasks()
            if task.state is ExecutionState.RECOVERY_REVIEW_REQUIRED
        ]
        for initial in [*incomplete_tasks, *review_required_tasks]:
            summary.scanned += 1
            task = self.store.get_task(initial.task_id)
            if task.state is ExecutionState.RECOVERY_REVIEW_REQUIRED:
                intervention = self.store.get_recovery_intervention(
                    task.recovery_intervention_id
                )
                summary.intervention_required.append(task.task_id)
                if intervention is not None and intervention not in summary.intervention_records:
                    summary.intervention_records.append(intervention)
                continue
            checkpoints = self.store.checkpoints_for_task(task.task_id)
            if checkpoints:
                latest = checkpoints[-1]
                if latest.attempt_id == task.attempt_id and latest.checkpoint_id != task.checkpoint_id:
                    def relink_checkpoint(current: TaskRecord) -> None:
                        current.checkpoint_id = latest.checkpoint_id
                        current.checkpoint_count = max(current.checkpoint_count, len(checkpoints))
                        current.updated_at = now
                    task, _ = self.store.update_task(task.task_id, relink_checkpoint)
                    self._emit(
                        "checkpoint_recovered",
                        task,
                        checkpoint_id=latest.checkpoint_id,
                        action="relinked_after_interrupted_atomic_write",
                    )
            escrow_result = (
                self.store.get_result(task.result_id)
                if task.result_id
                else self.store.get_result_by_execution_id(task.task_id)
            )
            if escrow_result is None:
                task_results = self.store.results_for_task(task.task_id)
                if task_results:
                    escrow_result = task_results[-1]

            if escrow_result is not None:
                if (
                    escrow_result.supervisor_state == ExecutionState.COMPLETED.value
                    or escrow_result.verification_status == VerificationStatus.PASSED
                ):
                    def repair_completed(current: TaskRecord) -> None:
                        current.result_id = escrow_result.result_id
                        current.state = ExecutionState.COMPLETED
                        current.verification_status = VerificationStatus.PASSED
                        if escrow_result.artifacts:
                            current.completion_artefacts = list(escrow_result.artifacts)
                        current.finished_at = escrow_result.completed_at or current.finished_at or now
                        current.lease_owner = ""
                        current.lease_token = ""
                        current.lease_expires_at = None
                        current.retry_not_before = None
                        if escrow_result.provider_metadata:
                            current.provider_metadata = {**current.provider_metadata, **escrow_result.provider_metadata}
                        current.updated_at = now
                    task, _ = self.store.update_task(task.task_id, repair_completed)
                    if task.attempt_id:
                        attempt = self.store.get_attempt(task.attempt_id)
                        if attempt:
                            attempt.state = "completed"
                            attempt.finished_at = task.finished_at
                            self.store.save_attempt(attempt)
                    summary.recovered.append(task.task_id)
                    self._emit("task_recovered", task, action="repaired_from_authoritative_escrow")
                    continue
                elif escrow_result.supervisor_state in {s.value for s in TERMINAL_STATES}:
                    target_term = ExecutionState(escrow_result.supervisor_state)
                    def repair_term(current: TaskRecord) -> None:
                        current.result_id = escrow_result.result_id
                        current.state = target_term
                        current.verification_status = escrow_result.verification_status
                        current.finished_at = escrow_result.completed_at or current.finished_at or now
                        current.lease_owner = ""
                        current.lease_token = ""
                        current.lease_expires_at = None
                        current.retry_not_before = None
                        if escrow_result.error_metadata:
                            if escrow_result.error_metadata.get("reason"):
                                current.failure_reason = str(escrow_result.error_metadata["reason"])
                            if escrow_result.error_metadata.get("recovery_reason"):
                                current.recovery_reason = str(escrow_result.error_metadata["recovery_reason"])
                        if escrow_result.provider_metadata:
                            current.provider_metadata = {**current.provider_metadata, **escrow_result.provider_metadata}
                        current.updated_at = now
                    task, _ = self.store.update_task(task.task_id, repair_term)
                    if task.attempt_id:
                        attempt = self.store.get_attempt(task.attempt_id)
                        if attempt:
                            attempt.state = target_term.value
                            attempt.finished_at = task.finished_at
                            self.store.save_attempt(attempt)
                    summary.recovered.append(task.task_id)
                    self._emit("task_recovered", task, action="repaired_from_authoritative_escrow")
                    continue
                elif (
                    escrow_result.status in {EscrowStatus.STORED, EscrowStatus.AVAILABLE, EscrowStatus.DELIVERED, EscrowStatus.ACKNOWLEDGED}
                    and (not task.result_id or task.state in {ExecutionState.RUNNING, ExecutionState.LEASED, ExecutionState.COMPLETED_PENDING_VERIFICATION})
                ):
                    def relink_result(current: TaskRecord) -> None:
                        current.result_id = escrow_result.result_id
                        current.state = ExecutionState.COMPLETED_PENDING_VERIFICATION
                        current.updated_at = now
                    task, _ = self.store.update_task(task.task_id, relink_result)
                    try:
                        recovered = self.verify_completion(task.task_id)
                        summary.recovered.append(task.task_id)
                        self._emit("task_recovered", recovered, action="completion_reverified")
                    except Exception:
                        summary.intervention_required.append(task.task_id)
                    continue
            if task.state == ExecutionState.CANCELLING:
                self.cancel(task.task_id, reason=task.cancellation_reason or "recovered pending cancellation")
                summary.recovered.append(task.task_id)
                continue
            if task.state == ExecutionState.COMPLETED_PENDING_VERIFICATION:
                try:
                    recovered = self.verify_completion(task.task_id)
                except CompletionVerificationError:
                    summary.intervention_required.append(task.task_id)
                else:
                    (summary.recovered if recovered.state == ExecutionState.COMPLETED else summary.intervention_required).append(task.task_id)
                    self._emit("task_recovered", recovered, action="completion_reverified")
                continue
            if task.state == ExecutionState.WAITING and task.waiting_inbox_item_id:
                summary.unchanged.append(task.task_id)
                continue
            human_wait_in_tree = any(
                candidate.root_task_id == task.root_task_id
                and candidate.state is ExecutionState.WAITING
                and bool(candidate.waiting_inbox_item_id)
                for candidate in self.store.list_tasks(incomplete_only=True)
            )
            if task.deadline_at is not None and task.deadline_at <= now and not human_wait_in_tree:
                if task.state not in TERMINAL_STATES:
                    task = self.transition(
                        task.task_id,
                        ExecutionState.FAILED,
                        reason="task wall-clock deadline exceeded during recovery",
                    )
                summary.intervention_required.append(task.task_id)
                continue
            active = task.state in {
                ExecutionState.LEASED,
                ExecutionState.RUNNING,
                ExecutionState.CHECKPOINTING,
                ExecutionState.WAITING,
            }
            if not active or task.lease_expires_at is None or task.lease_expires_at > now:
                summary.unchanged.append(task.task_id)
                continue
            attempt = self.store.get_attempt(task.attempt_id)
            if attempt:
                attempt.state = "lost"
                attempt.finished_at = now
                attempt.failure_reason = "lease expired before terminal result publication"
                attempt.recovery_reason = "startup recovery"
                self.store.save_attempt(attempt)
            self._emit("lease_expired", task, lease_owner=task.lease_owner)
            actions = self.store.actions_for_task(task.task_id)
            attempt_actions = (
                [a for a in actions if a.attempt_id == task.attempt_id]
                if task.attempt_id
                else actions
            )

            outcome, details = self.classify_lost_lease(
                task,
                now=now,
                actions=attempt_actions,
            )

            if outcome == LostLeaseOutcome.DEADLINE_EXPIRED:
                if task.state not in TERMINAL_STATES:
                    task = self.transition(
                        task.task_id,
                        ExecutionState.FAILED,
                        reason="task wall-clock deadline exceeded during recovery",
                    )
                summary.intervention_required.append(task.task_id)
                continue

            elif outcome == LostLeaseOutcome.RETRY_BUDGET_EXHAUSTED:
                if task.state not in TERMINAL_STATES:
                    def fail_budget(current: TaskRecord) -> None:
                        validate_transition(current.state, ExecutionState.BUDGET_EXHAUSTED)
                        current.state = ExecutionState.BUDGET_EXHAUSTED
                        current.finished_at = current.updated_at = now
                        current.failure_reason = "lease_loss retry budget is exhausted"
                        current.lease_owner = ""
                        current.lease_token = ""
                        current.lease_expires_at = None
                    task, _ = self.store.update_task(task.task_id, fail_budget)
                    self._emit("task_failed", task, reason=task.failure_reason)
                summary.intervention_required.append(task.task_id)
                continue

            elif outcome == LostLeaseOutcome.DURABLE_RESULT_AVAILABLE:
                if "result" in details:
                    result = details["result"]
                    if task.result_id != result.result_id:
                        def set_result(current: TaskRecord) -> None:
                            current.result_id = result.result_id
                            current.updated_at = now
                        task, _ = self.store.update_task(task.task_id, set_result)
                    if (
                        result.supervisor_state == ExecutionState.COMPLETED.value
                        or result.verification_status == VerificationStatus.PASSED
                    ):
                        def finish_recovered(current: TaskRecord) -> None:
                            current.state = ExecutionState.COMPLETED
                            current.verification_status = VerificationStatus.PASSED
                            if result.artifacts:
                                current.completion_artefacts = list(result.artifacts)
                            current.finished_at = result.completed_at or current.finished_at or now
                            current.lease_owner = ""
                            current.lease_token = ""
                            current.lease_expires_at = None
                            current.updated_at = now
                        task, _ = self.store.update_task(task.task_id, finish_recovered)
                        summary.recovered.append(task.task_id)
                        self._emit("task_recovered", task, action="durable_result_consumed")
                    elif result.supervisor_state in {s.value for s in TERMINAL_STATES}:
                        term_target = ExecutionState(result.supervisor_state)
                        def term_recovered(current: TaskRecord) -> None:
                            current.state = term_target
                            current.verification_status = result.verification_status
                            current.finished_at = result.completed_at or current.finished_at or now
                            current.lease_owner = ""
                            current.lease_token = ""
                            current.lease_expires_at = None
                            current.updated_at = now
                        task, _ = self.store.update_task(task.task_id, term_recovered)
                        summary.recovered.append(task.task_id)
                        self._emit("task_recovered", task, action="durable_result_consumed")
                    else:
                        def prep_verify(current: TaskRecord) -> None:
                            current.state = ExecutionState.COMPLETED_PENDING_VERIFICATION
                            current.updated_at = now
                        task, _ = self.store.update_task(task.task_id, prep_verify)
                        try:
                            recovered = self.verify_completion(task.task_id)
                            summary.recovered.append(task.task_id)
                            self._emit("task_recovered", recovered, action="durable_result_consumed")
                        except Exception:
                            summary.intervention_required.append(task.task_id)
                    continue
                else:
                    receipt_actions = [
                        action for action in attempt_actions
                        if action.request_state == ActionRequestState.SUCCEEDED and action.external_receipt
                    ]
                    if not receipt_actions:
                        summary.intervention_required.append(task.task_id)
                        continue
                    for action in receipt_actions:
                        action.request_state = ActionRequestState.RECONCILED
                        action.verification_state.update({
                            "receipt_consumed": True,
                            "receipt_consumed_at": now.isoformat(),
                            "receipt_lineage": {
                                "execution_id": action.execution_id,
                                "attempt_id": action.attempt_id,
                                "action_id": action.action_id,
                                "checkpoint_id": task.checkpoint_id,
                            },
                        })
                        action.updated_at = now
                        self.store.save_action(action)

                    def resume_after_receipt(current: TaskRecord) -> None:
                        current.resume_checkpoint_id = str(
                            receipt_actions[-1].verification_state.get("resume_checkpoint_id")
                            or current.checkpoint_id
                        )
                        current.resume_operation = str(
                            receipt_actions[-1].verification_state.get("next_stage")
                            or receipt_actions[-1].verification_state.get("resume_operation")
                            or "resume_after_durable_action_receipt"
                        )
                        current.retry_not_before = now
                        current.lease_owner = ""
                        current.lease_token = ""
                        current.lease_expires_at = None
                        current.state = ExecutionState.QUEUED
                        current.updated_at = now

                    task, _ = self.store.update_task(task.task_id, resume_after_receipt)
                    summary.recovered.append(task.task_id)
                    self._emit("task_recovered", task, action="durable_action_receipt_consumed")
                    continue

            elif outcome == LostLeaseOutcome.LOCAL_RECONCILIATION_REQUIRED:
                def fail_reconciliation(current: TaskRecord) -> None:
                    validate_transition(current.state, ExecutionState.FAILED)
                    current.state = ExecutionState.FAILED
                    current.finished_at = current.updated_at = now
                    current.failure_reason = "local mutation outcome requires manual reconciliation; no replay was scheduled"
                    current.recovery_reason = "lost lease local mutation evidence was inconclusive"
                    current.lease_owner = ""
                    current.lease_token = ""
                    current.lease_expires_at = None
                task, _ = self.store.update_task(task.task_id, fail_reconciliation)
                summary.intervention_required.append(task.task_id)
                self._emit("recovery_reconciliation_required", task)
                continue

            elif outcome == LostLeaseOutcome.SAFE_AUTOMATIC_RECOVERY:
                decision = details.get("decision") or self.retry_policy.automatic_recovery_decision(
                    task,
                    category=RetryCategory.LEASE_LOSS,
                    reason="automatic recovery after inspectable local work / safe lease loss",
                    actions=attempt_actions,
                    now=now,
                )
                if decision is None:
                    def fail_policy(current: TaskRecord) -> None:
                        validate_transition(current.state, ExecutionState.FAILED)
                        current.state = ExecutionState.FAILED
                        current.finished_at = current.updated_at = now
                        current.failure_reason = "automatic recovery policy blocked continuation"
                        current.lease_owner = ""
                        current.lease_token = ""
                        current.lease_expires_at = None
                    task, _ = self.store.update_task(task.task_id, fail_policy)
                    self._emit("task_failed", task, reason=task.failure_reason)
                    summary.intervention_required.append(task.task_id)
                    continue

                try:
                    task = self.retry(task.task_id, decision)
                except RetrySafetyError as exc:
                    def fail_invalid_recovery(current: TaskRecord) -> None:
                        validate_transition(current.state, ExecutionState.FAILED)
                        current.state = ExecutionState.FAILED
                        current.finished_at = current.updated_at = now
                        current.failure_reason = (
                            "automatic recovery decision failed validation; no fallback action "
                            f"was executed: {exc}"
                        )
                        current.recovery_reason = "manual recovery decision required"
                        current.lease_owner = ""
                        current.lease_token = ""
                        current.lease_expires_at = None
                    task, _ = self.store.update_task(task.task_id, fail_invalid_recovery)
                    self._emit("task_failed", task, reason=task.failure_reason)
                    summary.intervention_required.append(task.task_id)
                    continue
                self._emit("task_recovered", task, action=decision.action.value)
                summary.retry_scheduled.append(task.task_id)
                continue

            elif outcome == LostLeaseOutcome.POLICY_BLOCKED:
                def fail_policy(current: TaskRecord) -> None:
                    validate_transition(current.state, ExecutionState.FAILED)
                    current.state = ExecutionState.FAILED
                    current.finished_at = current.updated_at = now
                    current.failure_reason = details.get("reason", "automatic recovery policy blocked continuation")
                    current.lease_owner = ""
                    current.lease_token = ""
                    current.lease_expires_at = None
                task, _ = self.store.update_task(task.task_id, fail_policy)
                self._emit("task_failed", task, reason=task.failure_reason)
                summary.intervention_required.append(task.task_id)
                continue

            elif outcome == LostLeaseOutcome.UNKNOWN_EXTERNAL_OUTCOME:
                try:
                    intervention = self._require_ambiguous_lease_review(
                        task, now=now, actions=attempt_actions
                    )
                except RetrySafetyError:
                    summary.intervention_required.append(task.task_id)
                    continue
                summary.intervention_required.append(task.task_id)
                if intervention not in summary.intervention_records:
                    summary.intervention_records.append(intervention)
                continue
        return summary

    def parent_progress(self, task_id: str, *, now: datetime | None = None) -> ParentProgress:
        task = self.store.get_task(task_id)
        children = [self.store.get_task(child) for child in task.child_task_ids]
        completed = sum(child.state == ExecutionState.COMPLETED for child in children)
        failed = sum(
            child.state in {ExecutionState.FAILED, ExecutionState.RECOVERY_REVIEW_REQUIRED}
            for child in children
        )
        cancelled = sum(child.state == ExecutionState.CANCELLED for child in children)
        active_rows = [child for child in children if child.state not in TERMINAL_STATES]
        child_by_id = {child.task_id: child for child in children}
        blocking_dependencies = {
            child.task_id: [
                dependency_id
                for dependency_id in child.dependency_task_ids
                if dependency_id not in child_by_id
                or child_by_id[dependency_id].state != ExecutionState.COMPLETED
            ]
            for child in children
        }
        blocking_dependencies = {
            child_id: dependencies
            for child_id, dependencies in blocking_dependencies.items()
            if dependencies
        }
        timed_out = bool(task.deadline_at and (now or self.clock()) >= task.deadline_at)
        if task.wait_policy == WaitPolicy.FAIL_FAST:
            satisfied = not failed and not active_rows
        elif task.wait_policy == WaitPolicy.BEST_EFFORT:
            satisfied = not active_rows
        elif task.wait_policy == WaitPolicy.MINIMUM_SUCCESS_COUNT:
            satisfied = completed >= int(task.minimum_success_count or 1)
        elif task.wait_policy == WaitPolicy.DEPENDENCY_GRAPH:
            satisfied = not active_rows and not blocking_dependencies and not failed
        else:
            satisfied = bool(children) and completed == len(children)
        return ParentProgress(
            task_id=task_id,
            policy=task.wait_policy,
            total_children=len(children),
            completed=completed,
            failed=failed,
            cancelled=cancelled,
            active=len(active_rows),
            timed_out=timed_out,
            satisfied=satisfied and not timed_out,
            blocking_task_ids=[child.task_id for child in active_rows],
            blocking_dependencies=blocking_dependencies,
        )

    def reconnect_tree(self) -> int:
        """Repair missing parent child links without discarding existing records."""
        repaired = 0
        for task in self.store.list_tasks():
            if not task.parent_task_id:
                continue
            parent = self.store.get_task_or_none(task.parent_task_id)
            if parent is None or task.task_id in parent.child_task_ids:
                continue
            def link(current: TaskRecord) -> None:
                current.child_task_ids.append(task.task_id)
                current.updated_at = self.clock()
            self.store.update_task(parent.task_id, link)
            repaired += 1
        return repaired
