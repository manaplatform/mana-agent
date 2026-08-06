"""Persistent resource coordinator for gateway-owned specialist lanes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from mana_agent.compat import process_exists
from mana_agent.gateway.lanes import (
    ACTIVE_LANE_STATES,
    LockMode,
    LaneContract,
    LaneId,
    LanePriority,
    LaneTaskState,
    PRIORITY_ORDER,
    configured_lane_contracts,
    select_lane,
    validate_tool_permission,
)
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.multi_agent.core.types import TaskStatus
from mana_agent.workspaces.paths import workspace_dir
from mana_agent.evals.recorder import record_current
from mana_agent.config.settings import Settings
from mana_agent.execution_supervisor import (
    CompletionContract,
    CompletionContractType,
    ExecutionState as SupervisorState,
    ExecutionSupervisor,
    ExecutionSupervisorConfig,
    SideEffectClassification,
    RecoveryDecision,
)
from mana_agent.execution_supervisor.errors import (
    BudgetExceededError,
    CompletionVerificationError,
    ExecutionSupervisorError,
)
from mana_agent.execution_supervisor.models import (
    BudgetOverrunFinalizationDecision,
    ExecutionState,
    RecoveryAction,
)

if os.name == "nt":  # pragma: no cover - exercised on Windows CI
    import msvcrt
else:  # pragma: no cover - platform branch
    import fcntl


def _lock_process_file(handle: Any) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_process_file(handle: Any) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _completion_verification_failure(manifest: Mapping[str, Any]) -> str:
    verification = manifest.get("verification")
    if not isinstance(verification, Mapping):
        return "Durable completion verification failed without a persisted verification report."
    checks = verification.get("checks")
    checks = checks if isinstance(checks, list) else []
    failures: list[str] = []
    for raw in checks:
        if not isinstance(raw, Mapping) or raw.get("passed") is True:
            continue
        contract = str(raw.get("contract_type") or raw.get("verifier_type") or "contract")
        reference = str(raw.get("artifact_reference") or raw.get("path") or "").strip()
        reason = str(raw.get("failure_reason") or "condition not satisfied").strip()
        label = f"{contract} ({reference})" if reference else contract
        failures.append(f"{label}: {reason}")
    if not failures:
        return "Durable completion verification failed; the persisted report contains no passing completion projection."
    return "Durable completion verification failed: " + "; ".join(failures[:5])


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.01 * (2**attempt))
    finally:
        temporary.unlink(missing_ok=True)


class LaneCoordinatorError(RuntimeError):
    code = "lane_coordinator_error"


class LaneCapacityError(LaneCoordinatorError):
    code = "lane_capacity_unavailable"


class LaneBudgetError(LaneCoordinatorError):
    code = "lane_budget_exhausted"


class LaneLockTimeout(LaneCoordinatorError):
    code = "lane_lock_timeout"


class LaneHandoffError(LaneCoordinatorError):
    code = "lane_handoff_invalid"


_CONTROL_TRANSITIONS: dict[LaneTaskState, frozenset[LaneTaskState]] = {
    LaneTaskState.CREATED: frozenset({LaneTaskState.ROUTING, LaneTaskState.REJECTED, LaneTaskState.FAILED}),
    LaneTaskState.ROUTING: frozenset({LaneTaskState.QUEUED, LaneTaskState.REJECTED, LaneTaskState.FAILED}),
    LaneTaskState.QUEUED: frozenset({LaneTaskState.RUNNING, LaneTaskState.PAUSED, LaneTaskState.CANCELLING, LaneTaskState.BLOCKED, LaneTaskState.REJECTED}),
    LaneTaskState.RUNNING: frozenset({LaneTaskState.WAITING, LaneTaskState.BLOCKED, LaneTaskState.CANCELLING, LaneTaskState.VERIFYING, LaneTaskState.PENDING_BUDGET_DECISION, LaneTaskState.COMPLETED, LaneTaskState.FAILED}),
    LaneTaskState.WAITING: frozenset({LaneTaskState.QUEUED, LaneTaskState.RUNNING, LaneTaskState.PAUSED, LaneTaskState.BLOCKED, LaneTaskState.CANCELLING}),
    LaneTaskState.BLOCKED: frozenset({LaneTaskState.QUEUED, LaneTaskState.CANCELLING, LaneTaskState.FAILED, LaneTaskState.REJECTED}),
    LaneTaskState.PAUSED: frozenset({LaneTaskState.QUEUED, LaneTaskState.CANCELLING}),
    LaneTaskState.CANCELLING: frozenset({LaneTaskState.CANCELLED, LaneTaskState.FAILED}),
    LaneTaskState.HANDOFF: frozenset({LaneTaskState.QUEUED, LaneTaskState.CANCELLING, LaneTaskState.FAILED}),
    LaneTaskState.VERIFYING: frozenset({LaneTaskState.SELECTING_WINNER, LaneTaskState.APPLYING, LaneTaskState.COMPLETED, LaneTaskState.REJECTED, LaneTaskState.FAILED, LaneTaskState.CANCELLING}),
    LaneTaskState.SELECTING_WINNER: frozenset({LaneTaskState.APPLYING, LaneTaskState.REJECTED, LaneTaskState.FAILED, LaneTaskState.CANCELLING}),
    LaneTaskState.APPLYING: frozenset({LaneTaskState.COMPLETED, LaneTaskState.FAILED, LaneTaskState.CANCELLING}),
}

_CONTROL_TERMINAL_STATES = frozenset({
    LaneTaskState.COMPLETED, LaneTaskState.FAILED, LaneTaskState.CANCELLED,
    LaneTaskState.REJECTED, LaneTaskState.TIMED_OUT, LaneTaskState.INTERRUPTED,
    LaneTaskState.BUDGET_EXHAUSTED,
})


@dataclass(slots=True)
class LaneBudget:
    reserved_input_tokens: int = 0
    reserved_output_tokens: int = 0
    consumed_input_tokens: int = 0
    consumed_output_tokens: int = 0
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    estimated_cost_known: bool = False
    actual_cost_known: bool = False
    model_context_window: int = 0
    model_max_output_tokens: int = 0
    estimate_confidence: str = ""
    estimate_source: str = ""
    revisions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def reserved_tokens(self) -> int:
        return self.reserved_input_tokens + self.reserved_output_tokens

    @property
    def consumed_tokens(self) -> int:
        return self.consumed_input_tokens + self.consumed_output_tokens


@dataclass(slots=True)
class LaneHandoff:
    source_lane: LaneId
    target_lane: LaneId
    task_id: str
    reason: str
    artifacts: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    remaining_work: list[str] = field(default_factory=list)
    verification_state: dict[str, Any] = field(default_factory=dict)
    budget_consumed: LaneBudget = field(default_factory=LaneBudget)
    created_at: str = field(default_factory=_iso)


@dataclass(slots=True)
class LaneExecution:
    task_id: str
    root_task_id: str
    parent_task_id: str | None
    owning_lane: LaneId
    state: LaneTaskState
    normalized_intent: str
    repository_id: str
    workspace_id: str
    session_id: str
    target_files: list[str]
    priority: LanePriority
    budget: LaneBudget
    taskboard_task_id: str = ""
    worker_id: str = ""
    model: str = ""
    provider: str = ""
    routing_decision_id: str = ""
    accounting_reservation_ids: list[str] = field(default_factory=list)
    task_type: str = "single"
    capabilities: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    verification_state: dict[str, Any] = field(default_factory=dict)
    lane_history: list[dict[str, Any]] = field(default_factory=list)
    handoffs: list[LaneHandoff] = field(default_factory=list)
    duplicate_of: str | None = None
    last_heartbeat: str = field(default_factory=_iso)
    created_at: str = field(default_factory=_iso)
    updated_at: str = field(default_factory=_iso)
    error: str = ""
    progress_summary: str = ""
    current_tool_activity: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    cancellation_state: dict[str, Any] = field(default_factory=dict)
    final_result: dict[str, Any] = field(default_factory=dict)
    supervisor_attempt_id: str = ""
    supervisor_lease_token: str = ""
    checkpoint_id: str = ""
    trigger_turn_id: str = ""
    relation_type: str = "independent"
    previous_task_id: str = ""
    user_message_id: str = ""


@dataclass(slots=True)
class LockLease:
    lease_id: str
    task_id: str
    mode: LockMode
    workspace_id: str
    repository_id: str
    paths: list[str]
    owner_pid: int
    acquired_at: str
    expires_at: str


@dataclass(slots=True)
class LaneReservation:
    execution: LaneExecution
    duplicate: bool = False


class GatewayLockManager:
    """Lease-based central lock table with reader/writer compatibility."""

    def __init__(self, coordinator: "LaneCoordinator") -> None:
        self.coordinator = coordinator

    @staticmethod
    def _conflicts(left: LockLease, right: LockLease) -> bool:
        if left.task_id == right.task_id:
            return False
        if left.workspace_id != right.workspace_id:
            return False
        if LockMode.WORKSPACE_WRITE in {left.mode, right.mode}:
            return True
        same_repo = bool(left.repository_id and left.repository_id == right.repository_id)
        if not same_repo:
            return False
        if LockMode.REPOSITORY_WRITE in {left.mode, right.mode}:
            return True
        repo_modes = {LockMode.REPOSITORY_READ, LockMode.REPOSITORY_WRITE}
        if left.mode in repo_modes and right.mode in repo_modes:
            return False
        if (
            left.mode == LockMode.REPOSITORY_READ and right.mode == LockMode.FILE_WRITE
        ) or (
            right.mode == LockMode.REPOSITORY_READ and left.mode == LockMode.FILE_WRITE
        ):
            return True
        left_paths, right_paths = set(left.paths), set(right.paths)
        overlap = bool(left_paths.intersection(right_paths))
        if not overlap:
            return False
        return LockMode.FILE_WRITE in {left.mode, right.mode}

    def acquire(
        self,
        *,
        task_id: str,
        mode: LockMode,
        workspace_id: str,
        repository_id: str,
        paths: Sequence[str],
        timeout_seconds: float,
        lease_seconds: int,
    ) -> LockLease | None:
        if mode == LockMode.NONE:
            return None
        canonical = self.coordinator.canonical_paths(paths)
        requested = LockLease(
            lease_id=f"lock_{uuid.uuid4().hex}", task_id=task_id, mode=mode,
            workspace_id=workspace_id, repository_id=repository_id, paths=canonical,
            owner_pid=os.getpid(), acquired_at=_iso(),
            expires_at=_iso(_now() + timedelta(seconds=max(1, lease_seconds))),
        )
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        waiting_emitted = False
        self.coordinator.emit("lock.requested", task_id=task_id, lane_id=None, mode=mode.value)
        with self.coordinator._condition:
            while True:
                acquired = False
                with self.coordinator._process_state_lock():
                    self.coordinator._load_locks_file_locked()
                    self.coordinator._recover_stale_locked()
                    if not any(self._conflicts(requested, lease) for lease in self.coordinator._locks.values()):
                        self.coordinator._locks[requested.lease_id] = requested
                        self.coordinator._persist_locks_file_locked()
                        acquired = True
                if acquired:
                    self.coordinator.emit("lock.acquired", task_id=task_id, lane_id=None, mode=mode.value)
                    return requested
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LaneLockTimeout(f"Timed out waiting for {mode.value} lock")
                if not waiting_emitted:
                    self.coordinator.emit(
                        "lock.waiting", task_id=task_id, lane_id=None, mode=mode.value
                    )
                    waiting_emitted = True
                self.coordinator._condition.wait(timeout=min(remaining, 0.25))

    def release_task(self, task_id: str) -> None:
        with self.coordinator._condition:
            with self.coordinator._process_state_lock():
                self.coordinator._load_locks_file_locked()
                released = [key for key, value in self.coordinator._locks.items() if value.task_id == task_id]
                for key in released:
                    self.coordinator._locks.pop(key, None)
                if released:
                    self.coordinator._persist_locks_file_locked()
            if released:
                self.coordinator._condition.notify_all()
                self.coordinator.emit("lock.released", task_id=task_id, lane_id=None, count=len(released))

    def recover_stale(self) -> None:
        with self.coordinator._condition:
            with self.coordinator._process_state_lock():
                self.coordinator._load_locks_file_locked()
                before = len(self.coordinator._locks)
                self.coordinator._recover_stale_locked()
                if len(self.coordinator._locks) != before:
                    self.coordinator._persist_locks_file_locked()
            self.coordinator._condition.notify_all()


class LaneCoordinator:
    """Coordinates one owning specialist lane for each gateway task."""

    def __init__(
        self,
        root: str | Path,
        *,
        contracts: Mapping[str, Any] | Mapping[LaneId, LaneContract] | None = None,
        taskboard: TaskBoard | None = None,
        event_sink: Callable[..., None] | None = None,
        global_worker_limit: int = 8,
        provider_limits: Mapping[str, int] | None = None,
        session_token_budget: int | None = None,
        global_token_budget: int | None = None,
        execution_supervisor: ExecutionSupervisor | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.taskboard = taskboard or TaskBoard(self.root)
        if contracts and all(isinstance(key, LaneId) and isinstance(value, LaneContract) for key, value in contracts.items()):
            self.contracts = dict(contracts)  # type: ignore[arg-type]
        else:
            self.contracts = configured_lane_contracts(contracts)  # type: ignore[arg-type]
        self.event_sink = event_sink
        self.global_worker_limit = max(1, int(global_worker_limit))
        self.provider_limits = {str(key): max(1, int(value)) for key, value in (provider_limits or {}).items()}
        self.session_token_budget = session_token_budget
        self.global_token_budget = global_token_budget
        self._last_supervisor_ui_heartbeat: dict[str, float] = {}
        self._condition = threading.Condition(threading.RLock())
        self._executions: dict[str, LaneExecution] = {}
        self._locks: dict[str, LockLease] = {}
        self._waiters: list[dict[str, Any]] = []
        self._wait_sequence = 0
        supervisor_config = ExecutionSupervisorConfig.from_settings(Settings())
        self.execution_supervisor = execution_supervisor or ExecutionSupervisor(
            supervisor_config,
            event_sink=self._supervisor_event,
        )
        self.taskboard.set_task_id_reservation_checker(
            lambda task_id: self.execution_supervisor.store.get_task_or_none(task_id)
            is not None
        )
        self._supervisor_heartbeat_stops: dict[str, threading.Event] = {}
        self._supervisor_heartbeat_threads: dict[str, threading.Thread] = {}
        self._human_resume_dispatcher: Callable[[str, str, str, dict[str, Any]], None] | None = None
        self.lock_manager = GatewayLockManager(self)
        self.state_path = workspace_dir(self.taskboard.store.workspace_id) / "gateway" / "lane_coordinator.json"
        self.locks_path = self.state_path.with_name("lane_locks.json")
        self.guard_path = self.state_path.with_name("lane_coordinator.lock")
        self._load()
        self._migrate_legacy_supervisor_records()
        self.recover(supervise=self.execution_supervisor.config.startup_recovery)

    @property
    def store(self) -> Any:
        """Expose the durable supervisor store required by the inbox controller contract."""
        return self.execution_supervisor.store

    def set_human_resume_dispatcher(
        self,
        dispatcher: Callable[[str, str, str, dict[str, Any]], None] | None,
    ) -> None:
        """Register the branch-owned dispatcher for durable human responses."""
        self._human_resume_dispatcher = dispatcher

    def suspend_for_human_input(
        self,
        task_id: str,
        *,
        inbox_item_id: str,
        checkpoint_id: str,
        request_type: str,
    ) -> Any:
        """Suspend one gateway branch for its linked durable inbox item."""
        task = self.execution_supervisor.suspend_for_human_input(
            task_id,
            inbox_item_id=inbox_item_id,
            checkpoint_id=checkpoint_id,
            request_type=request_type,
        )
        self.transition(
            task_id,
            LaneTaskState.WAITING,
            reason=f"waiting for {request_type} inbox item {inbox_item_id}",
        )
        self._stop_supervisor_heartbeats(task_id)
        return task

    def resume_from_human_input(
        self,
        task_id: str,
        *,
        inbox_item_id: str,
        checkpoint_id: str,
        resume_claim_id: str,
        structured_response: dict[str, Any],
    ) -> Any:
        """Queue only the branch whose durable inbox response holds this claim."""
        task = self.execution_supervisor.resume_from_human_input(
            task_id,
            inbox_item_id=inbox_item_id,
            checkpoint_id=checkpoint_id,
            resume_claim_id=resume_claim_id,
            structured_response=structured_response,
        )
        with self._condition:
            execution = self._executions[task_id]
            execution.worker_id = ""
            execution.supervisor_attempt_id = ""
            execution.supervisor_lease_token = ""
            execution.updated_at = _iso()
            self._persist_locked()
        self.transition(task_id, LaneTaskState.QUEUED, reason="durable human response received")
        self._dispatch_human_resume(
            task_id,
            inbox_item_id=inbox_item_id,
            resume_claim_id=resume_claim_id,
            structured_response=structured_response,
        )
        return task

    def restore_human_wait(
        self,
        task_id: str,
        *,
        inbox_item_id: str,
        checkpoint_id: str,
        request_type: str,
    ) -> Any:
        """Restore an unresolved durable inbox wait without starting new work."""
        task = self.execution_supervisor.restore_human_wait(
            task_id,
            inbox_item_id=inbox_item_id,
            checkpoint_id=checkpoint_id,
            request_type=request_type,
        )
        if self.inspect_task(task_id).state is not LaneTaskState.WAITING:
            self.transition(
                task_id,
                LaneTaskState.WAITING,
                reason=f"restored {request_type} inbox wait {inbox_item_id}",
            )
        self._stop_supervisor_heartbeats(task_id)
        return task

    def _dispatch_human_resume(
        self,
        task_id: str,
        *,
        inbox_item_id: str,
        resume_claim_id: str,
        structured_response: dict[str, Any],
    ) -> None:
        dispatcher = self._human_resume_dispatcher
        if dispatcher is None:
            return

        def run() -> None:
            try:
                dispatcher(task_id, inbox_item_id, resume_claim_id, structured_response)
            except Exception as exc:
                execution = self._executions.get(task_id)
                if execution is not None:
                    execution.error = f"human resume dispatch failed: {type(exc).__name__}: {exc}"
                    execution.updated_at = _iso()
                    with self._condition:
                        self._persist_locked()
                self.emit(
                    "human_input.dispatch_failed",
                    task_id=task_id,
                    inbox_item_id=inbox_item_id,
                    resume_claim_id=resume_claim_id,
                    error_type=type(exc).__name__,
                )

        threading.Thread(
            target=run,
            name=f"mana-human-resume-{task_id}",
            daemon=True,
        ).start()

    def dispatch_queued_human_resume(
        self,
        task_id: str,
        *,
        inbox_item_id: str,
        resume_claim_id: str,
        structured_response: dict[str, Any],
    ) -> bool:
        """Dispatch a recovered response only while its exact branch is still queued."""
        task = self.execution_supervisor.store.get_task_or_none(task_id)
        execution = self._executions.get(task_id)
        if (
            task is None
            or task.state is not SupervisorState.QUEUED
            or task.waiting_inbox_item_id
            or execution is None
            or execution.state is not LaneTaskState.QUEUED
        ):
            return False
        self._dispatch_human_resume(
            task_id,
            inbox_item_id=inbox_item_id,
            resume_claim_id=resume_claim_id,
            structured_response=structured_response,
        )
        return True

    def _supervisor_event(self, event_type: str, payload: dict[str, Any]) -> None:
        task_id = str(payload.get("task_id") or "")
        if event_type == "heartbeat":
            now = time.monotonic()
            previous = self._last_supervisor_ui_heartbeat.get(task_id, 0.0)
            if now - previous < max(60.0, self.execution_supervisor.config.heartbeat_seconds * 4.0):
                return
            self._last_supervisor_ui_heartbeat[task_id] = now
        lane = self._executions.get(task_id)
        self.emit(
            event_type,
            task_id=task_id,
            lane_id=lane.owning_lane if lane else None,
            execution_supervisor=True,
            supervisor_event=payload,
        )

    def canonical_paths(self, paths: Sequence[str]) -> list[str]:
        resolved: list[str] = []
        for item in paths:
            path = Path(str(item))
            target = path if path.is_absolute() else self.root / path
            canonical = str(target.expanduser().resolve(strict=False))
            if canonical not in resolved:
                resolved.append(canonical)
        return sorted(resolved)

    def emit(self, event_type: str, *, task_id: str, lane_id: LaneId | None, **metadata: Any) -> None:
        payload = {"event_type": event_type, "task_id": task_id, "lane_id": lane_id.value if lane_id else None, **metadata}
        record_current(event_type, payload)
        try:
            self.taskboard.store.append_history({"event_type": event_type, "payload": payload, "created_at": _iso()})
        except OSError:
            pass
        if callable(self.event_sink):
            title = {
                "lane.queued": "Waiting for specialist lane",
                "lane.started": f"{lane_id.value.title() if lane_id else 'Lane'} work",
                "lock.waiting": "Waiting for repository lock",
                "lane.completed": "Specialist lane completed",
            }.get(event_type, event_type.replace(".", " ").title())
            try:
                self.event_sink(event_type, title, status=metadata.pop("status", "running"), metadata=payload)
            except Exception:
                pass

    def select_lane(self, *, entry_route: str = "", intent: str = "", model_lane: str | LaneId | None = None) -> LaneId:
        lane = select_lane(entry_route=entry_route, intent=intent, model_lane=model_lane)
        contract = self.contracts[lane]
        if not contract.enabled:
            raise LaneCoordinatorError(f"Selected specialist lane {lane.value} is disabled")
        return lane

    def reserve(
        self,
        *,
        normalized_intent: str,
        lane_id: LaneId,
        session_id: str,
        workspace_id: str,
        repository_id: str,
        target_files: Sequence[str] = (),
        parent_task_id: str | None = None,
        root_task_id: str | None = None,
        priority: LanePriority | None = None,
        model: str = "",
        requested_input_tokens: int = 0,
        requested_output_tokens: int = 0,
        estimated_cost: float | None = None,
        model_context_window: int = 0,
        model_max_output_tokens: int = 0,
        estimate_confidence: str = "",
        estimate_source: str = "",
        capabilities: Sequence[str] = (),
        routing_decision_id: str = "",
        provider: str = "",
        task_type: str = "single",
        taskboard_task_id: str | None = None,
        supersedes_execution_id: str = "",
        derived_from_execution_id: str = "",
        previous_execution_id: str = "",
        trigger_turn_id: str = "",
        relation_type: str = "independent",
        previous_task_id: str = "",
        user_message_id: str = "",
        supervision_contract_decision_id: str = "",
        side_effect_classification: SideEffectClassification | None = None,
        completion_contract: Sequence[CompletionContract] = (),
        compensation_strategy: str = "",
        important_constraints: Sequence[str] = (),
    ) -> LaneReservation:
        if not self.execution_supervisor.config.enabled:
            raise LaneCoordinatorError(
                "execution supervisor is disabled; no gateway task was created"
            )
        contract = self.contracts[lane_id]
        if contract.requires_repository and not repository_id:
            raise LaneCoordinatorError(f"Lane {lane_id.value} requires a repository identity")
        if contract.allowed_models and model not in contract.allowed_models:
            raise LaneCoordinatorError(f"Model {model or '<unset>'} is not allowed for lane {lane_id.value}")
        files = self.canonical_paths(target_files)
        fingerprint = _stable_hash({
            "intent": " ".join(normalized_intent.lower().split()), "repository_id": repository_id,
            "workspace_id": workspace_id, "target_files": files,
            "lane": lane_id.value, "parent_task_id": parent_task_id,
            "relation_type": relation_type,
        })
        # ``lane_id`` is itself a validated structured routing decision. Older
        # callers did not persist a separate decision ID, so derive a stable
        # audit reference from that explicit selection without rerouting.
        effective_routing_decision_id = routing_decision_id or f"lane-selection:{fingerprint}"
        selected_priority = priority or contract.default_priority
        with self._condition:
            self._wait_sequence += 1
            waiter = {
                "waiter_id": f"wait_{uuid.uuid4().hex}",
                "lane_id": lane_id.value,
                "priority": selected_priority.value,
                "sequence": self._wait_sequence,
                "created_at": _iso(),
            }
            self._waiters.append(waiter)
            waiter_persisted = False
            deadline = time.monotonic() + contract.timeout_seconds
            try:
                while True:
                    for active in self._executions.values():
                        active_fingerprint = _stable_hash({
                            "intent": " ".join(active.normalized_intent.lower().split()), "repository_id": active.repository_id,
                            "workspace_id": active.workspace_id,
                            "target_files": active.target_files, "lane": active.owning_lane.value,
                            "parent_task_id": active.parent_task_id,
                            "relation_type": active.relation_type,
                        })
                        explicit_identity_matches = (
                            not taskboard_task_id
                            or active.taskboard_task_id == taskboard_task_id
                        )
                        if (
                            active.state in ACTIVE_LANE_STATES
                            and active_fingerprint == fingerprint
                            and explicit_identity_matches
                        ):
                            self.emit("lane.duplicate_detected", task_id=active.task_id, lane_id=lane_id, duplicate_of=active.task_id)
                            return LaneReservation(active, duplicate=True)
                    capacity_available = True
                    try:
                        self._assert_capacity(contract, model)
                    except LaneCapacityError:
                        capacity_available = False
                    if capacity_available and self._next_waiter_id() == waiter["waiter_id"]:
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise LaneCapacityError(f"lane {lane_id.value} capacity wait timed out")
                    if not waiter_persisted:
                        self.emit(
                            "lane.queued",
                            task_id=waiter["waiter_id"],
                            lane_id=lane_id,
                            reason="capacity",
                        )
                        self._persist_locked()
                        waiter_persisted = True
                    self._condition.wait(timeout=min(remaining, 0.25))
            finally:
                self._waiters = [item for item in self._waiters if item["waiter_id"] != waiter["waiter_id"]]
                if waiter_persisted:
                    self._persist_locked()
            budget = LaneBudget(
                reserved_input_tokens=max(0, requested_input_tokens),
                reserved_output_tokens=max(0, requested_output_tokens),
                estimated_cost=max(0.0, float(estimated_cost or 0.0)),
                estimated_cost_known=estimated_cost is not None,
                model_context_window=max(0, int(model_context_window)),
                model_max_output_tokens=max(0, int(model_max_output_tokens)),
                estimate_confidence=estimate_confidence,
                estimate_source=estimate_source,
            )
            if parent_task_id:
                parent = self._executions.get(parent_task_id)
                if parent is None:
                    raise LaneBudgetError("parent task budget is unavailable")
                # Grow the active parent (and ancestors) before the child is
                # charged against session/global caps, so the expanded parent
                # envelope is included in the subsequent budget assertion.
                # Terminal parents do not constrain children (same policy as
                # recalculate_budget). Hard-fail only when expansion exceeds a
                # real lane/session/global cap.
                if parent.state not in {
                    LaneTaskState.COMPLETED,
                    LaneTaskState.FAILED,
                    LaneTaskState.CANCELLED,
                }:
                    self._ensure_parent_envelope_for_child_locked(
                        child_task_id=None,
                        parent_task_id=parent_task_id,
                        required_child_tokens=budget.reserved_tokens,
                        child_estimated_cost=(
                            float(budget.estimated_cost)
                            if budget.estimated_cost_known
                            else None
                        ),
                        reason="parent envelope for child reservation",
                    )
            self._assert_budget(contract, session_id, budget)
            if taskboard_task_id:
                task = self.taskboard.get_task(taskboard_task_id)
                expected_parent = self._executions[parent_task_id].taskboard_task_id if parent_task_id else None
                if task.parent_task_id != expected_parent:
                    raise LaneCoordinatorError("existing TaskBoard child does not match the selected lane parent")
                self.taskboard.add_files_to_inspect(task.task_id, files)
            elif parent_task_id:
                parent_execution = self._executions[parent_task_id]
                task = self.taskboard.create_child_task(
                    parent_execution.taskboard_task_id,
                    title=f"{contract.display_name}: {normalized_intent[:100]}",
                    user_request=normalized_intent,
                    owner_agent_id=f"lane:{lane_id.value}",
                    trigger_turn_id=trigger_turn_id,
                    relation_type=relation_type,
                    previous_task_id=previous_task_id or parent_task_id,
                )
                self.taskboard.add_files_to_inspect(task.task_id, files)
            else:
                task = self.taskboard.create_task(
                    title=f"{contract.display_name}: {normalized_intent[:100]}", user_request=normalized_intent,
                    normalized_goal=normalized_intent, owner_agent_id=f"lane:{lane_id.value}",
                    related_files=files, action_type=f"lane:{lane_id.value}", workspace_id=workspace_id,
                    session_id=session_id, repository_ids=[repository_id] if repository_id else [],
                    primary_repository_id=repository_id,
                    trigger_turn_id=trigger_turn_id,
                    relation_type=relation_type,
                    previous_task_id=previous_task_id,
                )
            task_id = task.task_id
            side_effect = side_effect_classification or (
                SideEffectClassification.READ_ONLY
                if lane_id in {LaneId.RESEARCH, LaneId.REVIEW, LaneId.VERIFY}
                and not contract.requires_write_access
                and set(capabilities).issubset({"repository_read"})
                else SideEffectClassification.UNKNOWN
            )
            initial_contract = list(completion_contract) or [
                CompletionContract(
                    contract_type=CompletionContractType.STRUCTURED_RESULT_VALID,
                    metadata={
                        "required_keys": ["lane_state", "verification_evidence_present"],
                        "expected_values": {"lane_state": "completed"},
                    },
                )
            ]
            contract_decision_id = supervision_contract_decision_id or effective_routing_decision_id
            self.execution_supervisor.create_task(
                task_id=task_id,
                parent_task_id=parent_task_id,
                task_type=task_type,
                assigned_agent=f"lane:{lane_id.value}",
                assigned_model=model,
                runtime_provider=provider,
                workspace_path=self.root,
                routing_decision_id=effective_routing_decision_id,
                side_effect_classification=side_effect,
                completion_contract=initial_contract,
                compensation_strategy=compensation_strategy,
                dependency_task_ids=task.depends_on,
                token_budget=budget.reserved_tokens or None,
                estimated_cost=(budget.estimated_cost if budget.estimated_cost_known else None),
                model_context_window=budget.model_context_window,
                model_max_output_tokens=budget.model_max_output_tokens,
                estimate_confidence=budget.estimate_confidence,
                estimate_source=budget.estimate_source,
                monetary_budget=contract.cost_budget,
                execution_fingerprint=fingerprint,
                session_id=session_id,
                workspace_id=workspace_id,
                repository_id=repository_id,
                normalized_intent=normalized_intent,
                requested_operation=lane_id.value,
                target_resources=files,
                expected_output=task_type,
                important_constraints=important_constraints,
                field_provenance={
                    "side_effect_classification": "model_selected_lane_contract",
                    "completion_contract": "model_selected_lane_contract",
                    "target_resources": "model_selected" if files else "not_applicable_or_not_selected",
                    "important_constraints": (
                        "model_selected" if important_constraints else "not_applicable_or_not_selected"
                    ),
                },
                supervision_contract_decision_id=contract_decision_id,
                supersedes_execution_id=supersedes_execution_id,
                derived_from_execution_id=derived_from_execution_id,
                previous_execution_id=previous_execution_id,
                trigger_turn_id=trigger_turn_id,
                relation_type=relation_type,
                previous_task_id=previous_task_id,
                idempotency_key=(f"{session_id}:{user_message_id}" if user_message_id else ""),
            )
            self.execution_supervisor.queue(task_id)
            execution = LaneExecution(
                task_id=task_id,
                root_task_id=(root_task_id or (self._executions[parent_task_id].root_task_id if parent_task_id else task_id)),
                parent_task_id=parent_task_id,
                owning_lane=lane_id, state=LaneTaskState.QUEUED, normalized_intent=normalized_intent,
                repository_id=repository_id, workspace_id=workspace_id, session_id=session_id,
                target_files=files, priority=selected_priority, budget=budget,
                taskboard_task_id=task.task_id, model=model, capabilities=list(capabilities),
                routing_decision_id=effective_routing_decision_id, provider=provider, task_type=task_type,
                trigger_turn_id=trigger_turn_id, relation_type=relation_type, previous_task_id=previous_task_id,
                user_message_id=user_message_id,
                lane_history=[{"lane_id": lane_id.value, "state": "queued", "at": _iso()}],
            )
            self._executions[task_id] = execution
            self.taskboard.update_status(task_id, TaskStatus.ROUTED)
            self.taskboard.update_status(task_id, TaskStatus.QUEUED)
            self._persist_locked()
            self.emit("lane.queued", task_id=task_id, lane_id=lane_id)
            self.emit("task.created", task_id=task_id, lane_id=lane_id, parent_task_id=parent_task_id)
            self.emit("model.assigned", task_id=task_id, lane_id=lane_id, routing_decision_id=effective_routing_decision_id, provider=provider, model=model)
            self.emit("resource.reserved", task_id=task_id, lane_id=lane_id, budget=asdict(budget))
            return LaneReservation(execution)

    def start(self, reservation: LaneReservation) -> LaneExecution:
        execution = reservation.execution
        if reservation.duplicate:
            return execution
        contract = self.contracts[execution.owning_lane]
        mode = contract.lock_policy
        paths = execution.target_files
        if mode == LockMode.NONE and "repository_write" in execution.capabilities:
            mode = LockMode.FILE_WRITE
        elif mode == LockMode.NONE and "repository_read" in execution.capabilities:
            mode = LockMode.REPOSITORY_READ
        if mode in {LockMode.FILE_READ, LockMode.FILE_WRITE} and not paths:
            mode = LockMode.REPOSITORY_WRITE if contract.requires_write_access else LockMode.REPOSITORY_READ
        self.lock_manager.acquire(
            task_id=execution.task_id, mode=mode, workspace_id=execution.workspace_id,
            repository_id=execution.repository_id, paths=paths,
            timeout_seconds=float(contract.timeout_seconds), lease_seconds=contract.timeout_seconds + 30,
        )
        try:
            current = self.execution_supervisor.store.get_task(execution.task_id)
            if execution.supervisor_attempt_id and execution.supervisor_lease_token:
                supervised = self.execution_supervisor.resume_running(
                    execution.task_id,
                    attempt_id=execution.supervisor_attempt_id,
                    lease_token=execution.supervisor_lease_token,
                )
                lease_token = execution.supervisor_lease_token
            else:
                if current.state == SupervisorState.RETRY_SCHEDULED:
                    current = self.execution_supervisor.release_retry(execution.task_id)
                if current.state != SupervisorState.QUEUED:
                    raise LaneCoordinatorError(
                        "durable task is not ready for a new lease; recover or wait for the "
                        "active lease to expire before resuming"
                    )
                supervised, lease_token = self.execution_supervisor.acquire_lease(
                    execution.task_id,
                    owner=f"gateway:{os.getpid()}",
                    worker=f"gateway:{os.getpid()}:{threading.get_ident()}",
                )
                self.execution_supervisor.start(
                    execution.task_id,
                    attempt_id=supervised.attempt_id,
                    lease_token=lease_token,
                )
        except Exception:
            self.lock_manager.release_task(execution.task_id)
            raise
        with self._condition:
            execution.state = LaneTaskState.RUNNING
            execution.worker_id = f"gateway:{os.getpid()}:{threading.get_ident()}"
            execution.supervisor_attempt_id = supervised.attempt_id
            execution.supervisor_lease_token = lease_token
            execution.last_heartbeat = execution.updated_at = _iso()
            execution.lane_history.append({"lane_id": execution.owning_lane.value, "state": "running", "at": execution.updated_at})
            task_status = self.taskboard.get_task(execution.taskboard_task_id).status
            if task_status in {TaskStatus.QUEUED, TaskStatus.ROUTED, TaskStatus.WAITING_FOR_TOOLS}:
                self.taskboard.update_status(execution.taskboard_task_id, TaskStatus.IN_PROGRESS)
            self._persist_locked()
        self._start_supervisor_heartbeats(execution)
        self.emit("lane.started", task_id=execution.task_id, lane_id=execution.owning_lane)
        return execution

    def _start_supervisor_heartbeats(self, execution: LaneExecution) -> None:
        self._stop_supervisor_heartbeats(execution.task_id)
        stop = threading.Event()
        self._supervisor_heartbeat_stops[execution.task_id] = stop

        def renew() -> None:
            interval = self.execution_supervisor.config.heartbeat_seconds
            while not stop.wait(interval):
                try:
                    self.execution_supervisor.heartbeat(
                        execution.task_id,
                        attempt_id=execution.supervisor_attempt_id,
                        lease_token=execution.supervisor_lease_token,
                    )
                except Exception as exc:
                    execution.error = f"supervisor heartbeat failed: {exc}"
                    stop.set()

        heartbeat_thread = threading.Thread(
            target=renew,
            name=f"mana-supervisor-heartbeat-{execution.task_id}",
            daemon=True,
        )
        self._supervisor_heartbeat_threads[execution.task_id] = heartbeat_thread
        heartbeat_thread.start()

    def _stop_supervisor_heartbeats(self, task_id: str) -> None:
        stop = self._supervisor_heartbeat_stops.pop(task_id, None)
        if stop is not None:
            stop.set()
        heartbeat_thread = self._supervisor_heartbeat_threads.pop(task_id, None)
        if heartbeat_thread is not None and heartbeat_thread is not threading.current_thread():
            heartbeat_thread.join(timeout=min(5.0, self.execution_supervisor.config.lease_seconds / 2))

    def checkpoint(
        self,
        task_id: str,
        *,
        boundary: str,
        resume_payload: Mapping[str, Any] | None = None,
        completed_steps: Sequence[str] = (),
        pending_steps: Sequence[str] = (),
    ) -> str:
        execution = self._executions[task_id]
        checkpoint = self.execution_supervisor.checkpoint(
            task_id,
            attempt_id=execution.supervisor_attempt_id,
            lease_token=execution.supervisor_lease_token,
            resume_payload={"boundary": boundary, **dict(resume_payload or {})},
            completed_steps=completed_steps,
            pending_steps=pending_steps,
            child_execution_ids=[
                item.task_id
                for item in self._executions.values()
                if item.parent_task_id == task_id
            ],
            result_escrow_references=[
                item.result_id
                for item in self.execution_supervisor.store.unacknowledged_results(task_id)
            ],
            budget_snapshot=asdict(execution.budget),
            resume_cursor=boundary,
        )
        execution.checkpoint_id = checkpoint.checkpoint_id
        execution.updated_at = _iso()
        with self._condition:
            self._persist_locked()
        self.emit(
            "checkpoint.saved",
            task_id=task_id,
            lane_id=execution.owning_lane,
            checkpoint_id=checkpoint.checkpoint_id,
            boundary=boundary,
        )
        return checkpoint.checkpoint_id

    @contextmanager
    def execution(self, **kwargs: Any) -> Iterator[LaneReservation]:
        reservation = self.reserve(**kwargs)
        if reservation.duplicate:
            yield reservation
            return
        self.start(reservation)
        try:
            yield reservation
        except BaseException as exc:
            self.finish(reservation.execution.task_id, state=LaneTaskState.FAILED, error=str(exc))
            raise

    def finish(
        self,
        task_id: str,
        *,
        state: LaneTaskState = LaneTaskState.COMPLETED,
        changed_files: Sequence[str] = (),
        consumed_input_tokens: int = 0,
        consumed_output_tokens: int = 0,
        actual_cost: float | None = None,
        verification_state: Mapping[str, Any] | None = None,
        error: str = "",
    ) -> LaneExecution:
        with self._condition:
            execution = self._executions[task_id]
            execution.changed_files = self.canonical_paths(changed_files)
            execution_had_usage = execution.budget.consumed_tokens > 0
            incremental_usage = consumed_input_tokens > 0 or consumed_output_tokens > 0
            execution.budget.consumed_input_tokens += max(0, consumed_input_tokens)
            execution.budget.consumed_output_tokens += max(0, consumed_output_tokens)
            execution.budget.actual_cost += max(0.0, float(actual_cost or 0.0))
            if actual_cost is not None:
                execution.budget.actual_cost_known = (
                    not execution_had_usage or execution.budget.actual_cost_known
                )
            elif incremental_usage:
                execution.budget.actual_cost_known = False
            if execution.parent_task_id and execution.parent_task_id in self._executions:
                parent = self._executions[execution.parent_task_id]
                parent_had_usage = parent.budget.consumed_tokens > 0
                parent.budget.consumed_input_tokens += max(0, consumed_input_tokens)
                parent.budget.consumed_output_tokens += max(0, consumed_output_tokens)
                parent.budget.actual_cost += max(0.0, float(actual_cost or 0.0))
                if actual_cost is not None:
                    parent.budget.actual_cost_known = (
                        not parent_had_usage or parent.budget.actual_cost_known
                    )
                elif incremental_usage:
                    parent.budget.actual_cost_known = False
                parent.updated_at = _iso()
            execution.verification_state.update(dict(verification_state or {}))
            execution.error = error
            execution.updated_at = execution.last_heartbeat = _iso()
            accounted_input_tokens = execution.budget.consumed_input_tokens
            accounted_output_tokens = execution.budget.consumed_output_tokens
            accounted_actual_cost = execution.budget.actual_cost
        self._stop_supervisor_heartbeats(task_id)
        verified = False
        if state == LaneTaskState.COMPLETED:
            contracts: list[CompletionContract] = []
            for changed in execution.changed_files:
                path = Path(changed)
                try:
                    relative = path.relative_to(self.root)
                except ValueError:
                    relative = path
                if path.is_file():
                    contracts.append(
                        CompletionContract(
                            contract_type=CompletionContractType.FILE_EXISTS,
                            path=str(relative),
                            minimum_size=0,
                            require_attempt_change=True,
                        )
                    )
                elif path.is_dir():
                    contracts.append(
                        CompletionContract(
                            contract_type=CompletionContractType.DIRECTORY_EXISTS,
                            path=str(relative),
                            expected_kind="directory",
                            require_attempt_change=True,
                        )
                    )
                else:
                    contracts.append(
                        CompletionContract(
                            contract_type=CompletionContractType.GIT_DIFF_PRESENT,
                            path=str(relative),
                        )
                    )
            if not contracts:
                contracts.append(
                    CompletionContract(
                        contract_type=CompletionContractType.STRUCTURED_RESULT_VALID,
                        metadata={
                            "required_keys": ["lane_state", "verification_evidence_present"],
                            "expected_values": {
                                "lane_state": "completed",
                                "verification_evidence_present": True,
                            },
                        },
                    )
                )
            try:
                self.execution_supervisor.set_completion_contract(
                    task_id,
                    attempt_id=execution.supervisor_attempt_id,
                    lease_token=execution.supervisor_lease_token,
                    contracts=contracts,
                )
                supervised = self.execution_supervisor.submit_result(
                    task_id,
                    attempt_id=execution.supervisor_attempt_id,
                    lease_token=execution.supervisor_lease_token,
                    payload={
                        "lane_state": "completed",
                        "changed_files": list(execution.changed_files),
                        "verification_evidence_present": bool(
                            execution.verification_state or execution.evidence
                        ),
                        "verification_state": dict(execution.verification_state),
                        "chat_result": dict(execution.verification_state.get("chat_result") or {}),
                        "evidence": list(execution.evidence),
                        "token_usage": accounted_input_tokens + accounted_output_tokens,
                        "actual_cost": accounted_actual_cost,
                    },
                    token_usage=accounted_input_tokens + accounted_output_tokens,
                    actual_cost=(accounted_actual_cost if execution.budget.actual_cost_known else None),
                )
            except CompletionVerificationError as exc:
                execution.error = str(exc)
                state = LaneTaskState.VERIFYING
            except BudgetExceededError as exc:
                # submit_result has already projected the durable task to its
                # authoritative terminal budget state. Do not replace it with
                # a second terminal transition to FAILED.
                execution.error = str(exc)
                state = LaneTaskState.BUDGET_EXHAUSTED
            except ExecutionSupervisorError as exc:
                execution.error = str(exc)
                supervised = self.execution_supervisor.store.get_task(task_id)
                state = (
                    LaneTaskState.BUDGET_EXHAUSTED
                    if supervised.state == SupervisorState.BUDGET_EXHAUSTED
                    else LaneTaskState.FAILED
                )
                if supervised.state not in {
                    SupervisorState.FAILED,
                    SupervisorState.CANCELLED,
                    SupervisorState.COMPLETED,
                    SupervisorState.COMPLETED_PENDING_VERIFICATION,
                    SupervisorState.BUDGET_EXHAUSTED,
                }:
                    self.execution_supervisor.transition(
                        task_id,
                        SupervisorState.FAILED,
                        reason=f"lane completion supervision failed: {exc}",
                    )
            else:
                verified = supervised.state == SupervisorState.COMPLETED
                state = (
                    LaneTaskState.COMPLETED if verified
                    else LaneTaskState.PENDING_BUDGET_DECISION
                    if supervised.state is SupervisorState.PENDING_BUDGET_DECISION
                    else LaneTaskState.VERIFYING
                )
                if not verified:
                    if state is LaneTaskState.PENDING_BUDGET_DECISION:
                        execution.error = "A validated model budget-overrun decision is required before finalization."
                    else:
                        manifest = self.execution_supervisor.store.artifact_manifest(task_id) or {}
                        execution.error = _completion_verification_failure(manifest)
        elif state == LaneTaskState.CANCELLED:
            self.execution_supervisor.cancel(task_id, reason=error or "lane execution cancelled")
        elif state == LaneTaskState.BUDGET_EXHAUSTED:
            supervised = self.execution_supervisor.store.get_task(task_id)
            if supervised.state not in {
                SupervisorState.BUDGET_EXHAUSTED,
                SupervisorState.COMPLETED,
                SupervisorState.CANCELLED,
            }:
                self.execution_supervisor.transition(
                    task_id,
                    SupervisorState.BUDGET_EXHAUSTED,
                    reason=error or "canonical execution budget exhausted",
                )
        else:
            supervised = self.execution_supervisor.store.get_task(task_id)
            if supervised.state not in {SupervisorState.FAILED, SupervisorState.CANCELLED}:
                self.execution_supervisor.transition(
                    task_id,
                    SupervisorState.FAILED,
                    reason=error or f"lane execution ended as {state.value}",
                )
        execution.supervisor_lease_token = ""
        with self._condition:
            execution.state = state
            mapped_status = {
                LaneTaskState.COMPLETED: TaskStatus.DONE,
                LaneTaskState.VERIFYING: TaskStatus.NEEDS_REVIEW,
                LaneTaskState.CANCELLED: TaskStatus.CANCELLED,
                LaneTaskState.PENDING_BUDGET_DECISION: TaskStatus.NEEDS_REVIEW,
            }.get(state, TaskStatus.FAILED)
            reason = error or execution.error or f"lane execution ended as {state.value}"
            if mapped_status == TaskStatus.DONE:
                supervised = self.execution_supervisor.store.get_task(task_id)
                manifest = self.execution_supervisor.store.artifact_manifest(task_id) or {}
                self.taskboard.project_supervisor_completion(
                    execution.taskboard_task_id,
                    supervisor_task=supervised,
                    verification_evidence={
                        "result_id": supervised.result_id,
                        "verification": manifest.get("verification"),
                        "artefacts": manifest.get("artefacts", []),
                    },
                )
            else:
                self.taskboard.update_status(
                    execution.taskboard_task_id,
                    mapped_status,
                    reason=reason if mapped_status == TaskStatus.FAILED else None,
                )
            self._persist_locked()
        self.lock_manager.release_task(task_id)
        if state is LaneTaskState.PENDING_BUDGET_DECISION:
            # The result is durable and valid, but requires its separate
            # finalization decision. Never publish it as a lane failure: a
            # later accepted decision would otherwise leave the UI showing a
            # false execution error beside a successful response.
            event = "budget.overrun.decision.required"
            event_status = "waiting"
        else:
            event = {
                LaneTaskState.COMPLETED: "lane.completed", LaneTaskState.CANCELLED: "lane.cancelled",
                LaneTaskState.BUDGET_EXHAUSTED: "lane.budget_exhausted",
            }.get(state, "lane.failed" if state != LaneTaskState.VERIFYING else "verification.failed")
            event_status = "success" if verified else "error"
        self.emit(event, task_id=task_id, lane_id=execution.owning_lane, status=event_status)
        self.emit("resource.released", task_id=task_id, lane_id=execution.owning_lane)
        with self._condition:
            self._condition.notify_all()
        return execution

    def synchronize_usage(
        self,
        task_id: str,
        *,
        consumed_input_tokens: int = 0,
        consumed_output_tokens: int = 0,
        actual_cost: float | None = None,
        accounting_reservation_ids: Sequence[str] = (),
    ) -> LaneExecution:
        """Persist cumulative provider usage without double-counting a later finish."""
        with self._condition:
            execution = self._executions[task_id]
            input_delta = max(
                0,
                int(consumed_input_tokens)
                - execution.budget.consumed_input_tokens,
            )
            output_delta = max(
                0,
                int(consumed_output_tokens)
                - execution.budget.consumed_output_tokens,
            )
            cost_delta = max(0.0, float(actual_cost or 0.0) - execution.budget.actual_cost)
            execution.budget.consumed_input_tokens += input_delta
            execution.budget.consumed_output_tokens += output_delta
            execution.budget.actual_cost += cost_delta
            execution.budget.actual_cost_known = actual_cost is not None
            for reservation_id in accounting_reservation_ids:
                if reservation_id not in execution.accounting_reservation_ids:
                    execution.accounting_reservation_ids.append(reservation_id)
            if execution.parent_task_id and execution.parent_task_id in self._executions:
                parent = self._executions[execution.parent_task_id]
                parent_had_usage = parent.budget.consumed_tokens > 0
                parent.budget.consumed_input_tokens += input_delta
                parent.budget.consumed_output_tokens += output_delta
                parent.budget.actual_cost += cost_delta
                parent.budget.actual_cost_known = (
                    (not parent_had_usage or parent.budget.actual_cost_known)
                    and actual_cost is not None
                )
                parent.updated_at = _iso()
            execution.updated_at = _iso()
            self._persist_locked()
            if accounting_reservation_ids:
                self.execution_supervisor.record_accounting_reservations(
                    task_id, accounting_reservation_ids
                )
            return execution

    def recalculate_budget(
        self,
        task_id: str,
        *,
        forecast_input_tokens: int,
        forecast_output_tokens: int,
        forecast_cost: float | None,
        accounting_reservation_id: str = "",
        reason: str = "provider-call forecast",
    ) -> LaneExecution:
        """Atomically grow a reservation only when every immutable cap admits it.

        When the task has an active parent, the parent (and active ancestors)
        reserved envelope is expanded first so child growth adds into the parent
        budget instead of failing with "recalculated child budget exceeds the
        parent remaining budget". Terminal parents do not constrain children.
        """
        with self._condition:
            execution = self._executions[task_id]
            if execution.state not in ACTIVE_LANE_STATES:
                raise LaneBudgetError("task is not eligible for budget recalculation")
            budget = execution.budget
            next_input = max(budget.reserved_input_tokens, budget.consumed_input_tokens + max(0, int(forecast_input_tokens)))
            next_output = max(budget.reserved_output_tokens, budget.consumed_output_tokens + max(0, int(forecast_output_tokens)))
            next_total = next_input + next_output
            next_cost = max(budget.estimated_cost, budget.actual_cost + max(0.0, float(forecast_cost or 0.0)))
            contract = self.contracts[execution.owning_lane]
            if contract.token_budget is not None and next_total > contract.token_budget:
                raise LaneBudgetError("recalculated budget exceeds the lane token limit")
            if contract.cost_budget is not None and (forecast_cost is None or next_cost > contract.cost_budget):
                raise LaneBudgetError("recalculated budget exceeds the lane cost limit")
            active = [item for item in self._executions.values() if item.state in ACTIVE_LANE_STATES and item.task_id != task_id]
            if self.session_token_budget is not None and sum(item.budget.reserved_tokens for item in active if item.session_id == execution.session_id) + next_total > self.session_token_budget:
                raise LaneBudgetError("recalculated budget exceeds the session token limit")
            if self.global_token_budget is not None and sum(item.budget.reserved_tokens for item in active) + next_total > self.global_token_budget:
                raise LaneBudgetError("recalculated budget exceeds the global token limit")
            if execution.parent_task_id and execution.parent_task_id in self._executions:
                parent = self._executions[execution.parent_task_id]
                if parent.state not in {
                    LaneTaskState.COMPLETED,
                    LaneTaskState.FAILED,
                    LaneTaskState.CANCELLED,
                }:
                    self._ensure_parent_envelope_for_child_locked(
                        child_task_id=task_id,
                        parent_task_id=execution.parent_task_id,
                        required_child_tokens=next_total,
                        child_estimated_cost=forecast_cost,
                        reason="parent envelope for child recalculation",
                    )
            if next_total <= budget.reserved_tokens and next_cost <= budget.estimated_cost:
                return execution
            previous_total = budget.reserved_tokens
            try:
                self.execution_supervisor.revise_budget(
                    task_id,
                    token_budget=next_total,
                    estimated_cost=(next_cost if forecast_cost is not None else None),
                    reason=reason,
                    evidence={
                        "forecast_input_tokens": max(0, int(forecast_input_tokens)),
                        "forecast_output_tokens": max(0, int(forecast_output_tokens)),
                        "forecast_cost": forecast_cost,
                        "accounting_reservation_id": accounting_reservation_id,
                    },
                )
            except BudgetExceededError as exc:
                raise LaneBudgetError(str(exc)) from exc
            budget.reserved_input_tokens = next_input
            budget.reserved_output_tokens = next_output
            budget.estimated_cost = next_cost
            budget.estimated_cost_known = forecast_cost is not None
            budget.revisions.append({
                "reason": reason,
                "previous_reserved_tokens": previous_total,
                "revised_reserved_tokens": next_total,
                "forecast_input_tokens": max(0, int(forecast_input_tokens)),
                "forecast_output_tokens": max(0, int(forecast_output_tokens)),
                "forecast_cost": forecast_cost,
                "accounting_reservation_id": accounting_reservation_id,
                "at": _iso(),
            })
            execution.updated_at = _iso()
            self._persist_locked()
            self.emit("budget.recalculated", task_id=task_id, lane_id=execution.owning_lane, budget=asdict(budget))
            return execution

    def _ensure_parent_envelope_for_child_locked(
        self,
        *,
        child_task_id: str | None,
        parent_task_id: str,
        required_child_tokens: int,
        child_estimated_cost: float | None = None,
        reason: str = "parent envelope for child budget",
    ) -> None:
        """Grow parent (and active ancestors) so required_child_tokens fits.

        ``required_child_tokens`` is the full post-change total for the target
        child. Active siblings keep their current reservations; the revising
        child (when provided) is excluded so its previous reservation is not
        double-counted. Terminal parents are not expanded and do not block.

        Must be called while holding ``self._condition``.
        """
        parent_id: str | None = parent_task_id
        revising_id = str(child_task_id or "")
        required = max(0, int(required_child_tokens))
        cost = child_estimated_cost
        seen: set[str] = set()

        while parent_id and parent_id in self._executions and parent_id not in seen:
            seen.add(parent_id)
            parent = self._executions[parent_id]
            if parent.state in {
                LaneTaskState.COMPLETED,
                LaneTaskState.FAILED,
                LaneTaskState.CANCELLED,
            }:
                # Terminal parents do not constrain live children (matches reserve).
                return

            sibling_reserved = sum(
                item.budget.reserved_tokens
                for item in self._executions.values()
                if item.parent_task_id == parent_id
                and item.state in ACTIVE_LANE_STATES
                and item.task_id != revising_id
            )
            needed_total = (
                parent.budget.consumed_tokens
                + sibling_reserved
                + required
            )
            if needed_total <= parent.budget.reserved_tokens:
                return

            # Expand ancestors first so the parent's new total still fits.
            grandparent_id = parent.parent_task_id
            if grandparent_id and grandparent_id in self._executions:
                grandparent = self._executions[grandparent_id]
                if grandparent.state not in {
                    LaneTaskState.COMPLETED,
                    LaneTaskState.FAILED,
                    LaneTaskState.CANCELLED,
                }:
                    self._ensure_parent_envelope_for_child_locked(
                        child_task_id=parent_id,
                        parent_task_id=grandparent_id,
                        required_child_tokens=needed_total,
                        child_estimated_cost=cost,
                        reason=reason,
                    )

            self._grow_execution_reservation_locked(
                parent_id,
                target_total_tokens=needed_total,
                additional_cost=cost,
                reason=reason,
                evidence={
                    "for_child_task_id": revising_id or "",
                    "required_child_tokens": required,
                },
            )
            return

    def _grow_execution_reservation_locked(
        self,
        task_id: str,
        *,
        target_total_tokens: int,
        additional_cost: float | None,
        reason: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        """Increase one execution's reserved tokens/cost under lane/session/global caps.

        Must be called while holding ``self._condition``.
        """
        execution = self._executions[task_id]
        budget = execution.budget
        target = max(
            0,
            int(target_total_tokens),
            budget.reserved_tokens,
            budget.consumed_tokens,
        )
        current_output = max(
            budget.reserved_output_tokens,
            budget.consumed_output_tokens,
        )
        target_input = max(budget.reserved_input_tokens, target - current_output)
        if target_input + current_output < target:
            current_output = target - target_input
        next_total = target_input + current_output
        next_cost = budget.estimated_cost
        cost_known = budget.estimated_cost_known
        if additional_cost is not None or budget.estimated_cost_known:
            next_cost = max(
                0.0,
                float(budget.estimated_cost)
                + max(0.0, float(additional_cost or 0.0)),
            )
            cost_known = additional_cost is not None or budget.estimated_cost_known
        if next_total <= budget.reserved_tokens and (
            additional_cost is None or next_cost <= budget.estimated_cost
        ):
            return

        contract = self.contracts[execution.owning_lane]
        if contract.token_budget is not None and next_total > contract.token_budget:
            raise LaneBudgetError(
                "parent envelope expansion exceeds the lane token limit"
            )
        if contract.cost_budget is not None and (
            not cost_known or next_cost > contract.cost_budget
        ):
            raise LaneBudgetError(
                "parent envelope expansion exceeds the lane cost limit"
            )
        active = [
            item
            for item in self._executions.values()
            if item.state in ACTIVE_LANE_STATES and item.task_id != task_id
        ]
        if self.session_token_budget is not None and (
            sum(
                item.budget.reserved_tokens
                for item in active
                if item.session_id == execution.session_id
            )
            + next_total
            > self.session_token_budget
        ):
            raise LaneBudgetError(
                "parent envelope expansion exceeds the session token limit"
            )
        if self.global_token_budget is not None and (
            sum(item.budget.reserved_tokens for item in active) + next_total
            > self.global_token_budget
        ):
            raise LaneBudgetError(
                "parent envelope expansion exceeds the global token limit"
            )

        previous_total = budget.reserved_tokens
        try:
            self.execution_supervisor.revise_budget(
                task_id,
                token_budget=next_total,
                estimated_cost=(next_cost if cost_known else None),
                reason=reason,
                evidence=dict(evidence or {}),
            )
        except BudgetExceededError as exc:
            raise LaneBudgetError(str(exc)) from exc

        budget.reserved_input_tokens = target_input
        budget.reserved_output_tokens = current_output
        budget.estimated_cost = next_cost
        budget.estimated_cost_known = cost_known
        budget.revisions.append(
            {
                "reason": reason,
                "previous_reserved_tokens": previous_total,
                "revised_reserved_tokens": next_total,
                "forecast_input_tokens": max(
                    0, target_input - budget.consumed_input_tokens
                ),
                "forecast_output_tokens": max(
                    0, current_output - budget.consumed_output_tokens
                ),
                "forecast_cost": (next_cost if cost_known else None),
                "accounting_reservation_id": "",
                "at": _iso(),
                **{
                    key: value
                    for key, value in dict(evidence or {}).items()
                    if key
                    not in {
                        "reason",
                        "previous_reserved_tokens",
                        "revised_reserved_tokens",
                        "forecast_input_tokens",
                        "forecast_output_tokens",
                        "forecast_cost",
                        "accounting_reservation_id",
                        "at",
                    }
                },
            }
        )
        execution.updated_at = _iso()
        self.emit(
            "budget.recalculated",
            task_id=task_id,
            lane_id=execution.owning_lane,
            budget=asdict(budget),
        )

    def transition(
        self,
        task_id: str,
        state: LaneTaskState,
        *,
        reason: str = "",
        progress_summary: str = "",
    ) -> LaneExecution:
        """Apply one validated live-control transition to authoritative state."""

        with self._condition:
            execution = self._executions[task_id]
            if state == execution.state:
                return execution
            if state not in _CONTROL_TRANSITIONS.get(execution.state, frozenset()):
                raise LaneCoordinatorError(
                    f"Invalid task-state transition: {execution.state.value} -> {state.value}"
                )
            previous = execution.state
            if state in {
                LaneTaskState.WAITING,
                LaneTaskState.BLOCKED,
                LaneTaskState.PAUSED,
            }:
                supervised = self.execution_supervisor.store.get_task(task_id)
                if supervised.state in {SupervisorState.LEASED, SupervisorState.RUNNING}:
                    self.execution_supervisor.transition(
                        task_id,
                        SupervisorState.WAITING,
                        reason=reason,
                    )
            elif (
                state == LaneTaskState.RUNNING
                and execution.supervisor_attempt_id
                and execution.supervisor_lease_token
            ):
                self.execution_supervisor.resume_running(
                    task_id,
                    attempt_id=execution.supervisor_attempt_id,
                    lease_token=execution.supervisor_lease_token,
                )
            execution.state = state
            execution.updated_at = _iso()
            if reason:
                execution.error = reason
            if progress_summary:
                execution.progress_summary = progress_summary
            execution.lane_history.append({
                "lane_id": execution.owning_lane.value,
                "state": state.value,
                "previous_state": previous.value,
                "reason": reason,
                "at": execution.updated_at,
            })
            task = self.taskboard.get_task(execution.taskboard_task_id)
            mapped_task_status = {
                LaneTaskState.QUEUED: TaskStatus.QUEUED,
                LaneTaskState.RUNNING: TaskStatus.IN_PROGRESS,
                LaneTaskState.WAITING: TaskStatus.WAITING_FOR_TOOLS,
                LaneTaskState.BLOCKED: TaskStatus.BLOCKED,
                LaneTaskState.CANCELLED: TaskStatus.CANCELLED,
            }.get(state)
            if mapped_task_status is not None and task.status != mapped_task_status:
                self.taskboard.update_status(
                    execution.taskboard_task_id,
                    mapped_task_status,
                    reason=reason if mapped_task_status == TaskStatus.BLOCKED else None,
                )
            self._persist_locked()
        if state in _CONTROL_TERMINAL_STATES or state in {LaneTaskState.PAUSED, LaneTaskState.BLOCKED}:
            self.lock_manager.release_task(task_id)
        self.emit(
            f"task.{state.value}",
            task_id=task_id,
            lane_id=execution.owning_lane,
            previous_state=previous.value,
            reason=reason,
        )
        with self._condition:
            self._condition.notify_all()
        return execution

    def list_tasks(self, *, active_only: bool = False, session_id: str = "") -> tuple[LaneExecution, ...]:
        with self._condition:
            rows = tuple(
                execution for execution in self._executions.values()
                if (not active_only or execution.state not in _CONTROL_TERMINAL_STATES)
                and (not session_id or execution.session_id == session_id)
            )
        return tuple(sorted(rows, key=lambda item: (PRIORITY_ORDER[item.priority], item.created_at, item.task_id)))

    def inspect_task(self, task_id: str) -> LaneExecution:
        try:
            return self._executions[task_id]
        except KeyError as exc:
            raise LaneCoordinatorError(f"Unknown gateway task: {task_id}") from exc

    def resume_checkpoint(
        self,
        task_id: str,
        *,
        decision: RecoveryDecision,
        session_id: str,
    ) -> LaneReservation:
        """Requeue the exact checkpoint selected by a validated model decision."""
        execution = self.inspect_task(task_id)
        durable = self.execution_supervisor.store.get_task(task_id)
        if not durable.checkpoint_id or durable.checkpoint_id != decision.resume_checkpoint_id:
            raise LaneCoordinatorError(
                "model-selected checkpoint does not match the durable task checkpoint"
            )
        try:
            self.execution_supervisor.resume_checkpoint(task_id)
        except ExecutionSupervisorError as exc:
            raise LaneCoordinatorError(
                f"checkpoint recovery validation failed; no retry was executed: {exc}"
            ) from exc
        return self._retry_existing_task(
            execution,
            decision=decision,
            session_id=session_id,
            event_type="lane.checkpoint_resumed",
            checkpoint_id=decision.resume_checkpoint_id,
        )

    def retry_task(
        self,
        task_id: str,
        *,
        decision: RecoveryDecision,
        session_id: str,
    ) -> LaneReservation:
        """Requeue the exact stopped task selected by a validated model decision."""
        if decision.action is not RecoveryAction.RETRY or not decision.same_task_retry_authorized:
            raise LaneCoordinatorError(
                "same-task retry requires an explicit authorized retry decision"
            )
        execution = self.inspect_task(task_id)
        if execution.state not in {
            LaneTaskState.FAILED,
            LaneTaskState.INTERRUPTED,
            LaneTaskState.TIMED_OUT,
            LaneTaskState.BUDGET_EXHAUSTED,
            LaneTaskState.REJECTED,
        }:
            raise LaneCoordinatorError("model-selected task is not in a retryable stopped state")
        return self._retry_existing_task(
            execution,
            decision=decision,
            session_id=session_id,
            event_type="lane.task_retried",
            checkpoint_id="",
        )

    def replan_task(
        self,
        task_id: str,
        *,
        decision: RecoveryDecision,
        session_id: str,
    ) -> LaneReservation:
        """Requeue the same stopped task after a model-selected plan revision."""
        if decision.action is not RecoveryAction.REPLAN:
            raise LaneCoordinatorError("same-task replan requires an explicit replan decision")
        execution = self.inspect_task(task_id)
        if execution.state not in {
            LaneTaskState.FAILED,
            LaneTaskState.INTERRUPTED,
            LaneTaskState.TIMED_OUT,
            LaneTaskState.BUDGET_EXHAUSTED,
            LaneTaskState.REJECTED,
        }:
            raise LaneCoordinatorError("model-selected task is not in a replannable stopped state")
        return self._retry_existing_task(
            execution,
            decision=decision,
            session_id=session_id,
            event_type="lane.task_replanned",
            checkpoint_id="",
        )

    def _retry_existing_task(
        self,
        execution: LaneExecution,
        *,
        decision: RecoveryDecision,
        session_id: str,
        event_type: str,
        checkpoint_id: str,
    ) -> LaneReservation:
        task_id = execution.task_id
        try:
            scheduled = self.execution_supervisor.retry(task_id, decision)
        except ExecutionSupervisorError as exc:
            raise LaneCoordinatorError(
                f"same-task recovery validation failed; no retry was executed: {exc}"
            ) from exc
        retry_at = scheduled.retry_not_before
        if retry_at is not None:
            delay = max(0.0, (retry_at - _now()).total_seconds())
            if delay > self.contracts[execution.owning_lane].timeout_seconds:
                raise LaneCoordinatorError("checkpoint retry backoff exceeds the lane timeout")
            if delay:
                time.sleep(delay)
        try:
            self.execution_supervisor.release_retry(task_id)
        except ExecutionSupervisorError as exc:
            raise LaneCoordinatorError(
                f"checkpoint retry could not be released: {exc}"
            ) from exc
        with self._condition:
            execution.state = LaneTaskState.QUEUED
            execution.session_id = session_id
            execution.worker_id = ""
            execution.supervisor_attempt_id = ""
            execution.supervisor_lease_token = ""
            execution.error = ""
            execution.updated_at = _iso()
            execution.lane_history.append(
                {
                    "lane_id": execution.owning_lane.value,
                    "state": "queued",
                    "at": execution.updated_at,
                    "reason": decision.reason,
                    "checkpoint_id": checkpoint_id,
                    "recovery_decision_id": decision.decision_id,
                }
            )
            task = self.taskboard.get_task(execution.taskboard_task_id)
            if task.status is TaskStatus.FAILED:
                self.taskboard.reopen(task.task_id, reason=decision.reason)
            self.taskboard.add_decision(task.task_id, decision.decision_id)
            self._persist_locked()
        self.emit(
            event_type,
            task_id=task_id,
            lane_id=execution.owning_lane,
            checkpoint_id=checkpoint_id,
            recovery_decision_id=decision.decision_id,
        )
        return LaneReservation(execution)

    def pause(self, task_id: str, *, reason: str = "paused by coordinator") -> LaneExecution:
        return self.transition(task_id, LaneTaskState.PAUSED, reason=reason)

    def finalize_budget_overrun(
        self, decision: BudgetOverrunFinalizationDecision,
    ) -> LaneExecution:
        """Project only the validated supervisor finalization decision into the lane."""
        supervised = self.execution_supervisor.finalize_budget_overrun(decision)
        if supervised.state is ExecutionState.COMPLETED:
            return self.reconcile_authoritative_completion(decision.task_id)
        with self._condition:
            execution = self._executions[decision.task_id]
            if supervised.state is ExecutionState.PENDING_BUDGET_DECISION:
                execution.state = LaneTaskState.PENDING_BUDGET_DECISION
                self.taskboard.update_status(execution.taskboard_task_id, TaskStatus.NEEDS_REVIEW)
            else:
                execution.state = LaneTaskState.QUEUED
                self.taskboard.update_status(execution.taskboard_task_id, TaskStatus.QUEUED)
            execution.verification_state["budget_overrun"] = dict(supervised.budget_overrun)
            execution.updated_at = _iso()
            self._persist_locked()
        self.emit(
            "budget.overrun_finalized", task_id=decision.task_id,
            lane_id=execution.owning_lane, decision_id=decision.decision_id,
            action=decision.action.value,
        )
        return execution

    def reconcile_authoritative_completion(self, task_id: str) -> LaneExecution:
        """Repair the lane projection from an already completed supervisor record.

        This is deliberately deterministic: it does not make a provider call or
        authorize a result. It only exposes the supervisor completion that was
        already durably finalized by a validated decision.
        """
        supervised = self.execution_supervisor.store.get_task(task_id)
        if supervised.state is not ExecutionState.COMPLETED:
            raise LaneCoordinatorError(
                "authoritative completion reconciliation requires a completed supervisor task"
            )
        manifest = self.execution_supervisor.store.artifact_manifest(task_id) or {}
        with self._condition:
            execution = self._executions[task_id]
            self.taskboard.project_supervisor_completion(
                execution.taskboard_task_id,
                supervisor_task=supervised,
                verification_evidence={
                    "result_id": supervised.result_id,
                    "verification": manifest.get("verification"),
                    "artefacts": manifest.get("artefacts", []),
                },
            )
            execution.state = LaneTaskState.COMPLETED
            execution.error = ""
            execution.verification_state["budget_overrun"] = dict(supervised.budget_overrun)
            execution.updated_at = _iso()
            self._persist_locked()
        self.emit(
            "lane.supervisor_completion_reconciled",
            task_id=task_id,
            lane_id=execution.owning_lane,
        )
        return execution

    def resume(self, task_id: str) -> LaneExecution:
        return self.transition(task_id, LaneTaskState.QUEUED, reason="resumed by coordinator")

    def cancel_task(self, task_id: str, *, reason: str = "cancelled by coordinator") -> LaneExecution:
        execution = self.inspect_task(task_id)
        if execution.state in _CONTROL_TERMINAL_STATES:
            return execution
        self.transition(task_id, LaneTaskState.CANCELLING, reason=reason)
        self.execution_supervisor.cancel(task_id, reason=reason, propagate=False)
        self._stop_supervisor_heartbeats(task_id)
        execution.cancellation_state.update({"requested_at": _iso(), "reason": reason})
        result = self.transition(task_id, LaneTaskState.CANCELLED, reason=reason)
        try:
            self.taskboard.update_status(result.taskboard_task_id, TaskStatus.CANCELLED)
        except Exception:
            pass
        return result

    def cancel_tree(self, task_id: str, *, reason: str = "task tree cancelled") -> tuple[str, ...]:
        descendants: list[LaneExecution] = []
        pending = [task_id]
        while pending:
            parent = pending.pop()
            children = [item for item in self._executions.values() if item.parent_task_id == parent]
            descendants.extend(children)
            pending.extend(item.task_id for item in children)
        cancelled: list[str] = []
        for execution in reversed(descendants):
            if execution.state not in _CONTROL_TERMINAL_STATES:
                self.cancel_task(execution.task_id, reason=reason)
                cancelled.append(execution.task_id)
        if self.inspect_task(task_id).state not in _CONTROL_TERMINAL_STATES:
            self.cancel_task(task_id, reason=reason)
            cancelled.append(task_id)
        return tuple(cancelled)

    def reprioritize(self, task_id: str, priority: LanePriority) -> LaneExecution:
        with self._condition:
            execution = self._executions[task_id]
            if execution.state not in {LaneTaskState.QUEUED, LaneTaskState.WAITING, LaneTaskState.PAUSED}:
                raise LaneCoordinatorError("Only queued, waiting, or paused tasks can be reprioritized")
            execution.priority = priority
            execution.updated_at = _iso()
            self._persist_locked()
            self._condition.notify_all()
        self.emit("task.reprioritized", task_id=task_id, lane_id=execution.owning_lane, priority=priority.value)
        return execution

    def mark_blocked(self, task_id: str, *, reason: str) -> LaneExecution:
        if not reason.strip():
            raise LaneCoordinatorError("A blocked task requires an actionable reason")
        return self.transition(task_id, LaneTaskState.BLOCKED, reason=reason)

    def attach_evidence(self, task_id: str, evidence: Mapping[str, Any]) -> LaneExecution:
        with self._condition:
            execution = self._executions[task_id]
            execution.evidence.append({**dict(evidence), "attached_at": _iso()})
            execution.updated_at = _iso()
            self._persist_locked()
        self.emit("task.evidence_attached", task_id=task_id, lane_id=execution.owning_lane)
        return execution

    def request_verification(self, task_id: str, *, level: str = "standard") -> LaneExecution:
        execution = self.transition(task_id, LaneTaskState.VERIFYING, reason=f"verification requested: {level}")
        execution.verification_state.update({"level": level, "requested_at": _iso()})
        with self._condition:
            self._persist_locked()
        self.emit("verification.started", task_id=task_id, lane_id=execution.owning_lane, level=level)
        return execution

    def budget_usage(self, *, task_id: str = "", session_id: str = "") -> dict[str, Any]:
        rows = [
            item for item in self._executions.values()
            if (not task_id or item.task_id == task_id) and (not session_id or item.session_id == session_id)
        ]
        return {
            "reserved_tokens": sum(item.budget.reserved_tokens for item in rows),
            "consumed_tokens": sum(item.budget.consumed_tokens for item in rows),
            "estimated_cost": (sum(item.budget.estimated_cost for item in rows) if rows and all(item.budget.estimated_cost_known for item in rows) else None),
            "actual_cost": (sum(item.budget.actual_cost for item in rows) if rows and all(item.budget.actual_cost_known for item in rows) else None),
            "task_count": len(rows),
        }

    def handoff(self, handoff: LaneHandoff) -> LaneExecution:
        with self._condition:
            execution = self._executions[handoff.task_id]
            if execution.owning_lane != handoff.source_lane:
                raise LaneHandoffError("handoff source does not own the task")
            source_contract = self.contracts[handoff.source_lane]
            if handoff.target_lane not in source_contract.handoff_targets:
                raise LaneHandoffError(f"handoff {handoff.source_lane.value} -> {handoff.target_lane.value} is not allowed")
            target = self.contracts[handoff.target_lane]
            self._assert_capacity(target, execution.model, exclude_task_id=execution.task_id)
            if source_contract.token_budget is not None and execution.budget.consumed_tokens >= source_contract.token_budget:
                raise LaneBudgetError("task budget is exhausted; handoff was not started")
            execution.state = LaneTaskState.HANDOFF
            execution.handoffs.append(handoff)
            execution.changed_files = self.canonical_paths(handoff.changed_files)
            execution.verification_state.update(handoff.verification_state)
            self._persist_locked()
        self.emit("lane.handoff_requested", task_id=execution.task_id, lane_id=handoff.source_lane, target_lane=handoff.target_lane.value)
        self.lock_manager.release_task(execution.task_id)
        with self._condition:
            execution.owning_lane = handoff.target_lane
            execution.state = LaneTaskState.QUEUED
            execution.target_files = self.canonical_paths(handoff.changed_files or execution.target_files)
            execution.lane_history.append({"lane_id": handoff.target_lane.value, "state": "queued", "at": _iso(), "reason": handoff.reason})
            execution.updated_at = _iso()
            self._persist_locked()
        self.start(LaneReservation(execution))
        self.emit("lane.handoff_completed", task_id=execution.task_id, lane_id=handoff.target_lane, source_lane=handoff.source_lane.value)
        return execution

    def authorize_tool(self, task_id: str, tool_name: str) -> frozenset[str]:
        execution = self._executions[task_id]
        if execution.state != LaneTaskState.RUNNING:
            raise LaneCoordinatorError(f"Task {task_id} is not running")
        try:
            capabilities = validate_tool_permission(
                self.contracts[execution.owning_lane], tool_name,
                task_capabilities=tuple(execution.capabilities),
            )
        except PermissionError as exc:
            self.emit("lane.permission_denied", task_id=task_id, lane_id=execution.owning_lane, tool_name=tool_name, reason=str(exc))
            raise
        held_modes = {lease.mode for lease in self._locks.values() if lease.task_id == task_id}
        if "repository_write" in capabilities and not held_modes.intersection(
            {LockMode.FILE_WRITE, LockMode.REPOSITORY_WRITE, LockMode.WORKSPACE_WRITE}
        ):
            self.emit("lane.permission_denied", task_id=task_id, lane_id=execution.owning_lane, tool_name=tool_name, reason="required write lock is not held")
            raise LaneCoordinatorError(f"Tool {tool_name} requires a gateway write lock")
        if "repository_read" in capabilities and not held_modes.intersection(
            {LockMode.FILE_READ, LockMode.FILE_WRITE, LockMode.REPOSITORY_READ, LockMode.REPOSITORY_WRITE, LockMode.WORKSPACE_WRITE}
        ):
            self.emit("lane.permission_denied", task_id=task_id, lane_id=execution.owning_lane, tool_name=tool_name, reason="required read lock is not held")
            raise LaneCoordinatorError(f"Tool {tool_name} requires a gateway repository lock")
        lane_limit = self.contracts[execution.owning_lane].token_budget
        if (
            (lane_limit is not None and execution.budget.consumed_tokens >= lane_limit)
            or execution.budget.consumed_tokens >= execution.budget.reserved_tokens
        ):
            self.emit("lane.budget_exhausted", task_id=task_id, lane_id=execution.owning_lane)
            raise LaneBudgetError("task token budget is exhausted")
        return capabilities

    def can_create_subagent(self, task_id: str, *, child_lane: LaneId, target_files: Sequence[str] = ()) -> None:
        execution = self._executions[task_id]
        contract = self.contracts[execution.owning_lane]
        if not contract.can_create_subagents:
            raise LaneCoordinatorError(f"Lane {execution.owning_lane.value} cannot create subagents")
        children = [item for item in self._executions.values() if item.parent_task_id == task_id and item.state in ACTIVE_LANE_STATES]
        if len(children) >= contract.max_subagents:
            raise LaneCapacityError("parent task subagent limit reached")
        if child_lane not in contract.handoff_targets and child_lane != execution.owning_lane:
            raise LaneCoordinatorError(f"Lane {child_lane.value} is not an allowed child lane")
        if child_lane == LaneId.CODING and set(self.canonical_paths(target_files)).intersection(execution.target_files):
            raise LaneCoordinatorError("overlapping coding subagent files require an isolated worktree or exclusive lock")

    def recover(self, *, supervise: bool = True) -> None:
        if supervise:
            self.execution_supervisor.reconnect_tree()
            self.execution_supervisor.recover()
        for execution in self._executions.values():
            supervised = self.execution_supervisor.store.get_task_or_none(execution.task_id)
            if supervised is None:
                continue
            if supervised.state == SupervisorState.COMPLETED:
                manifest = self.execution_supervisor.store.artifact_manifest(execution.task_id) or {}
                board_task = self.taskboard.get_task(execution.taskboard_task_id)
                if board_task.status is not TaskStatus.DONE:
                    self.taskboard.project_supervisor_completion(
                        execution.taskboard_task_id,
                        supervisor_task=supervised,
                        verification_evidence={
                            "result_id": supervised.result_id,
                            "verification": manifest.get("verification"),
                            "artefacts": manifest.get("artefacts", []),
                        },
                    )
                execution.state = LaneTaskState.COMPLETED
            elif supervised.state == SupervisorState.BUDGET_EXHAUSTED:
                execution.state = LaneTaskState.BUDGET_EXHAUSTED
                execution.error = supervised.failure_reason or "execution budget exhausted"
        self.lock_manager.recover_stale()
        interrupted_task_ids: list[str] = []
        with self._condition:
            # Waiters have no live caller after process restart. Active read-only
            # executions remain available for explicit gateway revalidation;
            # abandoned queue positions must not block new work.
            self._waiters = []
            for execution in self._executions.values():
                if execution.state not in ACTIVE_LANE_STATES:
                    continue
                heartbeat = datetime.fromisoformat(execution.last_heartbeat)
                worker_parts = execution.worker_id.split(":")
                worker_missing = (
                    len(worker_parts) >= 2
                    and worker_parts[0] == "gateway"
                    and worker_parts[1].isdigit()
                    and not process_exists(int(worker_parts[1]))
                )
                expired = heartbeat + timedelta(seconds=self.contracts[execution.owning_lane].timeout_seconds + 30) < _now()
                if worker_missing or expired:
                    contract = self.contracts[execution.owning_lane]
                    execution.state = (
                        LaneTaskState.INTERRUPTED
                        if contract.requires_write_access
                        else LaneTaskState.QUEUED
                    )
                    execution.worker_id = ""
                    execution.error = (
                        "worker interrupted; repository mutations require revalidation"
                        if contract.requires_write_access
                        else "read-only work requeued after worker interruption"
                    )
                    execution.updated_at = _iso()
                    interrupted_task_ids.append(execution.task_id)
                    self.emit(
                        "lane.failed" if contract.requires_write_access else "lane.queued",
                        task_id=execution.task_id,
                        lane_id=execution.owning_lane,
                        reason="worker_interrupted",
                    )
            self._persist_locked()
        for task_id in interrupted_task_ids:
            self.lock_manager.release_task(task_id)

    def _recover_stale_locked(self) -> None:
        now = _now()
        expired = [key for key, lease in self._locks.items() if datetime.fromisoformat(lease.expires_at) <= now]
        for key in expired:
            lease = self._locks.pop(key)
            self.emit("lock.expired", task_id=lease.task_id, lane_id=None, lock_id=lease.lease_id)

    def _assert_capacity(self, contract: LaneContract, model: str, *, exclude_task_id: str = "") -> None:
        active = [
            item
            for item in self._executions.values()
            if item.task_id != exclude_task_id
            and item.state in ACTIVE_LANE_STATES
            # A queued record has not acquired a supervisor lease or started a
            # worker. It is durable recovery state, not active execution, and
            # must not permanently consume capacity after a process exit.
            and item.state != LaneTaskState.QUEUED
            # Completion verification is durable review work after the runtime
            # lease and gateway resource lock have been released. It must
            # prevent final success and equivalent duplicate execution, but it
            # must not occupy a worker, lane, or provider execution slot.
            and item.state != LaneTaskState.VERIFYING
        ]
        if len(active) >= self.global_worker_limit:
            raise LaneCapacityError("global gateway worker limit reached")
        if sum(item.owning_lane == contract.lane_id for item in active) >= contract.max_concurrent_jobs:
            raise LaneCapacityError(f"lane {contract.lane_id.value} concurrency limit reached")
        if model and model in self.provider_limits and sum(item.model == model for item in active) >= self.provider_limits[model]:
            raise LaneCapacityError(f"model/provider concurrency limit reached for {model}")

    def _next_waiter_id(self) -> str:
        now = _now()

        def score(item: dict[str, Any]) -> tuple[int, int]:
            priority = LanePriority(str(item["priority"]))
            created = datetime.fromisoformat(str(item["created_at"]))
            age_promotions = max(0, int((now - created).total_seconds() // 30))
            return (max(0, PRIORITY_ORDER[priority] - age_promotions), int(item["sequence"]))

        return str(min(self._waiters, key=score)["waiter_id"]) if self._waiters else ""

    def _assert_budget(self, contract: LaneContract, session_id: str, requested: LaneBudget) -> None:
        if (
            (contract.token_budget is not None and requested.reserved_tokens > contract.token_budget)
            or (
                contract.cost_budget is not None
                and (not requested.estimated_cost_known or requested.estimated_cost > contract.cost_budget)
            )
        ):
            raise LaneBudgetError(f"requested budget exceeds {contract.lane_id.value} lane limit")
        active = [
            item
            for item in self._executions.values()
            if item.state in ACTIVE_LANE_STATES
            and item.state != LaneTaskState.VERIFYING
        ]
        if self.session_token_budget is not None:
            used = sum(item.budget.reserved_tokens for item in active if item.session_id == session_id)
            if used + requested.reserved_tokens > self.session_token_budget:
                raise LaneBudgetError("session token budget exhausted")
        if self.global_token_budget is not None:
            used = sum(item.budget.reserved_tokens for item in active)
            if used + requested.reserved_tokens > self.global_token_budget:
                raise LaneBudgetError("global token budget exhausted")

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for item in payload.get("executions", []):
            try:
                budget = LaneBudget(**item.pop("budget"))
                handoffs = []
                for raw in item.pop("handoffs", []):
                    raw["source_lane"] = LaneId(raw["source_lane"])
                    raw["target_lane"] = LaneId(raw["target_lane"])
                    raw["budget_consumed"] = LaneBudget(**raw.get("budget_consumed", {}))
                    handoffs.append(LaneHandoff(**raw))
                item["owning_lane"] = LaneId(item["owning_lane"])
                item["state"] = LaneTaskState(item["state"])
                item["priority"] = LanePriority(item["priority"])
                execution = LaneExecution(budget=budget, handoffs=handoffs, **item)
                # Raw credentials are deliberately memory-only. Older state
                # files are scrubbed on the next atomic persistence.
                execution.supervisor_lease_token = ""
                self._executions[execution.task_id] = execution
            except (KeyError, TypeError, ValueError):
                continue
        self._waiters = [dict(item) for item in payload.get("waiters", []) if isinstance(item, dict)]
        self._wait_sequence = max((int(item.get("sequence", 0)) for item in self._waiters), default=0)
        lock_rows = payload.get("locks", [])
        if self.locks_path.exists():
            try:
                lock_rows = json.loads(self.locks_path.read_text(encoding="utf-8")).get("locks", [])
            except (OSError, json.JSONDecodeError, AttributeError):
                lock_rows = []
        for item in lock_rows:
            try:
                item["mode"] = LockMode(item["mode"])
                lease = LockLease(**item)
                self._locks[lease.lease_id] = lease
            except (KeyError, TypeError, ValueError):
                continue

    def _migrate_legacy_supervisor_records(self) -> None:
        """Preserve active pre-supervisor lanes without unsafe replay."""
        pending = [
            item for item in self._executions.values()
            if item.state in ACTIVE_LANE_STATES
            and self.execution_supervisor.store.get_task_or_none(item.task_id) is None
        ]
        while pending:
            progressed = False
            for execution in list(pending):
                if (
                    execution.parent_task_id
                    and self.execution_supervisor.store.get_task_or_none(execution.parent_task_id) is None
                ):
                    continue
                contract = self.contracts[execution.owning_lane]
                classification = (
                    SideEffectClassification.READ_ONLY
                    if execution.owning_lane in {LaneId.RESEARCH, LaneId.REVIEW, LaneId.VERIFY}
                    and not contract.requires_write_access
                    else SideEffectClassification.UNKNOWN
                )
                supervised = self.execution_supervisor.create_task(
                    task_id=execution.task_id,
                    parent_task_id=execution.parent_task_id,
                    task_type=execution.task_type,
                    assigned_agent=f"lane:{execution.owning_lane.value}",
                    assigned_model=execution.model,
                    runtime_provider=execution.provider,
                    workspace_path=self.root,
                    routing_decision_id=(
                        execution.routing_decision_id
                        or f"legacy-lane-state:{execution.task_id}"
                    ),
                    side_effect_classification=classification,
                    dependency_task_ids=(
                        self.taskboard.get_task(execution.taskboard_task_id).depends_on
                        if execution.taskboard_task_id in self.taskboard.tasks
                        else ()
                    ),
                    token_budget=execution.budget.reserved_tokens or None,
                    estimated_cost=(execution.budget.estimated_cost if execution.budget.estimated_cost_known else None),
                    model_context_window=execution.budget.model_context_window,
                    model_max_output_tokens=execution.budget.model_max_output_tokens,
                    estimate_confidence=execution.budget.estimate_confidence,
                    estimate_source=execution.budget.estimate_source,
                    monetary_budget=contract.cost_budget,
                )
                if classification == SideEffectClassification.READ_ONLY:
                    self.execution_supervisor.queue(supervised.task_id)
                    execution.state = LaneTaskState.QUEUED
                    execution.worker_id = ""
                    execution.error = "legacy read-only task queued for supervised recovery"
                else:
                    self.execution_supervisor.transition(
                        supervised.task_id,
                        SupervisorState.FAILED,
                        reason=(
                            "Legacy active task has no supervisor lease and may have mutated state; "
                            "manual recovery decision is required"
                        ),
                    )
                    execution.state = LaneTaskState.INTERRUPTED
                    execution.worker_id = ""
                    execution.error = "legacy task requires manual recovery decision"
                pending.remove(execution)
                progressed = True
            if not progressed:
                # Preserve unresolved taskboard lineage in the legacy store and
                # stop; never invent a replacement parent or fallback route.
                break

    def _persist_locked(self) -> None:
        executions = []
        for item in self._executions.values():
            serialized = asdict(item)
            serialized["supervisor_lease_token"] = ""
            executions.append(serialized)
        payload = {
            "schema_version": 2, "updated_at": _iso(),
            "executions": executions,
            "waiters": list(self._waiters),
            "locks": [],
        }
        _atomic_write_json(self.state_path, payload)

    @contextmanager
    def _process_state_lock(self) -> Iterator[None]:
        self.guard_path.parent.mkdir(parents=True, exist_ok=True)
        with self.guard_path.open("a+b") as handle:
            _lock_process_file(handle)
            try:
                yield
            finally:
                _unlock_process_file(handle)

    def _load_locks_file_locked(self) -> None:
        if not self.locks_path.exists():
            self._locks = {}
            return
        try:
            rows = json.loads(self.locks_path.read_text(encoding="utf-8")).get("locks", [])
        except (OSError, json.JSONDecodeError, AttributeError):
            rows = []
        loaded: dict[str, LockLease] = {}
        for raw in rows:
            try:
                item = dict(raw)
                item["mode"] = LockMode(item["mode"])
                lease = LockLease(**item)
                loaded[lease.lease_id] = lease
            except (KeyError, TypeError, ValueError):
                continue
        self._locks = loaded

    def _persist_locks_file_locked(self) -> None:
        payload = {
            "schema_version": 1,
            "updated_at": _iso(),
            "locks": [asdict(item) for item in self._locks.values()],
        }
        _atomic_write_json(self.locks_path, payload)

    @property
    def executions(self) -> tuple[LaneExecution, ...]:
        with self._condition:
            return tuple(self._executions.values())


__all__ = [
    "GatewayLockManager", "LaneBudget", "LaneBudgetError", "LaneCapacityError",
    "LaneCoordinator", "LaneCoordinatorError", "LaneExecution", "LaneHandoff",
    "LaneHandoffError", "LaneLockTimeout", "LaneReservation",
]
