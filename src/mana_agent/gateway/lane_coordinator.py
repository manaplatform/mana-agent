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
    LaneTaskState.RUNNING: frozenset({LaneTaskState.WAITING, LaneTaskState.BLOCKED, LaneTaskState.CANCELLING, LaneTaskState.VERIFYING, LaneTaskState.COMPLETED, LaneTaskState.FAILED}),
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
                self.coordinator.emit("lock.waiting", task_id=task_id, lane_id=None, mode=mode.value)
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
        self._condition = threading.Condition(threading.RLock())
        self._executions: dict[str, LaneExecution] = {}
        self._locks: dict[str, LockLease] = {}
        self._waiters: list[dict[str, Any]] = []
        self._wait_sequence = 0
        self.lock_manager = GatewayLockManager(self)
        self.state_path = workspace_dir(self.taskboard.store.workspace_id) / "gateway" / "lane_coordinator.json"
        self.locks_path = self.state_path.with_name("lane_locks.json")
        self.guard_path = self.state_path.with_name("lane_coordinator.lock")
        self._load()
        self.recover()

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
        estimated_cost: float = 0.0,
        capabilities: Sequence[str] = (),
        routing_decision_id: str = "",
        provider: str = "",
        task_type: str = "single",
        taskboard_task_id: str | None = None,
    ) -> LaneReservation:
        contract = self.contracts[lane_id]
        if contract.requires_repository and not repository_id:
            raise LaneCoordinatorError(f"Lane {lane_id.value} requires a repository identity")
        if contract.allowed_models and model not in contract.allowed_models:
            raise LaneCoordinatorError(f"Model {model or '<unset>'} is not allowed for lane {lane_id.value}")
        files = self.canonical_paths(target_files)
        fingerprint = _stable_hash({
            "intent": " ".join(normalized_intent.lower().split()), "repository_id": repository_id,
            "workspace_id": workspace_id, "session_id": session_id, "target_files": files,
            "lane": lane_id.value, "parent_task_id": parent_task_id,
        })
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
                            "workspace_id": active.workspace_id, "session_id": active.session_id,
                            "target_files": active.target_files, "lane": active.owning_lane.value,
                            "parent_task_id": active.parent_task_id,
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
                    self.emit("lane.queued", task_id=waiter["waiter_id"], lane_id=lane_id, reason="capacity")
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
                estimated_cost=max(0.0, estimated_cost),
            )
            self._assert_budget(contract, session_id, budget)
            if parent_task_id:
                parent = self._executions.get(parent_task_id)
                if parent is None:
                    raise LaneBudgetError("parent task budget is unavailable")
                remaining = max(0, parent.budget.reserved_tokens - parent.budget.consumed_tokens)
                if budget.reserved_tokens > remaining:
                    raise LaneBudgetError("child reservation exceeds the parent task's remaining budget")
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
                )
                self.taskboard.add_files_to_inspect(task.task_id, files)
            else:
                task = self.taskboard.create_task(
                    title=f"{contract.display_name}: {normalized_intent[:100]}", user_request=normalized_intent,
                    normalized_goal=normalized_intent, owner_agent_id=f"lane:{lane_id.value}",
                    related_files=files, action_type=f"lane:{lane_id.value}", workspace_id=workspace_id,
                    session_id=session_id, repository_ids=[repository_id] if repository_id else [],
                    primary_repository_id=repository_id,
                )
            task_id = task.task_id
            execution = LaneExecution(
                task_id=task_id,
                root_task_id=(root_task_id or (self._executions[parent_task_id].root_task_id if parent_task_id else task_id)),
                parent_task_id=parent_task_id,
                owning_lane=lane_id, state=LaneTaskState.QUEUED, normalized_intent=normalized_intent,
                repository_id=repository_id, workspace_id=workspace_id, session_id=session_id,
                target_files=files, priority=selected_priority, budget=budget,
                taskboard_task_id=task.task_id, model=model, capabilities=list(capabilities),
                routing_decision_id=routing_decision_id, provider=provider, task_type=task_type,
                lane_history=[{"lane_id": lane_id.value, "state": "queued", "at": _iso()}],
            )
            self._executions[task_id] = execution
            self.taskboard.update_status(task_id, TaskStatus.ROUTED)
            self.taskboard.update_status(task_id, TaskStatus.QUEUED)
            self._persist_locked()
            self.emit("lane.queued", task_id=task_id, lane_id=lane_id)
            self.emit("task.created", task_id=task_id, lane_id=lane_id, parent_task_id=parent_task_id)
            self.emit("model.assigned", task_id=task_id, lane_id=lane_id, routing_decision_id=routing_decision_id, provider=provider, model=model)
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
        with self._condition:
            execution.state = LaneTaskState.RUNNING
            execution.worker_id = f"gateway:{os.getpid()}:{threading.get_ident()}"
            execution.last_heartbeat = execution.updated_at = _iso()
            execution.lane_history.append({"lane_id": execution.owning_lane.value, "state": "running", "at": execution.updated_at})
            task_status = self.taskboard.get_task(execution.taskboard_task_id).status
            if task_status in {TaskStatus.QUEUED, TaskStatus.ROUTED, TaskStatus.WAITING_FOR_TOOLS}:
                self.taskboard.update_status(execution.taskboard_task_id, TaskStatus.IN_PROGRESS)
            self._persist_locked()
        self.emit("lane.started", task_id=execution.task_id, lane_id=execution.owning_lane)
        return execution

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
        actual_cost: float = 0.0,
        verification_state: Mapping[str, Any] | None = None,
        error: str = "",
    ) -> LaneExecution:
        with self._condition:
            execution = self._executions[task_id]
            execution.state = state
            execution.changed_files = self.canonical_paths(changed_files)
            execution.budget.consumed_input_tokens += max(0, consumed_input_tokens)
            execution.budget.consumed_output_tokens += max(0, consumed_output_tokens)
            execution.budget.actual_cost += max(0.0, actual_cost)
            if execution.parent_task_id and execution.parent_task_id in self._executions:
                parent = self._executions[execution.parent_task_id]
                parent.budget.consumed_input_tokens += max(0, consumed_input_tokens)
                parent.budget.consumed_output_tokens += max(0, consumed_output_tokens)
                parent.budget.actual_cost += max(0.0, actual_cost)
                parent.updated_at = _iso()
            execution.verification_state.update(dict(verification_state or {}))
            execution.error = error
            execution.updated_at = execution.last_heartbeat = _iso()
            mapped_status = {
                LaneTaskState.COMPLETED: TaskStatus.DONE,
                LaneTaskState.CANCELLED: TaskStatus.CANCELLED,
            }.get(state, TaskStatus.FAILED)
            reason = error or f"lane execution ended as {state.value}"
            self.taskboard.update_status(
                execution.taskboard_task_id,
                mapped_status,
                reason=reason if mapped_status == TaskStatus.FAILED else None,
            )
            self._persist_locked()
        self.lock_manager.release_task(task_id)
        event = {
            LaneTaskState.COMPLETED: "lane.completed", LaneTaskState.CANCELLED: "lane.cancelled",
            LaneTaskState.BUDGET_EXHAUSTED: "lane.budget_exhausted",
        }.get(state, "lane.failed")
        self.emit(event, task_id=task_id, lane_id=execution.owning_lane, status="success" if state == LaneTaskState.COMPLETED else "error")
        self.emit("resource.released", task_id=task_id, lane_id=execution.owning_lane)
        with self._condition:
            self._condition.notify_all()
        return execution

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

    def pause(self, task_id: str, *, reason: str = "paused by coordinator") -> LaneExecution:
        return self.transition(task_id, LaneTaskState.PAUSED, reason=reason)

    def resume(self, task_id: str) -> LaneExecution:
        return self.transition(task_id, LaneTaskState.QUEUED, reason="resumed by coordinator")

    def cancel_task(self, task_id: str, *, reason: str = "cancelled by coordinator") -> LaneExecution:
        execution = self.inspect_task(task_id)
        if execution.state in _CONTROL_TERMINAL_STATES:
            return execution
        self.transition(task_id, LaneTaskState.CANCELLING, reason=reason)
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
            "estimated_cost": sum(item.budget.estimated_cost for item in rows),
            "actual_cost": sum(item.budget.actual_cost for item in rows),
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
            if execution.budget.consumed_tokens >= source_contract.token_budget:
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
        if (
            execution.budget.consumed_tokens >= self.contracts[execution.owning_lane].token_budget
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

    def recover(self) -> None:
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
        active = [item for item in self._executions.values() if item.task_id != exclude_task_id and item.state in ACTIVE_LANE_STATES]
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
        if requested.reserved_tokens > contract.token_budget or requested.estimated_cost > contract.cost_budget:
            raise LaneBudgetError(f"requested budget exceeds {contract.lane_id.value} lane limit")
        active = [item for item in self._executions.values() if item.state in ACTIVE_LANE_STATES]
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

    def _persist_locked(self) -> None:
        payload = {
            "schema_version": 1, "updated_at": _iso(),
            "executions": [asdict(item) for item in self._executions.values()],
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
