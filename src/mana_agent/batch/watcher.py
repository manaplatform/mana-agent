"""Bounded provider polling using Mana's registered background-process runtime."""

from __future__ import annotations

import hashlib
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mana_agent.batch.manager import BatchLifecycleManager
from mana_agent.batch.operations import BatchOperationStore

logger = logging.getLogger(__name__)


@contextmanager
def watcher_lifecycle_lock(root: Path, process_root: Path):
    from mana_agent.execution_supervisor.store import _acquire_file_lock, _release_file_lock
    directory = process_root / "watcher_lifecycle"
    directory.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(str(root.resolve()).encode()).hexdigest() + ".lock"
    with (directory / name).open("a+b") as handle:
        _acquire_file_lock(handle)
        try:
            yield
        finally:
            _release_file_lock(handle)


def ensure_batch_watcher(root: Path, processes: Any) -> Any:
    root = root.resolve()
    singleton = "openai-batch:" + hashlib.sha256(str(root).encode()).hexdigest()
    with watcher_lifecycle_lock(root, processes.store.root):
        processes.recover_stale()
        for process in processes.list():
            if process.singleton_key == singleton and process.state in {"starting", "running"}:
                return process
        return processes.start("batch.watch", process_type="batch_watcher", arguments={"root": str(root)},
                               singleton_key=singleton, ownership="workspace", transient=False)


def transient_error(exc: Exception) -> bool:
    from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
    return isinstance(exc, (OSError, TimeoutError, ConnectionError, APIConnectionError,
                            APITimeoutError, RateLimitError, InternalServerError))


def poll_once(root: Path, manager: BatchLifecycleManager, resume: Any, failure: Any = None) -> bool:
    """Poll owned operations, then deliver ready wakes through the Supervisor."""
    operations = BatchOperationStore(manager.store)
    pending = operations.pending_runs(str(root.resolve()))
    retry_errors = []
    for run in pending:
        try:
            if run.get("final_delivery_pending") and not run.get("gateway_pending_operation"):
                resume(run["task_id"], run["last_operation_id"], run)
                continue
            operation = run.get("gateway_pending_operation") or run.get("pending_operation")
            if not operation:
                resume(run["task_id"], run.get("last_operation_id", ""), run)
                continue  # Supervisor decides whether an interrupted local phase is runnable.
            job = manager.poll_batch(operation["batch_id"])
            if job.status in {"results_imported", "verified", "failed", "expired", "cancelled"}:
                resume(run["task_id"], operation["operation_id"], run)
        except Exception as exc:
            if transient_error(exc):
                retry_errors.append(exc)
            elif failure:
                failure(run, exc)
            else:
                raise
    if retry_errors:
        raise retry_errors[0]
    return bool(operations.pending_runs(str(root.resolve())))


def watch(arguments: dict[str, str]) -> None:
    from mana_agent.gateway.chat_gateway import AgentChatGateway
    from mana_agent.gateway.config import ChatGatewayConfig
    root = Path(arguments["root"]).resolve()
    from mana_agent.execution.autonomous_controller import _publish_hub_event
    manager = BatchLifecycleManager(event_sink=_publish_hub_event)
    gateways: dict[str, Any] = {}
    backoff = 30

    def gateway_for(run: dict[str, Any]) -> Any:
        saved = run["resume"]
        session_id = saved["session_id"]
        if session_id not in gateways:
            gateways[session_id] = AgentChatGateway(root, config=ChatGatewayConfig(
                **saved.get("gateway_config", {}),
                session_id=session_id, model=run["model"], coding_model=run["model"],
                extra={"batch_watcher_process": True},
            ))
        return gateways[session_id]

    def resume(task_id: str, operation_id: str, run: dict[str, Any]) -> None:
        gateway = gateway_for(run)
        supervisor = gateway._lane_coordinator.execution_supervisor
        task = supervisor.get_task(task_id)
        if task.state.value == "completed_pending_verification" and run.get("final_delivery_pending"):
            # Recover a crash between result escrow and the existing verifier.
            # Verification is a separate phase, never another model/tool wave.
            task = supervisor.verify_completion(task_id)
        if task.state.value == "completed" and run.get("final_delivery_pending"):
            gateway.deliver_batch_result(task_id)
            return
        if task.state.value in {"cancelled", "failed", "completed", "budget_exhausted", "recovery_review_required"}:
            operation = run.get("gateway_pending_operation") or run.get("pending_operation")
            if task.state.value == "cancelled" and operation:
                job = manager.store.get_job(operation["batch_id"])
                if job and not job.is_terminal() and job.status != "cancelling":
                    manager.cancel_batch(job.batch_id)
            run["status"] = task.state.value
            run["final_delivery_pending"] = False
            BatchOperationStore(manager.store).save_run(task_id, run)
            return
        if task.waiting_inbox_item_id:
            return
        claimed_ready = task.state.value == "queued" and task.execution_policy.required
        if not claimed_ready and (task.state.value != "waiting" or task.wake_up_reference != operation_id):
            return  # Another worker owns the wake; its lease is authoritative.
        gateway.resume_batch_task(task_id, operation_id)

    def failure(run: dict[str, Any], exc: Exception) -> None:
        from mana_agent.gateway.lane_models import LaneTaskState
        from mana_agent.utils.redaction import redact_secrets
        gateway = gateway_for(run)
        reason = str(redact_secrets(str(exc)))
        current_task = gateway._lane_coordinator.execution_supervisor.get_task(run["task_id"])
        if current_task.state.value in {"running", "leased", "checkpointing"}:
            return  # Polling cannot fail another worker's active execution.
        gateway._lane_coordinator.ensure_recoverable_execution(run["task_id"])
        gateway._finish_lane(run["task_id"], state=LaneTaskState.FAILED, error=reason)
        saved = run["resume"]
        gateway._append_session_message(saved["session_id"], role="assistant",
            content="Batch execution could not continue: " + reason, turn_id=saved["turn_id"],
            message_id=f"batch_failure_{run['task_id']}", metadata={"goal_satisfied": False})
        current = BatchOperationStore(manager.store).load_run(run["task_id"])
        current["status"], current["final_delivery_pending"] = "failed", False
        BatchOperationStore(manager.store).save_run(run["task_id"], current)

    try:
        while True:
            try:
                if not poll_once(root, manager, resume, failure):
                    from mana_agent.background.commands import idle_retirement
                    retire = idle_retirement.get()
                    if retire is None:
                        raise RuntimeError("Batch watcher requires the registered background worker lifecycle")
                    if retire(lambda: not BatchOperationStore(manager.store).pending_runs(str(root))):
                        return
                    continue  # A submitter published another operation before retirement.
                backoff = 30
            except Exception as exc:
                if not transient_error(exc):
                    raise
                logger.warning("Batch watcher transport unavailable: %s", type(exc).__name__)
                backoff = min(300, backoff * 2)
            time.sleep(backoff)
    finally:
        for gateway in gateways.values():
            # Dispose worker-owned handles without closing the user's session.
            coding = getattr(gateway, "_coding_agent", None)
            if coding is not None and hasattr(coding, "close"):
                coding.close()
