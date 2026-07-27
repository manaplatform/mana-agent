from __future__ import annotations

from pathlib import Path
from typing import Any

from mana_agent.multi_agent.core.ids import new_queue_job_id
from mana_agent.multi_agent.core.types import ExecutionContext, QueueJob, QueueJobStatus, QueueJobType, enrich_event_identity, utc_now
from mana_agent.multi_agent.routing.hierarchy import HierarchyPolicy
from mana_agent.memory import MultiAgentMemoryService, normalize_file_path, stable_hash
from mana_agent.multi_agent.queue.locks import LockTable
from mana_agent.multi_agent.queue.scheduler import next_job
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.multi_agent.tools.tool_manager import ToolsManager
from mana_agent.execution.manager import ExecutionManager


class QueueManager:
    def __init__(
        self,
        root: str | Path = ".",
        *,
        taskboard: TaskBoard | None = None,
        tools_manager: ToolsManager | None = None,
        memory_service: MultiAgentMemoryService | None = None,
        hierarchy_policy: HierarchyPolicy | None = None,
        default_worker_agent_id: str = "agent_tool_worker_0001",
        execution_manager: ExecutionManager | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.memory_service = memory_service
        self.taskboard = taskboard or TaskBoard(root, memory_service=memory_service)
        self.tools_manager = tools_manager or ToolsManager(
            root, memory_service=memory_service, execution_manager=execution_manager
        )
        self.hierarchy_policy = hierarchy_policy or HierarchyPolicy(taskboard=self.taskboard)
        self.default_worker_agent_id = default_worker_agent_id
        self.jobs: dict[str, QueueJob] = {}
        self.locks = LockTable()

    def enqueue(self, job: QueueJob | None = None, **kwargs: Any) -> QueueJob:
        if job is None:
            task_id = str(kwargs.get("task_id") or "")
            requested_by = str(kwargs.get("requested_by_agent_id") or "")
            self.hierarchy_policy.assert_can_create_queue_job(requested_by, task_id=task_id)
            if not kwargs.get("approved_by_agent_id"):
                kwargs["approved_by_agent_id"] = self._approver_for_task(task_id) or kwargs.get("requested_by_agent_id")
            if not kwargs.get("purpose"):
                kwargs["purpose"] = f"Run {kwargs.get('job_type', 'tool')} for task {kwargs.get('task_id')}"
            kwargs.setdefault("assigned_worker_agent_id", self.default_worker_agent_id)
            self.hierarchy_policy.assert_can_assign_worker(
                requested_by,
                str(kwargs.get("assigned_worker_agent_id") or ""),
                task_id=task_id,
            )
            kwargs.setdefault("root_task_id", self._root_task_id(task_id))
            kwargs.setdefault("parent_task_id", self._parent_task_id(task_id))
            task_scope = self._task_scope(task_id)
            kwargs.setdefault("workspace_id", task_scope["workspace_id"])
            kwargs.setdefault("session_id", task_scope["session_id"])
            kwargs.setdefault("primary_repository_id", task_scope["primary_repository_id"])
            kwargs.setdefault("repository_ids", task_scope["repository_ids"])
            kwargs.setdefault("execution_repo_root", task_scope["execution_repo_root"])
            kwargs.setdefault("managed_workspace_id", task_scope["managed_workspace_id"])
            kwargs.setdefault("parent_agent_id", kwargs.get("requested_by_agent_id"))
            kwargs.setdefault("agent_role", "tool_worker")
            kwargs.setdefault("args_summary", self._args_summary(kwargs.get("payload") or {}))
            kwargs.setdefault("budget_reserved", self._default_budget_for(kwargs.get("job_type")))
            kwargs.setdefault("budget_reserved_ms", 30_000)
            kwargs.setdefault("fingerprint", self._fingerprint_from_kwargs(kwargs))
            kwargs.setdefault("related_files", self._related_files_from_kwargs(kwargs))
            # Parallel coding tasks lock per managed worktree, never share one checkout lock.
            if kwargs.get("requires_write_lock") and not kwargs.get("lock_key"):
                execution_root = str(kwargs.get("execution_repo_root") or "").strip()
                if execution_root:
                    kwargs["lock_key"] = f"worktree:{execution_root}"
            if self.memory_service is not None and not kwargs.get("memory_bundle_id"):
                bundle = self.memory_service.build_bundle(
                    agent_id=str(kwargs.get("requested_by_agent_id") or "agent_queue"),
                    agent_role="queue",
                    task_id=str(kwargs.get("task_id") or ""),
                    target_files=list(kwargs.get("related_files") or []),
                )
                kwargs["memory_bundle_id"] = bundle.bundle_id
            # Stamp execution root into payload for workers that only read payload.
            payload = dict(kwargs.get("payload") or {})
            if kwargs.get("execution_repo_root") and not payload.get("repo_root") and not payload.get("execution_repo_root"):
                payload["execution_repo_root"] = kwargs["execution_repo_root"]
                payload["repo_root"] = kwargs["execution_repo_root"]
                kwargs["payload"] = payload
            job = QueueJob(job_id=new_queue_job_id(), **kwargs)
        elif not job.fingerprint:
            self.hierarchy_policy.assert_can_create_queue_job(job.requested_by_agent_id, task_id=job.task_id)
            if not job.assigned_worker_agent_id:
                job.assigned_worker_agent_id = self.default_worker_agent_id
            if not job.approved_by_agent_id:
                job.approved_by_agent_id = self._approver_for_task(job.task_id) or job.requested_by_agent_id
            if not job.parent_agent_id:
                job.parent_agent_id = job.requested_by_agent_id
            if not job.agent_role:
                job.agent_role = "tool_worker"
            self.hierarchy_policy.assert_can_assign_worker(
                job.requested_by_agent_id,
                job.assigned_worker_agent_id,
                task_id=job.task_id,
            )
            if not job.root_task_id:
                job.root_task_id = self._root_task_id(job.task_id)
            if not job.parent_task_id:
                job.parent_task_id = self._parent_task_id(job.task_id)
            if not job.args_summary:
                job.args_summary = self._args_summary(job.payload)
            if not job.budget_reserved:
                job.budget_reserved = self._default_budget_for(job.job_type)
            if not job.budget_reserved_ms:
                job.budget_reserved_ms = 30_000
            job.fingerprint = self._fingerprint_for_job(job)
        self._reserve_budget(job)
        if self.memory_service is not None:
            accepted, existing_id = self.memory_service.register_queue_item(
                queue_item_id=job.job_id,
                fingerprint=job.fingerprint,
            )
            if not accepted:
                job.status = QueueJobStatus.CANCELLED
                job.duplicate_of = existing_id
                job.result_summary = f"skipped duplicate_of:{existing_id}"
                self.jobs[job.job_id] = job
                self.taskboard.add_queue_job(job.task_id, job.job_id)
                self.taskboard.add_evidence(job.task_id, f"Queue job {job.job_id} skipped as duplicate_of:{existing_id}")
                self._bump_memory_status(job.task_id, duplicate_of=existing_id)
                return job
        self.jobs[job.job_id] = job
        self.taskboard.add_queue_job(job.task_id, job.job_id)
        if job.job_type in {QueueJobType.SHELL, QueueJobType.RUN_TESTS, QueueJobType.RUN_LINT}:
            self.taskboard.add_verification_queue_job(job.task_id, job.job_id)
        self.taskboard.add_evidence(job.task_id, f"Queue job {job.job_id} queued for {job.job_type.value}: {job.purpose}")
        return job

    def claim_next(self, agent_id: str) -> QueueJob | None:
        job = next_job(list(self.jobs.values()))
        if job is None:
            return None
        job.status = QueueJobStatus.CLAIMED
        job.updated_at = utc_now()
        job.assigned_worker_agent_id = job.assigned_worker_agent_id or agent_id
        job.agent_id = agent_id
        if agent_id.startswith("subagent_"):
            job.subagent_id = agent_id
        elif not job.subagent_id and str(job.assigned_worker_agent_id or "").startswith("subagent_"):
            job.subagent_id = job.assigned_worker_agent_id
        job.payload = {**job.payload, "claimed_by_agent_id": agent_id}
        return job

    def run_next(self, worker_agent_id: str | None = None) -> QueueJob | None:
        worker_id = worker_agent_id or self.default_worker_agent_id
        job = self.claim_next(worker_id)
        if job is None:
            return None
        self.hierarchy_policy.assert_can_execute_tool(
            worker_id,
            job.job_type.value,
            task_id=job.task_id,
            queue_job_id=job.job_id,
            assigned_worker_agent_id=job.assigned_worker_agent_id,
        )
        if job.lock_key:
            lock_key = job.lock_key
        elif job.requires_write_lock:
            execution_root = str(job.execution_repo_root or "").strip()
            lock_key = f"worktree:{execution_root}" if execution_root else f"repository:{job.primary_repository_id}"
        else:
            lock_key = f"job:{job.job_id}"
        lock = self.locks.lock_for(lock_key)
        with lock:
            job.status = QueueJobStatus.RUNNING
            job.started_at = utc_now()
            job.updated_at = utc_now()
            self._record_tool_event(job, event_type="tool.started", worker_agent_id=worker_id)
            result = self.tools_manager.execute_job(job)
            job.result = result.result
            job.error = result.error
            job.result_summary = "ok" if result.ok else str(result.error or "failed")
            job.status = QueueJobStatus.DONE if result.ok else QueueJobStatus.FAILED
            job.ended_at = utc_now()
            job.token_usage = max(1, len(str(result.result or result.error or "")) // 4)
            if isinstance(job.result, dict):
                changed = job.result.get("changed_files") or job.result.get("files_changed") or []
                if isinstance(changed, list):
                    job.changed_files = [str(item) for item in changed]
                job.cache_status = "hit" if job.result.get("cache_hit") else "miss"
            job.updated_at = utc_now()
            self._record_tool_event(job, event_type="tool.finished", worker_agent_id=worker_id)
            self.taskboard.record_budget(
                job.task_id,
                {
                    "queue_job_id": job.job_id,
                    "agent_id": worker_id,
                    "requested_by_agent_id": job.requested_by_agent_id,
                    "budget_used_tokens": job.token_usage,
                    "budget_used_ms": self._duration_ms(job),
                    "action": "queue_job_finished",
                },
            )
            if isinstance(result.result, dict) and result.result.get("cache_hit"):
                self._bump_memory_status(job.task_id, cache_hit=True, file_read=job.job_type in {QueueJobType.REPO_READ, QueueJobType.REPO_BATCH_READ})
        if job.job_type in {QueueJobType.APPLY_PATCH} and job.result:
            changed = job.result.get("changed_files") or job.result.get("files_changed") or []
            if isinstance(changed, list):
                self.taskboard.add_files_touched(job.task_id, [str(item) for item in changed])
        if self.memory_service is not None:
            self.memory_service.update_task(job.task_id, status=job.status.value, result_summary=job.result_summary or "")
        return job

    def run_until_idle(self, max_jobs: int | None = None, *, worker_agent_id: str | None = None) -> list[QueueJob]:
        ran: list[QueueJob] = []
        while max_jobs is None or len(ran) < max_jobs:
            job = self.run_next(worker_agent_id=worker_agent_id)
            if job is None:
                break
            ran.append(job)
        return ran

    def get_job(self, job_id: str) -> QueueJob:
        return self.jobs[job_id]

    def cancel_job(self, job_id: str) -> None:
        job = self.jobs[job_id]
        job.status = QueueJobStatus.CANCELLED
        job.updated_at = utc_now()

    def jobs_for_task(self, task_id: str) -> list[QueueJob]:
        return [job for job in self.jobs.values() if job.task_id == task_id]

    def _fingerprint_from_kwargs(self, kwargs: dict[str, Any]) -> str:
        job_type = kwargs.get("job_type")
        tool = job_type.value if hasattr(job_type, "value") else str(job_type or "")
        return stable_hash(
            {
                "task_id": kwargs.get("task_id"),
                "tool": tool,
                "payload": kwargs.get("payload") or {},
                "related_files": kwargs.get("related_files") or self._related_files_from_kwargs(kwargs),
                "repository_ids": kwargs.get("repository_ids") or [],
            }
        )

    def _fingerprint_for_job(self, job: QueueJob) -> str:
        return stable_hash(
            {
                "task_id": job.task_id,
                "tool": job.job_type.value,
                "payload": job.payload,
                "related_files": job.related_files or self._related_files_for_payload(job.job_type, job.payload),
                "repository_ids": job.repository_ids,
            }
        )

    def _related_files_from_kwargs(self, kwargs: dict[str, Any]) -> list[str]:
        job_type = kwargs.get("job_type")
        payload = kwargs.get("payload") or {}
        return self._related_files_for_payload(job_type, payload)

    def _related_files_for_payload(self, job_type: Any, payload: dict[str, Any]) -> list[str]:
        kind = job_type.value if hasattr(job_type, "value") else str(job_type or "")
        root = Path(str(payload.get("execution_repo_root") or payload.get("repo_root") or self.root)).resolve()
        if kind == QueueJobType.REPO_READ.value and payload.get("path"):
            return [normalize_file_path(payload.get("path"), root=root)]
        if kind == QueueJobType.REPO_BATCH_READ.value:
            return [normalize_file_path(item, root=root) for item in payload.get("files") or payload.get("paths") or []]
        if kind == QueueJobType.APPLY_PATCH.value:
            return [normalize_file_path(payload.get("path"), root=root)] if payload.get("path") else []
        return []

    def _bump_memory_status(
        self,
        task_id: str,
        *,
        cache_hit: bool = False,
        file_read: bool = False,
        duplicate_of: str | None = None,
    ) -> None:
        try:
            task = self.taskboard.get_task(task_id)
        except KeyError:
            return
        status = dict(task.memory_status or {})
        if cache_hit:
            status["cache_hits"] = int(status.get("cache_hits") or 0) + 1
        if file_read:
            status["file_reads_reused"] = int(status.get("file_reads_reused") or 0) + 1
        if duplicate_of:
            status["duplicate_of"] = duplicate_of
        status["last_memory_check_at"] = utc_now()
        task.memory_status = status
        task.updated_at = utc_now()
        self.taskboard.save()

    def _root_task_id(self, task_id: str) -> str:
        try:
            return self.taskboard.get_task(task_id).root_task_id
        except KeyError:
            return task_id

    def _parent_task_id(self, task_id: str) -> str | None:
        try:
            return self.taskboard.get_task(task_id).parent_task_id
        except KeyError:
            return None

    def _task_scope(self, task_id: str) -> dict[str, Any]:
        try:
            task = self.taskboard.get_task(task_id)
        except KeyError:
            return {
                "workspace_id": "",
                "session_id": "",
                "primary_repository_id": "",
                "repository_ids": [],
                "execution_repo_root": "",
                "managed_workspace_id": "",
            }
        execution_root = str(getattr(task, "execution_repo_root", "") or getattr(task, "managed_worktree_path", "") or "").strip()
        return {
            "workspace_id": task.workspace_id,
            "session_id": task.session_id,
            "primary_repository_id": task.primary_repository_id,
            "repository_ids": list(task.repository_ids),
            "execution_repo_root": execution_root,
            "managed_workspace_id": str(getattr(task, "managed_workspace_id", "") or ""),
        }

    def _approver_for_task(self, task_id: str) -> str | None:
        try:
            task = self.taskboard.get_task(task_id)
        except KeyError:
            return None
        return task.supervisor_agent_id or task.owner_agent_id or task.approved_by_agent_id

    def _args_summary(self, payload: dict[str, Any]) -> str:
        text = str(payload or {})
        return text if len(text) <= 240 else text[:237] + "..."

    def _default_budget_for(self, job_type: Any) -> int:
        kind = job_type.value if hasattr(job_type, "value") else str(job_type or "")
        if kind in {QueueJobType.APPLY_PATCH.value, QueueJobType.SHELL.value, QueueJobType.RUN_TESTS.value, QueueJobType.RUN_LINT.value}:
            return 1200
        return 400

    def _reserve_budget(self, job: QueueJob) -> None:
        self.taskboard.record_budget(
            job.task_id,
            {
                "queue_job_id": job.job_id,
                "requested_by_agent_id": job.requested_by_agent_id,
                "approved_by_agent_id": job.approved_by_agent_id,
                "assigned_worker_agent_id": job.assigned_worker_agent_id,
                "budget_reserved_tokens": job.budget_reserved,
                "budget_reserved_ms": job.budget_reserved_ms,
                "action": "queue_job_reserved",
            },
        )

    def _record_tool_event(self, job: QueueJob, *, event_type: str, worker_agent_id: str) -> None:
        context = ExecutionContext(
            agent_id=worker_agent_id,
            subagent_id=job.subagent_id or (worker_agent_id if worker_agent_id.startswith("subagent_") else None),
            agent_role=job.agent_role or "tool_worker",
            parent_agent_id=job.parent_agent_id or job.requested_by_agent_id,
            requested_by_agent_id=job.requested_by_agent_id,
            queue_job_id=job.job_id,
            task_id=job.task_id,
            root_task_id=job.root_task_id or job.task_id,
            delegation_path=list(
                dict.fromkeys(
                    item
                    for item in [job.approved_by_agent_id, job.requested_by_agent_id, worker_agent_id]
                    if item
                )
            ),
            workspace_id=job.workspace_id or None,
            session_id=job.session_id or None,
            repository_id=job.primary_repository_id or None,
            managed_workspace_id=job.managed_workspace_id or None,
            execution_repo_root=job.execution_repo_root or None,
        )
        event = self.hierarchy_policy.approve_tool_event(
            enrich_event_identity(
                {
                "type": event_type,
                "approved_by_agent_id": job.approved_by_agent_id,
                "assigned_worker_agent_id": job.assigned_worker_agent_id,
                "tool_name": job.job_type.value,
                "tool_args": dict(job.payload),
                "budget_used": job.token_usage,
                "token_usage": job.token_usage,
                },
                context,
            )
        )
        self.taskboard.record_tool_event(job.task_id, event)

    def _duration_ms(self, job: QueueJob) -> int:
        if not job.started_at or not job.ended_at:
            return 0
        return max(0, int((job.ended_at - job.started_at).total_seconds() * 1000))
