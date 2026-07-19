"""Synchronous orchestration adapter backed by canonical memory operations.

This adapter preserves the existing multi-agent call shape in external mode.
It keeps only turn-local indexes in process; durable writes always target the
selected external backend and never the internal repository.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Coroutine, TypeVar

from mana_agent.memory.models import (
    MemoryContent,
    MemoryScope,
    MemorySearchRequest,
    MemoryWriteRequest,
)
from mana_agent.multi_agent.core.types import AgentRole
from mana_agent.services.memory_service import (
    TASK_ACTIVE_STATUSES,
    VOLATILE_TOOL_ARG_KEYS,
    WRITE_TOOL_NAMES,
    AgentDecisionMemoryRecord,
    FileReadMemoryRecord,
    ScopedMemoryBundle,
    TaskMemoryRecord,
    ToolExecutionMemoryRecord,
    VerificationMemoryRecord,
    normalize_file_path,
    normalize_text,
    normalize_tool_name,
    stable_hash,
    task_fingerprint,
    utc_iso,
)

T = TypeVar("T")


def run_sync(operation: Coroutine[Any, Any, T]) -> T:
    """Run a backend coroutine from both ordinary and event-loop threads."""
    running_loop = False
    try:
        asyncio.get_running_loop()
        running_loop = True
    except RuntimeError:
        pass
    if not running_loop:
        return asyncio.run(operation)
    result: list[T] = []
    failure: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(operation))
        except BaseException as exc:  # propagated unchanged to the caller
            failure.append(exc)

    thread = threading.Thread(target=runner, name="mana-memory-sync", daemon=True)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    return result[0]


def _summary(value: Any, *, max_chars: int = 800) -> str:
    return " ".join(str(value or "").split())[:max_chars]


class ExternalRuntimeMemory:
    def __init__(
        self,
        *,
        service: Any,
        root: Path,
        user_id: str,
        workspace_id: str,
        repository_id: str,
        session_id: str,
    ) -> None:
        self.service = service
        self.root = root
        self.user_id = user_id
        self.workspace_id = workspace_id
        self.repository_id = repository_id
        self.session_id = session_id
        self.task_records: dict[str, TaskMemoryRecord] = {}
        self.tool_executions: dict[str, ToolExecutionMemoryRecord] = {}
        self.agent_decisions: list[AgentDecisionMemoryRecord] = []
        self.verifications: list[VerificationMemoryRecord] = []
        self.queue_fingerprints: dict[str, str] = {}
        self.project_memory: list[dict[str, Any]] = []
        self.workspace_memory: list[dict[str, Any]] = []

    def _scope(self, *, agent_id: str = "", task_id: str = "") -> MemoryScope:
        return MemoryScope(
            user_id=self.user_id,
            agent_id=agent_id,
            session_id=self.session_id,
            workspace_id=self.workspace_id,
            repository_id=self.repository_id,
            task_id=task_id,
        )

    def _write(
        self,
        *,
        kind: str,
        content: str,
        agent_id: str = "",
        task_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        request = MemoryWriteRequest(
            content=MemoryContent(content),
            scope=self._scope(agent_id=agent_id, task_id=task_id),
            metadata={"mana_kind": kind, **dict(metadata or {})},
        )
        run_sync(self.service.add(request))

    def normalize_task(
        self,
        *,
        goal: str,
        action_type: str = "task",
        target_files: list[str] | None = None,
        expected_output: str = "",
        repository_ids: list[str] | None = None,
    ) -> tuple[str, str]:
        normalized_goal = normalize_text(goal)
        return normalized_goal, task_fingerprint(
            normalized_goal=normalized_goal,
            action_type=action_type,
            target_files=target_files,
            expected_output=expected_output,
            root=self.root,
            repository_ids=repository_ids,
        )

    def remember_repository_fact(self, fact: str) -> None:
        text = _summary(fact, max_chars=1200)
        payload = {"fact": text, "repository_id": self.repository_id}
        if text and payload not in self.project_memory:
            self._write(kind="repository_fact", content=text)
            self.project_memory.append(payload)

    def remember_workspace_fact(self, fact: str) -> None:
        text = _summary(fact, max_chars=1200)
        payload = {"fact": text, "workspace_id": self.workspace_id}
        if text and payload not in self.workspace_memory:
            self._write(kind="workspace_fact", content=text)
            self.workspace_memory.append(payload)

    def find_duplicate_task(self, fingerprint: str) -> TaskMemoryRecord | None:
        local = next(
            (
                record
                for record in self.task_records.values()
                if record.fingerprint == fingerprint and record.reuse_allowed
            ),
            None,
        )
        if local is not None:
            return local
        matches = run_sync(
            self.service.search(
                MemorySearchRequest(
                    query="task",
                    scope=self._scope(),
                    limit=1,
                    metadata={"mana_kind": "task", "fingerprint": fingerprint},
                )
            )
        )
        if not matches:
            return None
        try:
            payload = json.loads(matches[0].content.text)
            record = TaskMemoryRecord(**payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        self.task_records[record.task_id] = record
        return record

    def register_task(
        self,
        *,
        task_id: str,
        normalized_goal: str,
        fingerprint: str,
        assigned_agent_id: str = "",
        parent_agent_id: str = "",
        related_files: list[str] | None = None,
        repository_ids: list[str] | None = None,
    ) -> TaskMemoryRecord:
        duplicate = self.find_duplicate_task(fingerprint)
        record = TaskMemoryRecord(
            task_id=task_id,
            normalized_goal=normalized_goal,
            fingerprint=fingerprint,
            assigned_agent_id=assigned_agent_id,
            parent_agent_id=parent_agent_id,
            related_files=[
                normalize_file_path(item, root=self.root) for item in related_files or []
            ],
            duplicate_of=duplicate.task_id if duplicate else None,
            workspace_id=self.workspace_id,
            session_id=self.session_id,
            repository_ids=list(repository_ids or [self.repository_id]),
        )
        self._write(
            kind="task",
            content=json.dumps(asdict(record), sort_keys=True),
            agent_id=assigned_agent_id,
            task_id=task_id,
            metadata={"fingerprint": fingerprint},
        )
        self.task_records[task_id] = record
        return record

    def update_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        result_summary: str = "",
    ) -> None:
        record = self.task_records.get(task_id)
        if record is None:
            return
        if status:
            record.status = status
        if result_summary:
            record.result_summary = _summary(result_summary)
        record.updated_at = utc_iso()
        self._write(
            kind="task_update",
            content=json.dumps(asdict(record), sort_keys=True),
            agent_id=record.assigned_agent_id,
            task_id=task_id,
        )

    def register_queue_item(
        self,
        *,
        queue_item_id: str,
        fingerprint: str,
    ) -> tuple[bool, str | None]:
        existing = self.queue_fingerprints.get(fingerprint)
        if existing:
            return False, existing
        self.queue_fingerprints[fingerprint] = queue_item_id
        return True, None

    def read_file_with_memory(
        self,
        *,
        file_path: str,
        task_id: str,
        agent_id: str,
    ) -> tuple[str, FileReadMemoryRecord, bool]:
        resolved = (self.root / file_path).resolve()
        resolved.relative_to(self.root)
        content = resolved.read_text(encoding="utf-8", errors="replace")
        record = FileReadMemoryRecord(
            file_path=normalize_file_path(resolved, root=self.root),
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            mtime=resolved.stat().st_mtime,
            last_read_at=utc_iso(),
            last_read_by_agent_id=agent_id,
            task_id=task_id,
            content_summary=_summary(content, max_chars=1200),
            content=content,
        )
        return content, record, False

    def tool_cache_key(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        relevant_file_hashes: dict[str, str] | None = None,
    ) -> str:
        stable_args = {
            key: value
            for key, value in dict(args or {}).items()
            if key not in VOLATILE_TOOL_ARG_KEYS
        }
        return stable_hash(
            {
                "tool_name": normalize_tool_name(tool_name),
                "args": stable_args,
                "file_hashes": relevant_file_hashes or {},
            }
        )

    def get_reusable_tool_result(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        relevant_file_hashes: dict[str, str] | None = None,
    ) -> ToolExecutionMemoryRecord | None:
        normalized = normalize_tool_name(tool_name)
        if normalized in WRITE_TOOL_NAMES:
            return None
        key = self.tool_cache_key(
            tool_name=normalized,
            args=args,
            relevant_file_hashes=relevant_file_hashes,
        )
        record = self.tool_executions.get(key)
        return record if record and record.reusable and record.status == "ok" else None

    def record_tool_execution(self, **kwargs: Any) -> ToolExecutionMemoryRecord:
        normalized = normalize_tool_name(kwargs["tool_name"])
        key = self.tool_cache_key(
            tool_name=normalized,
            args=kwargs["args"],
            relevant_file_hashes=kwargs.get("relevant_file_hashes"),
        )
        record = ToolExecutionMemoryRecord(
            tool_name=normalized,
            normalized_args_hash=key,
            task_id=str(kwargs["task_id"]),
            agent_id=str(kwargs["agent_id"]),
            status=str(kwargs["status"]),
            result_summary=_summary(kwargs.get("result_summary")),
            started_at=utc_iso(),
            completed_at=utc_iso(),
            reusable=normalized not in WRITE_TOOL_NAMES,
            result=dict(kwargs.get("result") or {}),
        )
        self._write(
            kind="tool_execution",
            content=json.dumps(asdict(record), sort_keys=True),
            agent_id=record.agent_id,
            task_id=record.task_id,
        )
        self.tool_executions[key] = record
        return record

    def record_decision(self, **kwargs: Any) -> None:
        record = AgentDecisionMemoryRecord(
            agent_id=str(kwargs["agent_id"]),
            task_id=str(kwargs["task_id"]),
            decision_type=str(kwargs["decision_type"]),
            input_summary=_summary(kwargs.get("input_summary")),
            memory_used=list(kwargs.get("memory_used") or []),
            decision=_summary(kwargs.get("decision")),
            reason=_summary(kwargs.get("reason")),
        )
        self._write(
            kind="agent_decision",
            content=json.dumps(asdict(record), sort_keys=True),
            agent_id=record.agent_id,
            task_id=record.task_id,
        )
        self.agent_decisions.append(record)

    def record_verification(self, **kwargs: Any) -> VerificationMemoryRecord:
        record = VerificationMemoryRecord(
            task_id=str(kwargs["task_id"]),
            verifier_agent_id=str(kwargs["verifier_agent_id"]),
            checked_files=[
                normalize_file_path(item, root=self.root)
                for item in kwargs.get("checked_files") or []
            ],
            tests_run=list(kwargs.get("tests_run") or []),
            result=str(kwargs["result"]),
            findings=list(kwargs.get("findings") or []),
        )
        self._write(
            kind="verification",
            content=json.dumps(asdict(record), sort_keys=True),
            agent_id=record.verifier_agent_id,
            task_id=record.task_id,
        )
        self.verifications.append(record)
        return record

    def build_bundle(
        self,
        *,
        agent_id: str,
        agent_role: AgentRole | str,
        task_id: str,
        parent_task_id: str | None = None,
        target_files: list[str] | None = None,
    ) -> ScopedMemoryBundle:
        _ = target_files
        role = agent_role.value if isinstance(agent_role, AgentRole) else str(agent_role)
        privilege = self.privilege_for_role(role)
        current = self.task_records.get(task_id)
        active = [
            asdict(item)
            for item in self.task_records.values()
            if item.status in TASK_ACTIVE_STATUSES
            and (not current or item.task_id != current.task_id)
        ]
        completed = [
            asdict(item) for item in self.task_records.values() if item.status == "completed"
        ]
        return ScopedMemoryBundle(
            bundle_id=f"memory_bundle_{stable_hash({'agent_id': agent_id, 'task_id': task_id, 'at': utc_iso()})}",
            agent_id=agent_id,
            agent_role=role,
            privilege_level=privilege,
            task_id=task_id,
            parent_task_id=parent_task_id,
            relevant_project_memory=(
                list(self.workspace_memory[-4:]) + list(self.project_memory[-8:])
                if privilege in {"full", "planner"}
                else []
            ),
            relevant_task_memory=[
                asdict(item)
                for item in self.task_records.values()
                if item.task_id in {task_id, parent_task_id}
            ],
            previous_tool_results=[
                asdict(item)
                for item in self.tool_executions.values()
                if item.task_id == task_id
            ],
            active_related_tasks=active if privilege not in {"coding", "tool", "verifier"} else [],
            completed_related_tasks=completed if privilege not in {"coding", "tool"} else [],
            routing_hints=(
                [f"duplicate_of:{current.duplicate_of}"]
                if current and current.duplicate_of
                else []
            ),
            verification_history=[
                asdict(item) for item in self.verifications if item.task_id == task_id
            ],
            workspace_id=self.workspace_id,
            session_id=self.session_id,
            repository_ids=[self.repository_id],
        )

    @staticmethod
    def privilege_for_role(role: str) -> str:
        normalized = str(role or "").lower()
        if normalized in {"main", "head_decision"}:
            return "full"
        if normalized in {"planner", "taskboard", "queue", "queue_manager"}:
            return "planner" if normalized == "planner" else normalized.replace("_manager", "")
        if normalized in {"coding", "tool_worker", "tools_manager"}:
            return "coding" if normalized == "coding" else "tool"
        if normalized in {"verifier", "reviewer"}:
            return normalized
        return "restricted"
