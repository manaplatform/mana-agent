from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mana_agent.multi_agent.core.types import AgentRole

logger = logging.getLogger(__name__)


TASK_ACTIVE_STATUSES = {"pending", "running", "queued", "new", "planning", "in_progress"}
WRITE_TOOL_NAMES = {"apply_patch", "write_file", "create_file", "delete_file", "edit_file", "multi_edit_file"}
VOLATILE_TOOL_ARG_KEYS = {"claimed_by_agent_id", "approved_by_agent_id", "memory_bundle_id"}


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"[^\w\s./:-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_file_path(path: Any, *, root: str | Path | None = None) -> str:
    text = str(path or "").strip().replace("\\", "/")
    if not text:
        return ""
    base = Path(root).resolve() if root is not None else None
    try:
        resolved = (base / text).resolve() if base and not Path(text).is_absolute() else Path(text).resolve()
        if base is not None:
            try:
                return resolved.relative_to(base).as_posix()
            except ValueError:
                return resolved.as_posix()
        return resolved.as_posix()
    except Exception:
        return text.lstrip("./")


def normalize_tool_name(tool_name: Any) -> str:
    return normalize_text(tool_name).replace("-", "_")


def stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def task_fingerprint(
    *,
    normalized_goal: str,
    action_type: str,
    target_files: list[str] | None = None,
    expected_output: str = "",
    root: str | Path | None = None,
) -> str:
    return stable_hash(
        {
            "goal": normalize_text(normalized_goal),
            "action_type": normalize_text(action_type),
            "target_files": sorted(normalize_file_path(item, root=root) for item in target_files or [] if str(item).strip()),
            "expected_output": normalize_text(expected_output),
        }
    )


def _summary(value: Any, *, max_chars: int = 800) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)[:max_chars]


@dataclass
class TaskMemoryRecord:
    task_id: str
    normalized_goal: str
    fingerprint: str
    status: str = "pending"
    assigned_agent_id: str = ""
    parent_agent_id: str = ""
    related_files: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_iso)
    updated_at: str = field(default_factory=utc_iso)
    result_summary: str = ""
    duplicate_of: str | None = None
    reuse_allowed: bool = True


@dataclass
class FileReadMemoryRecord:
    file_path: str
    content_hash: str
    mtime: float
    last_read_at: str
    last_read_by_agent_id: str
    task_id: str
    content_summary: str
    full_content_cache_ref: str | None = None
    changed_since_last_read: bool = False
    content: str = ""

    @property
    def id(self) -> str:
        return f"file:{self.file_path}:{self.content_hash[:12]}"


@dataclass
class ToolExecutionMemoryRecord:
    tool_name: str
    normalized_args_hash: str
    task_id: str
    agent_id: str
    status: str
    result_summary: str
    started_at: str
    completed_at: str
    reusable: bool = True
    result: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentDecisionMemoryRecord:
    agent_id: str
    task_id: str
    decision_type: str
    input_summary: str
    memory_used: list[str]
    decision: str
    reason: str
    created_at: str = field(default_factory=utc_iso)


@dataclass
class VerificationMemoryRecord:
    task_id: str
    verifier_agent_id: str
    checked_files: list[str]
    tests_run: list[str]
    result: str
    findings: list[str]
    created_at: str = field(default_factory=utc_iso)


@dataclass
class ScopedMemoryBundle:
    bundle_id: str
    agent_id: str
    agent_role: str
    privilege_level: str
    task_id: str
    parent_task_id: str | None = None
    relevant_project_memory: list[dict[str, Any]] = field(default_factory=list)
    relevant_task_memory: list[dict[str, Any]] = field(default_factory=list)
    relevant_file_cache: list[dict[str, Any]] = field(default_factory=list)
    previous_tool_results: list[dict[str, Any]] = field(default_factory=list)
    duplicate_task_candidates: list[dict[str, Any]] = field(default_factory=list)
    active_related_tasks: list[dict[str, Any]] = field(default_factory=list)
    completed_related_tasks: list[dict[str, Any]] = field(default_factory=list)
    routing_hints: list[str] = field(default_factory=list)
    verification_history: list[dict[str, Any]] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        payload = asdict(self)
        return "Scoped memory bundle:\n" + json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


class MultiAgentMemoryService:
    def __init__(self, *, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()
        self.task_records: dict[str, TaskMemoryRecord] = {}
        self.file_reads: dict[str, FileReadMemoryRecord] = {}
        self.tool_executions: dict[str, ToolExecutionMemoryRecord] = {}
        self.agent_decisions: list[AgentDecisionMemoryRecord] = []
        self.verifications: list[VerificationMemoryRecord] = []
        self.queue_fingerprints: dict[str, str] = {}
        self.project_memory: list[dict[str, Any]] = []

    def normalize_task(
        self,
        *,
        goal: str,
        action_type: str = "task",
        target_files: list[str] | None = None,
        expected_output: str = "",
    ) -> tuple[str, str]:
        normalized_goal = normalize_text(goal)
        return normalized_goal, task_fingerprint(
            normalized_goal=normalized_goal,
            action_type=action_type,
            target_files=target_files,
            expected_output=expected_output,
            root=self.root,
        )

    def find_duplicate_task(self, fingerprint: str) -> TaskMemoryRecord | None:
        for record in self.task_records.values():
            if record.fingerprint == fingerprint and record.reuse_allowed:
                return record
        return None

    def register_task(
        self,
        *,
        task_id: str,
        normalized_goal: str,
        fingerprint: str,
        assigned_agent_id: str = "",
        parent_agent_id: str = "",
        related_files: list[str] | None = None,
    ) -> TaskMemoryRecord:
        duplicate = self.find_duplicate_task(fingerprint)
        record = TaskMemoryRecord(
            task_id=task_id,
            normalized_goal=normalized_goal,
            fingerprint=fingerprint,
            status="pending",
            assigned_agent_id=assigned_agent_id,
            parent_agent_id=parent_agent_id,
            related_files=[normalize_file_path(item, root=self.root) for item in related_files or []],
            duplicate_of=duplicate.task_id if duplicate else None,
        )
        self.task_records[task_id] = record
        logger.info("[memory] duplicate_task_hit task_id=%s duplicate_of=%s", task_id, record.duplicate_of or "")
        return record

    def update_task(self, task_id: str, *, status: str | None = None, result_summary: str = "") -> None:
        record = self.task_records.get(task_id)
        if record is None:
            return
        if status:
            record.status = status
        if result_summary:
            record.result_summary = _summary(result_summary)
        record.updated_at = utc_iso()

    def register_queue_item(self, *, queue_item_id: str, fingerprint: str) -> tuple[bool, str | None]:
        existing = self.queue_fingerprints.get(fingerprint)
        if existing:
            logger.info("[memory] queue_duplicate_rejected queue_item_id=%s existing=%s", queue_item_id, existing)
            return False, existing
        self.queue_fingerprints[fingerprint] = queue_item_id
        return True, None

    def _file_meta(self, path: str | Path) -> tuple[str, float, str]:
        resolved = (self.root / str(path)).resolve() if not Path(str(path)).is_absolute() else Path(str(path)).resolve()
        resolved.relative_to(self.root)
        content = resolved.read_bytes()
        return hashlib.sha256(content).hexdigest(), resolved.stat().st_mtime, content.decode("utf-8", errors="replace")

    def read_file_with_memory(self, *, file_path: str, task_id: str, agent_id: str) -> tuple[str, FileReadMemoryRecord, bool]:
        normalized = normalize_file_path(file_path, root=self.root)
        content_hash, mtime, content = self._file_meta(normalized)
        cached = self.file_reads.get(normalized)
        if cached and cached.content_hash == content_hash and cached.mtime == mtime:
            cached.last_read_at = utc_iso()
            cached.last_read_by_agent_id = agent_id
            cached.task_id = task_id
            cached.changed_since_last_read = False
            self.record_decision(
                agent_id=agent_id,
                task_id=task_id,
                decision_type="file_read_cache_hit",
                input_summary=normalized,
                memory_used=[cached.id],
                decision=f"Reused cached read for {normalized}",
                reason="file hash and mtime unchanged",
            )
            logger.info("[memory] file_cache_hit path=%s", normalized)
            return cached.content or cached.content_summary, cached, True
        record = FileReadMemoryRecord(
            file_path=normalized,
            content_hash=content_hash,
            mtime=mtime,
            last_read_at=utc_iso(),
            last_read_by_agent_id=agent_id,
            task_id=task_id,
            content_summary=_summary(content, max_chars=1200),
            changed_since_last_read=cached is not None,
            content=content,
        )
        self.file_reads[normalized] = record
        self.record_decision(
            agent_id=agent_id,
            task_id=task_id,
            decision_type="file_read_cache_miss",
            input_summary=normalized,
            memory_used=[cached.id] if cached else [],
            decision=f"Read {normalized} from repository",
            reason="no cache record" if cached is None else "file hash or mtime changed",
        )
        logger.info("[memory] file_cache_miss path=%s", normalized)
        return content, record, False

    def tool_cache_key(self, *, tool_name: str, args: dict[str, Any], relevant_file_hashes: dict[str, str] | None = None) -> str:
        stable_args = {key: value for key, value in dict(args or {}).items() if key not in VOLATILE_TOOL_ARG_KEYS}
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
        key = self.tool_cache_key(tool_name=normalized, args=args, relevant_file_hashes=relevant_file_hashes)
        record = self.tool_executions.get(key)
        if record and record.reusable and record.status == "ok":
            logger.info("[memory] tool_cache_hit tool=%s", normalized)
            return record
        return None

    def record_tool_execution(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        task_id: str,
        agent_id: str,
        status: str,
        result_summary: str,
        result: dict[str, Any] | None = None,
        relevant_file_hashes: dict[str, str] | None = None,
    ) -> ToolExecutionMemoryRecord:
        normalized = normalize_tool_name(tool_name)
        key = self.tool_cache_key(tool_name=normalized, args=args, relevant_file_hashes=relevant_file_hashes)
        reusable = normalized not in WRITE_TOOL_NAMES
        record = ToolExecutionMemoryRecord(
            tool_name=normalized,
            normalized_args_hash=key,
            task_id=task_id,
            agent_id=agent_id,
            status=status,
            result_summary=_summary(result_summary),
            started_at=utc_iso(),
            completed_at=utc_iso(),
            reusable=reusable,
            result=result or {},
        )
        self.tool_executions[key] = record
        return record

    def record_decision(
        self,
        *,
        agent_id: str,
        task_id: str,
        decision_type: str,
        input_summary: str,
        memory_used: list[str],
        decision: str,
        reason: str,
    ) -> None:
        self.agent_decisions.append(
            AgentDecisionMemoryRecord(
                agent_id=agent_id,
                task_id=task_id,
                decision_type=decision_type,
                input_summary=_summary(input_summary),
                memory_used=list(memory_used),
                decision=_summary(decision),
                reason=_summary(reason),
            )
        )

    def record_verification(
        self,
        *,
        task_id: str,
        verifier_agent_id: str,
        checked_files: list[str],
        tests_run: list[str],
        result: str,
        findings: list[str] | None = None,
    ) -> VerificationMemoryRecord:
        record = VerificationMemoryRecord(
            task_id=task_id,
            verifier_agent_id=verifier_agent_id,
            checked_files=[normalize_file_path(item, root=self.root) for item in checked_files],
            tests_run=list(tests_run),
            result=result,
            findings=list(findings or []),
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
        role = agent_role.value if isinstance(agent_role, AgentRole) else str(agent_role)
        privilege = self.privilege_for_role(role)
        target_set = {normalize_file_path(item, root=self.root) for item in target_files or [] if str(item).strip()}
        current = self.task_records.get(task_id)
        same_goal = [
            asdict(item)
            for item in self.task_records.values()
            if current and item.task_id != task_id and item.fingerprint == current.fingerprint
        ]
        active = [
            asdict(item)
            for item in self.task_records.values()
            if item.status in TASK_ACTIVE_STATUSES and (not current or item.task_id != current.task_id)
        ]
        completed = [asdict(item) for item in self.task_records.values() if item.status == "completed"]

        def file_allowed(record: FileReadMemoryRecord) -> bool:
            return privilege in {"full", "planner", "taskboard", "queue", "verifier"} or record.file_path in target_set

        project_memory = list(self.project_memory[-8:]) if privilege in {"full", "planner"} else []
        task_memory = [asdict(item) for item in self.task_records.values() if item.task_id in {task_id, parent_task_id}]
        file_cache = [asdict(record) for record in self.file_reads.values() if file_allowed(record)]
        if privilege in {"coding", "tool"}:
            active = []
            completed = []
        if privilege == "verifier":
            active = []
        bundle = ScopedMemoryBundle(
            bundle_id=f"memory_bundle_{stable_hash({'agent_id': agent_id, 'task_id': task_id, 'at': utc_iso()})}",
            agent_id=agent_id,
            agent_role=role,
            privilege_level=privilege,
            task_id=task_id,
            parent_task_id=parent_task_id,
            relevant_project_memory=project_memory,
            relevant_task_memory=task_memory,
            relevant_file_cache=file_cache,
            previous_tool_results=[asdict(item) for item in self.tool_executions.values() if item.task_id == task_id],
            duplicate_task_candidates=same_goal,
            active_related_tasks=active,
            completed_related_tasks=completed,
            routing_hints=[f"duplicate_of:{current.duplicate_of}"] if current and current.duplicate_of else [],
            verification_history=[asdict(item) for item in self.verifications if item.task_id == task_id],
        )
        logger.info("[memory] scoped_bundle_created bundle_id=%s agent_id=%s role=%s", bundle.bundle_id, agent_id, role)
        return bundle

    @staticmethod
    def privilege_for_role(role: str) -> str:
        normalized = str(role or "").lower()
        if normalized in {"main", "head_decision"}:
            return "full"
        if normalized == "planner":
            return "planner"
        if normalized == "taskboard":
            return "taskboard"
        if normalized in {"queue", "queue_manager"}:
            return "queue"
        if normalized == "coding":
            return "coding"
        if normalized == "tool":
            return "tool"
        if normalized == "verifier":
            return "verifier"
        return "scoped"


__all__ = [
    "AgentDecisionMemoryRecord",
    "FileReadMemoryRecord",
    "MultiAgentMemoryService",
    "ScopedMemoryBundle",
    "TaskMemoryRecord",
    "ToolExecutionMemoryRecord",
    "VerificationMemoryRecord",
    "normalize_file_path",
    "normalize_text",
    "normalize_tool_name",
    "stable_hash",
    "task_fingerprint",
]
