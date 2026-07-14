from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from mana_agent.multi_agent.core.types import AgentRole
from mana_agent.workspaces.paths import repository_dir, repository_id_for_path
from mana_agent.workspaces.paths import session_dir, workspace_dir
from mana_agent.workspaces.service import WorkspaceService
from mana_agent.workspaces.store import atomic_write_json

logger = logging.getLogger(__name__)


ReadMode = Literal["line", "full"]
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
    repository_ids: list[str] | None = None,
) -> str:
    return stable_hash(
        {
            "goal": normalize_text(normalized_goal),
            "action_type": normalize_text(action_type),
            "target_files": sorted(normalize_file_path(item, root=root) for item in target_files or [] if str(item).strip()),
            "expected_output": normalize_text(expected_output),
            "repository_ids": sorted(str(item) for item in repository_ids or []),
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
    workspace_id: str = ""
    session_id: str = ""
    repository_ids: list[str] = field(default_factory=list)


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
    workspace_id: str = ""
    session_id: str = ""
    repository_ids: list[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        payload = asdict(self)
        return "Scoped memory bundle:\n" + json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


class EvidenceMemory:
    """Append-only, run-scoped read evidence under ``.mana/runs/<run_id>``."""

    def __init__(self, *, repo_root: Path, run_id: str | None) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.run_id = str(run_id or "").strip()
        self.run_dir = (
            repository_dir(repository_id_for_path(self.repo_root)) / "runs" / self.run_id
            if self.run_id
            else None
        )
        self.path = self.run_dir / "read_evidence.jsonl" if self.run_dir else None
        self._index: dict[str, list[dict[str, Any]]] = {}
        self._loaded = False

    @staticmethod
    def content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def enabled(self) -> bool:
        return self.path is not None

    def normalize_path(self, path: str | Path) -> Path:
        requested = Path(path)
        resolved = requested if requested.is_absolute() else (self.repo_root / requested)
        resolved = resolved.resolve()
        resolved.relative_to(self.repo_root)
        return resolved

    def to_repo_rel(self, path: str | Path) -> str:
        return self.normalize_path(path).relative_to(self.repo_root).as_posix()

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.path is None or not self.path.exists():
            return
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            normalized = str(row.get("normalized_path") or "").strip()
            event = str(row.get("event") or "").strip()
            if not normalized:
                continue
            if event == "invalidate":
                self._index.pop(normalized, None)
                continue
            if event == "read":
                self._index.setdefault(normalized, []).insert(0, row)

    def _append(self, row: dict[str, Any]) -> None:
        if self.path is None or self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def read_files(self) -> set[str]:
        self._load()
        out: set[str] = set()
        for normalized in self._index:
            try:
                out.add(Path(normalized).relative_to(self.repo_root).as_posix())
            except ValueError:
                out.add(normalized)
        return out

    def invalidate(self, path: str | Path) -> None:
        if not self.enabled():
            return
        try:
            normalized = str(self.normalize_path(path))
        except Exception:
            return
        self._load()
        self._index.pop(normalized, None)
        self._append({"event": "invalidate", "timestamp": utc_iso(), "normalized_path": normalized})

    def invalidate_many(self, paths: list[str] | set[str] | tuple[str, ...]) -> None:
        for path in paths:
            self.invalidate(path)

    def _fresh_rows(self, resolved: Path) -> tuple[list[dict[str, Any]], bool]:
        self._load()
        normalized = str(resolved)
        rows = list(self._index.get(normalized, []))
        if not rows:
            return [], False
        stat = resolved.stat()
        fresh: list[dict[str, Any]] = []
        stale = False
        for row in rows:
            if int(row.get("stat_size", -1) or -1) != int(stat.st_size):
                stale = True
                continue
            if int(row.get("stat_mtime_ns", -1) or -1) != int(stat.st_mtime_ns):
                stale = True
                continue
            content = str(row.get("content") or "")
            if self.content_hash(content) != str(row.get("content_hash") or ""):
                stale = True
                continue
            fresh.append(row)
        if stale:
            self.invalidate(resolved)
            for row in reversed(fresh):
                self._index.setdefault(normalized, []).insert(0, row)
        return fresh, stale

    def lookup(
        self,
        *,
        resolved: Path,
        mode: ReadMode,
        start_line: int,
        end_line: int,
    ) -> tuple[dict[str, Any] | None, bool]:
        if not self.enabled():
            return None, False
        rows, invalidated = self._fresh_rows(resolved)
        if not rows:
            return None, invalidated
        full = next((row for row in rows if str(row.get("mode")) == "full"), None)
        if full is not None:
            return self._payload_from_row(full, mode=mode, start_line=start_line, end_line=end_line), invalidated
        if mode == "line":
            for row in rows:
                row_start = int(row.get("start_line", 1) or 1)
                row_end = int(row.get("end_line", row_start) or row_start)
                if row_start <= start_line and row_end >= end_line:
                    return self._payload_from_row(row, mode="line", start_line=start_line, end_line=end_line), invalidated
        return None, invalidated

    def _payload_from_row(
        self,
        row: dict[str, Any],
        *,
        mode: ReadMode,
        start_line: int,
        end_line: int,
    ) -> dict[str, Any]:
        row_mode = "full" if str(row.get("mode")) == "full" else "line"
        line_count = int(row.get("line_count", 0) or 0)
        if mode == "full":
            covered = [1, line_count]
            content = str(row.get("content") or "")
            actual_end = line_count
        elif row_mode == "full":
            actual_end = min(max(end_line, start_line), line_count)
            content = "\n".join(str(row.get("content") or "").splitlines()[start_line - 1 : actual_end])
            covered = [start_line, actual_end]
        else:
            row_start = int(row.get("start_line", 1) or 1)
            row_end = int(row.get("end_line", row_start) or row_start)
            actual_end = min(max(end_line, start_line), row_end)
            lines = str(row.get("content") or "").splitlines()
            slice_start = max(start_line, row_start) - row_start
            slice_end = min(actual_end, row_end) - row_start + 1
            content = "\n".join(lines[slice_start:max(slice_start, slice_end)])
            covered = [start_line, actual_end]
        return {
            "file_path": str(row.get("normalized_path") or ""),
            "normalized_path": str(row.get("normalized_path") or ""),
            "original_path": str(row.get("original_path") or ""),
            "mode": mode,
            "start_line": 1 if mode == "full" else start_line,
            "end_line": actual_end,
            "line_count": line_count,
            "content": content,
            "cache_hit": True,
            "source": "memory",
            "cache_source": "run_evidence_full" if row_mode == "full" else "run_evidence_range",
            "cache_invalidated": False,
            "full_file_cached": row_mode == "full",
            "covered_range": covered,
        }

    def store(
        self,
        *,
        original_path: str,
        resolved: Path,
        mode: ReadMode,
        start_line: int,
        end_line: int,
        line_count: int,
        content: str,
        summary: str,
    ) -> None:
        if not self.enabled():
            return
        stat = resolved.stat()
        row = {
            "event": "read",
            "timestamp": utc_iso(),
            "normalized_path": str(resolved),
            "original_path": str(original_path),
            "start_line": int(start_line),
            "end_line": int(end_line),
            "mode": mode,
            "content_hash": self.content_hash(content),
            "stat_size": int(stat.st_size),
            "stat_mtime": float(stat.st_mtime),
            "stat_mtime_ns": int(stat.st_mtime_ns),
            "line_count": int(line_count),
            "summary": summary,
            "content": content,
        }
        self._load()
        rows = self._index.setdefault(str(resolved), [])
        row_key = (mode, int(start_line), int(end_line))
        rows[:] = [
            item
            for item in rows
            if (str(item.get("mode")), int(item.get("start_line", 0) or 0), int(item.get("end_line", 0) or 0)) != row_key
        ]
        rows.insert(0, row)
        self._append(row)


class MultiAgentMemoryService:
    def __init__(
        self,
        *,
        root: str | Path = ".",
        workspace_id: str | None = None,
        repository_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        service = WorkspaceService()
        repo = service.register_repository(self.root)
        workspace = service.workspace_for_repository(repo.repository_id)
        self.workspace_id = workspace_id or workspace.workspace_id
        self.repository_id = repository_id or repo.repository_id
        self.session_id = str(session_id or "")
        self._repo_memory_path = repository_dir(self.repository_id) / "memory.json"
        self._workspace_memory_path = workspace_dir(self.workspace_id) / "memory.json"
        self._session_memory_path = (
            session_dir(self.session_id) / "memory.json"
            if self.session_id
            else repository_dir(self.repository_id) / "runtime-memory.json"
        )
        self.task_records: dict[str, TaskMemoryRecord] = {}
        self.tool_executions: dict[str, ToolExecutionMemoryRecord] = {}
        self.agent_decisions: list[AgentDecisionMemoryRecord] = []
        self.verifications: list[VerificationMemoryRecord] = []
        self.queue_fingerprints: dict[str, str] = {}
        self.project_memory: list[dict[str, Any]] = []
        self.workspace_memory: list[dict[str, Any]] = []
        self._load_persisted()

    @staticmethod
    def _read_payload(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _load_persisted(self) -> None:
        repo_payload = self._read_payload(self._repo_memory_path)
        workspace_payload = self._read_payload(self._workspace_memory_path)
        session_payload = self._read_payload(self._session_memory_path)
        self.project_memory = list(repo_payload.get("facts") or [])
        self.workspace_memory = list(workspace_payload.get("facts") or [])
        for item in session_payload.get("tasks") or []:
            try:
                record = TaskMemoryRecord(**item)
            except TypeError:
                continue
            self.task_records[record.task_id] = record
        for item in session_payload.get("tools") or []:
            try:
                record = ToolExecutionMemoryRecord(**item)
            except TypeError:
                continue
            self.tool_executions[record.normalized_args_hash] = record
        self.agent_decisions = [AgentDecisionMemoryRecord(**item) for item in session_payload.get("decisions") or []]
        self.verifications = [VerificationMemoryRecord(**item) for item in session_payload.get("verifications") or []]

    def _persist(self) -> None:
        atomic_write_json(
            self._repo_memory_path,
            {"schema_version": 1, "repository_id": self.repository_id, "facts": self.project_memory},
        )
        atomic_write_json(
            self._workspace_memory_path,
            {"schema_version": 1, "workspace_id": self.workspace_id, "facts": self.workspace_memory},
        )
        atomic_write_json(
            self._session_memory_path,
            {
                "schema_version": 1,
                "workspace_id": self.workspace_id,
                "session_id": self.session_id,
                "repository_id": self.repository_id,
                "tasks": [asdict(item) for item in self.task_records.values()],
                "tools": [asdict(item) for item in self.tool_executions.values()],
                "decisions": [asdict(item) for item in self.agent_decisions],
                "verifications": [asdict(item) for item in self.verifications],
            },
        )

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
        payload = {"fact": _summary(fact, max_chars=1200), "repository_id": self.repository_id}
        if payload["fact"] and payload not in self.project_memory:
            self.project_memory.append(payload)
            self._persist()

    def remember_workspace_fact(self, fact: str) -> None:
        payload = {"fact": _summary(fact, max_chars=1200), "workspace_id": self.workspace_id}
        if payload["fact"] and payload not in self.workspace_memory:
            self.workspace_memory.append(payload)
            self._persist()

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
        repository_ids: list[str] | None = None,
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
            workspace_id=self.workspace_id,
            session_id=self.session_id,
            repository_ids=list(repository_ids or [self.repository_id]),
        )
        self.task_records[task_id] = record
        self._persist()
        logger.debug("[memory] duplicate_task_hit task_id=%s duplicate_of=%s", task_id, record.duplicate_of or "")
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
        self._persist()

    def register_queue_item(self, *, queue_item_id: str, fingerprint: str) -> tuple[bool, str | None]:
        existing = self.queue_fingerprints.get(fingerprint)
        if existing:
            logger.debug("[memory] queue_duplicate_rejected queue_item_id=%s existing=%s", queue_item_id, existing)
            return False, existing
        self.queue_fingerprints[fingerprint] = queue_item_id
        return True, None

    def _file_meta(self, path: str | Path) -> tuple[str, float, str, str]:
        resolved = (self.root / str(path)).resolve() if not Path(str(path)).is_absolute() else Path(str(path)).resolve()
        resolved.relative_to(self.root)
        content = resolved.read_bytes()
        return (
            normalize_file_path(resolved, root=self.root),
            hashlib.sha256(content).hexdigest(),
            resolved.stat().st_mtime,
            content.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n"),
        )

    def read_file_with_memory(self, *, file_path: str, task_id: str, agent_id: str) -> tuple[str, FileReadMemoryRecord, bool]:
        normalized, content_hash, mtime, content = self._file_meta(file_path)
        record = FileReadMemoryRecord(
            file_path=normalized,
            content_hash=content_hash,
            mtime=mtime,
            last_read_at=utc_iso(),
            last_read_by_agent_id=agent_id,
            task_id=task_id,
            content_summary=_summary(content, max_chars=1200),
            content=content,
        )
        self.record_decision(
            agent_id=agent_id,
            task_id=task_id,
            decision_type="file_read_direct",
            input_summary=normalized,
            memory_used=[],
            decision=f"Read {normalized} from repository",
            reason="canonical read cache is EvidenceMemory; multi-agent memory does not store file content",
        )
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
            record.result = self._normalized_reusable_result(record.result, cache_hit=True, source="memory")
            logger.debug("[memory] tool_cache_hit tool=%s", normalized)
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
        stored_result = result or {}
        if reusable and status == "ok":
            stored_result = self._normalized_reusable_result(stored_result, cache_hit=False, source="tool")
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
            result=stored_result,
        )
        self.tool_executions[key] = record
        self._persist()
        return record

    @staticmethod
    def _normalized_reusable_result(
        result: dict[str, Any] | None,
        *,
        cache_hit: bool,
        source: str,
    ) -> dict[str, Any]:
        payload = dict(result or {})
        payload["cache_hit"] = cache_hit
        payload["source"] = source
        payload["cache_source"] = source
        return payload

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
        self._persist()

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
        self._persist()
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

        project_memory = (
            list(self.workspace_memory[-4:]) + list(self.project_memory[-8:])
            if privilege in {"full", "planner"}
            else []
        )
        task_memory = [asdict(item) for item in self.task_records.values() if item.task_id in {task_id, parent_task_id}]
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
            relevant_file_cache=[],
            previous_tool_results=[asdict(item) for item in self.tool_executions.values() if item.task_id == task_id],
            duplicate_task_candidates=same_goal,
            active_related_tasks=active,
            completed_related_tasks=completed,
            routing_hints=[f"duplicate_of:{current.duplicate_of}"] if current and current.duplicate_of else [],
            verification_history=[asdict(item) for item in self.verifications if item.task_id == task_id],
            workspace_id=self.workspace_id,
            session_id=self.session_id,
            repository_ids=[self.repository_id],
        )
        logger.debug("[memory] scoped_bundle_created bundle_id=%s agent_id=%s role=%s", bundle.bundle_id, agent_id, role)
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
    "EvidenceMemory",
    "FileReadMemoryRecord",
    "MultiAgentMemoryService",
    "ReadMode",
    "ScopedMemoryBundle",
    "TaskMemoryRecord",
    "ToolExecutionMemoryRecord",
    "VerificationMemoryRecord",
    "normalize_file_path",
    "normalize_text",
    "normalize_tool_name",
    "stable_hash",
    "task_fingerprint",
    "utc_iso",
]
