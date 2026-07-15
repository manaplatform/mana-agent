"""Central manager for Mana-managed Git worktrees used by coding agents."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable

from mana_agent.multi_agent.tools import git_tools
from mana_agent.multi_agent.worktrees.events import emit_workspace_event
from mana_agent.multi_agent.worktrees.models import (
    ManagedWorkspace,
    WorkspaceStatus,
    utc_now,
    validate_workspace_transition,
)
from mana_agent.multi_agent.worktrees.store import (
    ManagedWorkspaceStore,
    managed_worktree_checkouts_root,
)
from mana_agent.workspaces.paths import repository_id_for_path

_SAFE_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_BRANCH_PREFIX = "mana/"
EventSink = Callable[[str, dict[str, Any]], None]


class WorkspaceError(RuntimeError):
    """Raised when a managed workspace operation cannot continue safely."""


class WorkspaceManager:
    """Lifecycle owner for isolated coding worktrees.

    Flow integration point:
    Taskboard → QueueManager → WorkspaceManager → worktree root → CodingAgent
    → Verifier → Reviewer → merge_candidate (explicit merge only).
    """

    def __init__(
        self,
        source_repo_root: str | Path,
        *,
        repository_id: str | None = None,
        event_sink: EventSink | None = None,
        enabled: bool = True,
    ) -> None:
        self.source_repo_root = git_tools.resolve_repo_root(source_repo_root)
        self.repository_id = str(repository_id or repository_id_for_path(self.source_repo_root)).strip()
        self.store = ManagedWorkspaceStore(self.repository_id)
        self.enabled = bool(enabled)
        self._event_sink = event_sink

    # ------------------------------------------------------------------ public
    def create_for_task(
        self,
        task_id: str,
        *,
        title: str = "",
        assigned_agent_id: str = "",
        session_id: str = "",
        multi_agent_workspace_id: str = "",
        base_revision: str | None = None,
        reuse_existing: bool = True,
    ) -> ManagedWorkspace:
        """Create or reconnect a deterministic isolated workspace for a coding task."""

        if not self.enabled:
            raise WorkspaceError("managed worktrees are disabled")
        task_id = str(task_id or "").strip()
        if not task_id:
            raise WorkspaceError("task_id is required")

        existing = self.store.get_by_task_id(task_id)
        if existing is not None and reuse_existing:
            resumed = self.resume(task_id, assigned_agent_id=assigned_agent_id)
            self._emit("workspace.reused", resumed, summary=f"Reused workspace for task {task_id}")
            return resumed

        workspace_id = f"mws_{uuid.uuid4().hex[:16]}"
        task_slug = self.task_slug(task_id, title=title)
        branch_name = self._unique_branch_name(task_slug)
        worktree_path = self._unique_worktree_path(task_slug)
        base_rev = str(base_revision or self._git_out(["rev-parse", "HEAD"])).strip()
        if not base_rev:
            raise WorkspaceError("unable to resolve base revision for managed workspace")

        workspace = ManagedWorkspace(
            workspace_id=workspace_id,
            task_id=task_id,
            repository_id=self.repository_id,
            source_repo_root=str(self.source_repo_root),
            worktree_path=str(worktree_path),
            branch_name=branch_name,
            base_revision=base_rev,
            status=WorkspaceStatus.CREATING,
            assigned_agent_id=str(assigned_agent_id or ""),
            session_id=str(session_id or ""),
            multi_agent_workspace_id=str(multi_agent_workspace_id or ""),
            task_slug=task_slug,
            current_head=base_rev,
        )
        self.store.save(workspace)
        self._emit("workspace.created", workspace, summary=f"Creating managed worktree for task {task_id}")

        try:
            self._create_git_worktree(workspace)
            workspace.current_head = self._git_out(["rev-parse", "HEAD"], cwd=Path(workspace.worktree_path))
            workspace.status = WorkspaceStatus.READY
            workspace.updated_at = utc_now()
            self._record_local_event(workspace, "workspace.branch_created", {"branch": workspace.branch_name})
            self.store.save(workspace)
            self._emit(
                "workspace.branch_created",
                workspace,
                summary=f"Created branch {workspace.branch_name}",
                details={"branch": workspace.branch_name, "base_revision": workspace.base_revision},
            )
            self._emit("workspace.status_changed", workspace, summary="Workspace ready")
            return workspace
        except Exception as exc:
            workspace.status = WorkspaceStatus.FAILED
            workspace.last_error = str(exc)
            workspace.updated_at = utc_now()
            workspace.recovery_notes.append(f"creation failed: {exc}")
            self.store.save(workspace)
            self._emit("workspace.failed", workspace, status="error", summary=str(exc))
            raise WorkspaceError(f"failed to create managed worktree for task {task_id}: {exc}") from exc

    def list(self, *, reconcile: bool = False) -> list[ManagedWorkspace]:
        if reconcile:
            self.reconcile()
        rows = self.store.list()
        for item in rows:
            self._refresh_git_flags(item, persist=True)
        return self.store.list()

    def get_for_task(self, task_id: str) -> ManagedWorkspace:
        item = self.store.get_by_task_id(task_id)
        if item is None:
            raise WorkspaceError(f"no managed workspace for task {task_id}")
        return item

    def status(self, task_id: str, *, reconcile: bool = True) -> dict[str, Any]:
        if reconcile:
            try:
                self.reconcile(task_id=task_id)
            except WorkspaceError:
                pass
        workspace = self.get_for_task(task_id)
        self._refresh_git_flags(workspace, persist=True)
        report = workspace.status_report()
        report["git"] = self._git_state_payload(Path(workspace.worktree_path) if Path(workspace.worktree_path).exists() else self.source_repo_root)
        report["porcelain_worktrees"] = self._list_git_worktrees()
        return report

    def resume(self, task_id: str, *, assigned_agent_id: str = "") -> ManagedWorkspace:
        """Reconnect an interrupted task to its existing workspace when safe."""

        workspace = self.get_for_task(task_id)
        path = Path(workspace.worktree_path)
        notes: list[str] = []
        if assigned_agent_id:
            workspace.assigned_agent_id = assigned_agent_id

        if not path.exists():
            workspace.orphaned_metadata = True
            workspace.status = WorkspaceStatus.STALE
            notes.append("worktree path missing on disk; metadata retained")
            workspace.recovery_notes.extend(notes)
            workspace.updated_at = utc_now()
            self.store.save(workspace)
            self._emit("workspace.resumed", workspace, status="error", summary="Resume failed: path missing")
            raise WorkspaceError(
                f"cannot resume task {task_id}: worktree path missing ({workspace.worktree_path}). "
                "Metadata was retained; recreate only after explicit inspection."
            )

        if not self._is_registered_worktree(path):
            notes.append("path exists but is not registered in git worktree list")
            workspace.status = WorkspaceStatus.STALE
            workspace.orphaned_metadata = True
            workspace.recovery_notes.extend(notes)
            workspace.updated_at = utc_now()
            self.store.save(workspace)
            raise WorkspaceError(
                f"cannot resume task {task_id}: path is not a registered git worktree. "
                "Refusing to auto-repair to protect user data."
            )

        self._refresh_git_flags(workspace, persist=False)
        if workspace.dirty:
            workspace.status = WorkspaceStatus.DIRTY
            notes.append("workspace has uncommitted changes; retained for inspection")
            self._emit("workspace.dirty_detected", workspace, summary="Dirty workspace detected on resume")
        elif workspace.status in {
            WorkspaceStatus.INTERRUPTED,
            WorkspaceStatus.STALE,
            WorkspaceStatus.FAILED,
            WorkspaceStatus.CREATING,
        }:
            workspace.status = WorkspaceStatus.READY
            notes.append("workspace reconnected and marked ready")
        # leave active/review/merge states as-is

        workspace.current_head = self._git_out(["rev-parse", "HEAD"], cwd=path) or workspace.current_head
        workspace.recovery_notes.extend(notes)
        workspace.updated_at = utc_now()
        self.store.save(workspace)
        self._emit("workspace.resumed", workspace, summary=f"Resumed workspace for task {task_id}")
        return workspace

    def transition(
        self,
        task_id: str,
        status: WorkspaceStatus | str,
        *,
        agent_id: str = "",
        error: str = "",
        notes: list[str] | None = None,
        force: bool = False,
    ) -> ManagedWorkspace:
        workspace = self.get_for_task(task_id)
        nxt = WorkspaceStatus.parse(status)
        previous = workspace.status
        if previous != nxt:
            try:
                validate_workspace_transition(previous, nxt)
            except ValueError:
                if not force:
                    # Idempotent no-op when already at a terminal/retained state that
                    # cannot accept the requested transition; still record notes/errors.
                    if error:
                        workspace.last_error = error
                    if notes:
                        workspace.recovery_notes.extend(str(item) for item in notes)
                    workspace.updated_at = utc_now()
                    self.store.save(workspace)
                    raise WorkspaceError(
                        f"invalid workspace transition for task {task_id}: "
                        f"{previous.value} → {nxt.value}"
                    )
        workspace.status = nxt
        if agent_id:
            workspace.assigned_agent_id = agent_id
        if error:
            workspace.last_error = error
        if notes:
            workspace.recovery_notes.extend(str(item) for item in notes)
        self._refresh_git_flags(workspace, persist=False)
        workspace.updated_at = utc_now()
        self._record_local_event(
            workspace,
            "workspace.status_changed",
            {"from": previous.value, "to": nxt.value, "error": error},
        )
        self.store.save(workspace)
        kind = {
            WorkspaceStatus.VERIFYING: "workspace.verifying",
            WorkspaceStatus.REVIEWING: "workspace.reviewing",
            WorkspaceStatus.MERGE_CANDIDATE: "workspace.merge_candidate",
            WorkspaceStatus.MERGED: "workspace.merged",
            WorkspaceStatus.RETAINED: "workspace.retained",
            WorkspaceStatus.FAILED: "workspace.failed",
            WorkspaceStatus.CONFLICTED: "workspace.conflict",
            WorkspaceStatus.DIRTY: "workspace.dirty_detected",
        }.get(nxt, "workspace.status_changed")
        self._emit(kind, workspace, summary=f"{previous.value} → {nxt.value}")
        return workspace

    def mark_running(self, task_id: str, *, agent_id: str = "") -> ManagedWorkspace:
        return self.transition(task_id, WorkspaceStatus.RUNNING, agent_id=agent_id)

    def mark_verifying(self, task_id: str, *, agent_id: str = "") -> ManagedWorkspace:
        return self.transition(task_id, WorkspaceStatus.VERIFYING, agent_id=agent_id)

    def mark_reviewing(self, task_id: str, *, agent_id: str = "") -> ManagedWorkspace:
        return self.transition(task_id, WorkspaceStatus.REVIEWING, agent_id=agent_id)

    def mark_merge_candidate(self, task_id: str, *, review_result: dict[str, Any] | None = None) -> ManagedWorkspace:
        workspace = self.transition(task_id, WorkspaceStatus.MERGE_CANDIDATE)
        if review_result is not None:
            workspace.review_result = dict(review_result)
            workspace.updated_at = utc_now()
            self.store.save(workspace)
        return workspace

    def mark_failed(self, task_id: str, *, error: str, retain: bool = True) -> ManagedWorkspace:
        workspace = self.transition(task_id, WorkspaceStatus.FAILED, error=error)
        if retain:
            try:
                return self.transition(task_id, WorkspaceStatus.RETAINED, notes=[f"retained after failure: {error}"])
            except ValueError:
                return workspace
        return workspace

    def mark_interrupted(self, task_id: str, *, error: str = "") -> ManagedWorkspace:
        return self.transition(task_id, WorkspaceStatus.INTERRUPTED, error=error)

    def resolve_execution_root(self, task_id: str | None = None) -> Path:
        """Return the repository root tools/agents should use for a task."""

        if not task_id:
            return self.source_repo_root
        workspace = self.store.get_by_task_id(task_id)
        if workspace is None:
            return self.source_repo_root
        path = Path(workspace.worktree_path)
        if path.is_dir():
            return path.resolve()
        return self.source_repo_root

    def diff(self, task_id: str, *, stat: bool = False) -> dict[str, Any]:
        workspace = self.get_for_task(task_id)
        path = Path(workspace.worktree_path)
        if not path.is_dir():
            raise WorkspaceError(f"worktree path missing for task {task_id}")
        args = ["diff", f"{workspace.base_revision}...HEAD"]
        if stat:
            args.append("--stat")
        result = git_tools.run_git(args, repo_path=path)
        result["base_revision"] = workspace.base_revision
        result["branch"] = workspace.branch_name
        result["task_id"] = task_id
        result["worktree_path"] = str(path)
        return result

    def merge(
        self,
        task_id: str,
        *,
        explicit_user_intent: bool = False,
        allow_protected: bool = False,
        no_ff: bool = True,
    ) -> dict[str, Any]:
        """Merge the task branch into the source checkout with Git safety policy.

        Never force-pushes or rewrites history. Requires explicit validated intent.
        """

        if not explicit_user_intent:
            raise WorkspaceError(
                "merge refused: explicit user intent is required. "
                "Pass validated intent before merging a managed task branch."
            )
        workspace = self.get_for_task(task_id)
        if workspace.status not in {WorkspaceStatus.MERGE_CANDIDATE, WorkspaceStatus.READY, WorkspaceStatus.DIRTY}:
            # Allow merge from merge_candidate primarily; dirty/ready only with intent already checked.
            if workspace.status != WorkspaceStatus.MERGE_CANDIDATE:
                raise WorkspaceError(
                    f"merge refused: workspace status is {workspace.status.value}; "
                    "expected merge_candidate after successful review"
                )
        self._refresh_git_flags(workspace, persist=True)
        if workspace.dirty:
            raise WorkspaceError("merge refused: task worktree is dirty; commit or inspect before merge")

        source_state = git_tools.observe_git_state(self.source_repo_root)
        if source_state.status_porcelain.strip():
            raise WorkspaceError(
                "merge refused: source repository checkout is dirty; "
                "commit or stash user work before merging a managed task branch"
            )
        if source_state.operation_state:
            raise WorkspaceError(f"merge refused: source repository is mid-{source_state.operation_state}")

        args = ["merge"]
        if no_ff:
            args.append("--no-ff")
        args.extend(["-m", f"Merge managed task branch {workspace.branch_name} ({task_id})", workspace.branch_name])
        result = git_tools.run_git(args, repo_path=self.source_repo_root, allow_protected=allow_protected)
        workspace.merge_result = dict(result)
        workspace.updated_at = utc_now()
        if result.get("blocked"):
            workspace.last_error = str(result.get("error") or "merge blocked by git safety policy")
            self.store.save(workspace)
            self._emit("workspace.conflict", workspace, status="error", summary=workspace.last_error)
            raise WorkspaceError(workspace.last_error)
        if not result.get("ok"):
            stderr = str(result.get("stderr") or result.get("error") or "merge failed")
            if "conflict" in stderr.lower() or "CONFLICT" in str(result.get("stdout") or ""):
                workspace.status = WorkspaceStatus.CONFLICTED
                workspace.last_error = stderr
                self.store.save(workspace)
                self._emit("workspace.conflict", workspace, status="error", summary=stderr)
                raise WorkspaceError(f"merge conflict for task {task_id}: {stderr}")
            workspace.status = WorkspaceStatus.FAILED
            workspace.last_error = stderr
            self.store.save(workspace)
            raise WorkspaceError(f"merge failed for task {task_id}: {stderr}")

        workspace.status = WorkspaceStatus.MERGED
        workspace.current_head = self._git_out(["rev-parse", "HEAD"]) or workspace.current_head
        self.store.save(workspace)
        self._emit("workspace.merged", workspace, status="success", summary=f"Merged {workspace.branch_name}")
        return {
            "ok": True,
            "task_id": task_id,
            "branch": workspace.branch_name,
            "status": workspace.status.value,
            "git": result,
        }

    def remove(
        self,
        task_id: str,
        *,
        force: bool = False,
        delete_branch: bool = False,
        explicit_user_intent: bool = False,
    ) -> dict[str, Any]:
        """Remove a managed worktree safely. Refuses dirty/unmerged work without intent."""

        workspace = self.get_for_task(task_id)
        self._refresh_git_flags(workspace, persist=True)
        path = Path(workspace.worktree_path)
        unmerged = workspace.status not in {WorkspaceStatus.MERGED, WorkspaceStatus.RETAINED, WorkspaceStatus.FAILED}
        if (workspace.dirty or unmerged) and not (force and explicit_user_intent):
            raise WorkspaceError(
                "remove refused: worktree is dirty or unmerged. "
                "Provide explicit validated intent with force=true to destroy it, "
                "or leave it retained for inspection."
            )
        if workspace.status == WorkspaceStatus.MERGE_CANDIDATE and not (force and explicit_user_intent):
            raise WorkspaceError(
                "remove refused: workspace is a merge candidate. "
                "Merge or provide explicit force intent to discard."
            )

        cleanup: dict[str, Any] = {"task_id": task_id, "worktree_path": str(path), "branch": workspace.branch_name}
        if path.exists() and self._is_registered_worktree(path):
            # Never use --force unless explicit force intent; git worktree remove refuses dirty trees without force.
            args = ["worktree", "remove"]
            if force and explicit_user_intent:
                args.append("--force")
            args.append(str(path))
            result = git_tools.run_git(args, repo_path=self.source_repo_root)
            cleanup["worktree_remove"] = result
            if not result.get("ok") and not (force and explicit_user_intent):
                raise WorkspaceError(
                    f"git worktree remove failed: {result.get('stderr') or result.get('error') or 'unknown error'}"
                )
        elif path.exists() and force and explicit_user_intent:
            shutil.rmtree(path)
            cleanup["filesystem_removed"] = True
        elif path.exists():
            raise WorkspaceError(
                "remove refused: path exists but is not a registered managed worktree; "
                "never auto-delete unknown trees"
            )

        if delete_branch and force and explicit_user_intent:
            # Prefer soft branch delete; -D is protected.
            branch_result = git_tools.run_git(
                ["branch", "-d", workspace.branch_name],
                repo_path=self.source_repo_root,
            )
            cleanup["branch_delete"] = branch_result

        self.store.delete(workspace.workspace_id)
        cleanup["metadata_deleted"] = True
        cleanup["ok"] = True
        self._emit("workspace.cleanup", workspace, status="success", summary=f"Removed workspace for task {task_id}")
        return cleanup

    def reconcile(self, *, task_id: str | None = None) -> dict[str, Any]:
        """Reconcile persisted metadata with git worktree list and the filesystem."""

        porcelain = self._list_git_worktrees()
        registered_paths = {Path(item["path"]).resolve() for item in porcelain if item.get("path")}
        managed = self.store.list()
        if task_id:
            managed = [item for item in managed if item.task_id == task_id]

        orphaned_metadata: list[str] = []
        recovered: list[str] = []
        for item in managed:
            path = Path(item.worktree_path)
            if not path.exists():
                item.orphaned_metadata = True
                if item.status not in {WorkspaceStatus.MERGED, WorkspaceStatus.RETAINED}:
                    item.status = WorkspaceStatus.STALE
                item.recovery_notes.append("reconcile: worktree path missing")
                orphaned_metadata.append(item.task_id)
            elif path.resolve() not in registered_paths:
                item.orphaned_metadata = True
                item.status = WorkspaceStatus.STALE
                item.recovery_notes.append("reconcile: path not in git worktree list")
                orphaned_metadata.append(item.task_id)
            else:
                item.orphaned_metadata = False
                self._refresh_git_flags(item, persist=False)
            item.updated_at = utc_now()
            self.store.save(item)
            recovered.append(item.task_id)

        managed_paths = {Path(item.worktree_path).resolve() for item in self.store.list()}
        unmanaged = [
            str(item["path"])
            for item in porcelain
            if item.get("path")
            and Path(item["path"]).resolve() not in managed_paths
            and Path(item["path"]).resolve() != self.source_repo_root.resolve()
        ]
        report = {
            "ok": True,
            "repository_id": self.repository_id,
            "reconciled_tasks": recovered,
            "orphaned_metadata_tasks": orphaned_metadata,
            "unmanaged_git_worktrees": unmanaged,
            "note": "unmanaged git worktrees are never auto-deleted",
        }
        self._emit(
            "workspace.reconciled",
            ManagedWorkspace(
                workspace_id="reconcile",
                task_id=task_id or "",
                repository_id=self.repository_id,
                source_repo_root=str(self.source_repo_root),
                worktree_path="",
                branch_name="",
                base_revision="",
            ),
            summary=f"Reconciled {len(recovered)} managed workspace(s)",
            details=report,
        )
        return report

    def attach_to_taskboard(self, task: Any, workspace: ManagedWorkspace) -> None:
        """Copy workspace identity onto a TaskBoardItem (duck-typed)."""

        if task is None:
            return
        task.managed_workspace_id = workspace.workspace_id
        task.managed_branch = workspace.branch_name
        task.managed_worktree_path = workspace.worktree_path
        task.workspace_status = workspace.status.value
        task.base_revision = workspace.base_revision
        task.execution_repo_root = workspace.worktree_path

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def task_slug(task_id: str, *, title: str = "") -> str:
        base = str(title or "").strip().lower() or str(task_id).strip().lower()
        base = _SAFE_SLUG_RE.sub("-", base).strip("-._")
        if not base:
            base = "task"
        base = base[:48].strip("-._") or "task"
        digest = hashlib.sha256(str(task_id).encode("utf-8")).hexdigest()[:8]
        # Include short task identity so parallel tasks with similar titles stay deterministic.
        return f"{base}-{digest}"

    def _unique_branch_name(self, task_slug: str) -> str:
        candidate = f"{_BRANCH_PREFIX}{task_slug}"
        existing = self._local_branches()
        if candidate not in existing:
            return candidate
        # Collision: append numeric suffix while remaining deterministic for new creations
        # by hashing slug + index of first free name.
        for index in range(2, 1000):
            alt = f"{candidate}-{index}"
            if alt not in existing:
                return alt
        return f"{candidate}-{uuid.uuid4().hex[:6]}"

    def _unique_worktree_path(self, task_slug: str) -> Path:
        root = managed_worktree_checkouts_root(self.repository_id)
        candidate = root / task_slug
        if not candidate.exists():
            return candidate
        for index in range(2, 1000):
            alt = root / f"{task_slug}-{index}"
            if not alt.exists():
                return alt
        return root / f"{task_slug}-{uuid.uuid4().hex[:6]}"

    def _create_git_worktree(self, workspace: ManagedWorkspace) -> None:
        path = Path(workspace.worktree_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise WorkspaceError(f"worktree path already exists: {path}")
        # Prefer creating a new branch from the recorded base revision.
        args = ["worktree", "add", "-b", workspace.branch_name, str(path), workspace.base_revision]
        result = git_tools.run_git(args, repo_path=self.source_repo_root)
        if result.get("ok"):
            return
        stderr = str(result.get("stderr") or result.get("error") or "")
        # Branch may already exist from a previous partial attempt.
        if "already exists" in stderr.lower() or "already checked out" in stderr.lower():
            alt = git_tools.run_git(
                ["worktree", "add", str(path), workspace.branch_name],
                repo_path=self.source_repo_root,
            )
            if alt.get("ok"):
                return
            raise WorkspaceError(str(alt.get("stderr") or alt.get("error") or "worktree add failed"))
        raise WorkspaceError(stderr or "worktree add failed")

    def _local_branches(self) -> set[str]:
        result = git_tools.run_git(["branch", "--format=%(refname:short)"], repo_path=self.source_repo_root)
        if not result.get("ok"):
            return set()
        return {line.strip() for line in str(result.get("stdout") or "").splitlines() if line.strip()}

    def _git_out(self, args: list[str], *, cwd: Path | None = None) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd or self.source_repo_root),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    def _list_git_worktrees(self) -> list[dict[str, str]]:
        completed = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=str(self.source_repo_root),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            return []
        rows: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for line in completed.stdout.splitlines():
            if not line.strip():
                if current:
                    rows.append(current)
                    current = {}
                continue
            if line.startswith("worktree "):
                if current:
                    rows.append(current)
                current = {"path": line[len("worktree ") :].strip()}
            elif line.startswith("HEAD "):
                current["head"] = line[len("HEAD ") :].strip()
            elif line.startswith("branch "):
                ref = line[len("branch ") :].strip()
                current["branch"] = ref.replace("refs/heads/", "") if ref.startswith("refs/heads/") else ref
            elif line == "detached":
                current["detached"] = "true"
            elif line == "bare":
                current["bare"] = "true"
        if current:
            rows.append(current)
        return rows

    def _is_registered_worktree(self, path: Path) -> bool:
        target = path.resolve()
        return any(Path(item["path"]).resolve() == target for item in self._list_git_worktrees() if item.get("path"))

    def _refresh_git_flags(self, workspace: ManagedWorkspace, *, persist: bool) -> None:
        path = Path(workspace.worktree_path)
        if not path.is_dir():
            workspace.dirty = False
            if persist:
                workspace.updated_at = utc_now()
                self.store.save(workspace)
            return
        try:
            observation = git_tools.observe_git_state(path)
            workspace.dirty = bool(observation.status_porcelain.strip())
            workspace.current_head = observation.head or workspace.current_head
            if observation.operation_state == "merge":
                # conflict markers often appear as unmerged paths in porcelain
                if "UU" in observation.status_porcelain or "AA" in observation.status_porcelain:
                    if workspace.status not in {WorkspaceStatus.MERGED, WorkspaceStatus.RETAINED}:
                        workspace.status = WorkspaceStatus.CONFLICTED
            if observation.current_branch == "" and observation.head:
                workspace.recovery_notes.append("detached HEAD detected in managed worktree")
        except Exception as exc:
            workspace.last_error = str(exc)
        if persist:
            workspace.updated_at = utc_now()
            self.store.save(workspace)

    def _git_state_payload(self, root: Path) -> dict[str, Any]:
        try:
            return git_tools.observe_git_state(root).to_dict()
        except Exception as exc:
            return {"error": str(exc)}

    def _record_local_event(self, workspace: ManagedWorkspace, kind: str, details: dict[str, Any]) -> None:
        workspace.events.append(
            {
                "kind": kind,
                "at": utc_now().isoformat(),
                "details": details,
            }
        )
        # Cap local event history to keep metadata small.
        if len(workspace.events) > 100:
            workspace.events = workspace.events[-100:]

    def _emit(
        self,
        kind: str,
        workspace: ManagedWorkspace,
        *,
        status: str = "running",
        summary: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "task_id": workspace.task_id,
            "workspace_id": workspace.workspace_id,
            "repository_id": workspace.repository_id,
            "status": workspace.status.value if isinstance(workspace.status, WorkspaceStatus) else str(workspace.status),
            "branch": workspace.branch_name,
            "worktree_path": workspace.worktree_path,
            **dict(details or {}),
        }
        if self._event_sink is not None:
            try:
                self._event_sink(kind, payload)
            except Exception:
                pass
        emit_workspace_event(
            kind,
            task_id=workspace.task_id,
            workspace_id=workspace.workspace_id,
            repository_id=workspace.repository_id,
            session_id=workspace.session_id,
            status=status,
            summary=summary,
            details=payload,
            agent_id=workspace.assigned_agent_id or "workspace_manager",
        )


def coding_route_requires_worktree(route_name: str) -> bool:
    """Whether a multi-agent route should allocate an isolated coding worktree."""

    return str(route_name or "").strip().lower() in {"coding", "tool", "high_risk_tool"}
