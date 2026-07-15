"""Reviewer helpers for managed task branches vs recorded base revisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mana_agent.multi_agent.tools import git_tools
from mana_agent.multi_agent.worktrees.manager import WorkspaceError, WorkspaceManager
from mana_agent.multi_agent.worktrees.models import ManagedWorkspace


def review_task_branch(
    manager: WorkspaceManager,
    task_id: str,
    *,
    reviewer_agent_id: str = "",
    verification_passed: bool | None = None,
    hierarchy_ok: bool = True,
    extra_blockers: list[str] | None = None,
) -> dict[str, Any]:
    """Inspect task branch diff against the recorded base revision.

    Produces a structured review result. Successful reviews mark the workspace
    as a merge candidate; they never merge into the default branch.
    """

    workspace = manager.get_for_task(task_id)
    path = Path(workspace.worktree_path)
    if not path.is_dir():
        raise WorkspaceError(f"cannot review task {task_id}: worktree path missing")

    manager.mark_reviewing(task_id, agent_id=reviewer_agent_id)
    manager._refresh_git_flags(workspace, persist=True)  # noqa: SLF001 - shared manager internals

    diff = manager.diff(task_id, stat=True)
    full_diff = manager.diff(task_id, stat=False)
    head = git_tools.run_git(["rev-parse", "HEAD"], repo_path=path)
    branch = git_tools.run_git(["branch", "--show-current"], repo_path=path)
    status = git_tools.run_git(["status", "--porcelain=v1"], repo_path=path)

    failures: list[str] = []
    if extra_blockers:
        failures.extend(str(item) for item in extra_blockers if str(item).strip())
    if not hierarchy_ok:
        failures.append("hierarchy violations present")
    if verification_passed is False:
        failures.append("verification did not pass")
    if not diff.get("ok"):
        failures.append(str(diff.get("stderr") or diff.get("error") or "diff against base revision failed"))
    if workspace.dirty or str(status.get("stdout") or "").strip():
        failures.append("worktree is dirty; commit or clean before merge candidate promotion")
    if str(branch.get("stdout") or "").strip() and str(branch.get("stdout") or "").strip() != workspace.branch_name:
        failures.append(
            f"worktree branch mismatch: expected {workspace.branch_name}, "
            f"got {str(branch.get('stdout') or '').strip()}"
        )

    current_head = str(head.get("stdout") or "").strip()
    changed = bool(str(full_diff.get("stdout") or "").strip())
    approved = not failures and (changed or verification_passed is True)

    result = {
        "ok": approved,
        "approved": approved,
        "task_id": task_id,
        "workspace_id": workspace.workspace_id,
        "branch": workspace.branch_name,
        "base_revision": workspace.base_revision,
        "current_head": current_head,
        "diff_stat": str(diff.get("stdout") or ""),
        "has_changes": changed,
        "dirty": bool(str(status.get("stdout") or "").strip()),
        "failures": failures,
        "reviewer_agent_id": reviewer_agent_id,
        "summary": (
            f"Approved merge candidate for {workspace.branch_name} against {workspace.base_revision[:12]}"
            if approved
            else f"Review rejected for task {task_id}: {'; '.join(failures) or 'unknown reason'}"
        ),
    }

    workspace = manager.get_for_task(task_id)
    workspace.review_result = dict(result)
    workspace.current_head = current_head or workspace.current_head
    if approved:
        manager.mark_merge_candidate(task_id, review_result=result)
    else:
        manager.mark_failed(task_id, error=result["summary"], retain=True)
        # mark_failed may transition to retained; keep review payload
        retained = manager.get_for_task(task_id)
        retained.review_result = dict(result)
        manager.store.save(retained)
    return result


def structured_review_from_workspace(workspace: ManagedWorkspace) -> dict[str, Any]:
    return dict(workspace.review_result or {})
