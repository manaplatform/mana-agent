"""CLI for Mana-managed agent worktrees."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from mana_agent.multi_agent.worktrees import WorkspaceError, WorkspaceManager
from mana_agent.workspaces.paths import repository_id_for_path

worktree_app = typer.Typer(
    help="Manage isolated Git worktrees used by Mana coding agents.",
    no_args_is_help=True,
)


def _emit(payload) -> None:  # noqa: ANN001
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str))


def _resolve_root(root: str | None) -> Path:
    path = Path(root or ".").expanduser().resolve()
    if path.is_file():
        path = path.parent
    return path


def _manager(root: str | None) -> WorkspaceManager:
    repo = _resolve_root(root)
    return WorkspaceManager(repo, repository_id=repository_id_for_path(repo))


@worktree_app.command("list")
def worktree_list(
    root: str | None = typer.Option(None, "--root-dir", "--repo", help="Source repository checkout."),
    reconcile: bool = typer.Option(True, "--reconcile/--no-reconcile"),
    plain: bool = typer.Option(False, "--plain", help="Human-readable table instead of JSON."),
) -> None:
    """List managed workspaces for the repository (task, branch, status, path, agent, dirty)."""

    manager = _manager(root)
    rows = [item.list_row() for item in manager.list(reconcile=reconcile)]
    if plain:
        if not rows:
            typer.echo("No managed worktrees.")
            return
        typer.echo(
            f"{'TASK':<28} {'BRANCH':<36} {'STATUS':<16} {'DIRTY':<6} {'AGENT':<22} PATH"
        )
        for row in rows:
            typer.echo(
                f"{str(row.get('task_id') or ''):<28} "
                f"{str(row.get('branch') or ''):<36} "
                f"{str(row.get('status') or ''):<16} "
                f"{str(bool(row.get('dirty'))):<6} "
                f"{str(row.get('assigned_agent') or '')[:22]:<22} "
                f"{row.get('worktree_path') or ''}"
            )
        return
    _emit(rows)


@worktree_app.command("create")
def worktree_create(
    task_id: str = typer.Argument(..., help="Taskboard task id to isolate."),
    root: str | None = typer.Option(None, "--root-dir", "--repo"),
    title: str = typer.Option("", "--title", help="Optional title used in the branch slug."),
    agent_id: str = typer.Option("", "--agent-id", help="Optional assigned coding agent id."),
    resume_existing: bool = typer.Option(True, "--resume-existing/--no-resume-existing"),
) -> None:
    """Create a managed worktree and branch for a coding task."""

    manager = _manager(root)
    try:
        workspace = manager.create_for_task(
            task_id,
            title=title,
            assigned_agent_id=agent_id,
            reuse_existing=resume_existing,
        )
    except WorkspaceError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    _emit(workspace.status_report())


@worktree_app.command("status")
def worktree_status(
    task_id: str = typer.Argument(...),
    root: str | None = typer.Option(None, "--root-dir", "--repo"),
) -> None:
    """Show repository identity, base revision, HEAD, task/git state, and recovery info."""

    manager = _manager(root)
    try:
        _emit(manager.status(task_id))
    except WorkspaceError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@worktree_app.command("resume")
def worktree_resume(
    task_id: str = typer.Argument(...),
    root: str | None = typer.Option(None, "--root-dir", "--repo"),
    agent_id: str = typer.Option("", "--agent-id"),
) -> None:
    """Reconnect an interrupted task to its existing workspace when safe."""

    manager = _manager(root)
    try:
        workspace = manager.resume(task_id, assigned_agent_id=agent_id)
    except WorkspaceError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    _emit(workspace.status_report())


@worktree_app.command("diff")
def worktree_diff(
    task_id: str = typer.Argument(...),
    root: str | None = typer.Option(None, "--root-dir", "--repo"),
    stat: bool = typer.Option(False, "--stat", help="Show diffstat only."),
) -> None:
    """Show changes relative to the recorded task base revision."""

    manager = _manager(root)
    try:
        result = manager.diff(task_id, stat=stat)
    except WorkspaceError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    if result.get("ok"):
        stdout = str(result.get("stdout") or "")
        if stdout.strip():
            typer.echo(stdout)
        else:
            typer.echo("(no changes relative to base revision)")
        return
    _emit(result)
    raise typer.Exit(code=1)


@worktree_app.command("merge")
def worktree_merge(
    task_id: str = typer.Argument(...),
    root: str | None = typer.Option(None, "--root-dir", "--repo"),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm explicit user intent to merge the managed task branch into the source checkout.",
    ),
) -> None:
    """Merge a reviewed merge-candidate branch using the Git safety layer.

    Never force-pushes or rewrites history. Requires --yes for validated intent.
    """

    if not yes:
        typer.secho(
            "merge refused: pass --yes to confirm explicit user intent. "
            "Mana never silently merges managed task branches.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    manager = _manager(root)
    try:
        result = manager.merge(task_id, explicit_user_intent=True)
    except WorkspaceError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    _emit(result)


@worktree_app.command("remove")
def worktree_remove(
    task_id: str = typer.Argument(...),
    root: str | None = typer.Option(None, "--root-dir", "--repo"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Allow destructive cleanup of dirty/unmerged work when combined with --yes.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm explicit user intent for destructive cleanup.",
    ),
    delete_branch: bool = typer.Option(
        False,
        "--delete-branch",
        help="Also delete the local mana/* branch (requires --force --yes).",
    ),
) -> None:
    """Remove a managed worktree. Refuses dirty/unmerged cleanup without force+yes."""

    manager = _manager(root)
    try:
        result = manager.remove(
            task_id,
            force=force,
            delete_branch=delete_branch,
            explicit_user_intent=yes,
        )
    except WorkspaceError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    _emit(result)


@worktree_app.command("reconcile")
def worktree_reconcile(
    root: str | None = typer.Option(None, "--root-dir", "--repo"),
) -> None:
    """Reconcile persisted metadata with git worktree list and the filesystem."""

    manager = _manager(root)
    _emit(manager.reconcile())
