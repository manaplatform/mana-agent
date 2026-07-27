from __future__ import annotations

import json
from pathlib import Path

import typer

from mana_agent.workspaces.impact import ImpactService
from mana_agent.workspaces.models import WorkspaceSearchRequest
from mana_agent.workspaces.relationships import RelationshipService
from mana_agent.workspaces.search import WorkspaceSearchService
from mana_agent.workspaces.service import WorkspaceService
from mana_agent.workspaces.paths import repository_index_dir
from mana_agent.skills.adaptive import RepositoryIdentityService


workspace_app = typer.Typer(help="Manage multi-repository workspaces.")
repo_app = typer.Typer(help="Inspect and refresh registered repositories.")
session_app = typer.Typer(help="Manage isolated Mana-Agent chat sessions.")


def _emit(payload) -> None:  # noqa: ANN001
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str))


@workspace_app.command("create")
def workspace_create(
    name: str = typer.Argument(...),
    root: list[str] = typer.Option([], "--root", help="Root searched for Git repositories."),
    discover: bool = typer.Option(True, "--discover/--no-discover"),
) -> None:
    service = WorkspaceService()
    roots = root or [str(Path.cwd())]
    workspace = service.create_workspace(name, roots=roots, allowed_roots=roots, discover=discover)
    _emit(workspace.model_dump(mode="json"))


@workspace_app.command("list")
def workspace_list() -> None:
    _emit([item.model_dump(mode="json") for item in WorkspaceService().store.list_workspaces()])


@workspace_app.command("show")
def workspace_show(workspace_id: str) -> None:
    service = WorkspaceService()
    workspace = service.store.get_workspace(workspace_id)
    payload = workspace.model_dump(mode="json")
    payload["repositories"] = [service.store.get_repository(item).model_dump(mode="json") for item in workspace.repository_ids]
    payload["relationships"] = [item.model_dump(mode="json") for item in RelationshipService(service.store).list(workspace_id)]
    _emit(payload)


@workspace_app.command("discover")
def workspace_discover(workspace_id: str) -> None:
    _emit([item.model_dump(mode="json") for item in WorkspaceService().discover(workspace_id)])


@workspace_app.command("add-repo")
def workspace_add_repo(
    workspace_id: str,
    path: str,
    external: bool = typer.Option(False, "--external", help="Explicitly authorize a path outside discovery roots."),
) -> None:
    _emit(WorkspaceService().add_repository(workspace_id, path, external=external).model_dump(mode="json"))


@workspace_app.command("remove-repo")
def workspace_remove_repo(workspace_id: str, repository_id: str) -> None:
    _emit(WorkspaceService().remove_repository(workspace_id, repository_id).model_dump(mode="json"))


@workspace_app.command("relationships")
def workspace_relationships(workspace_id: str, refresh: bool = typer.Option(False, "--refresh")) -> None:
    service = WorkspaceService()
    relation_service = RelationshipService(service.store)
    workspace = service.store.get_workspace(workspace_id)
    rows = relation_service.detect(workspace) if refresh else relation_service.list(workspace_id)
    _emit([item.model_dump(mode="json") for item in rows])


@workspace_app.command("delete")
def workspace_delete(workspace_id: str) -> None:
    WorkspaceService().store.delete_workspace(workspace_id)
    _emit({"ok": True, "workspace_id": workspace_id, "repository_files_deleted": False})


@repo_app.command("list")
def repo_list() -> None:
    _emit([item.model_dump(mode="json") for item in WorkspaceService().store.list_repositories()])


@repo_app.command("show")
def repo_show(repository_id: str) -> None:
    _emit(WorkspaceService().store.get_repository(repository_id).model_dump(mode="json"))


@repo_app.command("refresh")
def repo_refresh(path_or_id: str) -> None:
    service = WorkspaceService()
    try:
        path = service.store.get_repository(path_or_id).canonical_path
    except FileNotFoundError:
        path = path_or_id
    _emit(service.register_repository(path, refresh=True).model_dump(mode="json"))


@repo_app.command("identity")
def repo_identity(
    root: str = typer.Option(".", "--repo", "--root-dir"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show the stable adaptive-skill identity for a checkout."""
    identity = RepositoryIdentityService().identify(root)
    if json_output:
        _emit(identity.model_dump(mode="json"))
    else:
        typer.echo(f"{identity.repository_id}  {identity.display_name}")


@repo_app.command("relink")
def repo_relink(repository_id: str, root: str = typer.Option(".", "--repo", "--root-dir")) -> None:
    _emit(RepositoryIdentityService().relink(repository_id, root).model_dump(mode="json"))


@repo_app.command("index")
def repo_index(repository_id: str, rebuild: bool = typer.Option(False, "--rebuild")) -> None:
    from mana_agent.commands.cli_internal import Settings, build_index_service

    service = WorkspaceService()
    repo = service.store.get_repository(repository_id)
    result = build_index_service(Settings()).index(
        repo.canonical_path,
        repository_index_dir(repository_id),
        rebuild=rebuild,
        repository_id=repository_id,
        repository_name=repo.name,
    )
    _emit(result)


@session_app.command("new")
def session_new(
    root: str = typer.Option(".", "--root-dir", "--repo"),
    workspace_id: str | None = typer.Option(None, "--workspace"),
) -> None:
    _emit(WorkspaceService().create_session(root, workspace_id=workspace_id).model_dump(mode="json"))


@session_app.command("list")
def session_list() -> None:
    _emit([item.model_dump(mode="json") for item in WorkspaceService().store.list_sessions()])


@session_app.command("show")
def session_show(session_id: str) -> None:
    _emit(WorkspaceService().store.get_session(session_id).model_dump(mode="json"))


@session_app.command("archive")
def session_archive(session_id: str) -> None:
    _emit(WorkspaceService().archive_session(session_id).model_dump(mode="json"))


@session_app.command("rename")
def session_rename(session_id: str, title: str) -> None:
    from mana_agent.sessions import SessionService

    _emit(SessionService().rename(session_id, title).model_dump(mode="json"))


@session_app.command("delete")
def session_delete(session_id: str, yes: bool = typer.Option(False, "--yes", help="Confirm permanent deletion.")) -> None:
    from mana_agent.sessions import SessionService

    if not yes:
        raise typer.BadParameter("Permanent session deletion requires --yes.")
    SessionService().delete(session_id)
    _emit({"ok": True, "session_id": session_id})


@session_app.command("switch")
def session_switch(session_id: str) -> None:
    session = WorkspaceService().store.get_session(session_id)
    _emit(
        {
            "session": session.model_dump(mode="json"),
            "resume_command": f"mana-agent chat --root-dir {json.dumps(session.cwd)} --session {session.session_id}",
        }
    )


def search_command(
    query: str = typer.Argument(...),
    workspace_id: str = typer.Option(..., "--workspace"),
    mode: str = typer.Option("text", "--mode"),
    repository: list[str] = typer.Option([], "--repo"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    if mode == "semantic":
        from mana_agent.commands.cli_internal import Settings, build_search_service

        semantic = build_search_service(Settings())
    else:
        semantic = None
    request = WorkspaceSearchRequest(
        workspace_id=workspace_id,
        query=query,
        mode=mode,  # type: ignore[arg-type]
        repository_ids=repository,
        limit=limit,
    )
    _emit(WorkspaceSearchService(semantic=semantic).search(request))


def impact_command(
    workspace_id: str = typer.Option(..., "--workspace"),
    repository_id: str = typer.Option(..., "--repo"),
    path: list[str] = typer.Option(..., "--path"),
    max_depth: int = typer.Option(3, "--max-depth"),
) -> None:
    _emit(ImpactService().analyze(workspace_id, repository_id, path, max_depth=max_depth).model_dump(mode="json"))
