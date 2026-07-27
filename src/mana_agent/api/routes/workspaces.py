from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Header
from pydantic import BaseModel, Field

from mana_agent.api.exceptions import ManaApiError
from mana_agent.api.services.job_service import ApiJobStore
from mana_agent.workspaces.impact import ImpactService
from mana_agent.workspaces.models import WorkspaceSearchRequest
from mana_agent.workspaces.paths import repository_index_dir
from mana_agent.workspaces.relationships import RelationshipService
from mana_agent.workspaces.search import WorkspaceSearchService
from mana_agent.workspaces.service import WorkspaceService


router = APIRouter(prefix="/api/v1", tags=["workspaces"])


class WorkspaceCreateRequest(BaseModel):
    name: str
    roots: list[str] = Field(default_factory=list)
    discover: bool = True


class RepositoryAddRequest(BaseModel):
    path: str
    external: bool = False


class SessionCreateRequest(BaseModel):
    cwd: str
    workspace_id: str | None = None


class SessionUpdateRequest(BaseModel):
    title: str


class SearchRequest(BaseModel):
    query: str
    mode: Literal["semantic", "text", "file", "symbol"] = "text"
    repository_ids: list[str] = Field(default_factory=list)
    limit: int = Field(default=20, ge=1, le=500)


class ImpactRequest(BaseModel):
    repository_id: str
    paths: list[str]
    max_depth: int = Field(default=3, ge=0, le=10)


def _allowed_roots() -> list[Path]:
    raw = str(os.getenv("MANA_WORKSPACE_ALLOWED_ROOTS") or "")
    values = [item.strip() for item in re.split(r"[," + re.escape(os.pathsep) + r"]", raw) if item.strip()]
    return [Path(item).expanduser().resolve() for item in values]


def _authorize_path(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    allowed = _allowed_roots()
    if not allowed:
        raise ManaApiError(403, "Workspace path API is disabled until MANA_WORKSPACE_ALLOWED_ROOTS is configured.")
    if not any(path == root or root in path.parents for root in allowed):
        raise ManaApiError(403, "Path is outside MANA_WORKSPACE_ALLOWED_ROOTS.")
    return path


def _require_mutation_token(authorization: str | None) -> None:
    expected = str(os.getenv("MANA_API_TOKEN") or "").strip()
    if expected and authorization != f"Bearer {expected}":
        raise ManaApiError(401, "A valid API bearer token is required.")


@router.get("/workspaces")
def list_workspaces() -> list[dict]:
    return [item.model_dump(mode="json") for item in WorkspaceService().store.list_workspaces()]


@router.post("/workspaces", status_code=201)
def create_workspace(payload: WorkspaceCreateRequest, authorization: str | None = Header(None)) -> dict:
    _require_mutation_token(authorization)
    roots = [_authorize_path(item) for item in payload.roots]
    workspace = WorkspaceService().create_workspace(
        payload.name,
        roots=roots,
        allowed_roots=roots,
        discover=payload.discover,
    )
    return workspace.model_dump(mode="json")


@router.get("/workspaces/{workspace_id}")
def get_workspace(workspace_id: str) -> dict:
    service = WorkspaceService()
    try:
        workspace = service.store.get_workspace(workspace_id)
    except FileNotFoundError as exc:
        raise ManaApiError(404, "Workspace not found.") from exc
    data = workspace.model_dump(mode="json")
    data["repositories"] = [service.store.get_repository(item).model_dump(mode="json") for item in workspace.repository_ids]
    return data


@router.delete("/workspaces/{workspace_id}")
def delete_workspace(workspace_id: str, authorization: str | None = Header(None)) -> dict:
    _require_mutation_token(authorization)
    WorkspaceService().store.delete_workspace(workspace_id)
    return {"ok": True, "repository_files_deleted": False}


@router.post("/workspaces/{workspace_id}/repositories", status_code=201)
def add_repository(workspace_id: str, payload: RepositoryAddRequest, authorization: str | None = Header(None)) -> dict:
    _require_mutation_token(authorization)
    path = _authorize_path(payload.path)
    return WorkspaceService().add_repository(workspace_id, path, external=payload.external).model_dump(mode="json")


@router.delete("/workspaces/{workspace_id}/repositories/{repository_id}")
def remove_repository(workspace_id: str, repository_id: str, authorization: str | None = Header(None)) -> dict:
    _require_mutation_token(authorization)
    return WorkspaceService().remove_repository(workspace_id, repository_id).model_dump(mode="json")


@router.post("/workspaces/{workspace_id}/discover")
def discover(workspace_id: str, authorization: str | None = Header(None)) -> list[dict]:
    _require_mutation_token(authorization)
    return [item.model_dump(mode="json") for item in WorkspaceService().discover(workspace_id)]


@router.get("/workspaces/{workspace_id}/relationships")
def relationships(workspace_id: str, refresh: bool = False) -> list[dict]:
    service = WorkspaceService()
    relation_service = RelationshipService(service.store)
    workspace = service.store.get_workspace(workspace_id)
    rows = relation_service.detect(workspace) if refresh else relation_service.list(workspace_id)
    return [item.model_dump(mode="json") for item in rows]


@router.post("/workspaces/{workspace_id}/search")
def search(workspace_id: str, payload: SearchRequest) -> dict:
    semantic = None
    if payload.mode == "semantic":
        from mana_agent.commands.cli_internal import Settings, build_search_service

        semantic = build_search_service(Settings())
    return WorkspaceSearchService(semantic=semantic).search(
        WorkspaceSearchRequest(
            workspace_id=workspace_id,
            query=payload.query,
            mode=payload.mode,
            repository_ids=payload.repository_ids,
            limit=payload.limit,
        )
    )


@router.post("/workspaces/{workspace_id}/impact")
def impact(workspace_id: str, payload: ImpactRequest) -> dict:
    return ImpactService().analyze(
        workspace_id,
        payload.repository_id,
        payload.paths,
        max_depth=payload.max_depth,
    ).model_dump(mode="json")


@router.get("/repositories")
def repositories() -> list[dict]:
    return [item.model_dump(mode="json") for item in WorkspaceService().store.list_repositories()]


@router.post("/repositories/{repository_id}/index", status_code=202)
def index_repository(repository_id: str, background: BackgroundTasks, authorization: str | None = Header(None)) -> dict:
    _require_mutation_token(authorization)
    service = WorkspaceService()
    repo = service.store.get_repository(repository_id)
    jobs = ApiJobStore()
    job = jobs.create("repository_index", {"repository_id": repository_id})

    def operation() -> dict:
        from mana_agent.commands.cli_internal import Settings, build_index_service

        return build_index_service(Settings()).index(
            repo.canonical_path,
            repository_index_dir(repository_id),
            repository_id=repository_id,
            repository_name=repo.name,
        )

    background.add_task(jobs.run, job["job_id"], operation)
    return job


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        return ApiJobStore().get(job_id)
    except FileNotFoundError as exc:
        raise ManaApiError(404, "Job not found.") from exc


@router.post("/sessions", status_code=201)
def create_session(payload: SessionCreateRequest, authorization: str | None = Header(None)) -> dict:
    from mana_agent.sessions import SessionService

    _require_mutation_token(authorization)
    cwd = _authorize_path(payload.cwd)
    return SessionService().create(cwd, workspace_id=payload.workspace_id, frontend="api").model_dump(mode="json")


@router.get("/sessions")
def list_sessions() -> list[dict]:
    from mana_agent.sessions import SessionService

    return [item.model_dump(mode="json") for item in SessionService().list()]


@router.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    from mana_agent.sessions import SessionService

    service = SessionService()
    try:
        record = service.workspaces.store.get_session(session_id)
    except FileNotFoundError as exc:
        raise ManaApiError(404, "Session not found.") from exc
    return {
        "session": service.summary(record).model_dump(mode="json"),
        "messages": [row.to_dict() for row in service.history.list(session_id, limit=5000)],
    }


@router.patch("/sessions/{session_id}")
def update_session(session_id: str, payload: SessionUpdateRequest, authorization: str | None = Header(None)) -> dict:
    from mana_agent.sessions import SessionService

    _require_mutation_token(authorization)
    try:
        return SessionService().rename(session_id, payload.title).model_dump(mode="json")
    except FileNotFoundError as exc:
        raise ManaApiError(404, "Session not found.") from exc


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, authorization: str | None = Header(None)) -> dict:
    from mana_agent.sessions import SessionService

    _require_mutation_token(authorization)
    try:
        SessionService().delete(session_id)
    except FileNotFoundError as exc:
        raise ManaApiError(404, "Session not found.") from exc
    return {"ok": True, "session_id": session_id}


@router.post("/sessions/{session_id}/activate")
def activate_session(session_id: str, authorization: str | None = Header(None)) -> dict:
    from mana_agent.sessions import SessionService

    _require_mutation_token(authorization)
    try:
        return SessionService().bind(session_id, frontend="api").model_dump(mode="json")
    except FileNotFoundError as exc:
        raise ManaApiError(404, "Session not found.") from exc
