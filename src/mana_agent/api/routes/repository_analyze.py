"""Repository-local analyze start/status/results for the dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Header
from pydantic import BaseModel, Field

from mana_agent.api.exceptions import ManaApiError
from mana_agent.api.services.job_service import ApiJobStore
from mana_agent.services.execution_event_hub import get_execution_event_hub
from mana_agent.ui.streamlit_helpers import find_mana_root, list_analysis_artifacts, safe_read_json
from mana_agent.utils.path_safety import (
    env_allowed_roots,
    resolve_user_path,
    resolve_within_allowed_roots,
)
from mana_agent.workspaces.paths import repository_analysis_dir, repository_id_for_path
from mana_agent.workspaces.service import WorkspaceService

router = APIRouter(prefix="/api/v1", tags=["analyze"])


def _require_mutation_token(authorization: str | None) -> None:
    expected = str(os.getenv("MANA_API_TOKEN") or "").strip()
    if expected and authorization != f"Bearer {expected}":
        raise ManaApiError(401, "A valid API bearer token is required.")


class RepositoryAnalyzeRequest(BaseModel):
    depth: Literal["quick", "normal", "full"] = "normal"
    language: Literal["en", "fa"] = "en"
    with_llm: bool = True
    conversation_id: str = ""
    root: str | None = None


def _local_default_roots() -> list[Path]:
    import tempfile

    candidates = [
        Path.home(),
        Path.cwd(),
        find_mana_root(),
        Path(tempfile.gettempdir()),
    ]
    roots: list[Path] = []
    for item in candidates:
        try:
            roots.append(item.resolve(strict=False))
        except OSError:
            continue
    return roots


def _authorize_root_path(raw: str) -> Path:
    allowed = env_allowed_roots() or _local_default_roots()
    try:
        return resolve_within_allowed_roots(raw, allowed, require_allowlist=True)
    except PermissionError as exc:
        raise ManaApiError(403, "Path is outside the configured allowlist.") from exc
    except ValueError as exc:
        raise ManaApiError(400, "Invalid path.") from exc


def _repo_root(repository_id: str, root: str | None = None) -> Path:
    if root:
        return find_mana_root(_authorize_root_path(root))
    try:
        repo = WorkspaceService().store.get_repository(repository_id)
        return resolve_user_path(repo.canonical_path)
    except FileNotFoundError:
        # Compatibility path-id repositories may not be registered yet.
        path = find_mana_root()
        if repository_id_for_path(path) == repository_id:
            return path
        raise ManaApiError(404, "Repository not found.")
    except ValueError as exc:
        raise ManaApiError(400, "Invalid repository path.") from exc


@router.post("/repositories/{repository_id}/analyze", status_code=202)
def start_repository_analyze(
    repository_id: str,
    payload: RepositoryAnalyzeRequest,
    background: BackgroundTasks,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    _require_mutation_token(authorization)
    root = _repo_root(repository_id, payload.root)
    # Prefer registry id when available.
    resolved_id = repository_id_for_path(root)
    jobs = ApiJobStore()
    job = jobs.create(
        "repository_analyze",
        {
            "repository_id": resolved_id,
            "root": str(root),
            "depth": payload.depth,
            "language": payload.language,
            "with_llm": payload.with_llm,
            "conversation_id": payload.conversation_id,
        },
    )
    hub = get_execution_event_hub()
    conversation_id = str(payload.conversation_id or "").strip()
    execution_id = job["job_id"]

    def operation() -> dict[str, Any]:
        from mana_agent.services.project_analyze_service import (
            ProjectAnalyzeOptions,
            ProjectAnalyzeService,
        )

        if conversation_id:
            hub.emit(
                "step.started",
                title="Analyze started",
                conversation_id=conversation_id,
                execution_id=execution_id,
                repository_id=resolved_id,
                message=f"depth={payload.depth}",
                status="running",
                metadata={"tool_name": "analyze"},
            )
        artifact_dir = repository_analysis_dir(resolved_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        llm_analyzer = None
        if payload.with_llm:
            try:
                from mana_agent.commands.cli_internal import _build_project_llm_analyzer

                llm_analyzer = _build_project_llm_analyzer()
            except Exception:
                llm_analyzer = None
        result = ProjectAnalyzeService().run(
            root,
            artifact_dir,
            options=ProjectAnalyzeOptions(depth=payload.depth, language=payload.language, output_format="both"),
            llm_analyzer=llm_analyzer,
        )
        artifacts = list(getattr(result, "artifacts", {}) or {})
        if conversation_id:
            hub.emit(
                "step.finished",
                title="Analyze finished",
                conversation_id=conversation_id,
                execution_id=execution_id,
                repository_id=resolved_id,
                message=f"artifacts={len(artifacts)}",
                status="success",
                metadata={"tool_name": "analyze", "artifacts": artifacts[:12]},
            )
        agent_context = safe_read_json(artifact_dir / "agent_context.json") or {}
        return {
            "repository_id": resolved_id,
            "artifact_dir": str(artifact_dir),
            "artifacts": artifacts[:20],
            "llm_used": llm_analyzer is not None,
            "last_analyzed_at": agent_context.get("last_analyzed_at"),
            "summary": {
                "project_summary": agent_context.get("project_summary"),
                "detected_stack": agent_context.get("detected_stack") or [],
                "risks": (agent_context.get("risks") or [])[:8],
                "recommended_tasks": (agent_context.get("recommended_tasks") or [])[:8],
            },
            "errors": list(getattr(result, "errors", []) or []),
        }

    def _run() -> None:
        try:
            jobs.run(job["job_id"], operation)
        except Exception as exc:
            if conversation_id:
                hub.emit(
                    "error",
                    title="Analyze failed",
                    conversation_id=conversation_id,
                    execution_id=execution_id,
                    repository_id=resolved_id,
                    message="Analyze failed.",
                    status="failed",
                    metadata={"tool_name": "analyze"},
                )
            raise

    background.add_task(_run)
    return job


@router.get("/repositories/{repository_id}/analysis")
def get_repository_analysis(repository_id: str, root: str | None = None) -> dict[str, Any]:
    artifact_dir = repository_analysis_dir(repository_id)
    path: Path | None = None
    try:
        path = _repo_root(repository_id, root)
        if not artifact_dir.exists():
            artifact_dir = repository_analysis_dir(repository_id_for_path(path))
    except ManaApiError:
        if not artifact_dir.exists():
            path = find_mana_root()
            if repository_id_for_path(path) == repository_id:
                artifact_dir = repository_analysis_dir(repository_id)
            elif not artifact_dir.exists():
                raise ManaApiError(404, "Repository analysis not found.")
    agent_context = safe_read_json(artifact_dir / "agent_context.json") or {}
    report = safe_read_json(artifact_dir / "report.json") or {}
    artifacts = list_analysis_artifacts(path) if path is not None else []
    if not artifacts and artifact_dir.exists():
        for f in sorted(artifact_dir.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
            if f.is_file() and f.suffix.lower() in {".md", ".json", ".html", ".txt"}:
                artifacts.append(
                    {
                        "path": str(f),
                        "name": f.name,
                        "type": f.suffix.lstrip(".").lower(),
                        "size": f.stat().st_size,
                    }
                )
    status = "ready" if artifact_dir.exists() and (agent_context or report or artifacts) else "missing"
    return {
        "ok": True,
        "status": status,
        "repository_id": repository_id,
        "artifact_dir": str(artifact_dir),
        "agent_context": agent_context,
        "report_preview": {
            "project_summary": report.get("project_summary") or agent_context.get("project_summary"),
            "generated_at": report.get("generated_at") or agent_context.get("last_analyzed_at"),
        },
        "artifacts": artifacts,
    }


@router.get("/repositories/{repository_id}/analysis/artifacts/{name}")
def get_analysis_artifact(repository_id: str, name: str, root: str | None = None) -> dict[str, Any]:
    safe_name = Path(name).name
    if safe_name != name or ".." in name:
        raise ManaApiError(400, "Invalid artifact name.")
    artifact_dir = repository_analysis_dir(repository_id)
    target = artifact_dir / safe_name
    if not target.exists() or not target.is_file():
        # Fall back to listing under root
        try:
            path = _repo_root(repository_id, root)
        except ManaApiError as exc:
            raise ManaApiError(404, "Artifact not found.") from exc
        arts = list_analysis_artifacts(path)
        match = next((item for item in arts if item.get("name") == safe_name), None)
        if not match:
            raise ManaApiError(404, "Artifact not found.")
        target = Path(str(match["path"]))
    text = target.read_text(encoding="utf-8", errors="replace")
    if target.suffix.lower() == ".json":
        try:
            return {"ok": True, "name": safe_name, "type": "json", "data": json.loads(text)}
        except json.JSONDecodeError:
            pass
    return {"ok": True, "name": safe_name, "type": target.suffix.lstrip(".") or "text", "content": text[:100_000]}
