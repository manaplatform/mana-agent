from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from mana_agent.tools.repository import find_symbols, list_files, repo_search
from mana_agent.workspaces.models import WorkspaceSearchRequest
from mana_agent.workspaces.paths import repository_index_dir
from mana_agent.workspaces.store import WorkspaceStore


class WorkspaceSearchService:
    def __init__(self, *, store: WorkspaceStore | None = None) -> None:
        self.store = store or WorkspaceStore()

    def search(self, request: WorkspaceSearchRequest) -> dict[str, Any]:
        workspace = self.store.get_workspace(request.workspace_id)
        repository_ids = request.repository_ids or list(workspace.repository_ids)
        unknown = set(repository_ids) - set(workspace.repository_ids)
        if unknown:
            raise PermissionError(f"repositories are outside workspace: {', '.join(sorted(unknown))}")
        results: list[dict[str, Any]] = []
        for repository_id in repository_ids:
            repo = self.store.get_repository(repository_id)
            root = Path(repo.canonical_path)
            if request.mode == "text":
                payload = repo_search(root, query=request.query, limit=request.limit)
                for item in payload.get("matches", []):
                    results.append(
                        {
                            **item,
                            "repository_id": repository_id,
                            "repository_name": repo.name,
                            "qualified_path": f"{repo.name}::{item['file']}",
                        }
                    )
            elif request.mode == "file":
                payload = list_files(root, glob="**/*", limit=max(request.limit * 10, 100))
                for rel in payload.get("files", []):
                    if request.query.lower() in rel.lower() or fnmatch.fnmatch(rel, request.query):
                        results.append(
                            {
                                "file": rel,
                                "repository_id": repository_id,
                                "repository_name": repo.name,
                                "qualified_path": f"{repo.name}::{rel}",
                            }
                        )
            elif request.mode == "symbol":
                payload = find_symbols(root, query=request.query, limit=request.limit)
                for item in payload.get("symbols", []):
                    results.append(
                        {
                            **item,
                            "repository_id": repository_id,
                            "repository_name": repo.name,
                            "qualified_path": f"{repo.name}::{item['file']}",
                        }
                    )
            if len(results) >= request.limit:
                break
        return {"mode": request.mode, "results": results[: request.limit], "warnings": []}
