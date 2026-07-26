from __future__ import annotations

import json
import re
from pathlib import Path

from mana_agent.compat import tomllib
from mana_agent.workspaces.models import RepositoryRecord, RepositoryRelationship, WorkspaceRecord
from mana_agent.workspaces.paths import workspace_dir
from mana_agent.workspaces.store import WorkspaceStore, atomic_write_json


def _package_names(repo: RepositoryRecord) -> set[str]:
    root = Path(repo.canonical_path)
    names: set[str] = {repo.name.lower()}
    package = root / "package.json"
    if package.is_file():
        try:
            name = str(json.loads(package.read_text(encoding="utf-8")).get("name") or "").lower()
            if name:
                names.add(name)
        except Exception:
            pass
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            name = str(tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project", {}).get("name") or "").lower()
            if name:
                names.add(name)
        except Exception:
            pass
    for filename, pattern in (
        ("Cargo.toml", r"(?m)^name\s*=\s*[\"']([^\"']+)[\"']"),
        ("pubspec.yaml", r"(?m)^name:\s*([^\s#]+)"),
        ("go.mod", r"(?m)^module\s+([^\s]+)"),
        ("composer.json", r'"name"\s*:\s*"([^"]+)"'),
    ):
        manifest = root / filename
        if manifest.is_file():
            match = re.search(pattern, manifest.read_text(encoding="utf-8", errors="ignore"))
            if match:
                names.add(match.group(1).strip().lower())
    return names


class RelationshipService:
    def __init__(self, store: WorkspaceStore | None = None) -> None:
        self.store = store or WorkspaceStore()

    def detect(self, workspace: WorkspaceRecord) -> list[RepositoryRelationship]:
        repos = [self.store.get_repository(item) for item in workspace.repository_ids]
        names = {repo.repository_id: _package_names(repo) for repo in repos}
        rows: dict[tuple[str, str, str], RepositoryRelationship] = {}
        for source in repos:
            root = Path(source.canonical_path)
            package = root / "package.json"
            dependencies: set[str] = set()
            if package.is_file():
                try:
                    payload = json.loads(package.read_text(encoding="utf-8"))
                    dependencies = {str(name).lower() for key in ("dependencies", "devDependencies", "peerDependencies") for name in (payload.get(key, {}) or {})}
                except Exception:
                    pass
            for target in repos:
                if source.repository_id == target.repository_id:
                    continue
                matched = dependencies & names[target.repository_id]
                if matched:
                    key = (source.repository_id, target.repository_id, "declared_dependency")
                    rows[key] = RepositoryRelationship(
                        workspace_id=workspace.workspace_id,
                        source_repository_id=source.repository_id,
                        target_repository_id=target.repository_id,
                        kind="declared_dependency",
                        evidence=[f"{source.name}::package.json dependency {name}" for name in sorted(matched)],
                    )
                for manifest_name in ("pyproject.toml", "Cargo.toml", "pubspec.yaml", "go.mod", "composer.json"):
                    manifest = root / manifest_name
                    if not manifest.is_file():
                        continue
                    text = manifest.read_text(encoding="utf-8", errors="ignore").lower()
                    exact = [
                        name
                        for name in names[target.repository_id]
                        if re.search(rf"(?<![\w@/.-]){re.escape(name)}(?![\w.-])", text)
                    ]
                    if not exact:
                        continue
                    kind = "path_dependency" if re.search(rf"{re.escape(exact[0])}[^\n]{{0,160}}path", text) or re.search(rf"path[^\n]{{0,160}}{re.escape(exact[0])}", text) else "declared_dependency"
                    key = (source.repository_id, target.repository_id, kind)
                    rows[key] = RepositoryRelationship(
                        workspace_id=workspace.workspace_id,
                        source_repository_id=source.repository_id,
                        target_repository_id=target.repository_id,
                        kind=kind,  # type: ignore[arg-type]
                        evidence=[f"{source.name}::{manifest_name} dependency {name}" for name in sorted(exact)],
                    )
            gitmodules = root / ".gitmodules"
            if gitmodules.is_file():
                text = gitmodules.read_text(encoding="utf-8", errors="ignore")
                for target in repos:
                    if target.repository_id != source.repository_id and target.remote_url and target.remote_url in text:
                        key = (source.repository_id, target.repository_id, "git_submodule")
                        rows[key] = RepositoryRelationship(
                            workspace_id=workspace.workspace_id,
                            source_repository_id=source.repository_id,
                            target_repository_id=target.repository_id,
                            kind="git_submodule",
                            evidence=[f"{source.name}::.gitmodules"],
                        )
        result = sorted(rows.values(), key=lambda item: (item.source_repository_id, item.target_repository_id, item.kind))
        atomic_write_json(
            workspace_dir(workspace.workspace_id) / "relationships.json",
            {"schema_version": 1, "relationships": [item.model_dump(mode="json") for item in result]},
        )
        return result

    def list(self, workspace_id: str) -> list[RepositoryRelationship]:
        path = workspace_dir(workspace_id) / "relationships.json"
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [RepositoryRelationship.model_validate(item) for item in payload.get("relationships", [])]
