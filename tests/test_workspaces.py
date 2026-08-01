from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mana_agent.multi_agent.runtime.tool_worker_process import ToolRunRequest, ToolWorkerClient, ToolWorkerProcessError
from mana_agent.services.memory_service import MultiAgentMemoryService
from mana_agent.workspaces.impact import ImpactService
from mana_agent.workspaces.models import WorkspaceSearchRequest
from mana_agent.workspaces.relationships import RelationshipService
from mana_agent.workspaces.routing import RepositoryScopeDecisionEngine
from mana_agent.workspaces.search import WorkspaceSearchService
from mana_agent.workspaces.service import WorkspaceService
from mana_agent.workspaces.store import atomic_write_json


def _git(path: Path, *args: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["git", *args], cwd=path, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def _repo(path: Path, files: dict[str, str] | None = None) -> Path:
    _git(path, "init")
    _git(path, "config", "user.name", "Mana Test")
    _git(path, "config", "user.email", "mana@example.test")
    for name, content in (files or {"README.md": "test\n"}).items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "initial")
    return path


def test_atomic_workspace_write_retries_transient_windows_access_denial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "session.json"
    atomic_write_json(target, {"status": "active"})
    real_replace = os.replace
    replace_calls = 0

    def transiently_denied(source: str | Path, destination: str | Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 1:
            raise PermissionError(13, "Access is denied")
        real_replace(source, destination)

    monkeypatch.setattr("mana_agent.workspaces.store.os.replace", transiently_denied)

    atomic_write_json(target, {"status": "closed"})

    assert replace_calls == 2
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "closed"}
    assert not list(tmp_path.glob(".session.json.*.tmp"))


def test_atomic_workspace_write_recovers_when_session_directory_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "sessions" / "session_test" / "session.json"
    real_replace = os.replace
    replace_calls = 0

    def removed_directory_once(source: str | Path, destination: str | Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 1:
            Path(source).unlink()
            target.parent.rmdir()
            raise FileNotFoundError(2, "No such file or directory", str(destination))
        real_replace(source, destination)

    monkeypatch.setattr("mana_agent.workspaces.store.os.replace", removed_directory_once)

    atomic_write_json(target, {"status": "active"})

    assert replace_calls == 2
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "active"}


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "mana-home"
    monkeypatch.setenv("MANA_HOME", str(home))
    return home


def test_workspace_discovers_repositories_and_preserves_stable_identity(tmp_path: Path, isolated_home: Path) -> None:
    root = tmp_path / "projects"
    first = _repo(root / "api")
    _repo(root / "web")
    service = WorkspaceService()
    workspace = service.create_workspace("product", roots=[root], allowed_roots=[root], discover=True)

    assert len(workspace.repository_ids) == 2
    initial = service.register_repository(first)
    refreshed = service.register_repository(first)
    assert refreshed.repository_id == initial.repository_id
    assert not (first / ".mana").exists()


def test_automatic_repository_and_workspace_are_reused_but_each_start_gets_a_new_session(
    tmp_path: Path, isolated_home: Path
) -> None:
    repo_path = _repo(tmp_path / "repo")

    first_service = WorkspaceService()
    first_repo = first_service.register_repository(repo_path)
    first_workspace = first_service.workspace_for_repository(first_repo.repository_id)
    first_session = first_service.restore_or_create_session(repo_path)

    second_service = WorkspaceService()
    second_repo = second_service.register_repository(repo_path)
    second_workspace = second_service.workspace_for_repository(second_repo.repository_id)
    second_session = second_service.restore_or_create_session(repo_path)

    assert second_repo.repository_id == first_repo.repository_id
    assert second_workspace.workspace_id == first_workspace.workspace_id
    assert second_session.session_id != first_session.session_id
    assert second_service.store.get_session(first_session.session_id).status == "abandoned"
    assert second_session.status == "active"
    assert len(second_service.store.list_repositories()) == 1
    assert len(second_service.store.list_workspaces()) == 1
    assert len(second_service.store.list_sessions()) == 2


def test_implicit_workspace_replaces_missing_legacy_repository_identity(
    tmp_path: Path, isolated_home: Path
) -> None:
    repo_path = _repo(tmp_path / "repo")
    service = WorkspaceService()
    repo = service.register_repository(repo_path)
    workspace = service.workspace_for_repository(repo.repository_id)
    workspace.repository_ids = ["repo_missing_legacy"]
    workspace.primary_repository_id = "repo_missing_legacy"
    service.store.save_workspace(workspace)

    restored = service.workspace_for_repository(repo.repository_id)

    assert restored.repository_ids == [repo.repository_id]
    assert restored.primary_repository_id == repo.repository_id


def test_session_context_removes_missing_non_primary_repository_references(
    tmp_path: Path, isolated_home: Path
) -> None:
    repo_path = _repo(tmp_path / "repo")
    service = WorkspaceService()
    repo = service.register_repository(repo_path)
    workspace = service.workspace_for_repository(repo.repository_id)
    workspace.repository_ids.append("repo_missing_secondary")
    service.store.save_workspace(workspace)
    session = service.create_session(repo_path, workspace_id=workspace.workspace_id)

    context = service.context_for_session(session.session_id)

    assert list(context.repositories) == [repo.repository_id]
    assert context.session.attached_repository_ids == [repo.repository_id]
    assert context.workspace.repository_ids == [repo.repository_id]


def test_session_context_does_not_hide_missing_primary_repository(
    tmp_path: Path, isolated_home: Path
) -> None:
    repo_path = _repo(tmp_path / "repo")
    service = WorkspaceService()
    repo = service.register_repository(repo_path)
    workspace = service.workspace_for_repository(repo.repository_id)
    session = service.create_session(repo_path, workspace_id=workspace.workspace_id)
    (isolated_home / "repositories" / repo.repository_id / "repository.json").unlink()

    with pytest.raises(FileNotFoundError):
        service.context_for_session(session.session_id)


def test_legacy_restore_api_abandons_active_sessions_and_opens_a_fresh_one(
    tmp_path: Path, isolated_home: Path
) -> None:
    repo_path = _repo(tmp_path / "repo")
    service = WorkspaceService()
    repo = service.register_repository(repo_path)
    workspace = service.workspace_for_repository(repo.repository_id)
    older = service.create_session(repo_path, workspace_id=workspace.workspace_id)
    newer = service.create_session(repo_path, workspace_id=workspace.workspace_id)

    restored = service.restore_or_create_session(repo_path)
    sessions = service.store.list_sessions()

    assert restored.session_id not in {older.session_id, newer.session_id}
    assert [item.session_id for item in sessions if item.status == "active"] == [restored.session_id]
    assert service.store.get_session(older.session_id).status == "abandoned"
    assert service.store.get_session(newer.session_id).status == "abandoned"


def test_sessions_and_repository_memory_are_isolated(tmp_path: Path, isolated_home: Path) -> None:
    repo_path = _repo(tmp_path / "repo")
    service = WorkspaceService()
    repo = service.register_repository(repo_path)
    workspace = service.workspace_for_repository(repo.repository_id)
    one = service.create_session(repo_path, workspace_id=workspace.workspace_id)
    two = service.create_session(repo_path, workspace_id=workspace.workspace_id)
    first = MultiAgentMemoryService(
        root=repo_path,
        workspace_id=workspace.workspace_id,
        repository_id=repo.repository_id,
        session_id=one.session_id,
    )
    normalized, fingerprint = first.normalize_task(goal="private session task", repository_ids=[repo.repository_id])
    first.register_task(task_id="task_one", normalized_goal=normalized, fingerprint=fingerprint)
    second = MultiAgentMemoryService(
        root=repo_path,
        workspace_id=workspace.workspace_id,
        repository_id=repo.repository_id,
        session_id=two.session_id,
    )
    assert "task_one" not in second.task_records


def test_cross_repo_search_qualifies_same_relative_path(tmp_path: Path, isolated_home: Path) -> None:
    first_path = _repo(tmp_path / "api", {"src/config.py": "WORKSPACE_NEEDLE = 1\n"})
    second_path = _repo(tmp_path / "worker", {"src/config.py": "WORKSPACE_NEEDLE = 2\n"})
    service = WorkspaceService()
    workspace = service.create_workspace("product", roots=[tmp_path], allowed_roots=[tmp_path])
    first = service.add_repository(workspace.workspace_id, first_path)
    second = service.add_repository(workspace.workspace_id, second_path)

    result = WorkspaceSearchService().search(
        WorkspaceSearchRequest(
            workspace_id=workspace.workspace_id,
            query="WORKSPACE_NEEDLE",
            mode="text",
            repository_ids=[first.repository_id, second.repository_id],
        )
    )
    paths = {item["qualified_path"] for item in result["results"]}
    assert paths == {"api::src/config.py", "worker::src/config.py"}


def test_relationship_and_impact_follow_declared_dependency(tmp_path: Path, isolated_home: Path) -> None:
    library_path = _repo(tmp_path / "shared", {"package.json": json.dumps({"name": "@demo/shared"})})
    app_path = _repo(
        tmp_path / "app",
        {"package.json": json.dumps({"name": "@demo/app", "dependencies": {"@demo/shared": "workspace:*"}})},
    )
    service = WorkspaceService()
    workspace = service.create_workspace("product", roots=[tmp_path], allowed_roots=[tmp_path])
    library = service.add_repository(workspace.workspace_id, library_path)
    app = service.add_repository(workspace.workspace_id, app_path)
    relationships = RelationshipService(service.store).detect(service.store.get_workspace(workspace.workspace_id))
    assert any(item.source_repository_id == app.repository_id and item.target_repository_id == library.repository_id for item in relationships)
    report = ImpactService(service.store).analyze(workspace.workspace_id, library.repository_id, ["src/index.ts"])
    assert [item.repository_id for item in report.affected] == [app.repository_id]


def test_model_scope_selects_multiple_repositories_and_validates_membership(tmp_path: Path, isolated_home: Path) -> None:
    first_path = _repo(tmp_path / "one")
    second_path = _repo(tmp_path / "two")
    service = WorkspaceService()
    workspace = service.create_workspace("product", roots=[tmp_path], allowed_roots=[tmp_path])
    first = service.add_repository(workspace.workspace_id, first_path)
    second = service.add_repository(workspace.workspace_id, second_path)
    session = service.create_session(first_path, workspace_id=workspace.workspace_id)
    context = service.context_for_session(session.session_id)

    class Model:
        def invoke(self, _messages):  # noqa: ANN001
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "repository_ids": [first.repository_id, second.repository_id],
                        "access_by_repository": {first.repository_id: "read", second.repository_id: "read"},
                        "requires_multi_repo": True,
                        "safe_to_continue": True,
                        "reason": "Both repositories implement the requested contract.",
                    }
                )
            )

    decision = RepositoryScopeDecisionEngine(Model()).decide(request="compare contracts", context=context)
    assert decision.repository_ids == [first.repository_id, second.repository_id]


def test_tool_worker_rejects_repository_scope_mismatch(tmp_path: Path, isolated_home: Path) -> None:
    client = ToolWorkerClient(
        api_key="test",
        model="test",
        session_id="test-session",
        repo_root=tmp_path,
        project_root=tmp_path,
        workspace_id="workspace_one",
        repository_id="repo_one",
    )
    with pytest.raises(ToolWorkerProcessError, match="repository does not match"):
        client.run_tools(
            ToolRunRequest(
                question="read",
                index_dir=str(tmp_path),
                repository_id="repo_two",
                workspace_id="workspace_one",
            )
        )
