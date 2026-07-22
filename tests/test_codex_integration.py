from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest

from mana_agent.coding.models import (
    CodingBackendDecision,
    CodingTask,
    CodingTaskResult,
    WorkspaceContext,
)
from mana_agent.coding.registry import CodingBackendDecisionError, CodingBackendRegistry
from mana_agent.coding.routing_policy import validate_backend_decision
from mana_agent.integrations.codex.backend import CodexCodingBackend
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.coding_agent_shim import CodexCodingAgentShim
from mana_agent.integrations.codex.event_adapter import adapt_codex_event
from mana_agent.integrations.codex.exceptions import CodexUnavailableError
from mana_agent.integrations.codex.health import check_codex_health
from mana_agent.integrations.codex.result_parser import parse_codex_result
from mana_agent.multi_agent.codex_pool import _scopes_overlap
from mana_agent.workspaces.preparation import RepositoryValidationError


class _Backend:
    name = "native"

    async def start(self) -> None: ...
    async def execute(self, task, workspace): ...
    async def stream(self, task, workspace):
        if False:
            yield None
    async def cancel(self, task_id: str) -> None: ...
    async def close(self) -> None: ...


class _FakeClient:
    running = True

    def __init__(self, command: tuple[str, ...], *, approval: bool = False) -> None:
        self.command = command
        self.approval = approval
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    async def start(self) -> None:
        return None

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((method, params))
        if method == "thread/start":
            return {"thread": {"id": "thread-1"}}
        if method == "thread/resume":
            return {"thread": {"id": params["threadId"]}}
        if method == "turn/start":
            return {"turn": {"id": "turn-1"}}
        return {}

    async def notifications(self, thread_id: str):
        assert thread_id == "thread-1"
        if self.approval:
            yield {"method": "approval/requestApproval", "params": {"threadId": thread_id}}
            return
        yield {
            "method": "item/completed",
            "params": {
                "threadId": thread_id,
                "item": {"type": "commandExecution", "command": "pytest -q"},
            },
        }
        yield {
            "method": "item/completed",
            "params": {
                "threadId": thread_id,
                "item": {"type": "agentMessage", "text": "Implemented the task."},
            },
        }
        yield {
            "method": "turn/completed",
            "params": {"threadId": thread_id, "turn": {"id": "turn-1"}, "usage": {"inputTokens": 10}},
        }

    async def interrupt(self, *, thread_id: str, turn_id: str) -> None:
        return None

    async def close(self) -> None:
        self.closed = True

    async def deny_server_request(self, request: dict[str, Any]) -> None:
        self.requests.append(("deny", request))


def _git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


def _task() -> CodingTask:
    return CodingTask(
        task_id="task-1",
        goal="Implement the requested change",
        allowed_files=["src/app.py"],
        acceptance_criteria=["Tests pass"],
        verification_commands=["pytest -q"],
    )


def _workspace(tmp_path: Path) -> WorkspaceContext:
    repository = tmp_path / "repository"
    worktree = tmp_path / "worktree"
    repository.mkdir()
    _git_repo(worktree)
    return WorkspaceContext(repository_path=repository, worktree_path=worktree, branch_name="mana/task-1")


def test_registry_executes_only_model_selected_backend() -> None:
    registry = CodingBackendRegistry()
    backend = _Backend()
    registry.register(backend)
    decision = CodingBackendDecision(
        decision_id="decision-1",
        coding_required=True,
        selected_backend="native",
        estimated_complexity="low",
        requires_repository_write=True,
        reasons=["The model selected the native backend"],
        safe_to_continue=True,
    )
    assert registry.resolve(decision) is backend


def test_registry_does_not_fallback_when_selected_backend_is_missing() -> None:
    registry = CodingBackendRegistry()
    registry.register(_Backend())
    decision = CodingBackendDecision(
        decision_id="decision-2",
        coding_required=True,
        selected_backend="codex",
        estimated_complexity="high",
        requires_repository_write=True,
        reasons=["The model selected Codex"],
        safe_to_continue=True,
    )
    with pytest.raises(CodingBackendDecisionError, match="No fallback backend was executed"):
        registry.resolve(decision)


def test_invalid_backend_decision_stops_safely() -> None:
    with pytest.raises(CodingBackendDecisionError, match="No backend was executed"):
        validate_backend_decision(
            {
                "decision_id": "decision-3",
                "coding_required": True,
                "estimated_complexity": "high",
                "requires_repository_write": True,
                "reasons": ["Coding is required"],
                "safe_to_continue": True,
            }
        )


def test_codex_backend_uses_thread_turn_protocol_and_normalizes_result(tmp_path: Path) -> None:
    fake: _FakeClient | None = None

    def factory(command: tuple[str, ...]) -> _FakeClient:
        nonlocal fake
        fake = _FakeClient(command)
        return fake

    backend = CodexCodingBackend(CodexSettings(enabled=True), client_factory=factory)
    result = asyncio.run(backend.execute(_task(), _workspace(tmp_path)))

    assert result.status == "completed"
    assert result.backend == "codex"
    assert result.tests_run == ["pytest -q"]
    assert result.tests_passed is True
    assert result.thread_id == "thread-1"
    assert fake is not None
    assert [method for method, _params in fake.requests] == ["thread/start", "turn/start"]
    turn_payload = fake.requests[1][1]
    assert turn_payload["cwd"].endswith("worktree")
    assert fake.requests[0][1]["sandbox"] == "workspace-write"
    assert turn_payload["sandbox"] == "workspace-write"
    assert "Do not commit, push, publish" in turn_payload["input"][0]["text"]


def test_codex_backend_translates_read_only_sandbox_for_protocol(tmp_path: Path) -> None:
    fake: _FakeClient | None = None

    def factory(command: tuple[str, ...]) -> _FakeClient:
        nonlocal fake
        fake = _FakeClient(command)
        return fake

    repository = tmp_path / "repository"
    repository.mkdir()
    workspace = WorkspaceContext(
        repository_path=repository,
        worktree_path=repository,
        sandbox="readOnly",
    )
    task = _task().model_copy(update={"requires_repository_write": False})
    backend = CodexCodingBackend(CodexSettings(enabled=True), client_factory=factory)

    result = asyncio.run(backend.execute(task, workspace))

    assert result.status == "completed"
    assert fake is not None
    assert fake.requests[0][1]["sandbox"] == "read-only"
    assert fake.requests[1][1]["sandbox"] == "read-only"


def test_codex_backend_resumes_persisted_thread(tmp_path: Path) -> None:
    fake: _FakeClient | None = None

    def factory(command: tuple[str, ...]) -> _FakeClient:
        nonlocal fake
        fake = _FakeClient(command)
        return fake

    backend = CodexCodingBackend(CodexSettings(enabled=True), client_factory=factory, resume_thread_id="thread-1")
    result = asyncio.run(backend.execute(_task(), _workspace(tmp_path)))

    assert result.status == "completed"
    assert fake is not None
    assert [method for method, _params in fake.requests] == ["thread/resume", "turn/start"]
    assert fake.requests[0][1]["threadId"] == "thread-1"


def test_codex_shim_failed_payload_retains_backend_error() -> None:
    result = CodingTaskResult(
        task_id="failed-task",
        worker_id="codex-test",
        backend="codex",
        status="failed",
        summary="Codex task did not complete.",
        errors=["turn/start rejected the sandbox value"],
    )

    payload = CodexCodingAgentShim._result_payload(
        result,
        events=[],
        workspace_path="",
    )

    assert payload["auto_execute_terminal_reason"] == "codex_failed"
    assert payload["answer"] == (
        "Codex task did not complete. Reason: turn/start rejected the sandbox value"
    )
    assert "turn/start rejected the sandbox value" in payload["warnings"]


class _ShimBackend:
    def __init__(self) -> None:
        self.tasks: list[CodingTask] = []
        self.workspaces: list[WorkspaceContext] = []
        self.results: dict[str, Any] = {}
        self.closed = False

    async def stream(self, task: CodingTask, workspace: WorkspaceContext):
        self.tasks.append(task)
        self.workspaces.append(workspace)
        yield adapt_codex_event(
            task.task_id,
            {"method": "turn/started", "params": {"threadId": "thread-shim"}},
        )
        self.results[task.task_id] = parse_codex_result(
            task=task,
            workspace=workspace,
            worker_id="codex-shim",
            thread_id="thread-shim",
            turn_id="turn-shim",
            changed_files=["README.md"] if task.requires_repository_write else [],
            notifications=[
                {
                    "method": "item/completed",
                    "params": {"item": {"type": "agentMessage", "text": "Codex completed the turn."}},
                },
                {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
            ],
        )

    def result_for(self, task_id: str):
        return self.results[task_id]

    async def close(self) -> None:
        self.closed = True


class _ShimWorkspaceManager:
    def __init__(self, worktree: Path) -> None:
        self.worktree = worktree
        self.transitions: list[str] = []

    def create_for_task(self, task_id: str, **_kwargs: Any):
        self.worktree.mkdir()
        return SimpleNamespace(
            task_id=task_id,
            worktree_path=str(self.worktree),
            branch_name="mana/codex-shim",
        )

    def transition(self, _task_id: str, status, **_kwargs: Any):
        value = getattr(status, "value", status)
        self.transitions.append(str(value))
        return None


def test_coding_agent_shim_delegates_plan_decision_to_one_read_only_codex_turn(tmp_path: Path) -> None:
    _git_repo(tmp_path)
    backend = _ShimBackend()
    shim = CodexCodingAgentShim(
        repo_root=tmp_path,
        codex_settings=CodexSettings(enabled=True),
        backend_factory=lambda: backend,
    )

    result = shim.generate("plan the auth refactor", auto_chat_mode="plan_only")

    assert result["backend"] == "codex"
    assert result["answer"] == "Codex completed the turn."
    assert backend.tasks[0].goal == "plan the auth refactor"
    assert backend.tasks[0].requires_repository_write is False
    assert backend.workspaces[0].sandbox == "readOnly"
    assert backend.closed is True
    assert shim.preview_execution_checklist("plan it")["prechecklist"] is None
    assert shim.get_active_flow_id() == "thread-shim"
    assert shim.checkpoint_flow() == "thread-shim"
    with pytest.raises(RuntimeError, match="Codex owns coding tool selection"):
        shim._tool_policy_for_request("plan it")


def test_coding_agent_shim_delegates_planning_editing_and_verification_to_codex_worktree(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    _git_repo(repository)
    (repository / "README.md").write_text("# Existing\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Mana Test",
            "-c",
            "user.email=mana@example.invalid",
            "commit",
            "-q",
            "-m",
            "initial",
        ],
        cwd=repository,
        check=True,
    )
    backend = _ShimBackend()
    manager = _ShimWorkspaceManager(tmp_path / "worktree")
    shim = CodexCodingAgentShim(
        repo_root=repository,
        codex_settings=CodexSettings(enabled=True, worktree_isolation=True),
        backend_factory=lambda: backend,
        workspace_manager_factory=lambda: manager,
    )

    result = shim.generate_auto_execute("fix the login bug", auto_chat_mode="edit")

    task = backend.tasks[0]
    assert task.goal == "fix the login bug"
    assert task.requires_repository_write is True
    assert "inspect, plan, implement, and verify" in task.requirements[0]
    assert backend.workspaces[0].sandbox == "workspaceWrite"
    assert result["changed_files"] == ["README.md"]
    assert result["workspace_path"] == str(manager.worktree)
    assert manager.transitions == ["running", "merge_candidate"]


def test_coding_agent_shim_writes_directly_in_the_repository_root_by_default(tmp_path: Path) -> None:
    _git_repo(tmp_path)
    backend = _ShimBackend()
    shim = CodexCodingAgentShim(
        repo_root=tmp_path,
        codex_settings=CodexSettings(enabled=True),
        backend_factory=lambda: backend,
        workspace_manager_factory=lambda: pytest.fail("direct Codex turns must not create a worktree"),
    )

    result = shim.generate_auto_execute("fix the login bug", auto_chat_mode="edit")

    assert backend.workspaces[0].repository_path == tmp_path.resolve()
    assert backend.workspaces[0].worktree_path == tmp_path.resolve()
    assert backend.workspaces[0].allow_in_place_write is True
    assert result["workspace_path"] == str(tmp_path.resolve())


def test_direct_codex_shim_rejects_unprepared_repository_before_backend_start(tmp_path: Path) -> None:
    shim = CodexCodingAgentShim(
        repo_root=tmp_path,
        codex_settings=CodexSettings(enabled=True),
        backend_factory=lambda: pytest.fail("Codex backend must not start"),
    )

    with pytest.raises(RepositoryValidationError, match="Codex boundary validation"):
        shim.generate_auto_execute("create the project", auto_chat_mode="edit")
    assert not (tmp_path / ".git").exists()


def test_codex_shim_preserves_selected_subdirectory_as_working_directory(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    selected = repository / "packages" / "app"
    selected.mkdir(parents=True)
    _git_repo(repository)
    backend = _ShimBackend()
    shim = CodexCodingAgentShim(
        repo_root=repository,
        working_directory=selected,
        codex_settings=CodexSettings(enabled=True),
        backend_factory=lambda: backend,
    )

    shim.generate("plan the app change", auto_chat_mode="plan_only")

    assert backend.workspaces[0].repository_path == repository.resolve()
    assert backend.workspaces[0].working_directory == selected.resolve()


def test_unborn_repository_uses_explicit_in_place_workspace_without_worktree(tmp_path: Path) -> None:
    _git_repo(tmp_path)
    backend = _ShimBackend()
    shim = CodexCodingAgentShim(
        repo_root=tmp_path,
        codex_settings=CodexSettings(enabled=True, worktree_isolation=True),
        backend_factory=lambda: backend,
        workspace_manager_factory=lambda: pytest.fail("an unborn repository cannot create a worktree"),
    )

    shim.generate_auto_execute("create the initial files", auto_chat_mode="edit")

    assert backend.workspaces[0].worktree_path == tmp_path.resolve()
    assert backend.workspaces[0].allow_in_place_write is True


def test_codex_backend_does_not_self_approve(tmp_path: Path) -> None:
    backend = CodexCodingBackend(
        CodexSettings(enabled=True),
        client_factory=lambda command: _FakeClient(command, approval=True),
    )
    result = asyncio.run(backend.execute(_task(), _workspace(tmp_path)))
    assert result.status == "failed"
    assert "did not elevate permissions" in result.errors[0]


def test_codex_events_are_mapped_to_mana_event_contract() -> None:
    event = adapt_codex_event(
        "task-1",
        {"method": "turn/completed", "params": {"threadId": "thread-1", "turn": {"id": "turn-1"}}},
    )
    assert event.event_type == "turn.completed"
    assert event.status == "success"
    assert event.thread_id == "thread-1"
    assert event.turn_id == "turn-1"


def test_writing_workspace_must_be_isolated(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    with pytest.raises(ValueError, match="isolated worktree"):
        WorkspaceContext(repository_path=repository, worktree_path=repository)


def test_pool_scope_overlap_is_conservative() -> None:
    assert _scopes_overlap(frozenset(), frozenset({"src/app.py"}))
    assert _scopes_overlap(frozenset({"src"}), frozenset({"src/app.py"}))
    assert not _scopes_overlap(frozenset({"docs"}), frozenset({"src/app.py"}))


def test_disabled_codex_health_is_explicit(tmp_path: Path) -> None:
    report = check_codex_health(CodexSettings(enabled=False), tmp_path)
    assert report.healthy is False
    assert "Codex integration is disabled" in report.errors


def test_codex_health_rejects_unrelated_codex_package(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "mana_agent.integrations.codex.health.shutil.which",
        lambda _command: "/fake/bin/codex",
    )
    monkeypatch.setattr(
        "mana_agent.integrations.codex.health.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="0.2.3\n",
            stderr="circular dependency warning\n",
        ),
    )

    report = check_codex_health(CodexSettings(enabled=True), tmp_path)

    assert report.healthy is False
    assert report.app_server_available is False
    assert any("not the official OpenAI Codex CLI" in error for error in report.errors)


def test_codex_health_requires_app_server_capability(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "mana_agent.integrations.codex.health.shutil.which",
        lambda _command: "/fake/bin/codex",
    )
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout="codex-cli 1.2.3\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="Usage: codex app-server [OPTIONS]\n", stderr=""),
        ]
    )
    monkeypatch.setattr(
        "mana_agent.integrations.codex.health.subprocess.run",
        lambda *args, **kwargs: next(responses),
    )

    report = check_codex_health(CodexSettings(enabled=True), tmp_path)

    assert report.healthy is True
    assert report.app_server_available is True


def test_production_backend_stops_when_codex_preflight_fails(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "mana_agent.integrations.codex.backend.check_codex_health",
        lambda settings, repository: SimpleNamespace(
            healthy=False,
            executable="/fake/bin/codex",
            errors=["configured executable is incompatible"],
        ),
    )
    backend = CodexCodingBackend(CodexSettings(enabled=True))

    with pytest.raises(
        CodexUnavailableError,
        match="Codex preflight failed.*No fallback backend was executed",
    ):
        asyncio.run(backend.start(tmp_path))

    assert backend._client is None


def test_failed_test_command_is_not_reported_as_passing(tmp_path: Path) -> None:
    result = parse_codex_result(
        task=_task(),
        workspace=_workspace(tmp_path),
        worker_id="worker-1",
        thread_id="thread-1",
        turn_id="turn-1",
        changed_files=[],
        notifications=[
            {
                "method": "item/completed",
                "params": {
                    "item": {
                        "type": "commandExecution",
                        "command": "pytest -q",
                        "exitCode": 1,
                        "status": "failed",
                    }
                },
            },
            {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
        ],
    )
    assert result.status == "completed"
    assert result.tests_passed is False
    assert result.warnings == ["Test command failed: pytest -q"]


def test_interrupted_turn_is_cancelled(tmp_path: Path) -> None:
    result = parse_codex_result(
        task=_task(),
        workspace=_workspace(tmp_path),
        worker_id="worker-1",
        thread_id="thread-1",
        turn_id="turn-1",
        changed_files=[],
        notifications=[
            {"method": "turn/completed", "params": {"turn": {"status": "interrupted"}}},
        ],
    )
    assert result.status == "cancelled"
