from __future__ import annotations

import json
from types import SimpleNamespace

from mana_agent.multi_agent.routing.agent_decision import AgentDecision
from mana_agent.multi_agent.routing.router import Router
from mana_agent.workspaces.routing import RepositoryScopeDecisionEngine


class _DecisionEngine:
    def __init__(self) -> None:
        self.kwargs = {}

    def decide(self, **kwargs):
        self.kwargs = kwargs
        return AgentDecision(
            intent="answer",
            confidence=1.0,
            verifier_passed=True,
            verifier_summary="valid",
            reasoning_summary="plain response",
        )


def test_main_router_loads_lowercase_agents_markdown(tmp_path) -> None:
    (tmp_path / "agents.md").write_text("Use this repository's formatting rules.\n", encoding="utf-8")
    engine = _DecisionEngine()

    result = Router(decision_engine=engine).route(
        task_id="task-1",
        user_request="Summarize this project.",
        repo_root=tmp_path,
    )

    assert result.route_name == "simple"
    assert "Use this repository's formatting rules." in engine.kwargs["repository_guidance"]


def test_main_router_reports_absent_agents_file(tmp_path) -> None:
    engine = _DecisionEngine()

    Router(decision_engine=engine).route(
        task_id="task-2",
        user_request="Summarize this project.",
        repo_root=tmp_path,
    )

    assert "No root-level AGENTS.md or agents.md file was found." in engine.kwargs["repository_guidance"]


def test_main_router_does_not_read_agent_guidance_through_outside_symlink(tmp_path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    private = tmp_path / "private.md"
    private.write_text("Do not inject this outside file.", encoding="utf-8")
    (repository / "AGENTS.md").symlink_to(private)
    engine = _DecisionEngine()

    Router(decision_engine=engine).route(
        task_id="task-3",
        user_request="Summarize this project.",
        repo_root=repository,
    )

    assert "Do not inject this outside file." not in engine.kwargs["repository_guidance"]
    assert "No root-level AGENTS.md or agents.md file was found." in engine.kwargs["repository_guidance"]


def test_repository_scope_decision_receives_guidance() -> None:
    class _ScopeModel:
        payload = None

        def invoke(self, messages):
            self.payload = json.loads(messages[-1].content)
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "repository_ids": ["repo-1"],
                        "access_by_repository": {"repo-1": "read"},
                        "relationship_depth": 0,
                        "requires_multi_repo": False,
                        "requires_verification": False,
                        "safe_to_continue": True,
                        "reason": "Use the primary repository.",
                    }
                )
            )

    context = SimpleNamespace(
        session=SimpleNamespace(
            attached_repository_ids=["repo-1", "repo-2"],
            primary_repository_id="repo-1",
            session_id="session-1",
        ),
        workspace=SimpleNamespace(workspace_id="workspace-1"),
        repositories={},
        validate_scope=lambda decision: decision,
    )
    model = _ScopeModel()

    RepositoryScopeDecisionEngine(model).decide(
        request="Summarize this project.",
        context=context,
        repository_guidance="Follow the repository instructions.",
    )

    assert model.payload["repository_guidance"] == "Follow the repository instructions."
