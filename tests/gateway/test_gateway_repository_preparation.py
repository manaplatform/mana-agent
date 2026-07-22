from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mana_agent.gateway.config import ChatGatewayConfig
from mana_agent.gateway.stack import build_chat_stack
from mana_agent.gateway.turn_engine import process_chat_turn
from mana_agent.integrations.codex.coding_agent_shim import CodexCodingAgentShim
from mana_agent.multi_agent.routing.agent_decision import AgentDecision
from mana_agent.workspaces.preparation import GitInitializationError


def test_gateway_stack_prepares_non_git_workspace_before_constructing_codex(tmp_path: Path) -> None:
    existing = tmp_path / "existing.txt"
    existing.write_text("preserve\n", encoding="utf-8")

    stack = build_chat_stack(
        tmp_path,
        ChatGatewayConfig(
            coding_agent=True,
            agent_tools=True,
            chat_service=SimpleNamespace(),
        ),
    )

    assert stack.prepared_repository is not None
    assert stack.prepared_repository.initialized is True
    assert isinstance(stack.coding_agent, CodexCodingAgentShim)
    assert stack.coding_agent.repo_root == tmp_path.resolve()
    assert stack.coding_agent.working_directory == tmp_path.resolve()
    assert existing.read_text(encoding="utf-8") == "preserve\n"


def test_gateway_stack_passes_parent_repository_root_and_selected_working_directory(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    selected = repository / "packages" / "app"
    selected.mkdir(parents=True)
    import subprocess

    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repository, check=True)

    stack = build_chat_stack(
        selected,
        ChatGatewayConfig(
            coding_agent=True,
            agent_tools=True,
            chat_service=SimpleNamespace(),
        ),
    )

    assert isinstance(stack.coding_agent, CodexCodingAgentShim)
    assert stack.coding_agent.repo_root == repository.resolve()
    assert stack.coding_agent.working_directory == selected.resolve()
    assert not (selected / ".git").exists()


def test_gateway_preparation_failure_stops_before_coding_agent_call(tmp_path: Path) -> None:
    called = False

    class CodingAgent:
        def generate(self, *_args, **_kwargs):
            nonlocal called
            called = True
            return {"answer": "unexpected"}

    def fail_preparation():
        raise GitInitializationError(
            tmp_path,
            "Git initialization",
            "simulated permission denied",
        )

    decision = AgentDecision(
        intent="edit",
        code_editing_needed=True,
        selected_tools=["apply_patch"],
        tool_inputs={},
        flow_action="none",
        reasoning_summary="coding is required",
        confidence=0.99,
        verifier_passed=True,
    )
    result = process_chat_turn(
        root=tmp_path,
        text="update the project",
        chat_service=SimpleNamespace(),
        ask_service=SimpleNamespace(),
        coding_agent=CodingAgent(),
        config=ChatGatewayConfig().normalized(),
        session_state={},
        agent_decision=decision,
        coding_workspace_preparer=fail_preparation,
    )

    assert called is False
    assert result.used_coding_agent is False
    assert "during Git initialization" in str(result.error)
    assert "coding agent was not started" in str(result.error).lower()


def test_repeated_gateway_stack_build_reuses_prepared_repository(tmp_path: Path) -> None:
    config = ChatGatewayConfig(
        coding_agent=True,
        agent_tools=True,
        chat_service=SimpleNamespace(),
    )
    first = build_chat_stack(tmp_path, config)
    second = build_chat_stack(tmp_path, config)

    assert first.prepared_repository is not None
    assert second.prepared_repository is not None
    assert first.prepared_repository.initialized is True
    assert second.prepared_repository.initialized is False
    assert first.repository_id == second.repository_id
