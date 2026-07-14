import logging
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mana_agent.analysis.models import AskResponse, SearchHit
from mana_agent.commands import cli

runner = CliRunner()


class DummySettings:
    openai_api_key = "test"
    openai_base_url = None
    openai_chat_model = "fake"
    openai_tool_worker_model = None
    openai_coding_planner_model = None
    openai_embed_model = "fake"
    default_top_k = 8
    coding_flow_max_turns = 5
    coding_flow_max_tasks = 20
    coding_plan_max_steps = 8
    coding_search_budget = 4
    coding_read_budget = 6
    coding_require_read_files = 2


class RecordingAskService:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.ask_agent = object()

    def ask(self, index_dir: str, question: str, k: int) -> AskResponse:
        _ = index_dir
        _ = k
        self.calls.append(question)
        return AskResponse(
            answer="Plan response",
            sources=[
                SearchHit(0.9, "/tmp/a.py", 1, 5, "a", "snippet"),
                SearchHit(0.8, "/tmp/b.py", 2, 8, "b", "snippet"),
            ],
        )


class FakeWorkerClient:
    def __init__(self, **_kwargs: object) -> None:
        return None

    def start(self) -> None:
        return None

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def stop(self) -> None:
        return None


class RecordingCodingAgent:
    calls: list[str] = []

    def __init__(self, **_kwargs: object) -> None:
        self.active = "flow-plan-test"

    def set_tools_manager_orchestrator(self, _orchestrator: object) -> None:
        return None

    def get_active_flow_id(self) -> str | None:
        return self.active

    def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
        _ = (request, flow_id)
        return False

    def generate(self, request: str, **_kwargs: object) -> dict:
        self.calls.append(request)
        return {
            "answer": "Plan response",
            "changed_files": [],
            "warnings": [],
            "diff": "",
            "flow_id": self.active,
            "actions_taken": [],
        }

    def generate_dir_mode(self, request: str, **kwargs: object) -> dict:
        return self.generate(request, **kwargs)


def test_chat_planning_mode_asks_questions_and_resets(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    RecordingCodingAgent.calls = []

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr(
        "mana_agent.commands.cli.build_ask_service",
        lambda _s, model_override=None: RecordingAskService(calls),
    )
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", RecordingCodingAgent)

    user_input = "\n".join(
        [
            "plan auth module",
            "success means no open implementation choices",
            "scope src/auth only; no schema changes",
            "markdown with milestones and tests",
            "plan billing module",
            "success means migration-safe rollout",
            "scope billing services and CLI flags",
            "include rollout and rollback checks",
            "quit",
        ]
    ) + "\n"

    result = runner.invoke(
        cli.app,
        ["chat", "--planning-max-questions", "3"],
        input=user_input,
    )

    assert result.exit_code == 0
    assert result.stdout.count("Planning question 1/3") == 2
    assert "Generating decision-complete plan" in result.stdout
    assert len(RecordingCodingAgent.calls) == 2
    assert "You are in planning mode." in RecordingCodingAgent.calls[0]
    assert "plan auth module" in RecordingCodingAgent.calls[0]
    assert "A3: markdown with milestones and tests" in RecordingCodingAgent.calls[0]
    assert "plan billing module" in RecordingCodingAgent.calls[1]


def test_main_warns_for_python_314(monkeypatch) -> None:
    monkeypatch.setattr("mana_agent.commands.cli.setup_logging", lambda **_: Path("/tmp/mana.log"))
    monkeypatch.setattr(cli.sys, "version_info", (3, 14, 0), raising=False)

    # main() is the Typer root callback; a subcommand is "invoked" so it does not
    # dispatch to chat. Only the Python 3.14 compatibility warning should fire.
    ctx = types.SimpleNamespace(invoked_subcommand="chat")
    with pytest.warns(UserWarning, match="Python 3.14"):
        cli.main(ctx, verbose=False, debug_llm=False, log_dir=None, output_dir=None)


def test_chat_planning_mode_uses_llm_generated_questions(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    generated_args: list[dict] = []

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr(
        "mana_agent.commands.cli.build_ask_service",
        lambda _s, model_override=None: RecordingAskService(calls),
    )

    def _fake_llm_question(
        *,
        ask_service,
        planning_request: str,
        prior_questions: list[str],
        prior_answers: list[str],
        asked_count: int,
        max_questions: int,
    ) -> str:
        _ = (ask_service, planning_request, max_questions)
        generated_args.append(
            {
                "prior_questions": list(prior_questions),
                "prior_answers": list(prior_answers),
                "asked_count": asked_count,
            }
        )
        return f"LLM question {asked_count + 1}?"

    monkeypatch.setattr("mana_agent.commands.chat_cli._generate_planning_question_llm", _fake_llm_question)
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", RecordingCodingAgent)

    user_input = "\n".join(
        [
            "plan auth module",
            "answer one",
            "answer two",
            "answer three",
            "quit",
        ]
    ) + "\n"

    result = runner.invoke(
        cli.app,
        ["chat", "--planning-max-questions", "3"],
        input=user_input,
    )

    assert result.exit_code == 0
    assert "LLM question 1?" in result.stdout
    assert "LLM question 2?" in result.stdout
    assert "LLM question 3?" in result.stdout
    assert generated_args[1]["prior_answers"] == ["answer one"]
    assert generated_args[2]["prior_answers"] == ["answer one", "answer two"]


def test_chat_planning_mode_falls_back_to_static_on_llm_question_failure(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr(
        "mana_agent.commands.cli.build_ask_service",
        lambda _s, model_override=None: RecordingAskService(calls),
    )
    monkeypatch.setattr(
        "mana_agent.commands.chat_cli._generate_planning_question_llm",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("llm question failed")),
    )
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", RecordingCodingAgent)

    user_input = "\n".join(
        [
            "plan auth module",
            "answer one",
            "answer two",
            "answer three",
            "quit",
        ]
    ) + "\n"

    result = runner.invoke(
        cli.app,
        ["chat", "--planning-max-questions", "3"],
        input=user_input,
    )

    assert result.exit_code == 0
    assert "What is the concrete goal and the success criteria?" in result.stdout


def test_planning_question_auth_failure_logs_once_and_uses_static_fallback(monkeypatch, tmp_path: Path, caplog) -> None:
    calls: list[str] = []
    llm_calls = 0

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr(
        "mana_agent.commands.cli.build_ask_service",
        lambda _s, model_override=None: RecordingAskService(calls),
    )

    def _raise_auth(**_kwargs):
        nonlocal llm_calls
        llm_calls += 1
        raise RuntimeError("Error code: 401 - Incorrect API key provided: test")

    monkeypatch.setattr("mana_agent.commands.chat_cli._generate_planning_question_llm", _raise_auth)
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", RecordingCodingAgent)

    user_input = "\n".join(
        [
            "plan auth module",
            "answer one",
            "answer two",
            "answer three",
            "quit",
        ]
    ) + "\n"

    with caplog.at_level(logging.WARNING, logger="mana_agent.commands.chat_cli"):
        result = runner.invoke(
            cli.app,
            ["chat", "--planning-max-questions", "3"],
            input=user_input,
        )

    assert result.exit_code == 0
    assert llm_calls == 1
    assert result.stdout.count("What is the concrete goal and the success criteria?") == 1
    assert "What should be in scope and out of scope?" in result.stdout
    assert caplog.text.count("Planning question generation failed; using static fallback") == 1
