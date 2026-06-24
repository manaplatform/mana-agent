from pathlib import Path

from typer.testing import CliRunner

from mana_agent.commands.cli import app

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


class _AskServiceWithAgent:
    def __init__(self) -> None:
        class _AskAgent:
            def __init__(self) -> None:
                self.tools: list[object] = []
                self.model = "fake"

            def ask(self, question: str, **kwargs: object) -> str:
                _ = (question, kwargs)
                return "ok"

        self.ask_agent = _AskAgent()


def test_flow_command_removed(tmp_path: Path) -> None:
    result = runner.invoke(app, ["flow", str(tmp_path)])
    assert result.exit_code != 0


def test_chat_startup_with_coding_memory_and_coding_agent_still_works(monkeypatch) -> None:
    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr(
        "mana_agent.commands.cli.build_ask_service",
        lambda _s, model_override=None: _AskServiceWithAgent(),
    )
    result = runner.invoke(
        app,
        ["chat", "--coding-memory"],
        input="quit\n",
    )
    assert result.exit_code == 0
    assert "Goodbye!" in result.stdout
