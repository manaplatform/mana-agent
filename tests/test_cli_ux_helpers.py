from __future__ import annotations

from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from mana_analyzer.analysis.models import AskResponseWithTrace, SearchHit, ToolInvocationTrace
from mana_analyzer.commands import cli

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


def test_render_turn_summary_and_transparency_sections() -> None:
    summary = cli._render_turn_summary(
        answer="Decision: Use deterministic fallback checklist.",
        sources_count=2,
        warnings_count=1,
        tool_steps=3,
        changed_files_count=2,
        has_diff=True,
    )
    assert "[bold]Summary[/bold]" in summary
    assert "changed_files: 2" in summary
    assert "diff: yes" in summary

    turn = cli.ChatTurnTelemetry(
        turn_index=1,
        timestamp="2026-02-27T10:00:00",
        question="implement flow updates",
        answer_text="Decision: Use deterministic fallback checklist.",
        sources=[],
        warnings=["patch-only loop detected"],
        trace=[
            {
                "tool_name": "semantic_search",
                "status": "ok",
                "duration_ms": 2.1,
                "args_summary": "query=flow",
            }
        ],
        decisions=[{"decision": "Use deterministic fallback checklist", "rationale": "Planner parse failed"}],
        changed_files=["src/mana_analyzer/commands/cli.py"],
        has_diff=True,
    )

    console = Console(record=True)
    cli._render_turn_transparency(console, turn=turn, history=[turn])
    rendered = console.export_text()
    assert "Summary" in rendered
    assert "Steps" in rendered
    assert "Decisions" in rendered
    assert "History" in rendered
    assert "Session History" in rendered


def test_render_coding_sections_contains_expected_blocks() -> None:
    console = Console(record=True)
    cli._render_coding_sections(
        console,
        {
            "plan": {
                "objective": "Ship flow command",
                "steps": [{"status": "in_progress", "title": "Wire command"}],
            },
            "progress": {"phase": "edit", "why": "working", "budgets": {"search_used": 1, "search_budget": 4, "read_used": 2, "read_budget": 6, "read_files_observed": 2, "required_read_files": 2}},
            "checklist": {"done": 1, "pending": 1, "blocked": 0, "total": 2},
            "actions_taken": [{"tool_name": "read_file", "status": "ok", "duration_ms": 1.2, "args_summary": "path=cli.py"}],
            "changed_files": ["src/mana_analyzer/commands/cli.py"],
            "static_analysis": {"finding_count": 0},
            "next_step": "Run targeted tests.",
            "warnings": ["planner fallback: deterministic checklist"],
        },
    )
    rendered = console.export_text()
    assert "Plan" in rendered
    assert "Progress" in rendered
    assert "Checklist" in rendered
    assert "Actions Taken" in rendered
    assert "Files Changed" in rendered
    assert "Verification" in rendered
    assert "Next Step" in rendered


def test_index_command_is_retired(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["index", str(tmp_path)])
    assert result.exit_code != 0


def test_ask_command_text_output_includes_mode_trace_warnings_and_sources(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService:
        def ask_with_tools(
            self,
            index_dir: str,
            question: str,
            k: int,
            max_steps: int = 6,
            timeout_seconds: int = 30,
        ) -> AskResponseWithTrace:
            _ = (index_dir, question, k, max_steps, timeout_seconds)
            return AskResponseWithTrace(
                answer="Tool-based answer",
                sources=[SearchHit(0.9, "/tmp/a.py", 1, 2, "a", "snippet")],
                mode="agent-tools",
                trace=[
                    ToolInvocationTrace(
                        tool_name="semantic_search",
                        args_summary="query=flow",
                        duration_ms=2.2,
                        status="ok",
                        output_preview="hit",
                    )
                ],
                warnings=["planner fallback: deterministic checklist"],
            )

    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr(
        "mana_analyzer.commands.cli._build_ask_service_compat",
        lambda settings, model_override=None, project_root=None: _FakeAskService(),
    )

    result = runner.invoke(
        cli.app,
        ["ask", "where is flow summary", "--index-dir", str(tmp_path / "idx"), "--agent-tools"],
    )
    assert result.exit_code == 0
    assert "Tool-based answer" in result.stdout
    assert "Mode: agent-tools" in result.stdout
    assert "Tool Trace:" in result.stdout
    assert "Warnings:" in result.stdout
    assert "Sources:" in result.stdout
