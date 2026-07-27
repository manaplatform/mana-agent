import json
from io import StringIO
from pathlib import Path
import pytest
from mana_agent.workspaces.paths import repository_dir, repository_id_for_path

from typer.testing import CliRunner

from mana_agent.analysis.models import AskResponse, AskResponseWithTrace, SearchHit
from mana_agent.commands.cli import _render_coding_sections, _sanitize_full_auto_answer_text, app
from mana_agent.commands.ui_helpers import emit_tool_event
from mana_agent.multi_agent.routing.agent_decision import AgentDecision

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_mana_home(tmp_path: Path, monkeypatch) -> None:
    """Keep persistent chat identity isolated between independent CLI tests."""
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana-home"))


class FakeIndexService:
    def index(self, target_path: str, index_dir: Path, rebuild: bool = False) -> dict:
        assert target_path
        assert index_dir
        return {
            "indexed_files": 1,
            "deleted_files": 0,
            "total_files": 1,
            "new_chunks": 2,
            "removed_chunks": 0,
            "index_dir": str(index_dir),
        }


class FakeSearchService:
    def search(self, index_dir: str, query: str, k: int) -> list[SearchHit]:
        assert index_dir
        assert query
        assert k
        return [
            SearchHit(
                score=0.99,
                file_path="/tmp/good.py",
                start_line=1,
                end_line=5,
                symbol_name="add",
                snippet="snippet",
            )
        ]


def test_chat_help_hides_manual_plan_execute_flags() -> None:
    result = runner.invoke(app, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--planning-mode" not in result.stdout
    assert "--auto-execute-plan" not in result.stdout
    assert "--no-auto-execute-plan" not in result.stdout
    assert "--full-auto" in result.stdout


def test_continue_help_accepts_root_dir_option(tmp_path: Path) -> None:
    result = runner.invoke(app, ["continue", "--root-dir", str(tmp_path), "--run-id", "abc123", "--help"])

    assert result.exit_code == 0
    assert "--root-dir" in result.stdout
    assert "--max-runtime-minu" in result.stdout
    assert "--max-cost" in result.stdout
    assert "--max-tool-c" in result.stdout


def test_analyze_command_is_public() -> None:
    result = runner.invoke(app, ["analyze", "--help"])

    assert result.exit_code == 0
    assert "--depth" in result.stdout
    assert "--max-files" in result.stdout


def test_continue_command_uses_root_dir_and_loops_until_complete(monkeypatch, tmp_path: Path) -> None:
    run_dir = repository_dir(repository_id_for_path(tmp_path)) / "runs" / "abc123"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text('{"goal": "finish", "flow_id": "flow"}\n')

    class _FakeWorkerClient:
        roots: list[str] = []

        def __init__(self, **kwargs: object) -> None:
            _FakeWorkerClient.roots.append(str(kwargs.get("repo_root", "")))

        def stop(self) -> None:
            return None

    class _FakeResult:
        def __init__(self, *, status: str, terminal_reason: str, answer: str) -> None:
            self.run_status = status
            self.terminal_reason = terminal_reason
            self.answer = answer
            self.run_id = "abc123"
            self.next_action = "read_file app/models.py"

    class _FakeOrchestrator:
        roots: list[str] = []
        calls = 0

        def __init__(self, **kwargs: object) -> None:
            _FakeOrchestrator.roots.append(str(kwargs.get("repo_root", "")))

        def resume_run(self, **kwargs: object) -> _FakeResult:
            assert kwargs["run_id"] == "abc123"
            _FakeOrchestrator.calls += 1
            if _FakeOrchestrator.calls == 1:
                return _FakeResult(status="needs_resume", terminal_reason="pass_cap_reached", answer="checkpoint")
            return _FakeResult(status="completed", terminal_reason="planner_finalize", answer="done")

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.QueueManager", _FakeOrchestrator)

    result = runner.invoke(app, ["continue", "--root-dir", str(tmp_path), "--run-id", "abc123"])

    assert result.exit_code == 0
    assert _FakeWorkerClient.roots == [str(tmp_path.resolve())]
    assert _FakeOrchestrator.roots == [str(tmp_path.resolve())]
    assert _FakeOrchestrator.calls == 2
    assert "Continuation checkpoint" in result.stdout
    assert "done" in result.stdout


class FakeAskService:
    def ask(self, index_dir: str, question: str, k: int) -> AskResponse:
        assert index_dir
        assert question
        assert k
        hit = SearchHit(0.8, "/tmp/good.py", 2, 4, "add", "snippet")
        return AskResponse(answer="Uses add. /tmp/good.py:2-4", sources=[hit])

    def ask_with_tools(
        self,
        index_dir: str,
        question: str,
        k: int,
        max_steps: int = 6,
        timeout_seconds: int = 30,
    ) -> AskResponse:
        assert index_dir
        assert question
        assert k
        assert max_steps > 0
        assert timeout_seconds > 0
        hit = SearchHit(0.9, "/tmp/good.py", 1, 3, "add", "snippet")
        return AskResponse(answer="Tool answer. /tmp/good.py:1-3", sources=[hit])

    def ask_dir_mode(self, index_dirs, question: str, k: int, root_dir: str) -> AskResponse:
        assert index_dirs
        assert question
        assert k
        assert root_dir
        hit = SearchHit(0.7, "/tmp/mono/pkg-a/a.py", 1, 2, "a", "snippet")
        return AskResponse(answer="Dir answer", sources=[hit], warnings=[])

    def ask_with_tools_dir_mode(
        self,
        index_dirs,
        question: str,
        k: int,
        max_steps: int = 6,
        timeout_seconds: int = 30,
        root_dir: str | None = None,
    ) -> AskResponseWithTrace:
        assert index_dirs
        assert question
        assert k
        assert max_steps > 0
        assert timeout_seconds > 0
        _ = root_dir
        hit = SearchHit(0.77, "/tmp/mono/pkg-a/a.py", 1, 2, "a", "snippet")
        return AskResponseWithTrace(answer="Dir tool answer", sources=[hit], mode="agent-tools", trace=[], warnings=[])


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


def test_pyproject_exposes_mana_agent_primary_script() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "mana-agent"' in pyproject
    assert 'mana-agent = "mana_agent.commands.cli:app"' in pyproject


def test_root_help_exposes_commands_and_no_legacy_branding() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "chat" in result.output
    # ask remains retired; analyze is public again as the repository intelligence command.
    assert "ask" not in result.output
    assert "analyze" in result.output
    assert "mana-agent" in result.output
    assert "mana-analyzer" not in result.output
    assert "analyzor" not in result.output


def test_chat_help_works() -> None:
    result = runner.invoke(app, ["chat", "--help"])
    assert result.exit_code == 0
    assert "chat [OPTIONS]" in result.output


def test_chat_prompt_direct_readme_version_edit_skips_heavy_setup(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "# Demo\n\nCurrent documented version: **v0.0.7**.\n",
        encoding="utf-8",
    )

    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("small direct edit should not initialize heavy chat dependencies")

    monkeypatch.setattr("mana_agent.commands.chat_cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.chat_cli.build_ask_service", _blocked)
    monkeypatch.setattr("mana_agent.commands.chat_cli.build_index_service", _blocked)
    monkeypatch.setattr("mana_agent.commands.chat_cli.CodingAgent", _blocked)
    monkeypatch.setattr("mana_agent.commands.chat_cli.ToolWorkerClient", _blocked)

    result = runner.invoke(
        app,
        ["chat", "update version in readme.md to 0.0.8", "--root-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "Verification skipped: docs-only one-line edit" in result.output
    assert "Current documented version: **v0.0.8**." in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_no_source_file_contains_analyzor() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    offenders = [
        str(path.relative_to(repo_root))
        for path in (repo_root / "src").rglob("*.py")
        if "analyzor" in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"'analyzor' still present in: {offenders}"


def test_root_command_non_tty_does_not_show_removed_mode_menu() -> None:
    result = runner.invoke(app, ["--no-banner"], input="4\n")

    assert result.exit_code == 2
    assert "Interactive chat requires a TTY" in result.output
    assert "Choose what you want to do" not in result.output
    assert "Analyze repo" not in result.output


class FakeStructureService:
    def __init__(self, include_tests: bool = False) -> None:
        self.include_tests = include_tests

    def analyze_project(self, target_path: str) -> object:
        assert target_path
        return type(
            "_Report",
            (),
            {
                "to_dict": lambda self: {
                    "project_root": "/tmp/project",
                    "modules": [],
                    "exports": [],
                    "data_structures": [],
                    "commands": [],
                }
            },
        )()

    def render_markdown(self, _report: object) -> str:
        return "# Project Structure Analysis\n\n## Modules\n"


class FakeDependencyReport:
    project_root = "/tmp/project"
    frameworks = ["Typer"]
    technologies = ["Typer", "LangChain"]
    package_managers = ["pip"]
    languages = ["python"]
    runtime_dependencies = ["typer"]
    dev_dependencies = ["pytest"]
    module_edges = []
    dependency_edges = []
    manifests = ["pyproject.toml"]

    def to_dict(self) -> dict:
        return {
            "project_root": self.project_root,
            "frameworks": self.frameworks,
            "technologies": self.technologies,
            "package_managers": self.package_managers,
            "languages": self.languages,
            "runtime_dependencies": self.runtime_dependencies,
            "dev_dependencies": self.dev_dependencies,
            "module_edges": [],
            "dependency_edges": [],
            "manifests": self.manifests,
        }

    def to_dot(self) -> str:
        return 'digraph mana_agent { "a" -> "b"; }'

    def to_graphml(self) -> str:
        return "<graphml></graphml>"


class FakeDependencyService:
    def analyze(self, path: str) -> FakeDependencyReport:
        assert path
        return FakeDependencyReport()


class FakeDescribeService:
    def describe(
        self,
        path: str,
        max_files: int = 12,
        include_functions: bool = False,
        use_llm: bool = True,
        **_: object,
    ) -> object:
        assert path
        assert max_files > 0
        _ = include_functions
        _ = use_llm
        return type(
            "_DescribeReport",
            (),
            {
                "to_dict": lambda self: {
                    "project_root": "/tmp/project",
                    "selected_files": ["src/a.py"],
                    "descriptions": [
                        {
                            "file_path": "src/a.py",
                            "language": "python",
                            "symbols": ["add"],
                            "summary": "a summary",
                        }
                    ],
                    "architecture_summary": "arch",
                    "tech_summary": "tech",
                    "chain_steps": ["one", "two"],
                    "architecture_mermaid": "flowchart LR",
                    "architecture_data": {},
                    "metrics": {},
                }
            },
        )()

    def render_markdown(self, _report: object) -> str:
        return "# Repository Description\n\n## Architecture\n"


def test_cli_commands(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_index_service", lambda _s: FakeIndexService())
    monkeypatch.setattr("mana_agent.commands.cli.build_search_service", lambda _s: FakeSearchService())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.build_dependency_service", lambda: FakeDependencyService())
    monkeypatch.setattr("mana_agent.commands.cli.discover_subprojects", lambda root: [])
    monkeypatch.setattr("mana_agent.commands.cli.discover_index_dirs", lambda root: [Path(root) / ".mana/index"])

    # Legacy subcommands other than analyze must still error out.
    for retired in ("ask", "index", "search", "deps", "graph", "describe", "report", "flow"):
        result_retired = runner.invoke(app, [retired, str(tmp_path)])
        assert result_retired.exit_code != 0


def test_chat_blocks_edit_requests_without_coding_agent(monkeypatch, tmp_path: Path) -> None:
    class _NoCallAskService(FakeAskService):
        def ask(self, index_dir: str, question: str, k: int) -> AskResponse:  # pragma: no cover - must not run
            raise AssertionError("chat_service.ask should not be called for blocked edit requests")

        def ask_with_tools(  # pragma: no cover - must not run
            self,
            index_dir: str,
            question: str,
            k: int,
            max_steps: int = 6,
            timeout_seconds: int = 30,
        ) -> AskResponse:
            raise AssertionError("ask_with_tools should not be called for blocked edit requests")

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _NoCallAskService())

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--no-coding-agent"],
        input="please patch this file\nquit\n",
    )
    assert result.exit_code == 0
    assert "read-only for file edits" in result.stdout
    assert "--agent-tools" in result.stdout
    assert "--coding-agent" in result.stdout


def test_chat_zero_tool_response_renders_only_answer(monkeypatch, tmp_path: Path) -> None:
    """Regression: classic chat with no tool calls shows only the answer."""
    class _AskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

        def ask(self, index_dir: str, question: str, k: int) -> AskResponse:
            _ = (index_dir, question, k)
            return AskResponse(answer="Plain answer with no tools.", sources=[])

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _AskService())

    result = runner.invoke(
        app,
        ["chat", "--no-agent-tools", "--no-coding-agent", "--no-auto-execute-plan"],
        input="hello world\nquit\n",
    )
    assert result.exit_code == 0
    assert "Plain answer with no tools." in result.stdout
    assert "Answer" in result.stdout
    assert "Session History" not in result.stdout
    assert "Answer preview" not in result.stdout
    assert "No tool steps ran for this answer." not in result.stdout
    assert "No decisions were recorded for this turn." not in result.stdout
    assert "No prior turns in this session." not in result.stdout


def test_chat_tool_backed_response_omits_diagnostic_panels(monkeypatch, tmp_path: Path) -> None:
    """Regression: tool-backed answers keep telemetry internally but omit panels."""
    class _TracingAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

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
                answer="Tool-backed final answer.",
                sources=[],
                mode="agent-tools",
                trace=[
                    {
                        "tool_name": "read_file",
                        "status": "ok",
                        "duration_ms": 4.0,
                        "args_summary": "path='README.md'",
                    }
                ],
                warnings=[],
            )

    logged: list[dict] = []

    class _FakeRunLogger:
        def __init__(self, log_file=None) -> None:
            _ = log_file

        def log(self, payload: dict) -> None:
            logged.append(payload)

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _TracingAskService())
    monkeypatch.setattr("mana_agent.commands.cli.LlmRunLogger", _FakeRunLogger)

    result = runner.invoke(
        app,
        [
            "chat",
            "--agent-tools",
            "--no-coding-agent",
            "--root-dir",
            str(tmp_path),
        ],
        input="read the readme\nquit\n",
    )
    assert result.exit_code == 0
    assert "Tool-backed final answer." in result.stdout
    assert "Session History" not in result.stdout
    assert "Answer preview" not in result.stdout
    assert "No tool steps ran for this answer." not in result.stdout
    # Internal trace still logged for debugging/dashboard use.
    chat_rows = [row for row in logged if row.get("flow") == "chat"]
    assert chat_rows
    assert chat_rows[-1].get("tool_steps") == 1
    assert any(
        (item or {}).get("tool_name") == "read_file"
        for item in (chat_rows[-1].get("trace") or [])
        if isinstance(item, dict)
    )


def test_chat_normal_mode_renders_answer_without_diagnostic_panels(monkeypatch, tmp_path: Path) -> None:
    class _AskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _AskService())

    result = runner.invoke(
        app,
        [
            "chat",
            "--no-agent-tools",
            "--no-coding-agent",
            "--root-dir",
            str(tmp_path),
        ],
        input="first question\nsecond question\nquit\n",
    )
    assert result.exit_code == 0
    assert "Uses add. /tmp/good.py:2-4" in result.stdout
    assert result.stdout.count("Answer") >= 2
    assert "Session History" not in result.stdout
    assert "Answer preview" not in result.stdout
    # Diagnostic panel titles must not appear as post-response sections.
    assert "No tool steps ran for this answer." not in result.stdout
    assert "No decisions were recorded for this turn." not in result.stdout
    assert "No prior turns in this session." not in result.stdout


def test_chat_root_dir_applies_to_worker_and_coding_agent_in_classic_mode(monkeypatch, tmp_path: Path) -> None:
    class _AskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeWorkerClient:
        init_kwargs: dict[str, object] = {}

        def __init__(self, **kwargs: object) -> None:
            _FakeWorkerClient.init_kwargs = dict(kwargs)

        def stop(self) -> None:
            return None

    class _FakeCodingAgent:
        init_kwargs: dict[str, object] = {}

        def __init__(self, **kwargs: object) -> None:
            _FakeCodingAgent.init_kwargs = dict(kwargs)

        def get_active_flow_id(self) -> str | None:
            return None

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None, project_root=None: _AskService())
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--root-dir", str(tmp_path)],
        input="quit\n",
    )
    assert result.exit_code == 0
    assert _FakeWorkerClient.init_kwargs.get("repo_root") == tmp_path.resolve()
    assert _FakeWorkerClient.init_kwargs.get("project_root") == tmp_path.resolve()
    assert _FakeCodingAgent.init_kwargs.get("repo_root") == tmp_path.resolve()


def test_chat_ping_returns_pong_without_faiss_index(monkeypatch, tmp_path: Path) -> None:
    # ping is a direct command: it must answer without any FAISS index, RAG,
    # or coding-agent search, even when the project has never been indexed.
    class _AskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

        def ask_with_tools(self, *_a: object, **_k: object):
            raise AssertionError("ping must not trigger semantic search")

    class _FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def stop(self) -> None:
            return None

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def get_active_flow_id(self) -> str | None:
            return None

        def generate(self, *_a: object, **_k: object) -> dict:
            raise AssertionError("ping must not invoke the coding agent")

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None, project_root=None: _AskService())
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    # No .mana/index directory exists under tmp_path -> no FAISS index.
    result = runner.invoke(
        app,
        ["chat", "--root-dir", str(tmp_path)],
        input="ping\nquit\n",
    )
    assert result.exit_code == 0
    assert "pong" in result.stdout


def test_chat_root_dir_changes_default_index_dir_in_classic_mode(monkeypatch, tmp_path: Path) -> None:
    class _AskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def stop(self) -> None:
            return None

    class _FakeCodingAgent:
        seen_index_dir: str = ""

        def __init__(self, **_kwargs: object) -> None:
            return None

        def get_active_flow_id(self) -> str | None:
            return None

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, request: str, **kwargs: object) -> dict:
            _ = request
            _FakeCodingAgent.seen_index_dir = str(kwargs.get("index_dir") or "")
            return {
                "answer": "ok",
                "changed_files": [],
                "diff": "",
                "warnings": [],
                "flow_id": None,
                "plan": {"objective": "noop", "steps": []},
                "progress": {"phase": "answer", "why": "done", "tool_call_allowed": False},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None, project_root=None: _AskService())
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--root-dir", str(tmp_path)],
        input="tell me about this code\nquit\n",
    )
    assert result.exit_code == 0
    assert _FakeCodingAgent.seen_index_dir == str(
        (repository_dir(repository_id_for_path(tmp_path)) / "index").resolve()
    )


def test_chat_agent_tools_mode_renders_answer_without_diagnostic_panels(monkeypatch, tmp_path: Path) -> None:
    class _TracingAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

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
                answer="Decision: Use semantic search first",
                sources=[],
                mode="agent-tools",
                trace=[
                    {
                        "tool_name": "semantic_search",
                        "status": "ok",
                        "duration_ms": 3.5,
                        "args_summary": "query='planner'",
                    }
                ],
                warnings=[],
            )

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _TracingAskService())

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--no-coding-agent", "--no-auto-execute-plan"],
        input="tell me about the code\nquit\n",
    )
    assert result.exit_code == 0
    assert "Use semantic search first" in result.stdout
    assert "Answer" in result.stdout
    assert "Session History" not in result.stdout
    assert "No tool steps ran for this answer." not in result.stdout
    assert "No decisions were recorded for this turn." not in result.stdout


def test_chat_writes_llm_run_log_rows(monkeypatch, tmp_path: Path) -> None:
    class _AskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    rows: list[dict] = []

    class _FakeRunLogger:
        def __init__(self, log_file=None) -> None:
            _ = log_file

        def log(self, payload: dict) -> None:
            rows.append(payload)

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _AskService())
    monkeypatch.setattr("mana_agent.commands.cli.LlmRunLogger", _FakeRunLogger)

    result = runner.invoke(
        app,
        ["chat", "--no-agent-tools", "--no-coding-agent"],
        input="what is this project?\nquit\n",
    )
    assert result.exit_code == 0
    assert rows
    assert rows[0]["flow"] == "chat"
    assert rows[0]["mode"] == "classic"
    assert rows[0]["question"] == "what is this project?"


def test_flow_show_checkpoint_and_reset_commands(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"
            self.checkpointed: list[str] = []
            self.reset_ids: list[str] = []

        def get_active_flow_id(self) -> str | None:
            return self.active

        def flow_summary(self, flow_id: str | None = None):
            if not (flow_id or self.active):
                return None
            return {
                "flow_id": flow_id or self.active,
                "objective": "Implement parser retry flow",
                "constraints": ["Only touch src/ and tests/"],
                "open_tasks": ["add regression test"],
                "last_changed_files": ["src/mana_agent/services/ask_service.py"],
            }

        def checkpoint_flow(self, flow_id: str | None = None) -> str | None:
            target = flow_id or self.active
            if not target:
                return None
            self.checkpointed.append(target)
            return target

        def reset_flow(self, flow_id: str | None = None) -> str | None:
            target = flow_id or self.active
            if not target:
                return None
            self.reset_ids.append(target)
            self.active = None
            return target

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return {"answer": "ok", "changed_files": [], "warnings": [], "diff": "", "flow_id": self.active}

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return {"answer": "ok", "changed_files": [], "warnings": [], "diff": "", "flow_id": self.active}

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent", "--flow-id", "flow-123"],
        input="/flow show\n/flow checkpoint\n/flow reset\nquit\n",
    )
    assert result.exit_code == 0
    assert "Flow memory active" in result.stdout
    assert "Implement parser retry flow" in result.stdout


def test_chat_coding_agent_uses_worker_lifecycle_once(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeWorkerClient:
        start_calls = 0
        stop_calls = 0
        health_calls = 0

        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            _FakeWorkerClient.start_calls += 1

        def health(self) -> dict[str, str]:
            _FakeWorkerClient.health_calls += 1
            return {"status": "ok"}

        def stop(self) -> None:
            _FakeWorkerClient.stop_calls += 1

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-xyz"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return {"answer": "ok", "changed_files": [], "warnings": [], "diff": "", "flow_id": self.active}

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return {"answer": "ok", "changed_files": [], "warnings": [], "diff": "", "flow_id": self.active}

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="please edit file\nanother edit\nquit\n",
    )
    assert result.exit_code == 0
    assert _FakeWorkerClient.start_calls == 1
    assert _FakeWorkerClient.health_calls == 1
    assert _FakeWorkerClient.stop_calls == 1


def test_chat_plan_trigger_is_quiet_and_skips_conflict_prompt(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return True

        def flow_summary(self, flow_id: str | None = None):
            _ = flow_id
            return {
                "flow_id": self.active,
                "objective": "Do TODO.md section 1 and 2",
                "checklist": {
                    "objective": "Do TODO.md section 1 and 2",
                    "steps": [
                        {"status": "in_progress", "title": "Document public API"},
                        {"status": "pending", "title": "Add missing type hints"},
                    ],
                },
            }

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return {
                "answer": "ok",
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
            }

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate(*_args, **_kwargs)

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="implement plan.\nquit\n",
    )
    assert result.exit_code == 0
    assert "Executing active flow plan..." not in result.stdout
    assert "Document public API" not in result.stdout
    assert "Add missing type hints" not in result.stdout
    assert "Auto-Execute" not in result.stdout
    assert "This request appears to diverge from the active flow." not in result.stdout


def test_chat_plan_trigger_with_preview_keeps_progress_quiet(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def preview_execution_checklist(self, request: str, *, flow_id: str | None = None, flow_context: str | None = None) -> dict:
            _ = (request, flow_id, flow_context)
            return {
                "flow_id": self.active,
                "prechecklist": {
                    "objective": "Implement TODO sections",
                    "source": "planner",
                    "steps": [
                        {"id": "s1", "status": "in_progress", "title": "Inspect TODO.md section 1"},
                        {"id": "s2", "status": "pending", "title": "Apply TODO.md section 2 updates"},
                    ],
                },
                "prechecklist_source": "planner",
                "prechecklist_warning": "",
            }

        def generate_auto_execute(self, *_args: object, **_kwargs: object) -> dict:
            return {
                "answer": "done",
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "Implement TODO sections", "steps": []},
                "progress": {"phase": "answer", "why": "complete"},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "auto_execute_passes": 1,
                "auto_execute_terminal_reason": "manager_stop",
                "toolsmanager_requests_count": 1,
                "pass_logs": [],
                "planner_decisions": [],
            }

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate_auto_execute()

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate_auto_execute()

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="implement plan.\nquit\n",
    )
    assert result.exit_code == 0
    assert "Inspect TODO.md section 1" not in result.stdout
    assert "Apply TODO.md section 2 updates" not in result.stdout
    assert "Checklist source: planner" not in result.stdout
    assert "Executing active flow plan..." not in result.stdout


def test_chat_plan_trigger_preview_fallback_hides_warning_panel_in_quiet_mode(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def preview_execution_checklist(self, request: str, *, flow_id: str | None = None, flow_context: str | None = None) -> dict:
            _ = (request, flow_id, flow_context)
            return {
                "flow_id": self.active,
                "prechecklist": {
                    "objective": "Fallback objective",
                    "source": "deterministic_fallback",
                    "steps": [{"id": "s1", "status": "in_progress", "title": "Discover target file(s)"}],
                },
                "prechecklist_source": "deterministic_fallback",
                "prechecklist_warning": "Planner parse failed; using deterministic fallback checklist.",
            }

        def generate_auto_execute(self, *_args: object, **_kwargs: object) -> dict:
            return {
                "answer": "done",
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "Fallback objective", "steps": []},
                "progress": {"phase": "answer", "why": "complete"},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "auto_execute_passes": 1,
                "auto_execute_terminal_reason": "manager_stop",
                "toolsmanager_requests_count": 1,
                "pass_logs": [],
                "planner_decisions": [],
            }

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate_auto_execute()

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate_auto_execute()

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="implement plan.\nquit\n",
    )
    assert result.exit_code == 0
    assert "Planner Warning" not in result.stdout
    assert "deterministic fallback checklist" not in result.stdout.lower()


def test_chat_plan_trigger_auto_execute_without_coding_agent_hides_progress(monkeypatch, tmp_path: Path) -> None:
    class _FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def stop(self) -> None:
            return None

    class _FakeAutoResult:
        def model_dump(self) -> dict:
            return {
                "answer": "auto execution complete",
                "sources": [],
                "trace": [],
                "warnings": [],
                "changed_files": [],
                "plan": {"objective": "Implement plan", "steps": [{"id": "s1", "title": "Inspect"}]},
                "passes": 1,
                "terminal_reason": "manager_stop",
                "toolsmanager_requests_count": 1,
                "pass_logs": [
                    {
                        "pass_index": 1,
                        "requests_count": 1,
                        "request_fingerprints": ["abc123"],
                        "tool_steps": 0,
                        "warnings_delta": 0,
                        "expected_progress": "inspection complete",
                    }
                ],
            }

    class _FakeOrchestrator:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def preview_plan(self, **_kwargs: object) -> dict:
            return {
                "prechecklist": {
                    "objective": "Implement plan",
                    "source": "planner",
                    "steps": [{"id": "s1", "status": "in_progress", "title": "Inspect files"}],
                },
                "prechecklist_source": "planner",
                "prechecklist_warning": "",
                "warnings": [],
            }

        def run(self, **_kwargs: object):
            return _FakeAutoResult()

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.QueueManager", _FakeOrchestrator)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--no-coding-agent"],
        input="implement plan.\nquit\n",
    )
    assert result.exit_code == 0
    assert "Executing active flow plan..." not in result.stdout
    assert "Auto-executing plan" not in result.stdout
    assert "Auto-Execute" not in result.stdout
    assert "auto execution complete" in result.stdout


def test_chat_redis_backend_falls_back_to_local_executor_when_unavailable(monkeypatch, tmp_path: Path) -> None:
    class _FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def stop(self) -> None:
            return None

        def init_payload_dict(self) -> dict:
            return {
                "api_key": "x",
                "model": "fake-model",
                "project_root": str(tmp_path),
                "repo_root": str(tmp_path),
                "tools_only_strict": True,
            }

    class _FakeAutoResult:
        def model_dump(self) -> dict:
            return {
                "answer": "auto execution complete",
                "sources": [],
                "trace": [],
                "warnings": [],
                "changed_files": [],
                "plan": {"objective": "Implement plan", "steps": [{"id": "s1", "title": "Inspect"}]},
                "passes": 1,
                "terminal_reason": "manager_stop",
                "toolsmanager_requests_count": 1,
                "pass_logs": [],
                "planner_decisions": [],
            }

    class _FakeOrchestrator:
        init_kwargs: dict[str, object] = {}

        def __init__(self, **kwargs: object) -> None:
            _FakeOrchestrator.init_kwargs = dict(kwargs)

        def preview_plan(self, **_kwargs: object) -> dict:
            return {
                "prechecklist": {
                    "objective": "Implement plan",
                    "source": "planner",
                    "steps": [{"id": "s1", "status": "in_progress", "title": "Inspect files"}],
                },
                "prechecklist_source": "planner",
                "prechecklist_warning": "",
                "warnings": [],
            }

        def run(self, **_kwargs: object):
            return _FakeAutoResult()

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.QueueManager", _FakeOrchestrator)
    monkeypatch.setattr(
        "mana_agent.commands.cli.RedisRQToolsExecutor",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("redis unavailable")),
    )

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--no-coding-agent", "--tool-exec-backend", "redis"],
        input="implement plan.\nquit\n",
    )
    assert result.exit_code == 0
    assert "auto execution complete" in result.stdout
    assert _FakeOrchestrator.init_kwargs
    execution_config = _FakeOrchestrator.init_kwargs.get("execution_config")
    assert str(getattr(execution_config, "backend", "")) == "redis"
    assert _FakeOrchestrator.init_kwargs.get("executor").__class__.__name__ == "LocalToolsExecutor"


def test_chat_planning_mode_auto_executes_after_clarifications(monkeypatch, tmp_path: Path) -> None:
    codex_calls: list[str] = []

    class _PlanningLlm:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return type("_Msg", (), {"content": f"Clarification question {self.calls}?"})()

    class _AskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()
            self.qna_chain = type("_Qna", (), {"llm": _PlanningLlm()})()

    class _FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def stop(self) -> None:
            return None

    class _FakeAutoResult:
        def model_dump(self) -> dict:
            return {
                "answer": "auto execution complete",
                "sources": [],
                "trace": [],
                "warnings": [],
                "changed_files": [],
                "plan": {"objective": "Implement plan", "steps": [{"id": "s1", "title": "Inspect"}]},
                "passes": 1,
                "terminal_reason": "manager_stop",
                "toolsmanager_requests_count": 1,
                "pass_logs": [
                    {
                        "pass_index": 1,
                        "requests_count": 1,
                        "request_fingerprints": ["abc123"],
                        "tool_steps": 0,
                        "warnings_delta": 0,
                        "expected_progress": "inspection complete",
                    }
                ],
            }

    class _FakeOrchestrator:
        calls: list[str] = []

        def __init__(self, **_kwargs: object) -> None:
            _FakeOrchestrator.calls = []

        def run(self, **kwargs: object):
            _FakeOrchestrator.calls.append(str(kwargs.get("request", "")))
            return _FakeAutoResult()

    monkeypatch.setattr("mana_agent.commands.chat_cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.chat_cli.build_ask_service", lambda _s, model_override=None: _AskService())
    monkeypatch.setattr("mana_agent.commands.chat_cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.chat_cli.QueueManager", _FakeOrchestrator)
    monkeypatch.setattr(
        "mana_agent.integrations.codex.coding_agent_shim.CodexCodingAgentShim.generate_auto_execute",
        lambda _self, request, **_kwargs: (
            codex_calls.append(str(request))
            or {
                "answer": "Codex execution complete",
                "sources": [],
                "trace": [],
                "warnings": [],
                "changed_files": [],
                "passes": 1,
                "pass_logs": [],
                "auto_execute_terminal_reason": "completed",
            }
        ),
    )
    monkeypatch.setattr(
        "mana_agent.commands.chat_cli._generate_planning_question_llm",
        lambda **kwargs: f"Clarification question {int(kwargs['asked_count']) + 1}?",
    )

    result = runner.invoke(
        app,
        ["chat", "--planning-mode", "--auto-execute-plan"],
        input="plan auth module\nanswer one\nanswer two\nanswer three\nquit\n",
    )
    assert result.exit_code == 0
    assert "Codex execution complete" in result.stdout
    assert "Generating decision-complete plan..." not in result.stdout
    assert codex_calls
    assert not _FakeOrchestrator.calls


def test_chat_planning_mode_no_auto_execute_keeps_plan_only_behavior(monkeypatch, tmp_path: Path) -> None:
    class _PlanningLlm:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return type("_Msg", (), {"content": f"Clarification question {self.calls}?"})()

    class _AskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()
            self.qna_chain = type("_Qna", (), {"llm": _PlanningLlm()})()

    class _FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def stop(self) -> None:
            return None

    class _FakeCodingAgent:
        generate_calls: list[str] = []

        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-plan-only"

        def set_tools_manager_orchestrator(self, _orchestrator: object) -> None:
            return None

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, request: str, **_kwargs: object) -> dict:
            _FakeCodingAgent.generate_calls.append(request)
            return {
                "answer": "Plan:\\n1. inspect\\n2. edit\\n3. verify",
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "actions_taken": [],
            }

        def generate_auto_execute(self, *_args: object, **_kwargs: object) -> dict:  # pragma: no cover
            raise AssertionError("default planning requests should not auto-execute")

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _AskService())
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--planning-mode", "--no-auto-execute-plan"],
        input="plan auth module\nanswer one\nanswer two\nanswer three\nquit\n",
    )
    assert result.exit_code == 0
    assert "Generating decision-complete plan..." in result.stdout
    assert "Auto-executing plan" not in result.stdout
    assert _FakeCodingAgent.generate_calls


def test_flow_checklist_cli_view_renders_codex_sections(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def flow_summary(self, flow_id: str | None = None):
            _ = flow_id
            return {
                "flow_id": "flow-123",
                "objective": "Implement planner",
                "checklist": {
                    "objective": "Implement planner",
                    "steps": [
                        {"status": "in_progress", "title": "Inspect file"},
                        {"status": "pending", "title": "Apply patch"},
                    ],
                },
            }

        def checkpoint_flow(self, flow_id: str | None = None) -> str | None:
            return flow_id or self.active

        def reset_flow(self, flow_id: str | None = None) -> str | None:
            _ = flow_id
            return self.active

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return {
                "answer": "done",
                "changed_files": ["src/a.py"],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "Implement planner", "steps": [{"status": "done", "title": "Inspect file"}]},
                "progress": {"phase": "edit", "why": "gate passed", "budgets": {"search_used": 1, "search_budget": 4, "read_used": 2, "read_budget": 6, "read_files_observed": 2, "required_read_files": 2}},
                "checklist": {"done": 1, "pending": 0, "blocked": 0, "total": 1},
                "actions_taken": [{"tool_name": "read_file", "status": "ok", "duration_ms": 1.0, "args_summary": "x"}],
                "next_step": "Run verification",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate()

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="implement planner\n/flow checklist\nquit\n",
    )
    assert result.exit_code == 0
    assert "Plan" in result.stdout
    assert "Progress" in result.stdout
    assert "Checklist" in result.stdout
    assert "Next Step" in result.stdout
    assert "Flow Checklist" in result.stdout
    assert "done" in result.stdout
    assert "Session History" not in result.stdout
    assert "Answer preview" not in result.stdout
    assert "No decisions were recorded for this turn." not in result.stdout


def test_chat_coding_agent_answer_only_on_tools_only_fallback(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return {
                "answer": "Request blocked by tools-only worker policy.",
                "changed_files": [],
                "warnings": ["tools_only_violation: no successful tool calls"],
                "diff": "",
                "flow_id": self.active,
                "actions_taken": [],
                "actions_taken_total": 0,
                "render_mode": "answer_only",
                "fallback_reason": "tools_only_violation",
                "fallback_retry_attempted": True,
            }

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate()

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="update readme.md with new version\nquit\n",
    )
    assert result.exit_code == 0
    assert "Answer" in result.stdout
    assert "Request blocked by tools-only worker policy." in result.stdout
    assert "Summary" not in result.stdout
    assert "Steps" not in result.stdout
    assert "History" not in result.stdout
    assert "Next Step" not in result.stdout


def test_chat_coding_agent_answer_only_when_no_repo_edits(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return {
                "answer": "Analysis complete. No repository edits required.",
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "actions_taken": [
                    {
                        "tool_name": "read_file",
                        "status": "ok",
                        "duration_ms": 1.2,
                        "args_summary": "path='README.md' start=1 end=50",
                    }
                ],
                "actions_taken_total": 1,
                "plan": {"objective": "should not render", "steps": []},
                "progress": {"phase": "inspect", "why": "No edits needed"},
                "checklist": {"done": 1, "pending": 0, "blocked": 0, "total": 1},
                "next_step": "Done",
            }

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate()

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="analyze this src,give me best plan for upgrade\nquit\n",
    )
    assert result.exit_code == 0
    assert "Answer" in result.stdout
    assert "Analysis complete. No repository edits required." in result.stdout
    assert "Summary" not in result.stdout
    assert "Steps" not in result.stdout
    assert "History" not in result.stdout
    assert "Plan" not in result.stdout
    assert "Checklist" not in result.stdout
    assert "Next Step" not in result.stdout


def test_large_json_answer_is_rendered_as_sections_not_raw_blob(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            huge = '{"answer":"' + ("x" * 5000) + '"}'
            return {
                "answer": huge,
                "changed_files": ["src/example.py"],
                "warnings": [],
                "diff": "diff --git a/src/example.py b/src/example.py\n",
                "flow_id": self.active,
                "plan": {"objective": "obj", "steps": []},
                "progress": {"phase": "inspect", "why": "insufficient reads", "budgets": {"search_used": 2, "search_budget": 4, "read_used": 1, "read_budget": 6, "read_files_observed": 1, "required_read_files": 2}},
                "checklist": {"done": 0, "pending": 2, "blocked": 0, "total": 2},
                "actions_taken": [],
                "next_step": "Read one more file",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate()

        def flow_summary(self, flow_id: str | None = None):
            _ = flow_id
            return None

        def checkpoint_flow(self, flow_id: str | None = None) -> str | None:
            return flow_id

        def reset_flow(self, flow_id: str | None = None) -> str | None:
            return flow_id

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="implement\nquit\n",
    )
    assert result.exit_code == 0
    assert "Plan" in result.stdout
    assert "Next Step" in result.stdout


def test_chat_coding_agent_unlimited_mode_bypasses_default_step_cap(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        last_max_steps: int | None = None

        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, *_args: object, **kwargs: object) -> dict:
            _FakeCodingAgent.last_max_steps = int(kwargs.get("max_steps", 0) or 0)
            return {
                "answer": "done",
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "obj", "steps": []},
                "progress": {"phase": "edit", "why": "ok", "budgets": {"search_used": 0, "search_budget": 4, "read_used": 0, "read_budget": 6, "read_files_observed": 0, "required_read_files": 2}},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate_dir_mode(self, *_args: object, **kwargs: object) -> dict:
            return self.generate(*_args, **kwargs)

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent", "--agent-unlimited"],
        input="implement change\nquit\n",
    )
    assert result.exit_code == 0
    assert isinstance(_FakeCodingAgent.last_max_steps, int)
    assert _FakeCodingAgent.last_max_steps is not None
    assert _FakeCodingAgent.last_max_steps > 200


def test_chat_turn_log_preserves_actions_taken_total_when_trace_is_truncated(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return {
                "answer": "done",
                "changed_files": ["src/example.py"],
                "warnings": [],
                "diff": "diff --git a/src/example.py b/src/example.py\n",
                "flow_id": self.active,
                "plan": {"objective": "obj", "steps": []},
                "progress": {"phase": "edit", "why": "ok", "budgets": {"search_used": 0, "search_budget": 4, "read_used": 0, "read_budget": 6, "read_files_observed": 0, "required_read_files": 2}},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken_total": 37,
                "actions_taken_truncated": True,
                "actions_taken": [{"tool_name": "read_file", "status": "ok", "duration_ms": 1.0, "args_summary": "x"}],
                "next_step": "done",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate(*_args, **_kwargs)

    rows: list[dict] = []

    class _FakeRunLogger:
        def __init__(self, log_file=None) -> None:
            _ = log_file

        def log(self, payload: dict) -> None:
            rows.append(payload)

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)
    monkeypatch.setattr("mana_agent.commands.cli.LlmRunLogger", _FakeRunLogger)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="implement\nquit\n",
    )
    assert result.exit_code == 0
    assert "done" in result.stdout
    assert "Session History" not in result.stdout
    assert "Tool steps" not in result.stdout
    chat_rows = [row for row in rows if row.get("flow") == "chat"]
    assert chat_rows
    assert chat_rows[-1].get("tool_steps") == 37


def test_chat_renders_dynamic_plan_and_diagram_blocks_in_normal_path(monkeypatch, tmp_path: Path) -> None:
    class _DynamicAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

        def ask_with_tools(
            self,
            index_dir: str,
            question: str,
            k: int,
            max_steps: int = 6,
            timeout_seconds: int = 30,
        ) -> AskResponseWithTrace:
            _ = (index_dir, question, k, max_steps, timeout_seconds)
            payload = {
                "answer": "Dynamic answer",
                "ui_blocks": [
                    {
                        "type": "plan",
                        "title": "Architecture Plan",
                        "objective": "Ship dynamic UI",
                        "steps": [{"status": "in_progress", "title": "Render ui_blocks", "detail": "Use rich panel tables"}],
                    },
                    {
                        "type": "diagram",
                        "title": "Flow Diagram",
                        "format": "mermaid",
                        "content": "graph TD\nA-->B",
                    },
                ],
            }
            hit = SearchHit(0.9, "/tmp/good.py", 1, 3, "add", "snippet")
            return AskResponseWithTrace(
                answer=json.dumps(payload),
                sources=[hit],
                mode="agent-tools",
                trace=[],
                warnings=[],
            )

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _DynamicAskService())

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--no-coding-agent"],
        input="show dynamic\nquit\n",
    )
    assert result.exit_code == 0
    assert "Architecture Plan" in result.stdout
    assert "Render ui_blocks" in result.stdout
    assert "Flow Diagram" in result.stdout
    assert "graph TD" in result.stdout


def test_chat_inferrs_mermaid_diagram_block_and_renders_before_summary(monkeypatch, tmp_path: Path) -> None:
    class _MermaidAskService(FakeAskService):
        calls: list[str] = []

        def __init__(self) -> None:
            self.ask_agent = object()
            _MermaidAskService.calls = []

        def ask_with_tools(
            self,
            index_dir: str,
            question: str,
            k: int,
            max_steps: int = 6,
            timeout_seconds: int = 30,
        ) -> AskResponseWithTrace:
            _ = (index_dir, k, max_steps, timeout_seconds)
            _MermaidAskService.calls.append(question)
            answer = "```mermaid\ngraph TD\nA-->B\n```"
            return AskResponseWithTrace(answer=answer, sources=[], mode="agent-tools", trace=[], warnings=[])

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _MermaidAskService())

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--no-coding-agent"],
        input="show diagram\nquit\n",
    )
    assert result.exit_code == 0
    assert len(_MermaidAskService.calls) == 1
    assert "graph TD" in result.stdout
    assert "Diagram" in result.stdout
    assert "Session History" not in result.stdout
    assert "Answer preview" not in result.stdout


def test_chat_diagram_artifact_render_invokes_mermaid_renderer(monkeypatch, tmp_path: Path) -> None:
    class _DiagramAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

        def ask_with_tools(
            self,
            index_dir: str,
            question: str,
            k: int,
            max_steps: int = 6,
            timeout_seconds: int = 30,
        ) -> AskResponseWithTrace:
            _ = (index_dir, question, k, max_steps, timeout_seconds)
            payload = {
                "answer": "diagram answer",
                "ui_blocks": [
                    {
                        "type": "diagram",
                        "title": "Flow Diagram",
                        "format": "mermaid",
                        "content": "graph TD\nA-->B",
                    }
                ],
            }
            return AskResponseWithTrace(answer=json.dumps(payload), sources=[], mode="agent-tools", trace=[], warnings=[])

    calls: list[dict] = []

    def _fake_render_mermaid_artifact(
        content: str,
        *,
        output_dir: Path,
        title: str,
        image_format: str,
        timeout_seconds: int,
        project_root: Path | None = None,
    ):
        calls.append(
            {
                "content": content,
                "output_dir": output_dir,
                "title": title,
                "image_format": image_format,
                "timeout_seconds": timeout_seconds,
                "project_root": project_root,
            }
        )
        return (tmp_path / "flow.svg", None)

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _DiagramAskService())
    monkeypatch.setattr("mana_agent.commands.cli._render_mermaid_artifact", _fake_render_mermaid_artifact)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--no-coding-agent", "--diagram-output-dir", str(tmp_path), "--diagram-format", "svg"],
        input="show diagram\nquit\n",
    )
    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["content"] == "graph TD\nA-->B"
    assert calls[0]["image_format"] == "svg"
    assert "Diagram Artifact" in result.stdout
    assert "flow.svg" in result.stdout


def test_chat_no_diagram_render_images_skips_mermaid_artifact(monkeypatch, tmp_path: Path) -> None:
    class _DiagramAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

        def ask_with_tools(
            self,
            index_dir: str,
            question: str,
            k: int,
            max_steps: int = 6,
            timeout_seconds: int = 30,
        ) -> AskResponseWithTrace:
            _ = (index_dir, question, k, max_steps, timeout_seconds)
            payload = {
                "answer": "diagram answer",
                "ui_blocks": [
                    {
                        "type": "diagram",
                        "title": "Flow Diagram",
                        "format": "mermaid",
                        "content": "graph TD\nA-->B",
                    }
                ],
            }
            return AskResponseWithTrace(answer=json.dumps(payload), sources=[], mode="agent-tools", trace=[], warnings=[])

    calls: list[dict] = []

    def _fake_render_mermaid_artifact(
        content: str,
        *,
        output_dir: Path,
        title: str,
        image_format: str,
        timeout_seconds: int,
        project_root: Path | None = None,
    ):
        _ = (content, output_dir, title, image_format, timeout_seconds, project_root)
        calls.append({})
        return (tmp_path / "flow.svg", None)

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _DiagramAskService())
    monkeypatch.setattr("mana_agent.commands.cli._render_mermaid_artifact", _fake_render_mermaid_artifact)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--no-diagram-render-images"],
        input="show diagram\nquit\n",
    )
    assert result.exit_code == 0
    assert len(calls) == 0
    assert "Diagram Artifact" not in result.stdout


def test_chat_coding_path_prefers_dynamic_plan_over_static_plan_section(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            payload = {
                "answer": "dynamic",
                "ui_blocks": [
                    {
                        "type": "plan",
                        "title": "Dynamic Plan",
                        "objective": "DYNAMIC_OBJECTIVE",
                        "steps": [{"status": "done", "title": "done step"}],
                    }
                ],
            }
            return {
                "answer": json.dumps(payload),
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "STATIC_PLAN_SHOULD_NOT_RENDER", "steps": [{"status": "pending", "title": "static"}]},
                "progress": {"phase": "edit", "why": "ok", "budgets": {"search_used": 0, "search_budget": 4, "read_used": 0, "read_budget": 6, "read_files_observed": 0, "required_read_files": 2}},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate(*_args, **_kwargs)

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="implement\nquit\n",
    )
    assert result.exit_code == 0
    assert "DYNAMIC_OBJECTIVE" in result.stdout
    assert "STATIC_PLAN_SHOULD_NOT_RENDER" not in result.stdout


def test_chat_coding_path_inferrs_mermaid_diagram_block_and_renders_before_summary(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return {
                "answer": "```mermaid\ngraph LR\nX-->Y\n```",
                "changed_files": ["src/example.py"],
                "warnings": [],
                "diff": "diff --git a/src/example.py b/src/example.py\n",
                "flow_id": self.active,
                "plan": {"objective": "obj", "steps": []},
                "progress": {"phase": "edit", "why": "ok", "budgets": {"search_used": 0, "search_budget": 4, "read_used": 0, "read_budget": 6, "read_files_observed": 0, "required_read_files": 2}},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate(*_args, **_kwargs)

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="diagram\nquit\n",
    )
    assert result.exit_code == 0
    assert "graph LR" in result.stdout
    assert "Diagram" in result.stdout
    assert "Session History" not in result.stdout
    assert "Answer preview" not in result.stdout


def test_chat_ignores_malformed_ui_blocks_and_falls_back_to_answer(monkeypatch, tmp_path: Path) -> None:
    class _MalformedAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

        def ask_with_tools(
            self,
            index_dir: str,
            question: str,
            k: int,
            max_steps: int = 6,
            timeout_seconds: int = 30,
        ) -> AskResponseWithTrace:
            _ = (index_dir, question, k, max_steps, timeout_seconds)
            payload = {
                "answer": "Fallback answer",
                "ui_blocks": [
                    {"type": "diagram", "content": ""},
                    {"no_type": "x"},
                    "invalid",
                ],
            }
            return AskResponseWithTrace(answer=json.dumps(payload), sources=[], mode="agent-tools", trace=[], warnings=[])

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _MalformedAskService())

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--no-coding-agent"],
        input="tell me about this code\nquit\n",
    )
    assert result.exit_code == 0
    assert "Fallback answer" in result.stdout


def test_chat_handles_effective_ui_blocks_failure_without_crash(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def stop(self) -> None:
            return None

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-ui-blocks"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, request: str, **_kwargs: object) -> dict:
            _ = request
            payload = {
                "answer": "Fallback answer",
                "ui_blocks": [{"type": "diagram", "title": "Flow", "content": "graph LR; A-->B;"}],
            }
            return {
                "answer": json.dumps(payload),
                "changed_files": [],
                "diff": "",
                "warnings": [],
                "flow_id": self.active,
                "plan": {"objective": "Handle request", "steps": []},
                "progress": {"phase": "answer", "why": "done", "tool_call_allowed": False},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None, project_root=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)
    monkeypatch.setattr(
        "mana_agent.commands.cli._effective_ui_blocks",
        lambda _answer, _payload: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = runner.invoke(
        app,
        ["chat"],
        input="tell me about this code\nquit\n",
    )
    assert result.exit_code == 0
    assert "Fallback answer" in result.stdout


def test_chat_selection_flow_accepts_numeric_choice_and_synthesizes_follow_up(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        calls: list[str] = []

        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"
            _FakeCodingAgent.calls = []

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def _base_result(self, answer: str) -> dict:
            return {
                "answer": answer,
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "obj", "steps": []},
                "progress": {"phase": "inspect", "why": "ok", "budgets": {"search_used": 0, "search_budget": 4, "read_used": 0, "read_budget": 6, "read_files_observed": 0, "required_read_files": 2}},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate(self, question: str, *_args: object, **_kwargs: object) -> dict:
            _FakeCodingAgent.calls.append(question)
            if len(_FakeCodingAgent.calls) == 1:
                payload = {
                    "answer": "",
                    "ui_blocks": [
                        {
                            "type": "selection",
                            "id": "mode_select",
                            "prompt": "Pick a mode",
                            "options": [
                                {"id": "safe", "label": "Safe mode", "value": "safe"},
                                {"id": "fast", "label": "Fast mode", "value": "speed"},
                            ],
                        }
                    ],
                }
                return self._base_result(json.dumps(payload))
            return self._base_result("Selection applied")

        def generate_dir_mode(self, *args: object, **kwargs: object) -> dict:
            return self.generate(*args, **kwargs)

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="begin\n2\nquit\n",
    )
    assert result.exit_code == 0
    assert len(_FakeCodingAgent.calls) == 2
    assert _FakeCodingAgent.calls[1] == (
        'User selected "fast" for selection "mode_select" (value="speed"). Continue accordingly.'
    )


def test_chat_selection_flow_reprompts_on_invalid_choice(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        calls: list[str] = []

        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-123"
            _FakeCodingAgent.calls = []

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def _base_result(self, answer: str) -> dict:
            return {
                "answer": answer,
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "obj", "steps": []},
                "progress": {"phase": "inspect", "why": "ok", "budgets": {"search_used": 0, "search_budget": 4, "read_used": 0, "read_budget": 6, "read_files_observed": 0, "required_read_files": 2}},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate(self, question: str, *_args: object, **_kwargs: object) -> dict:
            _FakeCodingAgent.calls.append(question)
            if len(_FakeCodingAgent.calls) == 1:
                payload = {
                    "answer": "",
                    "ui_blocks": [
                        {
                            "type": "continue",
                            "prompt": "Continue current flow?",
                            "options": [
                                {"id": "continue", "label": "Continue"},
                                {"id": "new", "label": "Start new"},
                            ],
                        }
                    ],
                }
                return self._base_result(json.dumps(payload))
            return self._base_result("Handled")

        def generate_dir_mode(self, *args: object, **kwargs: object) -> dict:
            return self.generate(*args, **kwargs)

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="begin\nbad choice\nnew\nquit\n",
    )
    assert result.exit_code == 0
    assert "Invalid selection" in result.stdout
    assert len(_FakeCodingAgent.calls) == 2
    assert _FakeCodingAgent.calls[1] == (
        'User selected "new" for selection "continue_selection" (value="new"). Continue accordingly.'
    )


def test_chat_selection_flow_works_in_normal_agent_tools_path(monkeypatch, tmp_path: Path) -> None:
    class _SelectionAskService(FakeAskService):
        calls: list[str] = []

        def __init__(self) -> None:
            self.ask_agent = object()
            _SelectionAskService.calls = []

        def ask_with_tools(
            self,
            index_dir: str,
            question: str,
            k: int,
            max_steps: int = 6,
            timeout_seconds: int = 30,
        ) -> AskResponseWithTrace:
            _ = (index_dir, k, max_steps, timeout_seconds)
            _SelectionAskService.calls.append(question)
            if len(_SelectionAskService.calls) == 1:
                payload = {
                    "answer": "",
                    "ui_blocks": [
                        {
                            "type": "selection",
                            "id": "normal_select",
                            "prompt": "Pick one",
                            "options": [
                                {"id": "one", "label": "One"},
                                {"id": "two", "label": "Two"},
                            ],
                        }
                    ],
                }
                return AskResponseWithTrace(answer=json.dumps(payload), sources=[], mode="agent-tools", trace=[], warnings=[])
            hit_a = SearchHit(0.8, "/tmp/a.py", 1, 2, "a", "snippet")
            hit_b = SearchHit(0.8, "/tmp/b.py", 3, 4, "b", "snippet")
            return AskResponseWithTrace(
                answer="Normal path handled",
                sources=[hit_a, hit_b],
                mode="agent-tools",
                trace=[],
                warnings=[],
            )

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _SelectionAskService())

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--no-coding-agent", "--no-auto-execute-plan"],
        input="begin\n2\nquit\n",
    )
    assert result.exit_code == 0
    assert len(_SelectionAskService.calls) == 2
    assert _SelectionAskService.calls[1] == (
        'User selected "two" for selection "normal_select" (value="two"). Continue accordingly.'
    )


def test_sanitize_full_auto_answer_text_replaces_confirmation_prompts() -> None:
    text = "If you want, I can continue with more edits. Reply yes to continue."
    cleaned = _sanitize_full_auto_answer_text(text, changed_files_count=0, terminal_reason="pass_cap_reached")
    assert cleaned.startswith("Status: executing")


def test_sanitize_full_auto_answer_text_replaces_synthetic_pass_cap_diagnostic() -> None:
    text = (
        "Auto-execute ended without a direct answer from tool runs.\n"
        "terminal_reason=pass_cap_reached\n"
        "passes=8\n"
        "toolsmanager_requests=19"
    )
    cleaned = _sanitize_full_auto_answer_text(text, changed_files_count=0, terminal_reason="pass_cap_reached")
    assert cleaned.startswith("Status: executing")
    assert "Auto-execute ended without a direct answer" not in cleaned


def test_sanitize_full_auto_answer_text_replaces_non_hard_blocker_prompts() -> None:
    text = "Blocker: I need a scope choice. Please choose option 1 or 2."
    cleaned = _sanitize_full_auto_answer_text(text, changed_files_count=0, terminal_reason="planner_finalize")
    assert cleaned.startswith("Status: executing")


def test_sanitize_full_auto_answer_text_replaces_non_hard_repository_access_prompts() -> None:
    text = (
        "I'm blocked on making a safe, accurate update because I need to read the current repository files first. "
        "Please share permission to proceed."
    )
    cleaned = _sanitize_full_auto_answer_text(text, changed_files_count=0, terminal_reason="planner_finalize")
    assert cleaned.startswith("Status: executing")


def test_render_coding_sections_shows_dynamic_read_budget_metadata() -> None:
    from rich.console import Console

    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    _render_coding_sections(
        console,
        {
            "plan": {},
            "progress": {
                "phase": "inspect",
                "why": "dynamic policy selected",
                "budgets": {
                    "search_used": 1,
                    "search_budget": 4,
                    "read_used": 2,
                    "read_budget": 5,
                    "required_read_files": 2,
                    "read_files_observed": 2,
                    "read_line_window": 900,
                    "dynamic_read_budget_used": True,
                    "dynamic_read_budget_fallback_used": False,
                },
            },
            "checklist": {"done": 1, "pending": 0, "blocked": 0, "total": 1},
            "next_step": "continue",
        },
        show_actions=False,
        show_warnings=False,
    )
    rendered = output.getvalue()
    assert "read-window: 900" in rendered
    assert "dynamic: True" in rendered
    assert "fallback: False" in rendered


def test_chat_coding_read_budget_cli_value_is_passed_to_coding_agent_cap(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def stop(self) -> None:
            return None

    class _FakeCodingAgent:
        init_kwargs: dict[str, object] = {}

        def __init__(self, **kwargs: object) -> None:
            _FakeCodingAgent.init_kwargs = dict(kwargs)
            self.active = "flow-budget-cap"

        def set_tools_manager_orchestrator(self, _orchestrator: object) -> None:
            return None

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate_auto_execute(self, *_args: object, **_kwargs: object) -> dict:
            return {
                "answer": "done",
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "Implement request", "steps": []},
                "progress": {"phase": "answer", "why": "complete"},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "auto_execute_passes": 1,
                "auto_execute_terminal_reason": "completed",
                "toolsmanager_requests_count": 1,
                "pass_logs": [],
                "planner_decisions": [],
            }

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        [
            "chat",
            "--agent-tools",
            "--coding-agent",
            "--execution-profile",
            "full-auto",
            "--coding-read-budget",
            "11",
        ],
        input="update readme.md\nquit\n",
    )
    assert result.exit_code == 0
    assert int(_FakeCodingAgent.init_kwargs.get("read_budget", 0) or 0) == 11
    assert bool(_FakeCodingAgent.init_kwargs.get("full_auto_mode", False)) is True


def test_chat_balanced_profile_keeps_coding_agent_non_full_auto_mode(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def stop(self) -> None:
            return None

    class _FakeCodingAgent:
        init_kwargs: dict[str, object] = {}

        def __init__(self, **kwargs: object) -> None:
            _FakeCodingAgent.init_kwargs = dict(kwargs)
            self.active = "flow-balanced-budget"

        def get_active_flow_id(self) -> str | None:
            return self.active

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        [
            "chat",
            "--agent-tools",
            "--coding-agent",
            "--execution-profile",
            "balanced",
            "--coding-read-budget",
            "7",
        ],
        input="quit\n",
    )
    assert result.exit_code == 0
    assert int(_FakeCodingAgent.init_kwargs.get("read_budget", 0) or 0) == 7
    assert bool(_FakeCodingAgent.init_kwargs.get("full_auto_mode", True)) is False


def test_chat_balanced_profile_auto_executes_clear_edit_requests(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def stop(self) -> None:
            return None

    class _FakeCodingAgent:
        auto_calls: list[dict[str, object]] = []

        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-balanced-auto"

        def set_tools_manager_orchestrator(self, _orchestrator: object) -> None:
            return None

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate_auto_execute(self, *_args: object, **kwargs: object) -> dict:
            _FakeCodingAgent.auto_calls.append(dict(kwargs))
            return {
                "answer": "done",
                "changed_files": ["README.md"],
                "warnings": [],
                "diff": "diff --git a/README.md b/README.md\n",
                "flow_id": self.active,
                "plan": {"objective": "Update README", "steps": []},
                "progress": {"phase": "answer", "why": "complete"},
                "checklist": {"done": 1, "pending": 0, "blocked": 0, "total": 1},
                "actions_taken": [],
                "next_step": "done",
                "auto_execute_passes": 1,
                "auto_execute_terminal_reason": "completed",
                "toolsmanager_requests_count": 1,
                "pass_logs": [],
                "planner_decisions": [],
            }

        def generate(self, *_args: object, **_kwargs: object) -> dict:  # pragma: no cover - should not be used
            raise AssertionError("balanced edit requests should route to generate_auto_execute")

    monkeypatch.setattr("mana_agent.commands.chat_cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.chat_cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.chat_cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.chat_cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--execution-profile", "balanced"],
        input="update README.md with the current CLI flags\nquit\n",
    )
    assert result.exit_code == 0
    assert _FakeCodingAgent.auto_calls
    assert int(_FakeCodingAgent.auto_calls[0].get("pass_cap", 0) or 0) == 4


def test_chat_full_auto_profile_forces_auto_execute_for_edit_requests(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def stop(self) -> None:
            return None

    class _FakeCodingAgent:
        auto_calls: list[dict[str, object]] = []

        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-fa-1"

        def set_tools_manager_orchestrator(self, _orchestrator: object) -> None:
            return None

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate_auto_execute(self, *_args: object, **kwargs: object) -> dict:
            _FakeCodingAgent.auto_calls.append(dict(kwargs))
            return {
                "answer": "done",
                "changed_files": ["src/example.py"],
                "warnings": [],
                "diff": "diff --git a/src/example.py b/src/example.py\n",
                "flow_id": self.active,
                "plan": {"objective": "Implement request", "steps": []},
                "progress": {"phase": "answer", "why": "complete"},
                "checklist": {"done": 1, "pending": 0, "blocked": 0, "total": 1},
                "actions_taken": [],
                "next_step": "done",
                "auto_execute_passes": 1,
                "auto_execute_terminal_reason": "completed",
                "toolsmanager_requests_count": 1,
                "pass_logs": [],
                "planner_decisions": [],
            }

        def generate(self, *_args: object, **_kwargs: object) -> dict:  # pragma: no cover - should not be used
            raise AssertionError("full-auto should route edit requests to generate_auto_execute")

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:  # pragma: no cover - should not be used
            raise AssertionError("full-auto should route edit requests to generate_auto_execute")

    monkeypatch.setattr("mana_agent.commands.chat_cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.chat_cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.chat_cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.chat_cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--execution-profile", "full-auto"],
        input="update readme.md with full cli flags and descriptions\nquit\n",
    )
    assert result.exit_code == 0
    assert _FakeCodingAgent.auto_calls
    assert int(_FakeCodingAgent.auto_calls[0].get("pass_cap", 0) or 0) == 10
    assert "Mana-Agent" in result.stdout
    assert "mode chat" in result.stdout
    assert "execution profile" not in result.stdout


def test_chat_full_auto_conflict_is_auto_continued(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def stop(self) -> None:
            return None

    class _FakeCodingAgent:
        auto_calls = 0

        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-conflict-1"

        def set_tools_manager_orchestrator(self, _orchestrator: object) -> None:
            return None

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return True

        def generate_auto_execute(self, *_args: object, **_kwargs: object) -> dict:
            _FakeCodingAgent.auto_calls += 1
            return {
                "answer": "done",
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "Implement request", "steps": []},
                "progress": {"phase": "answer", "why": "complete"},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "auto_execute_passes": 1,
                "auto_execute_terminal_reason": "completed",
                "toolsmanager_requests_count": 1,
                "pass_logs": [],
                "planner_decisions": [],
            }

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate_auto_execute()

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate_auto_execute()

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent", "--execution-profile", "full-auto"],
        input="please modify src/example.py\nquit\n",
    )
    assert result.exit_code == 0
    assert _FakeCodingAgent.auto_calls == 1
    assert "This request appears to diverge from the active flow." not in result.stdout


def test_chat_model_starts_distinct_work_without_control_prompt(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        calls: list[dict[str, object]] = []
        reset_ids: list[str] = []

        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-existing"
            _FakeCodingAgent.calls = []
            _FakeCodingAgent.reset_ids = []

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = request
            return flow_id == "flow-existing"

        def reset_flow(self, flow_id: str | None = None) -> str | None:
            target = flow_id or self.active
            if target:
                _FakeCodingAgent.reset_ids.append(target)
            self.active = None
            return target

        def generate(self, question: str, **kwargs: object) -> dict:
            _FakeCodingAgent.calls.append(
                {
                    "question": question,
                    "flow_id": kwargs.get("flow_id"),
                }
            )
            self.active = "flow-new"
            return {
                "answer": "updated",
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "new request", "steps": []},
                "progress": {"phase": "answer", "why": "done", "tool_call_allowed": False},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate_dir_mode(self, *args: object, **kwargs: object) -> dict:
            return self.generate(*args, **kwargs)

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)
    monkeypatch.setattr(
        "mana_agent.commands.chat_cli._decide_chat_route",
        lambda **_kwargs: AgentDecision(
            intent="edit",
            confidence=1.0,
            selected_tools=["repo_search", "read_file", "apply_patch"],
            repo_context_needed=True,
            code_editing_needed=True,
            flow_action="new",
            verifier_passed=True,
        ),
    )

    request = "add .mana to .gitignore"
    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input=f"{request}\nquit\n",
    )

    assert result.exit_code == 0
    assert "This request appears to diverge from the active flow." not in result.stdout
    assert _FakeCodingAgent.reset_ids == ["flow-existing"]
    assert _FakeCodingAgent.calls == [{"question": request, "flow_id": None}]


def test_chat_new_topic_resets_flow_but_keeps_history(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        calls: list[dict[str, object]] = []
        reset_ids: list[str] = []

        def __init__(self, **_kwargs: object) -> None:
            self.active: str | None = "flow-existing"
            _FakeCodingAgent.calls = []
            _FakeCodingAgent.reset_ids = []

        def get_active_flow_id(self) -> str | None:
            return self.active

        def reset_flow(self, flow_id: str | None = None) -> str | None:
            target = flow_id or self.active
            if target:
                _FakeCodingAgent.reset_ids.append(target)
            self.active = None
            return target

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, question: str, **kwargs: object) -> dict:
            _FakeCodingAgent.calls.append({"question": question, "flow_id": kwargs.get("flow_id")})
            self.active = f"flow-{len(_FakeCodingAgent.calls)}"
            return {
                "answer": f"updated {len(_FakeCodingAgent.calls)}",
                "changed_files": ["README.md"],
                "warnings": [],
                "diff": "diff --git a/README.md b/README.md\n",
                "flow_id": self.active,
                "plan": {"objective": question, "steps": []},
                "progress": {"phase": "answer", "why": "done", "tool_call_allowed": False},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate_dir_mode(self, *args: object, **kwargs: object) -> dict:
            return self.generate(*args, **kwargs)

    rendered_history_lengths: list[int] = []

    def _capture_turn_transparency(_console: object, *, turn: object, history: list[object]) -> None:
        _ = turn
        rendered_history_lengths.append(len(history))

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)
    monkeypatch.setattr("mana_agent.commands.chat_cli._render_turn_transparency", _capture_turn_transparency)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="change README title\nnew topic chat\nchange pyproject description\nquit\n",
    )

    assert result.exit_code == 0
    assert "Started new chat topic; flow reset: flow-1" in result.stdout
    assert _FakeCodingAgent.reset_ids == ["flow-1"]
    assert _FakeCodingAgent.calls[0] == {
        "question": "change README title",
        "flow_id": "flow-existing",
    }
    assert _FakeCodingAgent.calls[1]["flow_id"] is None
    assert "User: change README title" in str(_FakeCodingAgent.calls[1]["question"])
    assert "Assistant: updated 1" in str(_FakeCodingAgent.calls[1]["question"])
    assert str(_FakeCodingAgent.calls[1]["question"]).endswith("change pyproject description")
    assert rendered_history_lengths == [1, 2]


def test_chat_explicit_new_topic_still_starts_new_flow(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        calls: list[dict[str, object]] = []
        reset_ids: list[str] = []

        def __init__(self, **_kwargs: object) -> None:
            self.active: str | None = "flow-existing"
            _FakeCodingAgent.calls = []
            _FakeCodingAgent.reset_ids = []

        def get_active_flow_id(self) -> str | None:
            return self.active

        def reset_flow(self, flow_id: str | None = None) -> str | None:
            target = flow_id or self.active
            if target:
                _FakeCodingAgent.reset_ids.append(target)
            self.active = None
            return target

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = request
            return flow_id == "flow-existing"

        def generate(self, question: str, **kwargs: object) -> dict:
            _FakeCodingAgent.calls.append({"question": question, "flow_id": kwargs.get("flow_id")})
            self.active = "flow-new"
            return {
                "answer": "updated",
                "changed_files": [".gitignore"],
                "warnings": [],
                "diff": "diff --git a/.gitignore b/.gitignore\n",
                "flow_id": self.active,
                "plan": {"objective": "new request", "steps": []},
                "progress": {"phase": "answer", "why": "done", "tool_call_allowed": False},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate_dir_mode(self, *args: object, **kwargs: object) -> dict:
            return self.generate(*args, **kwargs)

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    request = "add .mana to .gitignore"
    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input=f"{request}\nnew topic\nquit\n",
    )

    assert result.exit_code == 0
    assert "This request appears to diverge from the active flow." not in result.stdout
    assert "Started new chat topic; flow reset: flow-new" in result.stdout
    assert _FakeCodingAgent.reset_ids == ["flow-new"]
    assert _FakeCodingAgent.calls == [{"question": request, "flow_id": "flow-existing"}]


def test_chat_clear_still_clears_visible_history(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeCodingAgent:
        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-existing"

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate(self, question: str, **kwargs: object) -> dict:
            _ = kwargs
            return {
                "answer": f"updated {question}",
                "changed_files": ["README.md"],
                "warnings": [],
                "diff": "diff --git a/README.md b/README.md\n",
                "flow_id": self.active,
                "plan": {"objective": question, "steps": []},
                "progress": {"phase": "answer", "why": "done", "tool_call_allowed": False},
                "checklist": {"done": 0, "pending": 0, "blocked": 0, "total": 0},
                "actions_taken": [],
                "next_step": "done",
                "static_analysis": {"finding_count": 0, "findings": []},
            }

        def generate_dir_mode(self, *args: object, **kwargs: object) -> dict:
            return self.generate(*args, **kwargs)

    rendered_history_lengths: list[int] = []

    def _capture_turn_transparency(_console: object, *, turn: object, history: list[object]) -> None:
        _ = turn
        rendered_history_lengths.append(len(history))

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)
    monkeypatch.setattr("mana_agent.commands.chat_cli._render_turn_transparency", _capture_turn_transparency)

    result = runner.invoke(
        app,
        ["chat", "--agent-tools", "--coding-agent"],
        input="change README title\n/clear\nchange README subtitle\nquit\n",
    )

    assert result.exit_code == 0
    assert "Chat history cleared." in result.stdout
    assert rendered_history_lengths == [1, 2]


def test_chat_full_auto_pass_cap_auto_resumes_until_completion(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def stop(self) -> None:
            return None

    class _FakeCodingAgent:
        auto_calls = 0

        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-checkpoint-1"

        def set_tools_manager_orchestrator(self, _orchestrator: object) -> None:
            return None

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate_auto_execute(self, *_args: object, **_kwargs: object) -> dict:
            _FakeCodingAgent.auto_calls += 1
            emit_tool_event(
                "start",
                "tool_worker",
                args=f"cycle {_FakeCodingAgent.auto_calls}",
                event_id=f"cycle-{_FakeCodingAgent.auto_calls}",
            )
            emit_tool_event(
                "end",
                "tool_worker",
                duration=0.0,
                event_id=f"cycle-{_FakeCodingAgent.auto_calls}",
            )
            if _FakeCodingAgent.auto_calls == 1:
                return {
                    "answer": (
                        "Auto-execute ended without a direct answer from tool runs.\n"
                        "terminal_reason=pass_cap_reached\n"
                        "passes=2\n"
                        "toolsmanager_requests=1"
                    ),
                    "changed_files": [],
                    "warnings": [],
                    "diff": "",
                    "flow_id": self.active,
                    "plan": {
                        "objective": "Implement request",
                        "steps": [
                            {"status": "done", "title": "Inspect files"},
                            {"status": "pending", "title": "Apply edit"},
                        ],
                    },
                    "checklist": {"done": 1, "pending": 1, "blocked": 0, "total": 2},
                    "progress": {"phase": "answer", "why": "continue_execution"},
                    "next_step": "continue_execution",
                    "auto_execute_passes": 2,
                    "auto_execute_terminal_reason": "pass_cap_reached",
                    "toolsmanager_requests_count": 1,
                    "pass_logs": [
                        {
                            "pass_index": 1,
                            "planner_decision": "continue",
                            "planner_decision_reason": "pass one",
                        },
                        {
                            "pass_index": 2,
                            "planner_decision": "continue",
                            "planner_decision_reason": "pass two",
                        },
                    ],
                    "planner_decisions": [
                        {"pass_index": 1, "decision": "decide-one", "decision_reason": "pass one"},
                        {"pass_index": 2, "decision": "decide-two", "decision_reason": "pass two"},
                    ],
                }
            return {
                "answer": "done",
                "changed_files": ["src/example.py"],
                "warnings": [],
                "diff": "diff --git a/src/example.py b/src/example.py\n",
                "flow_id": self.active,
                "plan": {
                    "objective": "Implement request",
                    "steps": [{"status": "done", "title": "Finalize"}],
                },
                "checklist": {"done": 1, "pending": 0, "blocked": 0, "total": 1},
                "progress": {"phase": "answer", "why": "complete"},
                "next_step": "done",
                "auto_execute_passes": 1,
                "auto_execute_terminal_reason": "completed",
                "toolsmanager_requests_count": 1,
                "pass_logs": [],
                "planner_decisions": [],
            }

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate_auto_execute()

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate_auto_execute()

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        [
            "chat",
            "--agent-tools",
            "--coding-agent",
            "--execution-profile",
            "full-auto",
            "--full-auto-status-every",
            "2",
        ],
        input="please modify src/a.py\nquit\n",
    )
    assert result.exit_code == 0
    assert _FakeCodingAgent.auto_calls == 2
    assert "Tool activity" not in result.stdout
    assert result.stdout.count("─ tools ") == 1
    # "tool_worker" appears once per cycle in the tools panel + once per cycle from the
    # terminal tool event decoration lines written via InlineChatRenderer after console.print(activity).
    assert result.stdout.count("tool_worker") == 4
    assert result.stdout.count("Full-auto Checkpoint") == 1
    assert "checklist: done 1 | pending 1 | blocked 0 | total 2" in result.stdout
    assert "Status: executing (pass_cap_reached)." not in result.stdout
    assert "Auto-execute ended without a direct answer" not in result.stdout
    assert "done" in result.stdout


def test_chat_full_auto_checkpoint_window_is_non_overlapping(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def stop(self) -> None:
            return None

    class _FakeCodingAgent:
        auto_calls = 0

        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-checkpoint-window"

        def set_tools_manager_orchestrator(self, _orchestrator: object) -> None:
            return None

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate_auto_execute(self, *_args: object, **_kwargs: object) -> dict:
            _FakeCodingAgent.auto_calls += 1
            if _FakeCodingAgent.auto_calls == 1:
                return {
                    "answer": "working",
                    "changed_files": [],
                    "warnings": [],
                    "diff": "",
                    "flow_id": self.active,
                    "plan": {"objective": "Implement request", "steps": []},
                    "checklist": {"done": 0, "pending": 2, "blocked": 0, "total": 2},
                    "progress": {"phase": "answer", "why": "continue_execution"},
                    "next_step": "continue_execution",
                    "auto_execute_passes": 4,
                    "auto_execute_terminal_reason": "pass_cap_reached",
                    "toolsmanager_requests_count": 1,
                    "pass_logs": [
                        {"pass_index": 1, "planner_decision": "continue", "planner_decision_reason": "pass one"},
                        {"pass_index": 2, "planner_decision": "continue", "planner_decision_reason": "pass two"},
                        {"pass_index": 3, "planner_decision": "continue", "planner_decision_reason": "pass three"},
                        {"pass_index": 4, "planner_decision": "continue", "planner_decision_reason": "pass four"},
                    ],
                    "planner_decisions": [
                        {"pass_index": 1, "decision": "decide-one", "decision_reason": "pass one"},
                        {"pass_index": 2, "decision": "decide-two", "decision_reason": "pass two"},
                        {"pass_index": 3, "decision": "decide-three", "decision_reason": "pass three"},
                        {"pass_index": 4, "decision": "decide-four", "decision_reason": "pass four"},
                    ],
                }
            return {
                "answer": "done",
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "Implement request", "steps": []},
                "progress": {"phase": "answer", "why": "complete"},
                "next_step": "done",
                "auto_execute_passes": 1,
                "auto_execute_terminal_reason": "completed",
                "toolsmanager_requests_count": 1,
                "pass_logs": [],
                "planner_decisions": [],
            }

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate_auto_execute()

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate_auto_execute()

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        [
            "chat",
            "--agent-tools",
            "--coding-agent",
            "--execution-profile",
            "full-auto",
            "--full-auto-status-every",
            "2",
        ],
        input="please modify src/a.py\nquit\n",
    )
    assert result.exit_code == 0
    assert result.stdout.count("Full-auto Checkpoint") == 2
    assert result.stdout.count("decide-two") == 1
    assert result.stdout.count("decide-four") == 1


def test_chat_full_auto_checkpoint_can_be_disabled(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def stop(self) -> None:
            return None

    class _FakeCodingAgent:
        auto_calls = 0

        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-checkpoint-2"

        def set_tools_manager_orchestrator(self, _orchestrator: object) -> None:
            return None

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate_auto_execute(self, *_args: object, **_kwargs: object) -> dict:
            _FakeCodingAgent.auto_calls += 1
            if _FakeCodingAgent.auto_calls == 1:
                return {
                    "answer": "working",
                    "changed_files": [],
                    "warnings": [],
                    "diff": "",
                    "flow_id": self.active,
                    "plan": {"objective": "Implement request", "steps": [{"status": "pending", "title": "Inspect"}]},
                    "progress": {"phase": "answer", "why": "continue_execution"},
                    "next_step": "continue_execution",
                    "auto_execute_passes": 1,
                    "auto_execute_terminal_reason": "pass_cap_reached",
                    "toolsmanager_requests_count": 1,
                    "pass_logs": [{"pass_index": 1, "planner_decision": "continue", "planner_decision_reason": "pass"}],
                    "planner_decisions": [{"pass_index": 1, "decision": "decide-pass", "decision_reason": "pass"}],
                }
            return {
                "answer": "done",
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "Implement request", "steps": [{"status": "pending", "title": "Inspect"}]},
                "progress": {"phase": "answer", "why": "complete"},
                "next_step": "done",
                "auto_execute_passes": 1,
                "auto_execute_terminal_reason": "completed",
                "toolsmanager_requests_count": 1,
                "pass_logs": [],
                "planner_decisions": [],
            }

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate_auto_execute()

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate_auto_execute()

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        [
            "chat",
            "--agent-tools",
            "--coding-agent",
            "--execution-profile",
            "full-auto",
            "--full-auto-status-every",
            "0",
        ],
        input="please modify src/a.py\nquit\n",
    )
    assert result.exit_code == 0
    assert _FakeCodingAgent.auto_calls == 2
    assert "Full-auto Checkpoint" not in result.stdout


def test_chat_no_auto_continue_does_not_resume_pass_cap(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def stop(self) -> None:
            return None

    class _FakeCodingAgent:
        auto_calls = 0

        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-balanced-1"

        def set_tools_manager_orchestrator(self, _orchestrator: object) -> None:
            return None

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate_auto_execute(self, *_args: object, **_kwargs: object) -> dict:
            _FakeCodingAgent.auto_calls += 1
            return {
                "answer": "need more passes",
                "changed_files": [],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "Implement request", "steps": [{"status": "pending", "title": "Inspect"}]},
                "progress": {"phase": "answer", "why": "continue_execution"},
                "checklist": {"done": 0, "pending": 1, "blocked": 0, "total": 1},
                "actions_taken": [],
                "next_step": "continue_execution",
                "auto_execute_passes": 1,
                "auto_execute_terminal_reason": "pass_cap_reached",
                "toolsmanager_requests_count": 1,
                "pass_logs": [{"pass_index": 1, "planner_decision": "continue", "planner_decision_reason": "pass"}],
                "planner_decisions": [{"pass_index": 1, "decision": "decide-pass", "decision_reason": "pass"}],
            }

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate_auto_execute()

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate_auto_execute()

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        [
            "chat",
            "--agent-tools",
            "--coding-agent",
            "--execution-profile",
            "balanced",
            "--no-auto-continue",
            "--full-auto-status-every",
            "2",
        ],
        input="implement plan.\nquit\n",
    )
    assert result.exit_code == 0
    assert _FakeCodingAgent.auto_calls == 1
    assert "Full-auto Checkpoint" not in result.stdout


def test_chat_balanced_mode_auto_continues_pass_cap_by_default(monkeypatch, tmp_path: Path) -> None:
    class _FakeAskService(FakeAskService):
        def __init__(self) -> None:
            self.ask_agent = object()

    class _FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def stop(self) -> None:
            return None

    class _FakeCodingAgent:
        auto_calls = 0

        def __init__(self, **_kwargs: object) -> None:
            self.active = "flow-balanced-auto-continue"

        def set_tools_manager_orchestrator(self, _orchestrator: object) -> None:
            return None

        def get_active_flow_id(self) -> str | None:
            return self.active

        def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
            _ = (request, flow_id)
            return False

        def generate_auto_execute(self, *_args: object, **_kwargs: object) -> dict:
            _FakeCodingAgent.auto_calls += 1
            if _FakeCodingAgent.auto_calls == 1:
                return {
                    "answer": "need more passes",
                    "changed_files": [],
                    "warnings": [],
                    "diff": "",
                    "flow_id": self.active,
                    "plan": {"objective": "Implement request", "steps": [{"status": "pending", "title": "Inspect"}]},
                    "progress": {"phase": "answer", "why": "continue_execution"},
                    "checklist": {"done": 0, "pending": 1, "blocked": 0, "total": 1},
                    "actions_taken": [],
                    "next_step": "continue_execution",
                    "auto_execute_passes": 1,
                    "auto_execute_terminal_reason": "pass_cap_reached",
                    "toolsmanager_requests_count": 1,
                    "pass_logs": [{"pass_index": 1, "planner_decision": "continue", "planner_decision_reason": "pass"}],
                    "planner_decisions": [{"pass_index": 1, "decision": "decide-pass", "decision_reason": "pass"}],
                    "run_id": "balanced-run-1",
                }
            return {
                "answer": "done",
                "changed_files": ["src/a.py"],
                "warnings": [],
                "diff": "",
                "flow_id": self.active,
                "plan": {"objective": "Implement request", "steps": [{"status": "done", "title": "Inspect"}]},
                "progress": {"phase": "answer", "why": "complete"},
                "checklist": {"done": 1, "pending": 0, "blocked": 0, "total": 1},
                "actions_taken": [],
                "next_step": "done",
                "auto_execute_passes": 1,
                "auto_execute_terminal_reason": "completed",
                "toolsmanager_requests_count": 1,
                "pass_logs": [],
                "planner_decisions": [],
                "run_id": "balanced-run-1",
            }

        def generate(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate_auto_execute()

        def generate_dir_mode(self, *_args: object, **_kwargs: object) -> dict:
            return self.generate_auto_execute()

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: _FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.CodingAgent", _FakeCodingAgent)

    result = runner.invoke(
        app,
        [
            "chat",
            "--agent-tools",
            "--coding-agent",
            "--execution-profile",
            "balanced",
            "--full-auto-status-every",
            "2",
        ],
        input="implement plan.\nquit\n",
    )

    assert result.exit_code == 0
    assert _FakeCodingAgent.auto_calls == 2
    assert "done" in result.stdout
    assert "need more passes" not in result.stdout
    assert "Full-auto Checkpoint" not in result.stdout


def test_chat_full_auto_tools_manager_path_auto_resumes_docs_update_pass_cap(monkeypatch, tmp_path: Path) -> None:
    class _FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def stop(self) -> None:
            return None

    class _FakeAutoResult:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def model_dump(self) -> dict[str, object]:
            return dict(self.payload)

    class _FakeOrchestrator:
        run_calls = 0
        run_ids: list[object] = []

        def __init__(self, **_kwargs: object) -> None:
            return None

        def preview_plan(self, **_kwargs: object) -> dict:
            return {
                "prechecklist": {
                    "objective": "Find all models and update docs/models.md",
                    "source": "planner",
                    "steps": [{"id": "s1", "status": "in_progress", "title": "Find all models"}],
                },
                "prechecklist_source": "planner",
                "prechecklist_warning": "",
                "warnings": [],
            }

        def run(self, **_kwargs: object):
            _FakeOrchestrator.run_calls += 1
            _FakeOrchestrator.run_ids.append(_kwargs.get("run_id"))
            if _FakeOrchestrator.run_calls == 1:
                return _FakeAutoResult(
                    {
                        "answer": (
                            "Auto-execute ended without a direct answer from tool runs.\n"
                            "terminal_reason=pass_cap_reached\n"
                            "passes=2\n"
                            "toolsmanager_requests=1"
                        ),
                        "sources": [],
                        "trace": [],
                        "warnings": [],
                        "changed_files": [],
                        "plan": {
                            "objective": "Find all models and update docs/models.md",
                            "steps": [{"id": "s1", "title": "Find all models"}],
                        },
                        "passes": 2,
                        "terminal_reason": "pass_cap_reached",
                        "toolsmanager_requests_count": 1,
                        "pass_logs": [
                            {"pass_index": 1, "planner_decision": "continue", "planner_decision_reason": "pass one"},
                            {"pass_index": 2, "planner_decision": "continue", "planner_decision_reason": "pass two"},
                        ],
                        "planner_decisions": [
                            {"pass_index": 1, "decision": "decide-one", "decision_reason": "pass one"},
                            {"pass_index": 2, "decision": "decide-two", "decision_reason": "pass two"},
                        ],
                        "checklist": {"done": 1, "pending": 1, "blocked": 0, "total": 2},
                        "run_id": "persisted-run-1",
                    }
                )
            return _FakeAutoResult(
                {
                    "answer": "docs/models.md updated",
                    "sources": [],
                    "trace": [],
                    "warnings": [],
                    "changed_files": ["docs/models.md"],
                    "plan": {
                        "objective": "Find all models and update docs/models.md",
                        "steps": [{"id": "s1", "title": "Update docs/models.md"}],
                    },
                    "passes": 1,
                    "terminal_reason": "manager_stop",
                    "toolsmanager_requests_count": 1,
                    "pass_logs": [],
                    "planner_decisions": [],
                    "run_id": "persisted-run-1",
                }
            )

    monkeypatch.setattr("mana_agent.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_agent.commands.cli.build_ask_service", lambda _s, model_override=None: FakeAskService())
    monkeypatch.setattr("mana_agent.commands.cli.ToolWorkerClient", _FakeWorkerClient)
    monkeypatch.setattr("mana_agent.commands.cli.QueueManager", _FakeOrchestrator)

    result = runner.invoke(
        app,
        [
            "chat",
            "--agent-tools",
            "--no-coding-agent",
            "--execution-profile",
            "full-auto",
            "--full-auto-status-every",
            "2",
        ],
        input="execute plan: find all models and update docs/models.md\nquit\n",
    )
    assert result.exit_code == 0
    assert _FakeOrchestrator.run_calls == 2
    assert _FakeOrchestrator.run_ids == [None, "persisted-run-1"]
    assert result.stdout.count("Full-auto Checkpoint") == 1
    assert "docs/models.md updated" in result.stdout
    assert "Auto-execute ended without a direct answer" not in result.stdout
    assert "terminal_reason=pass_cap_reached" not in result.stdout
    assert "Status: executing (pass_cap_reached)." not in result.stdout
