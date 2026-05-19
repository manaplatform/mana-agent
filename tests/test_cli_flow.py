import json
from pathlib import Path

from typer.testing import CliRunner

from mana_analyzer.commands.cli import app
from mana_analyzer.services.coding_memory_service import CodingMemoryService
from mana_analyzer.analysis.models import Finding

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


class _FakeIndexService:
    def index(self, target_path: str, index_dir: Path, rebuild: bool = False, vectors: bool = True) -> dict:
        _ = (target_path, rebuild, vectors)
        return {"indexed_files": 0, "deleted_files": 0, "total_files": 0, "new_chunks": 0, "removed_chunks": 0, "index_dir": str(index_dir)}


class _FakeDependencyReport:
    project_root = "/tmp/project"
    frameworks: list[str] = []
    technologies: list[str] = []
    package_managers = ["pip"]
    languages = ["python"]
    runtime_dependencies: list[str] = []
    dev_dependencies: list[str] = []
    module_edges: list[object] = []
    dependency_edges: list[object] = []
    manifests: list[str] = []
    files: list[Path] = []

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
            "manifests": [],
        }

    def to_dot(self) -> str:
        return "digraph mana_analyzer {}"

    def to_graphml(self) -> str:
        return "<graphml></graphml>"


class _FakeDependencyService:
    def analyze(self, path: str) -> _FakeDependencyReport:
        _ = path
        return _FakeDependencyReport()

    def collect_inventory(self, path: str) -> list[object]:
        _ = path
        return []


class _FakeDescribeService:
    def describe(self, *args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        return type("_Report", (), {"to_dict": lambda self: {"architecture_summary": "arch", "tech_summary": "tech", "descriptions": []}})()


class _FakeStructureService:
    def __init__(self, include_tests: bool = True) -> None:
        self.include_tests = include_tests

    def analyze_project(self, path: str) -> object:
        _ = path
        return type("_Structure", (), {"to_dict": lambda self: {"project_root": "/tmp/project", "language_counts": {}}})()


def _patch_analyze_dependencies(monkeypatch) -> None:
    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_index_service", lambda _s: _FakeIndexService())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_dependency_service", lambda: _FakeDependencyService())
    monkeypatch.setattr("mana_analyzer.commands.cli.build_analyze_service", lambda: type("_Analyze", (), {"analyze": lambda self, path: []})())
    monkeypatch.setattr(
        "mana_analyzer.commands.cli.build_llm_analyze_service",
        lambda _s, model_override=None: type("_LlmAnalyze", (), {"analyze": lambda self, path, static_findings, max_files=10: []})(),
    )
    monkeypatch.setattr("mana_analyzer.commands.cli.build_describe_service", lambda *_args, **_kwargs: _FakeDescribeService())
    monkeypatch.setattr("mana_analyzer.commands.cli.StructureService", _FakeStructureService)


def _seed_flow(project_root: Path) -> str:
    service = CodingMemoryService(project_root=project_root, max_turns=5, max_tasks=20)
    flow_id = service.ensure_flow(flow_id=None, request="Implement flow summary command and docs")
    service.record_turn(
        flow_id=flow_id,
        user_request="Implement flow summary command and docs",
        effective_prompt="system prompt",
        agent_answer=(
            "Decision: Keep persistence schema unchanged\n"
            "- [x] Wire flow command\n"
            "- [ ] Write docs\n"
        ),
        changed_files=["src/mana_analyzer/commands/cli.py", "README.md"],
        warnings=["write_file fallback was used once"],
        static_findings=["missing-docstring: src/mana_analyzer/services/index_service.py:27"],
        checklist={
            "objective": "Implement flow visibility and docs",
            "steps": [
                {"status": "done", "title": "Wire flow command"},
                {"status": "blocked", "title": "Finish docs"},
            ],
        },
        transitions=[
            {"from_phase": "discover", "to_phase": "edit", "reason": "files identified"},
            {"from_phase": "edit", "to_phase": "blocked", "reason": "waiting for docs update"},
        ],
    )
    return flow_id


def test_flow_command_removed(tmp_path: Path) -> None:
    result = runner.invoke(app, ["flow", str(tmp_path)])
    assert result.exit_code != 0


def test_analyze_flow_no_active_flow(monkeypatch, tmp_path: Path) -> None:
    _patch_analyze_dependencies(monkeypatch)
    result = runner.invoke(app, ["analyze", str(tmp_path)])
    assert result.exit_code == 0
    assert "No active coding flow found." in result.stdout


def test_analyze_flow_output_includes_core_sections(monkeypatch, tmp_path: Path) -> None:
    _patch_analyze_dependencies(monkeypatch)
    _seed_flow(tmp_path)
    result = runner.invoke(app, ["analyze", str(tmp_path), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    summary = payload["flow"]["summary"]
    assert payload["flow"]["active"] is True
    assert "Implement flow summary command and docs" in summary["objective"]
    assert isinstance(summary.get("checklist"), dict)
    assert isinstance(summary.get("open_tasks"), list)


def test_analyze_flow_includes_blocked_transition_reason(monkeypatch, tmp_path: Path) -> None:
    _patch_analyze_dependencies(monkeypatch)
    _seed_flow(tmp_path)
    result = runner.invoke(app, ["analyze", str(tmp_path), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["flow"]["summary"]["last_blocked_reason"] == "waiting for docs update"


def test_chat_startup_with_coding_memory_and_coding_agent_still_works(monkeypatch) -> None:
    monkeypatch.setattr("mana_analyzer.commands.cli.Settings", lambda: DummySettings())
    monkeypatch.setattr(
        "mana_analyzer.commands.cli.build_ask_service",
        lambda _s, model_override=None: _AskServiceWithAgent(),
    )
    result = runner.invoke(
        app,
        ["chat", "--coding-memory"],
        input="quit\n",
    )
    assert result.exit_code == 0
    assert "Goodbye!" in result.stdout
