from __future__ import annotations

import json
from pathlib import Path

from mana_agent.services.project_analyze_service import ProjectAnalyzeOptions, ProjectAnalyzeService


def _sample_repo(root: Path) -> Path:
    (root / "src" / "demo").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "docs").mkdir()
    (root / "node_modules").mkdir()
    (root / ".mana" / "analyze").mkdir(parents=True)
    (root / "src" / "demo" / "__main__.py").write_text(
        "from dataclasses import dataclass\n\n"
        "IMPORTANT = 1\n\n"
        "@dataclass\n"
        "class User:\n"
        "    name: str\n\n"
        "def run(value: int) -> int:\n"
        "    return value + IMPORTANT\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_demo.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (root / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\n"
        "name = 'demo'\n"
        "dependencies = ['typer>=0.12', 'openai>=1']\n"
        "[project.optional-dependencies]\n"
        "dev = ['pytest>=8']\n"
        "[project.scripts]\n"
        "demo = 'demo.__main__:run'\n",
        encoding="utf-8",
    )
    (root / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest"}, "dependencies": {"react": "^18"}, "devDependencies": {"vitest": "^1"}}),
        encoding="utf-8",
    )
    (root / ".env").write_text("OPENAI_API_KEY=secret-value\n", encoding="utf-8")
    (root / "node_modules" / "ignored.py").write_text("def ignored(): pass\n", encoding="utf-8")
    (root / ".mana" / "analyze" / "old.json").write_text("{}", encoding="utf-8")
    (root / "large.py").write_text("x = 1\n" * 1000, encoding="utf-8")
    (root / "image.bin").write_bytes(b"\x00\x01binary")
    return root


def test_project_analyze_writes_required_artifacts_and_valid_json(tmp_path: Path) -> None:
    repo = _sample_repo(tmp_path)
    out_dir = repo / ".mana" / "analyze"

    result = ProjectAnalyzeService().run(
        repo,
        out_dir,
        options=ProjectAnalyzeOptions(max_file_size_kb=1),
    )

    required = {
        "report.md",
        "report.json",
        "agent_context.json",
        "inventory.json",
        "symbols.json",
        "dependencies.json",
        "architecture.md",
        "risks.json",
        "recommendations.md",
    }
    assert required <= set(result.artifacts)
    assert result.errors == []
    for name in ("report.json", "agent_context.json", "inventory.json", "symbols.json", "dependencies.json", "risks.json"):
        json.loads((out_dir / name).read_text(encoding="utf-8"))


def test_project_analyze_records_selected_language_in_report_and_evidence(tmp_path: Path) -> None:
    repo = _sample_repo(tmp_path)
    result = ProjectAnalyzeService().run(
        repo,
        repo / ".mana" / "analyze",
        options=ProjectAnalyzeOptions(language="fa"),
    )

    assert result.report["analysis_language"] == "fa"
    assert result.report["llm_analysis"]["available"] is False
    evidence = json.loads((repo / ".mana" / "analyze" / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["language"] == "fa"


def test_project_analyze_inventory_ignores_noise_and_classifies_files(tmp_path: Path) -> None:
    repo = _sample_repo(tmp_path)
    result = ProjectAnalyzeService().run(repo, repo / ".mana" / "analyze", options=ProjectAnalyzeOptions(max_file_size_kb=1))
    inventory = result.report["inventory"]
    paths = {item["path"]: item for item in inventory["files"]}

    assert "node_modules/ignored.py" not in paths
    assert ".mana/analyze/old.json" not in paths
    assert paths["src/demo/__main__.py"]["category"] == "source_code"
    assert paths["tests/test_demo.py"]["category"] == "test"
    assert paths["docs/guide.md"]["category"] == "documentation"
    assert paths["pyproject.toml"]["category"] == "config"
    assert any(item["path"] == "large.py" for item in inventory["large_skipped_files"])
    assert any(item["path"] == "image.bin" for item in inventory["binary_skipped_files"])


def test_project_analyze_dependencies_entrypoints_symbols_and_secret_safety(tmp_path: Path) -> None:
    repo = _sample_repo(tmp_path)
    result = ProjectAnalyzeService().run(repo, repo / ".mana" / "analyze")

    dependencies = result.report["dependencies"]
    assert "typer" in dependencies["runtime_dependencies"]
    assert "pytest" in dependencies["dev_dependencies"]
    assert "openai" in dependencies["llm_agent_tooling_packages"]
    assert "React" in dependencies["framework_packages"]

    entrypoints = result.report["entrypoints"]
    assert any(item["name"] == "demo" and item["type"] == "cli" for item in entrypoints)
    assert any(item["name"] == "__main__" or item["file"] == "src/demo/__main__.py" for item in entrypoints)

    symbols = result.report["symbols"]["symbols"]
    assert any(item["name"] == "User" and item["kind"] == "model" for item in symbols)
    assert any(item["name"] == "run" and item["kind"] == "function" for item in symbols)

    report_text = (repo / ".mana" / "analyze" / "report.md").read_text(encoding="utf-8")
    assert "secret-value" not in report_text
    assert ".env" in result.report["inventory"]["secret_bearing_config"]
