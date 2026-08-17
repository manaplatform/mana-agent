from __future__ import annotations

import json
from pathlib import Path

import pytest

from mana_agent.commands import cli_internal
from mana_agent.services.project_analyze_service import ProjectAnalyzeOptions, ProjectAnalyzeService
from mana_agent.services.project_llm_analyze_service import (
    AnalyzeEvidence,
    LLMAnalyzeResult,
    ModelConfig,
    build_evidence,
    build_llm_analyzer,
    generate_llm_analysis,
)


def _sample_repo(root: Path) -> Path:
    (root / "src" / "demo").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "demo" / "__main__.py").write_text(
        "def run():\n    return 1\n", encoding="utf-8"
    )
    (root / "tests" / "test_demo.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\ndependencies = ['typer', 'openai']\n"
        "[project.scripts]\ndemo = 'demo.__main__:run'\n",
        encoding="utf-8",
    )
    (root / ".env").write_text("OPENAI_API_KEY=super-secret-value\n", encoding="utf-8")
    return root


def _fake_llm_result() -> LLMAnalyzeResult:
    return LLMAnalyzeResult(
        available=True,
        model="fake-model",
        project_summary="Demo is a tiny CLI.",
        detected_stack_explanation="Python + typer.",
        architecture_explanation="One CLI layer.",
        agent_workflow="user -> tool -> verify.",
        analyze_workflow="scan -> evidence -> llm -> artifacts.",
        onboarding_summary="Run demo.",
        important_files=[{"file": "src/demo/__main__.py", "why": "entrypoint", "evidence": "scripts"}],
        risk_analysis=[{"title": "No risks", "severity": "Low", "evidence": "n/a", "why_it_matters": "n/a", "recommended_fix": "n/a"}],
        recommendations=["Add more tests"],
        next_tasks=[{"title": "Add CLI test", "priority": "High", "files": ["tests/test_demo.py"], "acceptance_criteria": ["covers run()"], "verification_command": "pytest -q"}],
    )


def test_build_evidence_is_compact_and_secret_safe(tmp_path: Path) -> None:
    repo = _sample_repo(tmp_path)
    report = ProjectAnalyzeService().run(repo, repo / ".mana" / "analyze").report
    evidence = build_evidence(report, depth="quick")

    assert isinstance(evidence, AnalyzeEvidence)
    assert evidence.project_name == "demo"
    blob = json.dumps(evidence.to_dict())
    assert "super-secret-value" not in blob
    assert ".env" in evidence.secret_bearing_config


def test_analyze_uses_injected_llm_and_saves_output(tmp_path: Path) -> None:
    repo = _sample_repo(tmp_path)
    out_dir = repo / ".mana" / "analyze"
    seen: dict[str, object] = {}

    def fake_analyzer(evidence: AnalyzeEvidence, depth: str, root: Path) -> LLMAnalyzeResult:
        seen["evidence"] = evidence
        seen["depth"] = depth
        return _fake_llm_result()

    result = ProjectAnalyzeService().run(
        repo, out_dir, options=ProjectAnalyzeOptions(depth="normal"), llm_analyzer=fake_analyzer
    )

    # Called with compact evidence.
    assert isinstance(seen["evidence"], AnalyzeEvidence)
    assert seen["evidence"].language == "en"
    assert result.errors == []

    # LLM output saved into the artifacts.
    report_md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Demo is a tiny CLI." in report_md
    assert "## 1. Executive Summary" in report_md

    report_json = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report_json["llm_analysis"]["available"] is True
    assert report_json["llm_analysis"]["summary"] == "Demo is a tiny CLI."

    agent_context = json.loads((out_dir / "agent_context.json").read_text(encoding="utf-8"))
    assert agent_context["llm_available"] is True
    assert agent_context["project_summary"] == "Demo is a tiny CLI."
    assert agent_context["recommended_tasks"][0]["title"] == "Add CLI test"

    assert (out_dir / "evidence.json").exists()
    assert (out_dir / "llm_summary.md").exists()
    assert "fake-model" in (out_dir / "llm_summary.md").read_text(encoding="utf-8")


def test_analyze_passes_persian_language_to_llm_evidence(tmp_path: Path) -> None:
    repo = _sample_repo(tmp_path)
    seen: dict[str, object] = {}

    def fake_analyzer(evidence: AnalyzeEvidence, depth: str, root: Path) -> LLMAnalyzeResult:
        seen["language"] = evidence.language
        return _fake_llm_result()

    ProjectAnalyzeService().run(
        repo,
        repo / ".mana" / "analyze",
        options=ProjectAnalyzeOptions(language="fa"),
        llm_analyzer=fake_analyzer,
    )

    assert seen["language"] == "fa"


def test_analyze_llm_failure_falls_back_without_crashing(tmp_path: Path) -> None:
    repo = _sample_repo(tmp_path)
    out_dir = repo / ".mana" / "analyze"

    def boom(evidence: AnalyzeEvidence, depth: str, root: Path) -> LLMAnalyzeResult:
        raise RuntimeError("model exploded")

    result = ProjectAnalyzeService().run(repo, out_dir, llm_analyzer=boom)

    # Pipeline still produced valid artifacts.
    assert (out_dir / "report.md").exists()
    assert (out_dir / "evidence.json").exists()
    json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    report_md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "LLM analysis unavailable" in report_md
    # Surfaced as a warning (requested but failed).
    assert any("LLM analysis unavailable" in err for err in result.errors)


def test_generate_llm_analysis_without_key_returns_fallback(tmp_path: Path) -> None:
    repo = _sample_repo(tmp_path)
    report = ProjectAnalyzeService().run(repo, repo / ".mana" / "analyze").report
    evidence = build_evidence(report)

    result = generate_llm_analysis(evidence, "normal", repo, ModelConfig(api_key="", model="x"))
    assert result.available is False
    assert result.error


def test_build_llm_analyzer_returns_none_without_key() -> None:
    assert build_llm_analyzer(None) is None
    assert build_llm_analyzer(ModelConfig(api_key="  ", model="x")) is None
    assert build_llm_analyzer(ModelConfig(api_key="sk-test", model="x")) is not None


def test_cli_analyze_uses_user_config_instead_of_repository_dotenv(monkeypatch) -> None:
    captured: dict[str, ModelConfig] = {}

    monkeypatch.setattr(
        cli_internal,
        "load_effective_settings",
        lambda *, include_env: (
            {
            "OPENAI_API_KEY": "user-config-key",
            "OPENAI_CHAT_MODEL": "gpt-5.4-mini",
            "OPENAI_BASE_URL": "https://user-config.example/v1",
            }
            if include_env is False
            else pytest.fail("Analyze must not load environment or repository .env settings")
        ),
    )
    monkeypatch.setattr(
        cli_internal,
        "build_llm_analyzer",
        lambda config: captured.setdefault("config", config),
    )

    cli_internal._build_project_llm_analyzer()

    assert captured["config"] == ModelConfig(
        api_key="user-config-key",
        model="gpt-5.4-mini",
        base_url="https://user-config.example/v1",
    )


def test_architecture_is_project_derived_not_static_template(tmp_path: Path) -> None:
    # A project whose folders are nothing like the mana-agent layout.
    (tmp_path / "shop" / "api").mkdir(parents=True)
    (tmp_path / "shop" / "models").mkdir()
    (tmp_path / "shop" / "billing").mkdir()
    (tmp_path / "shop" / "__init__.py").write_text('"""Online shop backend."""\n', encoding="utf-8")
    (tmp_path / "shop" / "api" / "routes.py").write_text(
        "from shop.billing import charge\n\ndef checkout():\n    return charge()\n", encoding="utf-8"
    )
    (tmp_path / "shop" / "models" / "order.py").write_text("class Order:\n    total: int\n", encoding="utf-8")
    (tmp_path / "shop" / "billing" / "__init__.py").write_text(
        '"""Payment + billing logic."""\n\nfrom shop.models.order import Order\n\n\ndef charge():\n    return Order()\n',
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'shop'\n", encoding="utf-8")

    report = ProjectAnalyzeService().run(tmp_path, tmp_path / ".mana" / "analyze").report
    arch = report["architecture"]
    areas = {section["area"] for section in arch["sections"]}

    # Areas are this project's real directories — not a fixed mana-agent template.
    assert any(area.endswith("api") for area in areas)
    assert any(area.endswith("models") for area in areas)
    assert any(area.endswith("billing") for area in areas)
    assert not any("tools_manager" in area or "coding agent" in area.lower() for area in areas)

    # Responsibilities come from real package docstrings.
    by_area = {section["area"]: section for section in arch["sections"]}
    billing = next(section for area, section in by_area.items() if area.endswith("billing"))
    assert "billing" in billing["responsibility"].lower()

    # Cross-area dependencies are derived from real imports (api -> billing).
    api = next(section for area, section in by_area.items() if area.endswith("api"))
    assert any(dep.endswith("billing") for dep in api["dependencies_on_other_parts"])
