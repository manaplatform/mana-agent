"""Layer 2 of ``/analyze``: turn deterministic evidence into an LLM-written report.

The deterministic collector (:mod:`mana_agent.services.project_analyze_service`)
scans the repository and produces structured evidence. This module compacts that
evidence into :class:`AnalyzeEvidence`, sends it to the model, and parses the
model's JSON answer into :class:`LLMAnalyzeResult`.

This is the unified analyze-LLM layer. It supersedes the per-file LLM analyzer
(``mana_agent.multi_agent.runtime.analyze_chain.AnalyzeChain`` + ``LlmAnalyzeService``): instead
of prompting the model file-by-file, it sends one compact, whole-project evidence
payload. The deterministic core of the old engine (``PythonStaticAnalyzer``) is
merged into :mod:`mana_agent.services.project_analyze_service` and its findings
arrive here as part of the evidence (``risks`` + ``static_analysis_summary``), so
the old engine's signal informs the new project-level analysis.

Design contract:
- The LLM never scans files directly; it only analyzes the compact evidence here.
- Evidence is fully project-derived (real folders, imports, symbols, findings);
  nothing is hardcoded to a particular repository.
- ``generate_llm_analysis`` never raises. On any failure (no model configured,
  network error, bad JSON) it returns a result with ``available=False`` and an
  ``error`` message so the analyze pipeline can fall back deterministically.
- Secrets are never sent: evidence is built from the deterministic report, which
  already redacts values and references secret-bearing files by name only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Callable

logger = logging.getLogger(__name__)

__all__ = [
    "ModelConfig",
    "AnalyzeEvidence",
    "LLMAnalyzeResult",
    "build_evidence",
    "generate_llm_analysis",
    "build_llm_analyzer",
    "LLMAnalyzer",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ModelConfig:
    """Connection details for the analyzer model."""

    api_key: str
    model: str
    base_url: str | None = None
    max_retries: int = 3
    request_timeout: float | None = None


@dataclass(slots=True)
class AnalyzeEvidence:
    """Compact, bounded evidence sent to the LLM as analysis input."""

    project_name: str
    root_path: str
    depth: str
    languages: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    package_managers: list[str] = field(default_factory=list)
    file_counts: dict[str, int] = field(default_factory=dict)
    source_folders: list[str] = field(default_factory=list)
    test_folders: list[str] = field(default_factory=list)
    docs_folders: list[str] = field(default_factory=list)
    script_folders: list[str] = field(default_factory=list)
    important_config_files: list[str] = field(default_factory=list)
    entrypoints: list[dict[str, Any]] = field(default_factory=list)
    dependencies: dict[str, Any] = field(default_factory=dict)
    important_symbols: list[dict[str, Any]] = field(default_factory=list)
    architecture_areas: list[dict[str, Any]] = field(default_factory=list)
    agent_workflow: dict[str, str] = field(default_factory=dict)
    risks: list[dict[str, Any]] = field(default_factory=list)
    static_analysis: dict[str, int] = field(default_factory=dict)
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)
    secret_bearing_config: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "root_path": self.root_path,
            "depth": self.depth,
            "languages": self.languages,
            "frameworks": self.frameworks,
            "package_managers": self.package_managers,
            "file_counts": self.file_counts,
            "source_folders": self.source_folders,
            "test_folders": self.test_folders,
            "docs_folders": self.docs_folders,
            "script_folders": self.script_folders,
            "important_config_files": self.important_config_files,
            "entrypoints": self.entrypoints,
            "dependencies": self.dependencies,
            "important_symbols": self.important_symbols,
            "architecture_areas": self.architecture_areas,
            "agent_workflow": self.agent_workflow,
            "risks": self.risks,
            "static_analysis_summary": self.static_analysis,
            "recommendations": self.recommendations,
            "verification_commands": self.verification_commands,
            # By name only — never values.
            "secret_bearing_config": self.secret_bearing_config,
        }


@dataclass(slots=True)
class LLMAnalyzeResult:
    """Parsed LLM analysis. ``available=False`` means a fallback was used."""

    available: bool = False
    error: str | None = None
    model: str = ""
    project_summary: str = ""
    detected_stack_explanation: str = ""
    repository_overview: str = ""
    architecture_explanation: str = ""
    important_files: list[dict[str, Any]] = field(default_factory=list)
    cli_commands_explanation: str = ""
    agent_workflow: str = ""
    analyze_workflow: str = ""
    important_symbols_overview: str = ""
    risk_analysis: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    next_tasks: list[dict[str, Any]] = field(default_factory=list)
    onboarding_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "error": self.error,
            "model": self.model,
            "summary": self.project_summary,
            "detected_stack": self.detected_stack_explanation,
            "repository_overview": self.repository_overview,
            "architecture": self.architecture_explanation,
            "important_files": self.important_files,
            "cli_commands": self.cli_commands_explanation,
            "agent_workflow": self.agent_workflow,
            "analyze_workflow": self.analyze_workflow,
            "important_symbols": self.important_symbols_overview,
            "risk_analysis": self.risk_analysis,
            "recommendations": self.recommendations,
            "next_tasks": self.next_tasks,
            "onboarding_summary": self.onboarding_summary,
        }


# ---------------------------------------------------------------------------
# Evidence builder (deterministic report -> compact evidence)
# ---------------------------------------------------------------------------


def _trim(text: Any, limit: int = 280) -> str:
    value = str(text or "").strip().replace("\n", " ")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def build_evidence(report: dict[str, Any], *, depth: str = "normal") -> AnalyzeEvidence:
    """Compact the deterministic analyze report into bounded LLM evidence."""
    inventory = report.get("inventory", {}) or {}
    dependencies = report.get("dependencies", {}) or {}
    symbols = report.get("symbols", {}) or {}
    architecture = report.get("architecture", {}) or {}
    risks = report.get("risks", {}) or {}
    recommendations = report.get("recommendations", {}) or {}

    entrypoints = [
        {
            "name": item.get("name"),
            "type": item.get("type"),
            "file": item.get("file"),
            "line": item.get("line"),
            "command": _trim(item.get("command"), 120),
            "description": _trim(item.get("description"), 120),
        }
        for item in (report.get("entrypoints", []) or [])[:25]
    ]

    important_symbols = [
        {
            "name": item.get("name"),
            "kind": item.get("kind"),
            "file": item.get("file"),
            "line": item.get("line"),
            "signature": _trim(item.get("signature"), 160),
            "docstring": _trim(item.get("docstring"), 160),
        }
        for item in (symbols.get("important_symbols", []) or [])[:40]
    ]

    architecture_areas = [
        {
            "area": item.get("area"),
            "responsibility": _trim(item.get("responsibility"), 200),
            "related_files": (item.get("related_files", []) or [])[:8],
            "risk_notes": item.get("risk_notes", []) or [],
        }
        for item in (architecture.get("sections", []) or [])
    ]

    # Prioritize curated, higher-severity risks so the bounded sample the LLM
    # sees is representative rather than dominated by high-volume static findings
    # (e.g. thousands of missing-docstring hits merged from the static engine).
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    ranked_risks = sorted(
        (risks.get("items", []) or []),
        key=lambda item: severity_rank.get(str(item.get("severity", "low")).lower(), 3),
    )
    risk_items = [
        {
            "title": item.get("title"),
            "severity": item.get("severity"),
            "file": item.get("file"),
            "line": item.get("line"),
            "evidence": _trim(item.get("evidence"), 200),
            "why_it_matters": _trim(item.get("why_it_matters"), 200),
            "recommended_fix": _trim(item.get("recommended_fix"), 200),
        }
        for item in ranked_risks[:40]
    ]

    rec_items = [
        {
            "title": item.get("title"),
            "priority": item.get("priority"),
            "files": (item.get("files", []) or [])[:6],
            "reason": _trim(item.get("reason"), 200),
            "acceptance_criteria": item.get("acceptance_criteria", []) or [],
            "verification": item.get("verification"),
        }
        for item in (recommendations.get("items", []) or [])[:20]
    ]

    compact_dependencies = {
        "runtime": (dependencies.get("runtime_dependencies", []) or [])[:60],
        "dev": (dependencies.get("dev_dependencies", []) or [])[:60],
        "lock_files": dependencies.get("lock_files", []) or [],
        "framework_packages": dependencies.get("framework_packages", []) or [],
        "testing_packages": dependencies.get("testing_packages", []) or [],
        "llm_agent_tooling_packages": dependencies.get("llm_agent_tooling_packages", []) or [],
        "warnings": dependencies.get("warnings", []) or [],
    }

    return AnalyzeEvidence(
        project_name=str(inventory.get("project_name") or Path(report.get("root_path", ".")).name),
        root_path=str(report.get("root_path", "")),
        depth=depth,
        languages=inventory.get("detected_languages", []) or [],
        frameworks=inventory.get("detected_frameworks", []) or [],
        package_managers=inventory.get("package_managers", []) or [],
        file_counts={
            "total": inventory.get("total_files", 0),
            "source": inventory.get("source_files_count", 0),
            "test": inventory.get("test_files_count", 0),
            "config": inventory.get("config_files_count", 0),
            "documentation": inventory.get("documentation_files_count", 0),
        },
        source_folders=inventory.get("source_folders", []) or [],
        test_folders=inventory.get("test_folders", []) or [],
        docs_folders=inventory.get("docs_folders", []) or [],
        script_folders=inventory.get("script_folders", []) or [],
        important_config_files=(inventory.get("important_config_files", []) or [])[:40],
        entrypoints=entrypoints,
        dependencies=compact_dependencies,
        important_symbols=important_symbols,
        architecture_areas=architecture_areas,
        agent_workflow=architecture.get("agent_workflow", {}) or {},
        risks=risk_items,
        static_analysis=risks.get("static_analysis", {}) or {},
        recommendations=rec_items,
        verification_commands=report.get("verification_commands", []) or [],
        secret_bearing_config=inventory.get("secret_bearing_config", []) or [],
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


_RESULT_FIELDS: dict[str, str] = {
    "project_summary": "project_summary",
    "detected_stack_explanation": "detected_stack_explanation",
    "repository_overview": "repository_overview",
    "architecture_explanation": "architecture_explanation",
    "cli_commands_explanation": "cli_commands_explanation",
    "agent_workflow": "agent_workflow",
    "analyze_workflow": "analyze_workflow",
    "important_symbols_overview": "important_symbols_overview",
    "onboarding_summary": "onboarding_summary",
}


def _strip_json_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence (``` or ```json) and the trailing fence.
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else ""
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[: -3]
    return stripped.strip()


def _parse_result(text: str, model: str) -> LLMAnalyzeResult:
    payload = json.loads(_strip_json_fences(text))
    if not isinstance(payload, dict):
        raise ValueError("LLM analysis output is not a JSON object")

    result = LLMAnalyzeResult(available=True, model=model)
    for field_name, key in _RESULT_FIELDS.items():
        setattr(result, field_name, str(payload.get(key, "") or "").strip())

    result.important_files = [item for item in (payload.get("important_files") or []) if isinstance(item, dict)][:30]
    result.risk_analysis = [item for item in (payload.get("risk_analysis") or []) if isinstance(item, dict)][:40]
    result.next_tasks = [item for item in (payload.get("next_tasks") or []) if isinstance(item, dict)][:15]
    result.recommendations = [str(item).strip() for item in (payload.get("recommendations") or []) if str(item).strip()][:25]
    return result


class LLMAnalyzer:
    """Thin wrapper around a chat model that turns evidence into analysis JSON."""

    def __init__(self, model_config: ModelConfig) -> None:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        from mana_agent.multi_agent.runtime.prompts import (
            PROJECT_ANALYZE_HUMAN_TEMPLATE,
            PROJECT_ANALYZE_SYSTEM_PROMPT,
        )

        self.config = model_config
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PROJECT_ANALYZE_SYSTEM_PROMPT),
                ("human", PROJECT_ANALYZE_HUMAN_TEMPLATE),
            ]
        )
        kwargs: dict[str, Any] = {"api_key": model_config.api_key, "model": model_config.model}
        if model_config.base_url:
            kwargs["base_url"] = model_config.base_url
        if model_config.request_timeout:
            kwargs["timeout"] = model_config.request_timeout
        self.llm = ChatOpenAI(**kwargs)

    def run(self, evidence: AnalyzeEvidence) -> LLMAnalyzeResult:
        chain = self.prompt | self.llm
        evidence_json = json.dumps(evidence.to_dict(), ensure_ascii=False, indent=2)
        inputs = {
            "project_name": evidence.project_name,
            "depth": evidence.depth,
            "evidence_json": evidence_json,
        }
        attempts = max(1, int(self.config.max_retries))
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = chain.invoke(inputs)
                return _parse_result(str(response.content), self.config.model)
            except Exception as exc:  # noqa: BLE001 - bounded retry, never crash analyze
                last_error = exc
                logger.warning(
                    "LLM project analysis attempt %d/%d failed: %s", attempt + 1, attempts, exc
                )
                if attempt < attempts - 1:
                    sleep(min(2.0 * (attempt + 1), 8.0))
        raise last_error if last_error else RuntimeError("LLM project analysis failed")


def generate_llm_analysis(
    evidence: AnalyzeEvidence,
    depth: str,
    repo_path: Path,
    model_config: ModelConfig,
) -> LLMAnalyzeResult:
    """Run the LLM analyzer over compact evidence. Never raises.

    Returns a result with ``available=False`` and a populated ``error`` when no
    model is configured or the model call/parse fails.
    """
    evidence.depth = depth or evidence.depth
    if not model_config or not str(model_config.api_key or "").strip():
        return LLMAnalyzeResult(available=False, error="No LLM model configured (missing API key).")
    started = perf_counter()
    try:
        analyzer = LLMAnalyzer(model_config)
        result = analyzer.run(evidence)
    except Exception as exc:  # noqa: BLE001 - surfaced as fallback, not a crash
        logger.warning("LLM project analysis unavailable for %s: %s", repo_path, exc)
        return LLMAnalyzeResult(available=False, error=f"LLM analysis failed: {exc}")
    logger.info(
        "LLM project analysis completed in %.0fms (model=%s)",
        (perf_counter() - started) * 1000,
        model_config.model,
    )
    return result


def build_llm_analyzer(
    model_config: ModelConfig | None,
) -> Callable[[AnalyzeEvidence, str, Path], LLMAnalyzeResult] | None:
    """Return an analyzer callable bound to ``model_config`` (or ``None``)."""
    if model_config is None or not str(model_config.api_key or "").strip():
        return None

    def _analyzer(evidence: AnalyzeEvidence, depth: str, repo_path: Path) -> LLMAnalyzeResult:
        return generate_llm_analysis(evidence, depth, repo_path, model_config)

    return _analyzer
