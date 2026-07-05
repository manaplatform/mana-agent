from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)

from mana_agent.multi_agent.runtime.run_logger import LlmRunLogger
from mana_agent.multi_agent.runtime.prompts import (
    DEEP_FLOW_SYSTEM_PROMPT,
    DEEP_FLOW_HUMAN_TEMPLATE,
)

logger = logging.getLogger(__name__)

FILE_SUMMARY_SYSTEM = """
You are a code summarizer.
Return strict JSON with keys: summary (string), symbols (array of strings).
Keep summary concise and factual.
""".strip()

FILE_SUMMARY_HUMAN = """
File path: {file_path}
Language: {language}
Source:
{source}
""".strip()

ARCH_SYSTEM = """
You are a software architecture analyst.
Return strict JSON with keys: architecture_summary (string), tech_summary (string).
Use only the provided dependency and file-summary data.
""".strip()

ARCH_HUMAN = """
Dependency report JSON:
{dependency_report}

File summaries JSON:
{file_summaries}
""".strip()

TECH_SYSTEM = """
You are a repository technology detector.
Return strict JSON with key frameworks as an array of short framework names.
""".strip()

TECH_HUMAN = """
Project sample files:
{samples}
""".strip()


class RepositoryMultiChain:
    _MAX_FRAMEWORKS = 24
    _MAX_DEPENDENCIES = 200
    _MAX_EDGES = 400
    _MAX_FILE_SUMMARIES = 20
    _MAX_SYMBOLS_PER_FILE = 24
    _MAX_SUMMARY_CHARS = 700

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.llm = self._create_llm(model)
        self.run_logger = LlmRunLogger()

        self.file_summary_prompt = ChatPromptTemplate.from_messages(
            [("system", FILE_SUMMARY_SYSTEM), ("human", FILE_SUMMARY_HUMAN)]
        )
        self.arch_prompt = ChatPromptTemplate.from_messages(
            [("system", ARCH_SYSTEM), ("human", ARCH_HUMAN)]
        )
        self.tech_prompt = ChatPromptTemplate.from_messages(
            [("system", TECH_SYSTEM), ("human", TECH_HUMAN)]
        )

    def _create_llm(self, model_name: str) -> ChatOpenAI:
        kwargs: dict[str, Any] = {"api_key": self.api_key, "model": model_name}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return ChatOpenAI(**kwargs)

    def update_model(self, new_model: str):
        if self.model != new_model:
            logger.info(f"Updating RepositoryMultiChain model from {self.model} to {new_model}")
            self.model = new_model
            self.llm = self._create_llm(new_model)

    @staticmethod
    def _safe_json(text: str) -> dict[str, Any]:
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        return {}

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        if max_chars < 4:
            return text[:max_chars]
        return text[: max_chars - 3].rstrip() + "..."

    @classmethod
    def _compact_dependency_report(cls, dependency_report: dict[str, Any]) -> dict[str, Any]:
        compact = {
            "project_root": dependency_report.get("project_root"),
            "package_managers": list(dependency_report.get("package_managers", []))[:cls._MAX_FRAMEWORKS],
            "frameworks": list(dependency_report.get("frameworks", []))[:cls._MAX_FRAMEWORKS],
            "technologies": list(dependency_report.get("technologies", []))[:cls._MAX_FRAMEWORKS],
            "runtime_dependencies": list(dependency_report.get("runtime_dependencies", []))[:cls._MAX_DEPENDENCIES],
            "dev_dependencies": list(dependency_report.get("dev_dependencies", []))[:cls._MAX_DEPENDENCIES],
            "manifests": list(dependency_report.get("manifests", []))[:cls._MAX_DEPENDENCIES],
            "languages": list(dependency_report.get("languages", []))[:cls._MAX_FRAMEWORKS],
        }
        module_edges = list(dependency_report.get("module_edges", []))[:cls._MAX_EDGES]
        dependency_edges = list(dependency_report.get("dependency_edges", []))[:cls._MAX_EDGES]
        compact["module_edges"] = [
            {"source": e.get("source"), "target": e.get("target"), "kind": e.get("kind")}
            for e in module_edges
            if isinstance(e, dict)
        ]
        compact["dependency_edges"] = [
            {"source": e.get("source"), "target": e.get("target"), "kind": e.get("kind")}
            for e in dependency_edges
            if isinstance(e, dict)
        ]
        return compact

    @classmethod
    def _compact_file_summaries(cls, file_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for item in file_summaries[:cls._MAX_FILE_SUMMARIES]:
            if not isinstance(item, dict):
                continue
            symbols = [str(v) for v in item.get("symbols", []) if str(v).strip()]
            compact.append({
                "file_path": str(item.get("file_path", "")),
                "language": str(item.get("language", "")),
                "symbols": symbols[:cls._MAX_SYMBOLS_PER_FILE],
                "summary": cls._truncate(str(item.get("summary", "")), cls._MAX_SUMMARY_CHARS),
            })
        return compact

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        # only retry on exceptions that are NOT IsADirectoryError
        retry=retry_if_exception(lambda exc: not isinstance(exc, IsADirectoryError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def summarize_file(self, file_path: Path, language: str, source: str) -> tuple[str, list[str]]:
        # SKIP directories immediately
        if file_path.is_dir():
            logger.warning(f"Skipping directory in summarize_file: {file_path}")
            return "No summary generated.", []

        chain = self.file_summary_prompt | self.llm
        started = perf_counter()
        response = chain.invoke({
            "file_path": str(file_path),
            "language": language,
            "source": source,
        })
        elapsed_ms = (perf_counter() - started) * 1000

        payload = self._safe_json(str(response.content))
        summary = str(payload.get("summary", "")).strip() or "No summary generated."
        symbols = [str(item) for item in payload.get("symbols", []) if str(item).strip()]

        self.run_logger.log({
            "flow": "repo-file-summary",
            "model": self.model,
            "file_path": str(file_path),
            "source_chars": len(source),
            "duration_ms": round(elapsed_ms, 3),
            "response": str(response.content),
        })
        return summary, symbols

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception(lambda exc: not isinstance(exc, IsADirectoryError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def synthesize_deep_flow_analysis(
        self,
        *,
        dependency_report,
        structure_summary: dict,
        findings_summary: dict,
        security_summary: dict,
        sampled_file_summaries: list[dict[str, Any]],
        line_target: int = 350,
        security_lens: str = "defensive-red-team",
    ) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", DEEP_FLOW_SYSTEM_PROMPT),
            ("human", DEEP_FLOW_HUMAN_TEMPLATE),
        ])

        # compact everything
        dep_dict = dependency_report.to_dict() if hasattr(dependency_report, "to_dict") else dependency_report
        dep = self._compact_dependency_report(dep_dict)
        compact_structure = {
            "total_files": structure_summary.get("total_files"),
            "language_counts": structure_summary.get("language_counts"),
            "hotspots": structure_summary.get("hotspots", [])[:15],
            "tree_markdown": self._truncate(structure_summary.get("tree_markdown", ""), 6000),
        }
        compact_findings = {
            "counts": findings_summary.get("counts"),
            "top_rules": findings_summary.get("top_rules", [])[:20],
            "by_severity": findings_summary.get("by_severity"),
        }
        compact_security = {}
        if isinstance(security_summary, dict):
            compact_security = {
                k: (v[:50] if isinstance(v, list) else v)
                for k, v in security_summary.items()
            }
        compact_files = self._compact_file_summaries(sampled_file_summaries)

        chain = prompt | self.llm
        response = chain.invoke({
            "security_lens": security_lens,
            "line_target": min(max(line_target, 200), 400),
            "dependency_report_json": json.dumps(dep, ensure_ascii=False),
            "structure_summary_json": json.dumps(compact_structure, ensure_ascii=False),
            "findings_summary_json": json.dumps(compact_findings, ensure_ascii=False),
            "security_summary_json": json.dumps(compact_security, ensure_ascii=False),
            "sampled_file_summaries_json": json.dumps(compact_files, ensure_ascii=False),
        })
        return str(response.content).strip()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception(lambda exc: not isinstance(exc, IsADirectoryError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def synthesize_architecture(
        self,
        dependency_report: dict[str, Any],
        file_summaries: list[dict[str, Any]],
    ) -> tuple[str, str]:
        dep = self._compact_dependency_report(dependency_report)
        files = self._compact_file_summaries(file_summaries)
        chain = self.arch_prompt | self.llm
        started = perf_counter()
        response = chain.invoke({
            "dependency_report": json.dumps(dep, ensure_ascii=False),
            "file_summaries": json.dumps(files, ensure_ascii=False),
        })
        elapsed_ms = (perf_counter() - started) * 1000

        payload = self._safe_json(str(response.content))
        architecture = str(payload.get("architecture_summary", "")).strip() or "Architecture summary unavailable."
        tech = str(payload.get("tech_summary", "")).strip() or "Technology summary unavailable."

        self.run_logger.log({
            "flow": "repo-architecture-summary",
            "model": self.model,
            "summary_count": len(files),
            "duration_ms": round(elapsed_ms, 3),
            "response": str(response.content),
        })
        return architecture, tech

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception(lambda exc: not isinstance(exc, IsADirectoryError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def detect_frameworks_from_samples(self, samples: list[dict[str, str]]) -> list[str]:
        if not samples:
            return []
        chain = self.tech_prompt | self.llm
        started = perf_counter()
        response = chain.invoke({"samples": json.dumps(samples, ensure_ascii=False)})
        elapsed_ms = (perf_counter() - started) * 1000

        payload = self._safe_json(str(response.content))
        frameworks = sorted({
            str(item).strip() for item in payload.get("frameworks", [])
            if str(item).strip()
        })

        self.run_logger.log({
            "flow": "repo-tech-detection",
            "model": self.model,
            "sample_count": len(samples),
            "duration_ms": round(elapsed_ms, 3),
            "response": str(response.content),
        })
        return frameworks
