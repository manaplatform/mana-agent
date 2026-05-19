from __future__ import annotations

import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from mana_analyzer.analysis.models import CodeDescription, DescribeReport
from mana_analyzer.dependencies.dependency_service import DependencyService
from mana_analyzer.utils.io import language_for_path


class DescribeService:
    def __init__(
        self,
        dependency_service: DependencyService,
        summary_executor: Any | None = None,
        llm_chain: Any | None = None,
        include_tests: bool = False,
    ) -> None:
        self.dependency_service = dependency_service
        self.summary_executor = summary_executor
        self.llm_chain = llm_chain
        self.include_tests = include_tests

    def describe(
        self,
        root: str | Path,
        max_files: int = 12,
        include_functions: bool = False,
        use_llm: bool = True,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        modified_since: datetime | None = None,
        include_docstrings: bool = True,
        use_cache: bool = True,
    ) -> DescribeReport:
        _ = use_cache
        project_root = Path(root).resolve()
        if project_root.is_file():
            project_root = project_root.parent

        dependency_result = self.dependency_service.analyze(project_root)
        all_files = list(getattr(dependency_result, "files", []) or [])
        selected = self._select_files(
            project_root,
            all_files,
            max_files=max_files,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            modified_since=modified_since,
        )

        cache_path = project_root / ".mana_cache" / "describe_cache.json"
        cache = self._load_cache(cache_path) if use_cache else {}
        next_cache: dict[str, Any] = {}
        descriptions: list[CodeDescription] = []
        cache_hits = 0
        for path in selected:
            language = language_for_path(path)
            source = self._read_source(path)
            rel_path = str(path.relative_to(project_root))
            mtime_ns = path.stat().st_mtime_ns
            cached = cache.get(rel_path)
            if cached and cached.get("mtime_ns") == mtime_ns:
                cache_hits += 1
                descriptions.append(
                    CodeDescription(
                        file_path=rel_path,
                        language=str(cached.get("language") or language),
                        symbols=[str(item) for item in cached.get("symbols", [])],
                        summary=str(cached.get("summary") or ""),
                        entrypoint=bool(cached.get("entrypoint", False)),
                        symbol_docs=dict(cached.get("symbol_docs") or {}),
                    )
                )
                next_cache[rel_path] = cached
                continue

            symbols = self._extract_symbols(path, include_functions=include_functions)
            symbol_docs = self._extract_symbol_docs(path) if include_docstrings else {}
            summary = ""

            if use_llm and self.llm_chain is not None and source:
                summary, llm_symbols = self._summarize_with_llm(path, language, source)
                if llm_symbols:
                    symbols = sorted(set(symbols) | set(llm_symbols))

            if not summary:
                summary = self._local_summary(path, language, symbols)

            description = CodeDescription(
                file_path=rel_path,
                language=language,
                symbols=symbols,
                summary=summary,
                entrypoint=path.name in {"main.py", "cli.py", "app.py", "server.py"} or "commands" in path.parts,
                symbol_docs=symbol_docs,
            )
            descriptions.append(description)
            next_cache[rel_path] = {"mtime_ns": mtime_ns, **description.to_dict()}

        if use_cache:
            self._write_cache(cache_path, next_cache)

        dep_payload = dependency_result.to_dict() if hasattr(dependency_result, "to_dict") else {}
        architecture_summary, tech_summary = self._architecture_summary(dep_payload, descriptions, use_llm=use_llm)

        return DescribeReport(
            project_root=str(project_root),
            selected_files=[item.file_path for item in descriptions],
            descriptions=descriptions,
            architecture_summary=architecture_summary,
            tech_summary=tech_summary,
            chain_steps=[
                "dependency-analysis",
                "file-selection",
                "llm-file-summary" if use_llm and self.llm_chain is not None else "local-file-summary",
                "architecture-synthesis",
            ],
            architecture_mermaid=self._architecture_mermaid(dep_payload),
            architecture_data={
                "languages": dep_payload.get("languages", []),
                "frameworks": dep_payload.get("frameworks", []),
                "module_edges": len(dep_payload.get("module_edges", []) or []),
                "dependency_edges": len(dep_payload.get("dependency_edges", []) or []),
            },
            metrics={
                "all_files": len(all_files),
                "selected_files": len(selected),
                "cache_hits": cache_hits,
            },
        )

    def synthesize_deep_flow_analysis(self, *args: Any, **kwargs: Any) -> Any:
        if self.llm_chain is None or not hasattr(self.llm_chain, "synthesize_deep_flow_analysis"):
            raise RuntimeError("LLM describe chain does not provide synthesize_deep_flow_analysis.")
        return self.llm_chain.synthesize_deep_flow_analysis(*args, **kwargs)

    def render_markdown(self, report: DescribeReport | dict[str, Any]) -> str:
        payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        lines = ["# Repository Description", "", "## Architecture", ""]
        lines.append(str(payload.get("architecture_summary") or "Architecture summary unavailable."))
        lines.extend(["", "## Technology", ""])
        lines.append(str(payload.get("tech_summary") or "Technology summary unavailable."))
        lines.extend(["", "## File Summaries", ""])
        for item in payload.get("descriptions", []) or payload.get("files", []) or []:
            lines.append(
                f"- `{item.get('file_path') or item.get('path')}` ({item.get('language', 'text')}) - {item.get('summary', '')}"
            )
        if len(lines) == 7:
            lines.append("- none")
        return "\n".join(lines) + "\n"

    def merge_llm_framework_hints(self, report: Any, frameworks: list[str]) -> Any:
        if hasattr(report, "frameworks"):
            report.frameworks = sorted(set(getattr(report, "frameworks", []) or []) | set(frameworks))
        if hasattr(report, "technologies"):
            report.technologies = sorted(set(getattr(report, "technologies", []) or []) | set(frameworks))
        return report

    @staticmethod
    def _read_source(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""

    @staticmethod
    def _load_cache(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _write_cache(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _select_files(
        self,
        root: Path,
        files: list[Path],
        *,
        max_files: int,
        include_patterns: list[str] | None,
        exclude_patterns: list[str] | None,
        modified_since: datetime | None,
    ) -> list[Path]:
        selected = []
        for path in sorted(files, key=lambda item: str(item.relative_to(root))):
            rel = str(path.relative_to(root))
            if not self.include_tests and ("/tests/" in f"/{rel}" or rel.startswith("tests/")):
                continue
            if include_patterns and not any(path.match(pattern) or Path(rel).match(pattern) for pattern in include_patterns):
                continue
            if exclude_patterns and any(path.match(pattern) or Path(rel).match(pattern) for pattern in exclude_patterns):
                continue
            if modified_since is not None and datetime.fromtimestamp(path.stat().st_mtime) < modified_since:
                continue
            selected.append(path)
            if len(selected) >= max(1, max_files):
                break
        return selected

    @staticmethod
    def _extract_symbols(path: Path, *, include_functions: bool) -> list[str]:
        if path.suffix != ".py":
            return []
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
        except SyntaxError:
            return []
        symbols: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                symbols.append(node.name)
            elif include_functions and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(node.name)
        return sorted(set(symbols))

    @staticmethod
    def _extract_symbol_docs(path: Path) -> dict[str, str]:
        if path.suffix != ".py":
            return {}
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
        except SyntaxError:
            return {}
        docs: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if doc:
                    docs[node.name] = doc.splitlines()[0][:240]
        return docs

    def _summarize_with_llm(self, path: Path, language: str, source: str) -> tuple[str, list[str]]:
        if hasattr(self.llm_chain, "summarize_file"):
            summary, symbols = self.llm_chain.summarize_file(path, language, source[:12000])
            return str(summary).strip(), [str(item) for item in symbols if str(item).strip()]
        if hasattr(self.llm_chain, "summarize_files_batch"):
            result = self.llm_chain.summarize_files_batch(
                [{"file_path": str(path), "language": language, "source": source[:12000]}]
            )
            if result:
                summary, raw_symbols = result[0]
                return str(summary).strip(), self._normalize_llm_symbols(raw_symbols)
        return "", []

    @staticmethod
    def _normalize_llm_symbols(raw_symbols: Any) -> list[str]:
        if isinstance(raw_symbols, dict):
            values: list[str] = []
            for items in raw_symbols.values():
                if isinstance(items, list):
                    values.extend(str(item) for item in items)
            return sorted({item for item in values if item.strip()})
        if isinstance(raw_symbols, list):
            return sorted({str(item) for item in raw_symbols if str(item).strip()})
        return []

    @staticmethod
    def _local_summary(path: Path, language: str, symbols: list[str]) -> str:
        if symbols:
            return f"{path.name} is a {language} file defining {', '.join(symbols[:8])}."
        return f"{path.name} is a {language} source/configuration file."

    def _architecture_summary(
        self,
        dependency_report: dict[str, Any],
        descriptions: list[CodeDescription],
        *,
        use_llm: bool,
    ) -> tuple[str, str]:
        file_summaries = [item.to_dict() for item in descriptions]
        if use_llm and self.llm_chain is not None and hasattr(self.llm_chain, "synthesize_architecture"):
            try:
                return self.llm_chain.synthesize_architecture(dependency_report, file_summaries)
            except Exception:
                pass

        languages = ", ".join(dependency_report.get("languages", []) or []) or "unknown"
        frameworks = ", ".join(dependency_report.get("frameworks", []) or []) or "none detected"
        architecture = (
            f"Repository contains {len(file_summaries)} summarized files. "
            f"Detected languages: {languages}. Internal and external dependency edges are captured in the graph section."
        )
        tech = f"Technologies: {frameworks}."
        return architecture, tech

    @staticmethod
    def _architecture_mermaid(dependency_report: dict[str, Any]) -> str:
        edges = dependency_report.get("module_edges", []) or []
        lines = ["flowchart LR"]
        for edge in edges[:20]:
            lines.append(f'  "{edge.get("source", "source")}" --> "{edge.get("target", "target")}"')
        if len(lines) == 1:
            lines.append('  "Project" --> "Dependencies"')
        return "\n".join(lines)
