from __future__ import annotations

import ast
import configparser
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from mana_agent.compat import tomllib
from mana_agent.dependencies.dependency_service import FRAMEWORK_SIGNALS, _normalize_dep_name
from mana_agent.services.project_llm_analyze_service import (
    AnalyzeEvidence,
    LLMAnalyzeResult,
    build_evidence,
)
from mana_agent.utils.io import language_for_path, load_ignore_patterns

LLMAnalyzerFn = Callable[[AnalyzeEvidence, str, Path], LLMAnalyzeResult]


DEFAULT_IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "coverage",
    ".mana",
}

NOISY_FILES = {".DS_Store"}
CONFIG_NAMES = {
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "poetry.lock",
    "package.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "package-lock.json",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".env",
}
LOCK_FILES = {"poetry.lock", "pnpm-lock.yaml", "yarn.lock", "package-lock.json", "Pipfile.lock"}
DEPENDENCY_MANIFESTS = {
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "poetry.lock",
    "package.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "package-lock.json",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
}
SOURCE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".swift",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".php",
    ".rb",
    ".sh",
}
DOC_SUFFIXES = {".md", ".rst", ".txt", ".adoc"}
STATIC_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".webp",
    ".css",
    ".scss",
    ".html",
}
SECRET_NAME_RE = re.compile(r"(^|[._-])(env|secret|token|key|credential|private)([._-]|$)", re.I)
SECRET_VALUE_RE = re.compile(
    r"(?i)(api[_-]?key|token|password|secret|authorization|private[_-]?key)\s*[:=]\s*['\"]?[^'\"\s]+"
)

# Generic folder-name conventions used to label architecture areas. These are
# common across projects (not specific to any one repo); the *actual* area names
# always come from the project's real directories, and the responsibility prefers
# a real package docstring when one is present. Anything not listed falls back to
# a derived "Modules under `<dir>`" label, so unknown projects still get areas.
GENERIC_FOLDER_ROLES: dict[str, str] = {
    "commands": "Command-line entrypoints and command handlers.",
    "command": "Command handlers.",
    "cli": "Command-line interface layer.",
    "services": "Application and business-logic services.",
    "service": "Application services.",
    "models": "Data models and schemas.",
    "model": "Data models.",
    "schemas": "Data schemas / validation.",
    "api": "API endpoints and request handling.",
    "routes": "HTTP routes / controllers.",
    "controllers": "Request controllers.",
    "views": "View / presentation layer.",
    "templates": "Templates / presentation.",
    "utils": "Shared utility helpers.",
    "util": "Shared utility helpers.",
    "helpers": "Shared helper functions.",
    "lib": "Library / support code.",
    "core": "Core domain logic.",
    "config": "Configuration loading.",
    "settings": "Configuration / settings.",
    "db": "Database access layer.",
    "database": "Database access layer.",
    "migrations": "Database migrations.",
    "tests": "Automated tests.",
    "test": "Automated tests.",
    "docs": "Documentation.",
    "doc": "Documentation.",
    "scripts": "Operational / utility scripts.",
    "tools": "Tooling and integrations.",
    "parsers": "Parsing logic.",
    "renderers": "Rendering / output formatting.",
    "handlers": "Event / request handlers.",
    "middleware": "Middleware components.",
    "components": "UI components.",
    "hooks": "Hooks / extensions.",
    "store": "State / store management.",
    "stores": "State / store management.",
    "vector_store": "Vector store / embeddings storage.",
    "llm": "LLM client and model orchestration.",
    "ai": "AI / model orchestration.",
    "agents": "Agent orchestration.",
    "analysis": "Static analysis and checks.",
    "dependencies": "Dependency parsing / analysis.",
    "describe": "Description / summarization flows.",
    "workers": "Background workers / jobs.",
    "tasks": "Background tasks / jobs.",
    "jobs": "Background jobs.",
    "events": "Event handling.",
    "domain": "Domain model and logic.",
    "infra": "Infrastructure / adapters.",
    "infrastructure": "Infrastructure / adapters.",
    "adapters": "External-system adapters.",
    "repositories": "Data repositories.",
    "repository": "Data repositories.",
}


@dataclass(slots=True)
class ProjectAnalyzeOptions:
    depth: str = "normal"
    output_format: str = "both"
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    max_files: int = 5000
    max_file_size_kb: int = 512


@dataclass(slots=True)
class ProjectAnalyzeResult:
    output_dir: Path
    artifacts: dict[str, Path]
    report: dict[str, Any]
    errors: list[str] = field(default_factory=list)


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_text(path: Path, *, max_bytes: int = 1_000_000) -> str:
    data = path.read_bytes()[:max_bytes]
    return data.decode("utf-8", errors="ignore")


def _is_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:2048]
    except OSError:
        return True
    return b"\0" in chunk


def _matches_path_filter(rel_path: str, filters: Iterable[str]) -> bool:
    parts = rel_path.split("/")
    for raw in filters:
        item = str(raw or "").strip().strip("/")
        if not item:
            continue
        if rel_path == item or rel_path.startswith(item + "/") or item in parts:
            return True
    return False


def _dependency_name(raw: str) -> str:
    return _normalize_dep_name(str(raw).strip().strip("\"'"))


def _line_for_text(path: Path, needle: str) -> int:
    try:
        for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            if needle in line:
                return idx
    except OSError:
        return 1
    return 1


class ProjectAnalyzeService:
    """Build reusable project intelligence artifacts without reading noisy trees."""

    def run(
        self,
        root_dir: str | Path,
        output_dir: str | Path,
        *,
        options: ProjectAnalyzeOptions | None = None,
        llm_analyzer: LLMAnalyzerFn | None = None,
    ) -> ProjectAnalyzeResult:
        options = options or ProjectAnalyzeOptions()
        root = Path(root_dir).resolve()
        if root.is_file():
            root = root.parent
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        inventory = self.build_inventory(root, options)
        dependencies = self.build_dependencies(root, inventory)
        entrypoints = self.detect_entrypoints(root, inventory)
        symbols = self.extract_symbols(root, inventory, options)
        architecture = self.build_architecture(root, inventory, symbols, dependencies)
        risks = self.detect_risks(root, inventory)
        recommendations = self.build_recommendations(inventory, dependencies, risks)
        report = self.build_report(root, inventory, dependencies, entrypoints, symbols, architecture, risks, recommendations)

        # Layer 2: compact evidence -> LLM analysis (with deterministic fallback).
        evidence = build_evidence(report, depth=options.depth)
        llm_result = self._run_llm_analysis(evidence, options.depth, root, llm_analyzer)
        report["llm_analysis"] = llm_result.to_dict()

        artifacts = self.write_artifacts(
            out_dir,
            report=report,
            inventory=inventory,
            dependencies=dependencies,
            symbols=symbols,
            architecture=architecture,
            risks=risks,
            recommendations=recommendations,
            evidence=evidence,
            llm_result=llm_result,
            options=options,
        )
        errors = self.validate_artifacts(artifacts)
        # Surface an LLM warning only when analysis was actually requested; a
        # deterministic-only run (no analyzer injected) is an expected mode.
        if llm_analyzer is not None and not llm_result.available and llm_result.error:
            errors.append(f"LLM analysis unavailable: {llm_result.error}")
        return ProjectAnalyzeResult(output_dir=out_dir, artifacts=artifacts, report=report, errors=errors)

    def _run_llm_analysis(
        self,
        evidence: AnalyzeEvidence,
        depth: str,
        root: Path,
        llm_analyzer: LLMAnalyzerFn | None,
    ) -> LLMAnalyzeResult:
        """Invoke the injected analyzer; never let an LLM error break analyze."""
        if llm_analyzer is None:
            return LLMAnalyzeResult(available=False, error="LLM analyzer not provided.")
        try:
            return llm_analyzer(evidence, depth, root)
        except Exception as exc:  # noqa: BLE001 - fallback, do not crash analyze
            return LLMAnalyzeResult(available=False, error=f"LLM analyzer raised: {exc}")

    def build_inventory(self, root: Path, options: ProjectAnalyzeOptions) -> dict[str, Any]:
        ignore_patterns = set(DEFAULT_IGNORE_DIRS) | set(options.exclude)
        gitignore_patterns = load_ignore_patterns(root)
        files: list[dict[str, Any]] = []
        large_skipped: list[dict[str, Any]] = []
        binary_skipped: list[dict[str, Any]] = []
        ignored_dirs: set[str] = set()
        source_folders: set[str] = set()
        test_folders: set[str] = set()
        docs_folders: set[str] = set()
        script_folders: set[str] = set()
        generated_folders: set[str] = set()
        config_files: list[str] = []
        entrypoint_paths: set[str] = set()
        max_bytes = max(1, options.max_file_size_kb) * 1024

        for current, dirs, names in os.walk(root):
            current_path = Path(current)
            rel_dir = "." if current_path == root else _rel(current_path, root)
            kept_dirs: list[str] = []
            for dirname in dirs:
                rel_child = dirname if rel_dir == "." else f"{rel_dir}/{dirname}"
                if dirname in ignore_patterns or _matches_path_filter(rel_child, ignore_patterns):
                    ignored_dirs.add(rel_child)
                    continue
                if any(_matches_path_filter(rel_child, [pattern]) for pattern in gitignore_patterns):
                    ignored_dirs.add(rel_child)
                    continue
                kept_dirs.append(dirname)
            dirs[:] = kept_dirs

            for name in sorted(names):
                if name in NOISY_FILES:
                    continue
                path = current_path / name
                rel_path = _rel(path, root)
                if options.include and not _matches_path_filter(rel_path, options.include):
                    continue
                if _matches_path_filter(rel_path, ignore_patterns) or any(
                    _matches_path_filter(rel_path, [pattern]) for pattern in gitignore_patterns
                ):
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if len(files) >= max(1, options.max_files):
                    large_skipped.append({"path": rel_path, "size_bytes": size, "reason": "max_files_reached"})
                    continue
                if size > max_bytes:
                    large_skipped.append({"path": rel_path, "size_bytes": size, "reason": "max_file_size_exceeded"})
                    continue
                if _is_binary(path):
                    binary_skipped.append({"path": rel_path, "size_bytes": size})
                    continue

                category = self.classify_file(rel_path, path)
                language = language_for_path(path)
                is_test = category == "test"
                is_entrypoint = self._is_entrypoint_path(rel_path)
                if is_entrypoint:
                    entrypoint_paths.add(rel_path)
                record = {
                    "path": rel_path,
                    "category": category,
                    "language": language,
                    "size_bytes": size,
                    "is_entrypoint": is_entrypoint,
                    "is_test": is_test,
                }
                files.append(record)
                top = rel_path.split("/", 1)[0]
                if category == "source_code":
                    source_folders.add(top)
                elif category == "test":
                    test_folders.add(top)
                elif category == "documentation":
                    docs_folders.add(top)
                elif category == "script":
                    script_folders.add(top)
                elif category == "generated":
                    generated_folders.add(top)
                elif category == "config":
                    config_files.append(rel_path)

        languages = sorted({item["language"] for item in files if item["language"] != "unknown"})
        return {
            "project_name": self._project_name(root),
            "root_path": str(root),
            "detected_languages": languages,
            "detected_frameworks": [],
            "package_managers": [],
            "important_config_files": sorted(config_files),
            "entrypoints": sorted(entrypoint_paths),
            "source_folders": sorted(source_folders),
            "test_folders": sorted(test_folders),
            "docs_folders": sorted(docs_folders),
            "script_folders": sorted(script_folders),
            "generated_folders": sorted(generated_folders),
            "ignored_folders": sorted(ignored_dirs),
            "total_files": len(files),
            "source_files_count": sum(1 for item in files if item["category"] == "source_code"),
            "test_files_count": sum(1 for item in files if item["category"] == "test"),
            "config_files_count": sum(1 for item in files if item["category"] == "config"),
            "documentation_files_count": sum(1 for item in files if item["category"] == "documentation"),
            "large_skipped_files": large_skipped,
            "binary_skipped_files": binary_skipped,
            "files": files,
            "ignore_rules": sorted(DEFAULT_IGNORE_DIRS | set(options.exclude)),
            "secret_bearing_config": sorted(item["path"] for item in files if SECRET_NAME_RE.search(Path(item["path"]).name)),
        }

    def classify_file(self, rel_path: str, path: Path) -> str:
        name = path.name
        suffix = path.suffix.lower()
        lower = rel_path.lower()
        parts = set(lower.split("/"))
        if name in LOCK_FILES:
            return "lockfile"
        if name in CONFIG_NAMES or suffix in {".toml", ".yaml", ".yml", ".ini", ".cfg"} or name.startswith(".env"):
            return "config"
        if "migrations" in parts:
            return "migration"
        if "tests" in parts or "__tests__" in parts or name.startswith("test_") or name.endswith("_test.py"):
            return "test"
        if "generated" in parts or name.endswith(".generated.py") or ".min." in name:
            return "generated"
        if suffix in DOC_SUFFIXES or "docs" in parts:
            return "documentation"
        if "scripts" in parts or suffix in {".sh", ".bash", ".zsh"}:
            return "script"
        if suffix in STATIC_SUFFIXES:
            return "static_asset"
        if suffix in SOURCE_SUFFIXES:
            return "source_code"
        if lower.startswith(".mana/"):
            return "artifact"
        return "unknown"

    def build_dependencies(self, root: Path, inventory: dict[str, Any]) -> dict[str, Any]:
        manifests = [root / item["path"] for item in inventory["files"] if Path(item["path"]).name in DEPENDENCY_MANIFESTS]
        runtime: set[str] = set()
        dev: set[str] = set()
        managers: set[str] = set()
        lock_files: list[str] = []
        warnings: list[str] = []
        package_json_seen = False
        pyproject_seen = False
        requirements_seen = False

        for manifest in manifests:
            name = manifest.name
            rel_path = _rel(manifest, root)
            if name in LOCK_FILES:
                lock_files.append(rel_path)
            if name == "pyproject.toml":
                pyproject_seen = True
                managers.add("pip")
                rt, dv = self._deps_from_pyproject(manifest)
                runtime.update(rt)
                dev.update(dv)
            elif name == "requirements.txt":
                requirements_seen = True
                managers.add("pip")
                runtime.update(self._deps_from_requirements(manifest))
            elif name == "setup.cfg":
                managers.add("pip")
                rt, dv = self._deps_from_setup_cfg(manifest)
                runtime.update(rt)
                dev.update(dv)
            elif name == "setup.py":
                managers.add("pip")
                runtime.update(self._deps_from_setup_py(manifest))
            elif name == "Pipfile":
                managers.add("pipenv")
                rt, dv = self._deps_from_pipfile(manifest)
                runtime.update(rt)
                dev.update(dv)
            elif name == "package.json":
                package_json_seen = True
                managers.add("npm")
                rt, dv = self._deps_from_package_json(manifest)
                runtime.update(rt)
                dev.update(dv)
            elif name.startswith("docker-compose"):
                managers.add("docker-compose")
            elif name == "Dockerfile":
                managers.add("docker")

        if package_json_seen and not any(Path(item).name in {"package-lock.json", "pnpm-lock.yaml", "yarn.lock"} for item in lock_files):
            warnings.append("package.json detected without npm/pnpm/yarn lock file.")
        if (pyproject_seen or requirements_seen) and "poetry.lock" not in {Path(item).name for item in lock_files}:
            warnings.append("Python dependency manifest detected without poetry.lock; pip may still use requirements pinning.")

        all_deps = runtime | dev
        frameworks = sorted({label for key, label in FRAMEWORK_SIGNALS.items() if _dependency_name(key) in all_deps})
        testing = sorted(dep for dep in all_deps if dep in {"pytest", "unittest", "jest", "vitest", "mocha", "playwright"})
        llm = sorted(dep for dep in all_deps if dep in {"openai", "langchain", "langchain-openai", "langchain-community", "anthropic", "faiss-cpu"})
        inventory["detected_frameworks"] = frameworks
        inventory["package_managers"] = sorted(managers)
        return {
            "package_managers": sorted(managers),
            "runtime_dependencies": sorted(runtime),
            "dev_dependencies": sorted(dev),
            "lock_files": sorted(lock_files),
            "warnings": warnings,
            "framework_packages": frameworks,
            "testing_packages": testing,
            "llm_agent_tooling_packages": llm,
            "manifests": sorted(_rel(item, root) for item in manifests),
        }

    def detect_entrypoints(self, root: Path, inventory: dict[str, Any]) -> list[dict[str, Any]]:
        entrypoints: list[dict[str, Any]] = []
        by_path = {item["path"]: root / item["path"] for item in inventory["files"]}
        pyproject = by_path.get("pyproject.toml")
        if pyproject:
            try:
                payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                for name, command in (payload.get("project", {}).get("scripts", {}) or {}).items():
                    entrypoints.append(self._entry(name, "cli", "pyproject.toml", _line_for_text(pyproject, name), command, "Python project script"))
                console_scripts = (
                    payload.get("project", {})
                    .get("entry-points", {})
                    .get("console_scripts", {})
                    or payload.get("tool", {}).get("setuptools", {}).get("entry-points", {}).get("console_scripts", {})
                    or {}
                )
                for name, command in console_scripts.items():
                    entrypoints.append(self._entry(name, "cli", "pyproject.toml", _line_for_text(pyproject, name), command, "console_scripts entrypoint"))
            except Exception:
                pass
        setup_cfg = by_path.get("setup.cfg")
        if setup_cfg:
            parser = configparser.ConfigParser()
            try:
                parser.read(setup_cfg, encoding="utf-8")
                if parser.has_section("options.entry_points"):
                    for raw in parser.get("options.entry_points", "console_scripts", fallback="").splitlines():
                        if "=" in raw:
                            name, command = [piece.strip() for piece in raw.split("=", 1)]
                            entrypoints.append(self._entry(name, "cli", "setup.cfg", _line_for_text(setup_cfg, name), command, "setup.cfg console script"))
            except Exception:
                pass
        for rel_path, path in by_path.items():
            name = path.name
            if name in {"__main__.py", "main.py", "app.py", "server.py", "manage.py"}:
                entrypoints.append(self._entry(path.stem, "module" if name == "__main__.py" else "script", rel_path, 1, rel_path, f"{name} entrypoint"))
            if name == "package.json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    for script_name, command in (payload.get("scripts", {}) or {}).items():
                        entrypoints.append(self._entry(script_name, "script", rel_path, _line_for_text(path, script_name), command, "package.json script"))
                except Exception:
                    pass
            if name == "Dockerfile":
                for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                    stripped = line.strip()
                    if stripped.startswith(("CMD", "ENTRYPOINT")):
                        entrypoints.append(self._entry(stripped.split()[0].lower(), "docker", rel_path, idx, stripped, "Docker image startup"))
            if name.startswith("docker-compose"):
                for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                    if re.match(r"^\s{2}[A-Za-z0-9_.-]+:\s*$", line):
                        service = line.strip().rstrip(":")
                        entrypoints.append(self._entry(service, "service", rel_path, idx, service, "docker-compose service"))
        return self._dedupe_entrypoints(entrypoints)

    def extract_symbols(self, root: Path, inventory: dict[str, Any], options: ProjectAnalyzeOptions) -> dict[str, Any]:
        symbols: list[dict[str, Any]] = []
        python_files = [item for item in inventory["files"] if item["language"] == "python" and item["category"] in {"source_code", "test", "script"}]
        limit = 80 if options.depth == "quick" else 240 if options.depth == "normal" else 1000
        for record in python_files[:limit]:
            path = root / record["path"]
            try:
                source = path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source, filename=str(path))
            except Exception:
                continue
            for node in ast.walk(tree):
                symbol = self._symbol_from_node(node, record["path"], source)
                if symbol:
                    symbols.append(symbol)
        high = [item for item in symbols if item["importance"] == "high"]
        return {"symbols": symbols, "important_symbols": high[:80], "stats": {"python_files_scanned": min(len(python_files), limit), "symbols_count": len(symbols)}}

    def build_architecture(
        self,
        root: Path,
        inventory: dict[str, Any],
        symbols: dict[str, Any],
        dependencies: dict[str, Any],
    ) -> dict[str, Any]:
        """Derive architecture areas from the *actual* project structure.

        Areas are the project's real directories (grouped one level under the
        detected source root), their responsibilities prefer real package
        docstrings, and cross-area dependencies are computed from real intra-
        project imports. Nothing here is hardcoded to a specific repository.
        """
        code_records = [
            item for item in inventory["files"]
            if item["category"] in {"source_code", "test", "script"}
        ]
        source_root = self._detect_source_root(inventory)
        # Map each file -> its architecture area key (a real directory).
        file_area: dict[str, str] = {}
        area_files: dict[str, list[str]] = {}
        for record in code_records:
            area = self._area_key(record["path"], source_root)
            file_area[record["path"]] = area
            area_files.setdefault(area, []).append(record["path"])

        import_edges = self._build_import_graph(root, inventory, file_area)
        important_symbols = symbols.get("important_symbols", [])

        sections: list[dict[str, Any]] = []
        for area in sorted(area_files, key=lambda key: (-len(area_files[key]), key)):
            related = sorted(area_files[area])
            area_symbols = [item for item in important_symbols if file_area.get(item["file"]) == area][:20]
            depends_on = sorted(import_edges.get(area, set()) - {area})
            sections.append(
                {
                    "area": area,
                    "responsibility": self._area_responsibility(root, area, related),
                    "related_files": related[:20],
                    "file_count": len(related),
                    "important_classes_functions": area_symbols,
                    "dependencies_on_other_parts": depends_on,
                    "risk_notes": self._area_risk_notes(related),
                }
            )
        workflow = self._project_workflow(inventory, area_files, source_root)
        return {"sections": sections, "source_root": source_root, "agent_workflow": workflow}

    def _detect_source_root(self, inventory: dict[str, Any]) -> str:
        """Return the directory that holds most source files (e.g. ``src/pkg``)."""
        counts: dict[str, int] = {}
        for item in inventory["files"]:
            if item["category"] != "source_code":
                continue
            parts = item["path"].split("/")
            # Prefer a two-level root (src/<pkg>) when present, else the top folder.
            if len(parts) >= 3 and parts[0] in {"src", "lib", "app", "source"}:
                key = "/".join(parts[:2])
            elif len(parts) >= 2:
                key = parts[0]
            else:
                key = ""
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return ""
        return max(counts, key=lambda key: (counts[key], key))

    def _area_key(self, rel_path: str, source_root: str) -> str:
        """Map a file path to its architecture area (one real directory level)."""
        path = rel_path
        if source_root and path.startswith(source_root + "/"):
            remainder = path[len(source_root) + 1:]
            parts = remainder.split("/")
            return f"{source_root}/{parts[0]}" if len(parts) > 1 else source_root
        parts = path.split("/")
        return parts[0] if len(parts) > 1 else "(root)"

    def _area_responsibility(self, root: Path, area: str, related: list[str]) -> str:
        """Prefer a real package docstring; else a generic role; else a derived label."""
        init_path = root / area / "__init__.py"
        doc = self._first_docstring(init_path)
        if not doc:
            # Fall back to the docstring of the largest module in the area.
            for rel in related:
                doc = self._first_docstring(root / rel)
                if doc:
                    break
        if doc:
            first = doc.strip().splitlines()[0].strip()
            if first:
                return first[:200]
        leaf = area.rsplit("/", 1)[-1].lower()
        if leaf in GENERIC_FOLDER_ROLES:
            return GENERIC_FOLDER_ROLES[leaf]
        return f"Modules under `{area}`."

    def _first_docstring(self, path: Path) -> str:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
        except Exception:
            return ""
        return ast.get_docstring(tree) or ""

    def _build_import_graph(
        self,
        root: Path,
        inventory: dict[str, Any],
        file_area: dict[str, str],
    ) -> dict[str, set[str]]:
        """Compute area -> set(areas) edges from real intra-project imports."""
        # Index importable modules by dotted path so imports can resolve to files.
        # Register both the raw path and src-layout variants (drop a leading
        # ``src.``/``lib.``/``app.`` segment) because the importable module name
        # usually omits the source-root folder under a src layout.
        module_to_area: dict[str, str] = {}

        def register(dotted: str, area: str) -> None:
            module_to_area.setdefault(dotted, area)
            if dotted.endswith(".__init__"):
                module_to_area.setdefault(dotted[: -len(".__init__")], area)

        for rel_path, area in file_area.items():
            if not rel_path.endswith(".py"):
                continue
            dotted = rel_path[:-3].replace("/", ".")
            register(dotted, area)
            head = dotted.split(".", 1)
            if len(head) == 2 and head[0] in {"src", "lib", "app", "source"}:
                register(head[1], area)

        def resolve(name: str) -> str | None:
            parts = name.split(".")
            for cut in range(len(parts), 0, -1):
                candidate = ".".join(parts[:cut])
                if candidate in module_to_area:
                    return module_to_area[candidate]
            return None

        edges: dict[str, set[str]] = {}
        for rel_path, area in file_area.items():
            if not rel_path.endswith(".py"):
                continue
            try:
                tree = ast.parse((root / rel_path).read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module]
                for name in names:
                    target = resolve(name)
                    if target and target != area:
                        edges.setdefault(area, set()).add(target)
        return edges

    def detect_risks(self, root: Path, inventory: dict[str, Any]) -> dict[str, Any]:
        """Detect project-derived risks.

        All findings come from the actual files: generic, language-agnostic text
        rules plus the project's real Python static-analysis findings (merged
        from the original analyze engine, ``PythonStaticAnalyzer``). There are no
        repository-specific hardcoded patterns.
        """
        risks: list[dict[str, Any]] = []
        text_records = [
            item for item in inventory["files"]
            if item["category"] in {"source_code", "test", "script", "config", "documentation"} and item["size_bytes"] <= 512 * 1024
        ]
        seen_hashes: dict[str, str] = {}
        for record in text_records:
            path = root / record["path"]
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
            if digest in seen_hashes and record["category"] == "source_code":
                risks.append(self._risk("Duplicate file candidate", "low", record["path"], 1, f"Similar content to {seen_hashes[digest]}", "Can confuse future code search and maintenance.", "Remove or document duplicate ownership."))
            else:
                seen_hashes[digest] = record["path"]
            for idx, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                low = stripped.lower()
                if any(tag in stripped for tag in ("TODO", "FIXME", "HACK")):
                    risks.append(self._risk("TODO/FIXME/HACK comment", "low", record["path"], idx, stripped[:180], "Marks known incomplete or fragile work.", "Triage or convert into tracked tasks."))
                if "while true" in low or re.search(r"\bwhile\s+1\s*:", low):
                    risks.append(self._risk("Unbounded loop candidate", "medium", record["path"], idx, stripped[:180], "Loops without an obvious exit can hang the program.", "Add explicit limits, timeout, or break conditions."))
                if "subprocess" in low and "timeout" not in low:
                    risks.append(self._risk("Subprocess without local timeout evidence", "medium", record["path"], idx, stripped[:180], "Long-running commands can stall execution.", "Pass and enforce a timeout around subprocess calls."))
                if "except exception" in low and ("pass" in low or "continue" in low):
                    risks.append(self._risk("Broad exception swallowing", "medium", record["path"], idx, stripped[:180], "Failures can be hidden from report and callers.", "Log or return structured warnings."))
                if "rm -rf" in low or "git reset --hard" in low:
                    risks.append(self._risk("Unsafe shell command", "high", record["path"], idx, stripped[:180], "Destructive commands can remove user work.", "Require explicit confirmation and safer targeted operations."))
                if SECRET_VALUE_RE.search(stripped) and not Path(record["path"]).name.startswith(".env"):
                    risks.append(self._risk("Secrets exposure risk", "high", record["path"], idx, SECRET_VALUE_RE.sub("<redacted>", stripped[:180]), "Secrets in tracked files can leak credentials.", "Move secret values to ignored environment storage and rotate exposed credentials."))
            if record["category"] == "source_code" and text.count("\n") > 700:
                risks.append(self._risk("Large module", "medium", record["path"], 1, f"{text.count(chr(10)) + 1} lines", "Large modules are harder for humans and agents to reason about.", "Split by responsibility once behavior is covered by tests."))
        test_count = inventory.get("test_files_count", 0)
        if test_count == 0:
            risks.append(self._risk("Missing tests", "high", "", 1, "No test files detected.", "Changes are harder to verify safely.", "Add focused tests around core workflows."))

        static_findings = self._static_analysis_risks(root, inventory)
        risks.extend(static_findings["items"])
        return {"items": risks[:300], "static_analysis": static_findings["summary"]}

    def _static_analysis_risks(self, root: Path, inventory: dict[str, Any]) -> dict[str, Any]:
        """Merge the original analyze engine's Python static findings as risks.

        Reuses :class:`PythonStaticAnalyzer` (the deterministic core of the prior
        analyze system) so the unified analyze surfaces the same findings — but
        only for the project's real Python files, never a fixed template.
        """
        try:
            from mana_agent.analysis.checks import PythonStaticAnalyzer
        except Exception:  # noqa: BLE001 - static analysis is best-effort
            return {"items": [], "summary": {}}
        analyzer = PythonStaticAnalyzer()
        severity_map = {"error": "high", "warning": "medium", "info": "low"}
        items: list[dict[str, Any]] = []
        summary: dict[str, int] = {}
        python_files = [
            item for item in inventory["files"]
            if item["language"] == "python" and item["category"] in {"source_code", "test", "script"}
        ]
        for record in python_files:
            try:
                findings = analyzer.analyze_file(root / record["path"])
            except Exception:  # noqa: BLE001 - never let one file break analyze
                continue
            for finding in findings:
                summary[finding.rule_id] = summary.get(finding.rule_id, 0) + 1
                items.append(
                    self._risk(
                        finding.rule_id.replace("-", " "),
                        severity_map.get(str(finding.severity).lower(), "low"),
                        record["path"],
                        int(getattr(finding, "line", 1) or 1),
                        str(getattr(finding, "message", ""))[:180],
                        "Detected by static analysis of this file.",
                        "Review and resolve the reported static-analysis issue.",
                    )
                )
        return {"items": items, "summary": summary}

    def build_recommendations(self, inventory: dict[str, Any], dependencies: dict[str, Any], risks: dict[str, Any]) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        risk_items = risks.get("items", [])
        if any("Tools-only" in item["title"] for item in risk_items):
            items.append(self._recommendation(
                "Add regression test for tools-only violation recovery",
                "high",
                ["tests/", "src/"],
                "Tools-only strict paths are present and can block useful progress.",
                ["A test proves read/discovery jobs are not rejected by mutation-only policy.", "A failing mutation still reports no changed files clearly."],
                "pytest tests/test_tools_manager.py tests/test_tool_worker_process.py -q",
            ))
        if dependencies.get("warnings"):
            items.append(self._recommendation(
                "Clarify dependency lock policy",
                "medium",
                dependencies.get("manifests", []),
                "Dependency manifests have lock-file warnings.",
                ["Document the intended package manager and lock workflow.", "CI verifies dependency installation from the chosen files."],
                "python -m compileall src",
            ))
        if inventory.get("test_files_count", 0) == 0:
            items.append(self._recommendation(
                "Add smoke tests for the main entrypoints",
                "high",
                inventory.get("entrypoints", []),
                "No test files were detected.",
                ["Each detected CLI or service entrypoint has at least one smoke test."],
                "pytest -q",
            ))
        if any(item["title"] == "Subprocess without local timeout evidence" for item in risk_items):
            files = sorted({item["file"] for item in risk_items if item["title"] == "Subprocess without local timeout evidence"})[:10]
            items.append(self._recommendation(
                "Add timeout around long-running verification commands",
                "high",
                files,
                "Subprocess usage without nearby timeout evidence can stall agent runs.",
                ["All command execution helpers accept and enforce a timeout.", "Timeout behavior has a focused test."],
                "pytest tests -q",
            ))
        items.append(self._recommendation(
            "Load agent_context.json into future chat context",
            "medium",
            [".mana/analyze/agent_context.json"],
            "The compact context artifact is designed to make later coding-agent turns faster and better grounded.",
            ["Chat startup checks for .mana/analyze/agent_context.json.", "Loaded context is bounded and never includes secrets."],
            "pytest tests/commands tests/integration -q",
        ))
        return {"items": items}

    def build_report(
        self,
        root: Path,
        inventory: dict[str, Any],
        dependencies: dict[str, Any],
        entrypoints: list[dict[str, Any]],
        symbols: dict[str, Any],
        architecture: dict[str, Any],
        risks: dict[str, Any],
        recommendations: dict[str, Any],
    ) -> dict[str, Any]:
        verification_commands = self._verification_commands(inventory, dependencies)
        return {
            "project_summary": f"{inventory['project_name']} contains {inventory['total_files']} scanned files across {', '.join(inventory['detected_languages']) or 'unknown languages'}.",
            "root_path": str(root),
            "inventory": inventory,
            "dependencies": dependencies,
            "entrypoints": entrypoints,
            "symbols": symbols,
            "architecture": architecture,
            "risks": risks,
            "recommendations": recommendations,
            "verification_commands": verification_commands,
            "generated_at": _now_iso(),
        }

    def write_artifacts(
        self,
        output_dir: Path,
        *,
        report: dict[str, Any],
        inventory: dict[str, Any],
        dependencies: dict[str, Any],
        symbols: dict[str, Any],
        architecture: dict[str, Any],
        risks: dict[str, Any],
        recommendations: dict[str, Any],
        evidence: AnalyzeEvidence | None = None,
        llm_result: LLMAnalyzeResult | None = None,
        options: ProjectAnalyzeOptions,
    ) -> dict[str, Path]:
        llm_result = llm_result or LLMAnalyzeResult(available=False, error="LLM analyzer not provided.")
        agent_context = self._agent_context(report, llm_result)
        artifacts = {
            "report.md": output_dir / "report.md",
            "report.json": output_dir / "report.json",
            "agent_context.json": output_dir / "agent_context.json",
            "inventory.json": output_dir / "inventory.json",
            "symbols.json": output_dir / "symbols.json",
            "dependencies.json": output_dir / "dependencies.json",
            "architecture.md": output_dir / "architecture.md",
            "risks.json": output_dir / "risks.json",
            "recommendations.md": output_dir / "recommendations.md",
            "evidence.json": output_dir / "evidence.json",
            "llm_summary.md": output_dir / "llm_summary.md",
        }
        json_payloads = {
            "report.json": report,
            "agent_context.json": agent_context,
            "inventory.json": inventory,
            "symbols.json": symbols,
            "dependencies.json": dependencies,
            "risks.json": risks,
            "evidence.json": (evidence.to_dict() if evidence is not None else build_evidence(report, depth=options.depth).to_dict()),
        }
        for name, payload in json_payloads.items():
            artifacts[name].write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        artifacts["report.md"].write_text(self.render_report_markdown(report, llm_result), encoding="utf-8")
        artifacts["architecture.md"].write_text(self.render_architecture_markdown(architecture, llm_result), encoding="utf-8")
        artifacts["recommendations.md"].write_text(self.render_recommendations_markdown(recommendations, llm_result), encoding="utf-8")
        artifacts["llm_summary.md"].write_text(self.render_llm_summary_markdown(llm_result), encoding="utf-8")
        return {name: path for name, path in artifacts.items() if path.exists()}

    def validate_artifacts(self, artifacts: dict[str, Path]) -> list[str]:
        errors: list[str] = []
        for required in ("report.json", "agent_context.json", "inventory.json", "symbols.json", "dependencies.json", "risks.json", "evidence.json"):
            path = artifacts.get(required)
            if path is None or not path.exists():
                errors.append(f"Missing required JSON artifact: {required}")
                continue
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"Invalid JSON artifact {required}: {exc}")
        return errors

    _ARTIFACT_PURPOSES: tuple[tuple[str, str], ...] = (
        ("report.md", "Human-readable senior-engineer report (this file)."),
        ("report.json", "Full machine-readable report: deterministic data + LLM analysis."),
        ("agent_context.json", "Compact, bounded context loaded into chat/coding-agent turns."),
        ("evidence.json", "Compact structured evidence used as input to the LLM analyzer."),
        ("llm_summary.md", "LLM-written narrative summary and onboarding notes."),
        ("inventory.json", "File inventory, classifications, folders, and counts."),
        ("symbols.json", "Extracted Python classes/functions/commands with locations."),
        ("dependencies.json", "Runtime/dev dependencies, lock files, and tooling packages."),
        ("architecture.md", "Architecture map with layers, files, and LLM explanation."),
        ("risks.json", "Detected risks with severity, evidence, and fixes."),
        ("recommendations.md", "Prioritized recommendations and next coding tasks."),
    )

    def render_report_markdown(self, report: dict[str, Any], llm: LLMAnalyzeResult | None = None) -> str:
        llm = llm or LLMAnalyzeResult(available=False)
        inventory = report["inventory"]
        dependencies = report["dependencies"]
        architecture = report["architecture"]
        risks = report["risks"]
        recommendations = report["recommendations"]
        symbols = report["symbols"]
        entrypoints = report["entrypoints"]

        lines = ["# Project Analysis Report", ""]
        if llm.available:
            lines += [f"_Generated with LLM analysis (model: {llm.model or 'unknown'})._", ""]
        else:
            reason = llm.error or "no model configured"
            lines += [f"> ⚠️ **LLM analysis unavailable** — deterministic fallback report. Reason: {reason}", ""]

        # 1. Executive Summary
        lines += ["## 1. Executive Summary", llm.project_summary or report["project_summary"], ""]

        # 2. Detected Stack
        lines += ["## 2. Detected Stack"]
        if llm.detected_stack_explanation:
            lines += [llm.detected_stack_explanation, ""]
        lines += ["| Aspect | Detected |", "| --- | --- |"]
        lines.append(f"| Languages | {', '.join(inventory['detected_languages']) or 'not detected'} |")
        lines.append(f"| Frameworks | {', '.join(inventory['detected_frameworks']) or 'not detected'} |")
        lines.append(f"| Package managers | {', '.join(inventory['package_managers']) or 'not detected'} |")
        lines.append(f"| Testing | {', '.join(dependencies.get('testing_packages', [])) or 'not detected'} |")
        lines.append(f"| LLM / agent tooling | {', '.join(dependencies.get('llm_agent_tooling_packages', [])) or 'not detected'} |")
        lines.append("")
        lines += ["| Metric | Value |", "| --- | --- |"]
        for key in ("total_files", "source_files_count", "test_files_count", "config_files_count", "documentation_files_count"):
            lines.append(f"| {key.replace('_', ' ').title()} | {inventory.get(key, 0)} |")
        lines.append("")

        # 3. Repository Overview
        lines += ["## 3. Repository Overview"]
        if llm.repository_overview:
            lines += [llm.repository_overview, ""]
        lines.append("Key folders:")
        for label, key in (("source", "source_folders"), ("tests", "test_folders"), ("docs", "docs_folders"), ("scripts", "script_folders")):
            folders = inventory.get(key, [])
            if folders:
                lines.append(f"- **{label}**: " + ", ".join(f"`{name}`" for name in folders))
        lines.append("")

        # 4. Important Files
        lines += ["## 4. Important Files", "| File | Why It Matters | Evidence |", "| --- | --- | --- |"]
        important_files = llm.important_files or [
            {"file": path, "why": "Config or entrypoint detected by the scanner.", "evidence": "inventory"}
            for path in (inventory.get("important_config_files", []) + inventory.get("entrypoints", []))[:20]
        ]
        for item in important_files[:25]:
            lines.append(f"| `{item.get('file', '')}` | {self._cell(item.get('why'))} | {self._cell(item.get('evidence'))} |")
        if not important_files:
            lines.append("| not detected | | |")
        lines.append("")

        # 5. Entrypoints and Commands
        lines += ["## 5. Entrypoints and Commands"]
        if llm.cli_commands_explanation:
            lines += [llm.cli_commands_explanation, ""]
        lines += ["| Name | Type | File | Line | Command |", "| --- | --- | --- | ---: | --- |"]
        for item in entrypoints[:40]:
            lines.append(f"| {item['name']} | {item['type']} | `{item['file']}` | {item['line']} | `{self._cell(item['command'])}` |")
        if not entrypoints:
            lines.append("| not detected | | | | |")
        lines.append("")

        # 6. Architecture Map
        lines += ["## 6. Architecture Map"]
        if llm.architecture_explanation:
            lines += [llm.architecture_explanation, ""]
        for section in architecture.get("sections", []):
            lines.append(f"### {section['area']}")
            lines.append(section["responsibility"])
            lines.append("Related files: " + ", ".join(f"`{path}`" for path in section["related_files"][:8]))
            if section.get("risk_notes"):
                lines.append("Risk notes: " + "; ".join(section["risk_notes"]))
            lines.append("")

        # 7. Agent Workflow
        lines += ["## 7. Agent Workflow"]
        if llm.agent_workflow:
            lines += [llm.agent_workflow, ""]
        for question, answer in architecture.get("agent_workflow", {}).items():
            lines.append(f"- **{question}** {answer}")
        lines.append("")

        # 8. Analyze Workflow
        lines += ["## 8. Analyze Workflow"]
        lines += [llm.analyze_workflow or (
            "When `/analyze` runs the pipeline (1) scans the repository deterministically while "
            "skipping noisy folders, (2) collects structured evidence, (3) sends compact evidence "
            "to the LLM, (4) generates this report, (5) writes JSON + Markdown artifacts under "
            "`.mana/analyze/`, and (6) loads `agent_context.json` into chat context."
        ), ""]

        # 9. Dependencies
        lines += ["## 9. Dependencies", f"- Runtime: {', '.join(dependencies['runtime_dependencies'][:40]) or 'none detected'}", f"- Dev: {', '.join(dependencies['dev_dependencies'][:40]) or 'none detected'}", f"- Lock files: {', '.join(dependencies['lock_files']) or 'none detected'}"]
        if dependencies.get("warnings"):
            lines.extend(f"- ⚠️ {warning}" for warning in dependencies["warnings"])
        lines.append("")

        # 10. Symbols Overview
        lines += ["## 10. Symbols Overview"]
        if llm.important_symbols_overview:
            lines += [llm.important_symbols_overview, ""]
        lines += [f"- Python files scanned: {symbols['stats']['python_files_scanned']}", f"- Symbols extracted: {symbols['stats']['symbols_count']}"]
        for item in symbols.get("important_symbols", [])[:25]:
            lines.append(f"- `{item['name']}` ({item['kind']}) `{item['file']}:{item['line']}`")
        lines.append("")

        # 11. Risks and Problems
        lines += ["## 11. Risks and Problems"]
        risk_rows = llm.risk_analysis or risks.get("items", [])
        for item in risk_rows[:40]:
            severity = str(item.get("severity", "")).title() or "Info"
            location = f"`{item['file']}:{item.get('line', 1)}`" if item.get("file") else "repository"
            lines.append(f"- **{severity}** {item.get('title', '')} — {location}")
            if item.get("evidence"):
                lines.append(f"  - Evidence: {self._cell(item['evidence'])}")
            if item.get("why_it_matters"):
                lines.append(f"  - Why it matters: {self._cell(item['why_it_matters'])}")
            if item.get("recommended_fix"):
                lines.append(f"  - Recommended fix: {self._cell(item['recommended_fix'])}")
        if not risk_rows:
            lines.append("- No concrete risks detected.")
        static_summary = risks.get("static_analysis", {}) or {}
        if static_summary:
            lines.append("")
            lines.append("Static-analysis findings (from the merged static engine):")
            for rule, count in sorted(static_summary.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"- `{rule}`: {count}")
        lines.append("")

        # 12. Recommendations
        lines += ["## 12. Recommendations"]
        if llm.recommendations:
            lines.extend(f"- {item}" for item in llm.recommendations)
        else:
            for item in recommendations.get("items", []):
                lines.append(f"- **{item['priority']}** {item['title']}: {item['reason']}")
        lines.append("")

        # 13. Next Coding Tasks
        lines += ["## 13. Next Coding Tasks"]
        next_tasks = llm.next_tasks or [
            {
                "title": item["title"],
                "priority": item["priority"],
                "files": item["files"],
                "acceptance_criteria": item["acceptance_criteria"],
                "verification_command": item["verification"],
            }
            for item in recommendations.get("items", [])[:8]
        ]
        for task in next_tasks[:10]:
            lines.append(f"### {task.get('title', 'Task')}")
            lines.append(f"- Priority: {task.get('priority', 'Medium')}")
            files = task.get("files", []) or []
            lines.append("- Files likely involved: " + (", ".join(f"`{f}`" for f in files) or "not detected"))
            criteria = task.get("acceptance_criteria", []) or []
            if criteria:
                lines.append("- Acceptance criteria:")
                lines.extend(f"  - {c}" for c in criteria)
            if task.get("verification_command"):
                lines.append(f"- Verification: `{task['verification_command']}`")
            lines.append("")
        if not next_tasks:
            lines.append("- No next tasks generated.")
            lines.append("")

        # 14. Generated Artifacts
        lines += ["## 14. Generated Artifacts", "| Artifact | Purpose |", "| --- | --- |"]
        for name, purpose in self._ARTIFACT_PURPOSES:
            lines.append(f"| `.mana/analyze/{name}` | {purpose} |")
        lines.append("")
        return "\n".join(lines)

    def _cell(self, value: Any) -> str:
        """Sanitize a value for a single Markdown table cell."""
        return str(value or "").replace("\n", " ").replace("|", "\\|").strip()

    def render_llm_summary_markdown(self, llm: LLMAnalyzeResult) -> str:
        if not llm.available:
            return (
                "# LLM Summary\n\n"
                f"> LLM analysis unavailable: {llm.error or 'no model configured'}.\n\n"
                "This report was generated deterministically. Configure an API key and re-run "
                "`mana-agent analyze .` to add the LLM-written analysis.\n"
            )
        lines = ["# LLM Summary", "", f"_Model: {llm.model or 'unknown'}_", ""]
        lines += ["## Project Summary", llm.project_summary or "not detected", ""]
        lines += ["## Detected Stack", llm.detected_stack_explanation or "not detected", ""]
        lines += ["## Architecture", llm.architecture_explanation or "not detected", ""]
        lines += ["## Agent Workflow", llm.agent_workflow or "not detected", ""]
        lines += ["## Developer Onboarding", llm.onboarding_summary or "not detected", ""]
        return "\n".join(lines)

    def render_architecture_markdown(self, architecture: dict[str, Any], llm: LLMAnalyzeResult | None = None) -> str:
        llm = llm or LLMAnalyzeResult(available=False)
        lines = ["# Architecture Map", ""]
        if llm.architecture_explanation:
            lines += ["## Overview", llm.architecture_explanation, ""]
        for section in architecture.get("sections", []):
            lines += [f"## {section['area']}", section["responsibility"], ""]
            lines.append("Related files:")
            lines.extend(f"- `{path}`" for path in section.get("related_files", []))
            lines.append("")
            if section.get("important_classes_functions"):
                lines.append("Important symbols:")
                for symbol in section["important_classes_functions"]:
                    lines.append(f"- `{symbol['name']}` `{symbol['file']}:{symbol['line']}`")
                lines.append("")
            if section.get("dependencies_on_other_parts"):
                lines.append("Dependencies on other parts:")
                lines.extend(f"- {item}" for item in section["dependencies_on_other_parts"])
                lines.append("")
            if section.get("risk_notes"):
                lines.append("Risk notes:")
                lines.extend(f"- {item}" for item in section["risk_notes"])
                lines.append("")
        lines.append("## Agent Workflow")
        for question, answer in architecture.get("agent_workflow", {}).items():
            lines.append(f"- **{question}** {answer}")
        lines.append("")
        return "\n".join(lines)

    def render_recommendations_markdown(self, recommendations: dict[str, Any], llm: LLMAnalyzeResult | None = None) -> str:
        llm = llm or LLMAnalyzeResult(available=False)
        lines = ["# Recommendations", ""]
        if llm.recommendations:
            lines += ["## LLM Recommendations"]
            lines.extend(f"- {item}" for item in llm.recommendations)
            lines.append("")
        if llm.next_tasks:
            lines += ["## Next Coding Tasks (LLM)"]
            for task in llm.next_tasks:
                lines.append(f"### {task.get('title', 'Task')}")
                lines.append(f"- Priority: {task.get('priority', 'Medium')}")
                files = task.get("files", []) or []
                lines.append("- Files: " + (", ".join(f"`{f}`" for f in files) or "not detected"))
                criteria = task.get("acceptance_criteria", []) or []
                if criteria:
                    lines.append("- Acceptance criteria:")
                    lines.extend(f"  - {c}" for c in criteria)
                if task.get("verification_command"):
                    lines.append(f"- Verification: `{task['verification_command']}`")
                lines.append("")
        lines += ["## Deterministic Recommendations", ""]
        for item in recommendations.get("items", []):
            lines += [f"### {item['title']}", f"- Priority: {item['priority']}", f"- Reason: {item['reason']}", f"- Verification: `{item['verification']}`", "- Acceptance criteria:"]
            lines.extend(f"  - {criterion}" for criterion in item["acceptance_criteria"])
            lines.append("")
        return "\n".join(lines)

    def _project_name(self, root: Path) -> str:
        pyproject = root / "pyproject.toml"
        if pyproject.exists():
            try:
                return str(tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project", {}).get("name") or root.name)
            except Exception:
                return root.name
        package_json = root / "package.json"
        if package_json.exists():
            try:
                return str(json.loads(package_json.read_text(encoding="utf-8")).get("name") or root.name)
            except Exception:
                return root.name
        return root.name

    def _is_entrypoint_path(self, rel_path: str) -> bool:
        name = Path(rel_path).name
        return name in {"__main__.py", "main.py", "app.py", "server.py", "manage.py", "Dockerfile", "docker-compose.yml", "docker-compose.yaml"}

    def _deps_from_pyproject(self, path: Path) -> tuple[set[str], set[str]]:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        runtime = {_dependency_name(item) for item in payload.get("project", {}).get("dependencies", []) or []}
        dev: set[str] = set()
        optional = payload.get("project", {}).get("optional-dependencies", {}) or {}
        for group in ("dev", "test", "tests"):
            dev.update(_dependency_name(item) for item in optional.get(group, []) or [])
        return {item for item in runtime if item and item != "python"}, {item for item in dev if item and item != "python"}

    def _deps_from_requirements(self, path: Path) -> set[str]:
        deps: set[str] = set()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            item = line.strip()
            if item and not item.startswith(("#", "-")):
                dep = _dependency_name(item)
                if dep:
                    deps.add(dep)
        return deps

    def _deps_from_setup_cfg(self, path: Path) -> tuple[set[str], set[str]]:
        parser = configparser.ConfigParser()
        parser.read(path, encoding="utf-8")
        runtime = {_dependency_name(item) for item in parser.get("options", "install_requires", fallback="").splitlines()}
        dev = {_dependency_name(item) for item in parser.get("options.extras_require", "dev", fallback="").splitlines()}
        return {item for item in runtime if item}, {item for item in dev if item}

    def _deps_from_setup_py(self, path: Path) -> set[str]:
        text = _safe_text(path)
        deps: set[str] = set()
        for match in re.findall(r"install_requires\s*=\s*\[(.*?)\]", text, flags=re.S):
            deps.update(_dependency_name(item) for item in re.findall(r"['\"]([^'\"]+)['\"]", match))
        return {item for item in deps if item}

    def _deps_from_pipfile(self, path: Path) -> tuple[set[str], set[str]]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        runtime: set[str] = set()
        dev: set[str] = set()
        section = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                section = stripped.strip("[]")
                continue
            if "=" in stripped and section in {"packages", "dev-packages"}:
                target = dev if section == "dev-packages" else runtime
                target.add(_dependency_name(stripped.split("=", 1)[0]))
        return {item for item in runtime if item}, {item for item in dev if item}

    def _deps_from_package_json(self, path: Path) -> tuple[set[str], set[str]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runtime = {_dependency_name(item) for item in (payload.get("dependencies", {}) or {})}
        dev = {_dependency_name(item) for item in (payload.get("devDependencies", {}) or {})}
        return {item for item in runtime if item}, {item for item in dev if item}

    def _entry(self, name: str, typ: str, file: str, line: int, command: str, description: str) -> dict[str, Any]:
        return {"name": str(name), "type": typ, "file": file, "line": int(line), "command": str(command), "description": description}

    def _dedupe_entrypoints(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str, str, str]] = set()
        out: list[dict[str, Any]] = []
        for item in entries:
            key = (item["name"], item["type"], item["file"], item["command"])
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _symbol_from_node(self, node: ast.AST, rel_path: str, source: str) -> dict[str, Any] | None:
        kind = ""
        name = ""
        if isinstance(node, ast.AsyncFunctionDef):
            kind = "async_function"
            name = node.name
        elif isinstance(node, ast.FunctionDef):
            kind = "function"
            name = node.name
        elif isinstance(node, ast.ClassDef):
            bases = {ast.unparse(base).split(".")[-1] for base in node.bases}
            decorators = {ast.unparse(dec).split("(")[0].split(".")[-1] for dec in node.decorator_list}
            kind = "model" if bases & {"BaseModel", "Model"} else "class"
            if "dataclass" in decorators:
                kind = "model"
            name = node.name
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            names = [item for item in names if item.isupper() or item in {"app", "router"}]
            if not names:
                return None
            kind = "tool" if any("tool" in item.lower() for item in names) else "constant"
            name = names[0]
        else:
            return None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = [ast.unparse(dec) for dec in node.decorator_list]
            if any("command" in dec or "callback" in dec for dec in decorators):
                kind = "command"
            if any("tool" in dec.lower() for dec in decorators) or name.endswith("_tool"):
                kind = "tool"
        line = int(getattr(node, "lineno", 1))
        try:
            signature = ast.get_source_segment(source, node).splitlines()[0].strip() if ast.get_source_segment(source, node) else name
        except Exception:
            signature = name
        importance = "high" if kind in {"class", "command", "model", "tool"} or name in {"run", "main", "handle_analyze_command"} else "medium" if not name.startswith("_") else "low"
        return {
            "name": name,
            "kind": kind,
            "file": rel_path,
            "line": line,
            "signature": signature,
            "docstring": ast.get_docstring(node) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else "",
            "importance": importance,
        }

    def _area_risk_notes(self, related: list[str]) -> list[str]:
        notes: list[str] = []
        if len(related) > 12:
            notes.append("Area spans many files; keep contracts documented and tested.")
        return notes

    def _project_workflow(
        self,
        inventory: dict[str, Any],
        area_files: dict[str, list[str]],
        source_root: str,
    ) -> dict[str, str]:
        """Build a project-derived 'how this codebase runs' map from real evidence.

        Every question is answered from the actual detected entrypoints and the
        project's real directory areas (matched by generic folder conventions),
        so the result reflects *this* project rather than a fixed template. Only
        questions with concrete evidence are included.
        """
        def areas_matching(*keywords: str) -> list[str]:
            hits: list[str] = []
            for area in area_files:
                leaf = area.rsplit("/", 1)[-1].lower()
                if any(keyword in leaf for keyword in keywords):
                    hits.append(area)
            return sorted(hits)

        def files_for(areas: list[str], limit: int = 5) -> str:
            paths: list[str] = []
            for area in areas:
                paths.extend(sorted(area_files.get(area, []))[:limit])
            return ", ".join(f"`{path}`" for path in paths[:limit]) if paths else ""

        workflow: dict[str, str] = {}

        entrypoints = inventory.get("entrypoints", [])
        if entrypoints:
            workflow["Where does execution start?"] = ", ".join(f"`{path}`" for path in entrypoints[:6])

        cli_areas = areas_matching("command", "cli")
        if files_for(cli_areas):
            workflow["Where is the command / CLI layer?"] = files_for(cli_areas)

        core_areas = areas_matching("service", "core", "lib", "domain", "app")
        if files_for(core_areas):
            workflow["Where does the core application logic live?"] = files_for(core_areas)

        data_areas = areas_matching("model", "db", "database", "store", "repository", "schema")
        if files_for(data_areas):
            workflow["Where is data modeled / persisted?"] = files_for(data_areas)

        io_areas = areas_matching("api", "route", "controller", "handler", "view")
        if files_for(io_areas):
            workflow["Where are requests / I/O handled?"] = files_for(io_areas)

        integration_areas = areas_matching("llm", "ai", "agent", "tool", "adapter", "client", "integration")
        if files_for(integration_areas):
            workflow["Where are external integrations?"] = files_for(integration_areas)

        config_areas = areas_matching("config", "settings")
        config_files = [item for item in inventory.get("important_config_files", [])][:5]
        config_evidence = files_for(config_areas) or (", ".join(f"`{path}`" for path in config_files) if config_files else "")
        if config_evidence:
            workflow["Where is configuration loaded?"] = config_evidence

        test_areas = areas_matching("test")
        test_folders = inventory.get("test_folders", [])
        test_evidence = files_for(test_areas) or (", ".join(f"`{folder}/`" for folder in test_folders) if test_folders else "")
        if test_evidence:
            workflow["Where are the tests?"] = test_evidence

        return workflow

    def _risk(self, title: str, severity: str, file: str, line: int, evidence: str, why: str, fix: str) -> dict[str, Any]:
        return {"title": title, "severity": severity, "file": file, "line": int(line), "evidence": evidence, "why_it_matters": why, "recommended_fix": fix}

    def _recommendation(
        self,
        title: str,
        priority: str,
        files: list[str],
        reason: str,
        acceptance: list[str],
        verification: str,
    ) -> dict[str, Any]:
        return {"title": title, "priority": priority, "files": files, "reason": reason, "acceptance_criteria": acceptance, "verification": verification}

    def _verification_commands(self, inventory: dict[str, Any], dependencies: dict[str, Any]) -> list[str]:
        commands = ["python -m compileall ."]
        if inventory.get("test_files_count", 0):
            commands.append("pytest")
        if "npm" in dependencies.get("package_managers", []):
            commands.append("npm test")
        return commands

    def _agent_context(self, report: dict[str, Any], llm: LLMAnalyzeResult | None = None) -> dict[str, Any]:
        llm = llm or LLMAnalyzeResult(available=False)
        inventory = report["inventory"]
        # Prefer LLM next tasks; fall back to deterministic recommendations.
        recommended_tasks = llm.next_tasks or [
            {
                "title": item["title"],
                "priority": item["priority"],
                "files": item["files"],
                "acceptance_criteria": item["acceptance_criteria"],
                "verification_command": item["verification"],
            }
            for item in report["recommendations"].get("items", [])[:20]
        ]
        risks = llm.risk_analysis or report["risks"].get("items", [])[:60]
        return {
            "project_summary": llm.project_summary or report["project_summary"],
            "detected_stack": sorted(set(inventory.get("detected_languages", []) + inventory.get("detected_frameworks", []) + inventory.get("package_managers", []))),
            "architecture_summary": llm.architecture_explanation or "",
            "important_files": (
                llm.important_files
                or [{"file": path} for path in (inventory.get("important_config_files", []) + inventory.get("entrypoints", []))[:60]]
            ),
            "entrypoints": report["entrypoints"][:40],
            "important_symbols": report["symbols"].get("important_symbols", [])[:80],
            "agent_workflow": llm.agent_workflow or "",
            "onboarding_summary": llm.onboarding_summary or "",
            "risks": risks[:60],
            "recommended_tasks": recommended_tasks[:20],
            "verification_commands": report.get("verification_commands", []),
            "generated_artifacts": [f".mana/analyze/{name}" for name, _ in self._ARTIFACT_PURPOSES],
            "llm_available": llm.available,
            "ignore_rules": inventory.get("ignore_rules", []),
            "last_analyzed_at": report["generated_at"],
        }
