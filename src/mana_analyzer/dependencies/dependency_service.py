from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path
from typing import Iterable

from mana_analyzer.analysis.models import DependencyEdge, DependencyGraphReport
from mana_analyzer.models import DependencyPackageRef
from mana_analyzer.utils.io import iter_source_files, language_for_path


FRAMEWORK_SIGNALS = {
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "typer": "Typer",
    "click": "Click",
    "react": "React",
    "vite": "Vite",
    "next": "Next.js",
    "vue": "Vue",
    "express": "Express",
    "langchain": "LangChain",
    "flutter": "Flutter",
    "flutter_test": "Flutter",
    "@nestjs/common": "NestJS",
    "@nestjs/core": "NestJS",
    "nestjs": "NestJS",
}


def _normalize_dep_name(raw: str) -> str:
    item = str(raw).strip().lower()
    if not item:
        return ""
    item = re.split(r"[<>=!~\[]", item)[0]
    return item.strip()


def _parse_python_req(raw: str) -> tuple[str, str]:
    item = str(raw).split(";", 1)[0].strip()
    match = re.match(r"^\s*([A-Za-z0-9_.\-]+)(.*)$", item)
    if not match:
        return "", ""
    return _normalize_dep_name(match.group(1)), match.group(2).strip()


def _detect_exact_version(spec: str) -> str | None:
    match = re.match(r"^\s*==\s*v?(\d+\.\d+\.\d+(?:[.\-+][0-9A-Za-z.\-+]+)?)\s*$", spec or "")
    return match.group(1) if match else None


def _looks_external(name: str) -> bool:
    return bool(name) and not name.startswith(".")


class DependencyService:
    """Discover source files, manifests, dependency inventory, and import graph edges."""

    def analyze(self, root: str | Path) -> DependencyGraphReport:
        project_root = Path(root).resolve()
        if project_root.is_file():
            project_root = project_root.parent

        files = iter_source_files(project_root)
        manifests = self._discover_manifests(project_root)
        runtime_deps, dev_deps, package_managers = self._manifest_dependencies(manifests)

        module_edges, dependency_edges = self._build_import_edges(project_root, files)
        observed_imports = {edge.target for edge in dependency_edges}
        frameworks = self._match_frameworks(runtime_deps | dev_deps | observed_imports)
        technologies = sorted(set(frameworks) | package_managers)

        return DependencyGraphReport(
            files=files,
            project_root=str(project_root),
            package_managers=sorted(package_managers),
            frameworks=frameworks,
            technologies=technologies,
            runtime_dependencies=sorted(runtime_deps),
            dev_dependencies=sorted(dev_deps),
            module_edges=sorted(module_edges, key=lambda e: (e.source, e.target, e.kind)),
            dependency_edges=sorted(dependency_edges, key=lambda e: (e.source, e.target, e.kind)),
            manifests=sorted(str(item.relative_to(project_root)) for item in manifests),
            languages=self._detect_languages(files),
        )

    def collect_inventory(self, target_path: str | Path) -> list[DependencyPackageRef]:
        root = Path(target_path).resolve()
        if root.is_file():
            root = root.parent

        refs: list[DependencyPackageRef] = []
        for manifest in self._discover_manifests(root):
            runtime, dev = self._manifest_inventory(manifest)
            for ref in runtime + dev:
                try:
                    ref.manifest_path = str(manifest.relative_to(root))
                except ValueError:
                    ref.manifest_path = str(manifest)
                refs.append(ref)

        seen: set[tuple[str, str, str, str, str]] = set()
        deduped: list[DependencyPackageRef] = []
        for ref in refs:
            key = (ref.ecosystem, ref.name, ref.scope, ref.manifest_path, ref.version_spec_raw)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ref)
        return sorted(deduped, key=lambda item: (item.ecosystem, item.name, item.scope, item.manifest_path))

    @staticmethod
    def _detect_languages(files: list[Path]) -> list[str]:
        return sorted({lang for lang in (language_for_path(item) for item in files) if lang and lang != "unknown"})

    @staticmethod
    def _discover_manifests(root: Path) -> list[Path]:
        names = {
            "pyproject.toml",
            "requirements.txt",
            "package.json",
            "go.mod",
            "Cargo.toml",
            "pubspec.yaml",
            "composer.json",
            "Gemfile",
            "nest-cli.json",
        }
        excluded = {".git", ".venv", "venv", "node_modules", "dist", "build", "__pycache__", ".mana"}
        manifests: list[Path] = []
        for path in root.rglob("*"):
            if not path.is_file() or path.name not in names:
                continue
            if any(part in excluded for part in path.relative_to(root).parts):
                continue
            manifests.append(path)
        return sorted(manifests)

    def _manifest_dependencies(self, manifests: Iterable[Path]) -> tuple[set[str], set[str], set[str]]:
        runtime: set[str] = set()
        dev: set[str] = set()
        managers: set[str] = set()
        for manifest in manifests:
            rt, dv = self._manifest_dependency_names(manifest)
            runtime.update(rt)
            dev.update(dv)
            managers.add(self._package_manager_for_manifest(manifest))
        return runtime, dev, managers

    def _manifest_dependency_names(self, manifest: Path) -> tuple[set[str], set[str]]:
        runtime_refs, dev_refs = self._manifest_inventory(manifest)
        return {ref.name for ref in runtime_refs}, {ref.name for ref in dev_refs}

    def _manifest_inventory(self, manifest: Path) -> tuple[list[DependencyPackageRef], list[DependencyPackageRef]]:
        name = manifest.name
        if name == "pyproject.toml":
            return self._inventory_pyproject(manifest)
        if name == "requirements.txt":
            return self._inventory_requirements(manifest)
        if name == "package.json":
            return self._inventory_package_json(manifest)
        if name == "pubspec.yaml":
            return self._inventory_pubspec(manifest)
        return [], []

    @staticmethod
    def _package_manager_for_manifest(manifest: Path) -> str:
        return {
            "pyproject.toml": "pip",
            "requirements.txt": "pip",
            "package.json": "npm",
            "go.mod": "go",
            "Cargo.toml": "cargo",
            "pubspec.yaml": "pub",
            "composer.json": "composer",
            "Gemfile": "bundler",
            "nest-cli.json": "npm",
        }.get(manifest.name, manifest.name)

    @staticmethod
    def _inventory_pyproject(path: Path) -> tuple[list[DependencyPackageRef], list[DependencyPackageRef]]:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        project = payload.get("project", {})
        runtime: list[DependencyPackageRef] = []
        dev: list[DependencyPackageRef] = []
        for raw in project.get("dependencies", []) or []:
            dep, spec = _parse_python_req(raw)
            if dep and dep != "python":
                runtime.append(DependencyPackageRef(dep, "PyPI", "runtime", str(path), "pip", spec, _detect_exact_version(spec)))
        optional = project.get("optional-dependencies", {}) or {}
        for raw in optional.get("dev", []) or []:
            dep, spec = _parse_python_req(raw)
            if dep and dep != "python":
                dev.append(DependencyPackageRef(dep, "PyPI", "dev", str(path), "pip", spec, _detect_exact_version(spec)))
        return runtime, dev

    @staticmethod
    def _inventory_requirements(path: Path) -> tuple[list[DependencyPackageRef], list[DependencyPackageRef]]:
        refs: list[DependencyPackageRef] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            item = line.strip()
            if not item or item.startswith(("#", "-")):
                continue
            dep, spec = _parse_python_req(item)
            if dep:
                refs.append(DependencyPackageRef(dep, "PyPI", "runtime", str(path), "pip", spec, _detect_exact_version(spec)))
        return refs, []

    @staticmethod
    def _inventory_package_json(path: Path) -> tuple[list[DependencyPackageRef], list[DependencyPackageRef]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runtime = [
            DependencyPackageRef(_normalize_dep_name(dep), "npm", "runtime", str(path), "npm", str(spec), None)
            for dep, spec in (payload.get("dependencies", {}) or {}).items()
            if _normalize_dep_name(dep)
        ]
        dev = [
            DependencyPackageRef(_normalize_dep_name(dep), "npm", "dev", str(path), "npm", str(spec), None)
            for dep, spec in (payload.get("devDependencies", {}) or {}).items()
            if _normalize_dep_name(dep)
        ]
        return runtime, dev

    @staticmethod
    def _inventory_pubspec(path: Path) -> tuple[list[DependencyPackageRef], list[DependencyPackageRef]]:
        runtime: list[DependencyPackageRef] = []
        dev: list[DependencyPackageRef] = []
        section = ""
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped in {"dependencies:", "dev_dependencies:"}:
                section = stripped[:-1]
                continue
            if section and ":" in stripped and not stripped.startswith("-"):
                dep = _normalize_dep_name(stripped.split(":", 1)[0])
                if not dep:
                    continue
                scope = "dev" if section == "dev_dependencies" else "runtime"
                target = dev if scope == "dev" else runtime
                target.append(DependencyPackageRef(dep, "Pub", scope, str(path), "pub", "", None))
        return runtime, dev

    @staticmethod
    def _module_name(root: Path, path: Path) -> str:
        relative = path.relative_to(root)
        parts = list(relative.parts)
        if parts and parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        if parts and parts[-1] in {"__init__", "index"}:
            parts = parts[:-1]
        return ".".join(parts) if parts else str(relative)

    def _build_import_edges(self, root: Path, files: list[Path]) -> tuple[list[DependencyEdge], list[DependencyEdge]]:
        module_edges: list[DependencyEdge] = []
        dependency_edges: list[DependencyEdge] = []
        file_to_module = {path: self._module_name(root, path) for path in files}
        known_modules = set(file_to_module.values())
        known_roots = {part for module in known_modules for part in module.split(".") if part}

        for file_path, module_name in file_to_module.items():
            imports = self._extract_imports(file_path)
            for item in imports:
                root_name = item.split("/")[0].split(".")[0]
                if not root_name:
                    continue
                is_internal = item in known_modules or root_name in known_modules or root_name in known_roots
                if is_internal:
                    target = item if item in known_modules else next(
                        (module for module in known_modules if module.endswith(item) or module.endswith(f".{item}")),
                        root_name,
                    )
                    module_edges.append(DependencyEdge(module_name, target, "module-import", str(file_path)))
                elif _looks_external(root_name):
                    dependency_edges.append(DependencyEdge(module_name, root_name, "external-import", str(file_path)))
        return module_edges, dependency_edges

    @staticmethod
    def _extract_imports(path: Path) -> list[str]:
        if path.suffix == ".py":
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
            except SyntaxError:
                return []
            imports: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            return imports
        if path.suffix.lower() in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
            source = path.read_text(encoding="utf-8", errors="ignore")
            return re.findall(r"(?:from|require\()\s*['\"]([^'\"]+)['\"]", source)
        return []

    @staticmethod
    def _match_frameworks(names: set[str]) -> list[str]:
        found = set()
        lowered = {_normalize_dep_name(item) for item in names}
        for signal, label in FRAMEWORK_SIGNALS.items():
            if signal in lowered:
                found.add(label)
        return sorted(found)
