from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, List
from pathlib import Path


@dataclass(slots=True)
class CodeSymbol:
    kind: str
    name: str
    signature: str
    docstring: str
    file_path: str
    start_line: int
    end_line: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CodeChunk:
    id: str
    text: str
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str
    symbol_kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Finding:
    rule_id: str
    severity: str
    message: str
    file_path: str
    line: int
    column: int
    architecture_summary: str = ""
    technology_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchHit:
    score: float
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SourceGroup:
    index_dir: str
    subproject_root: str
    sources: list[SearchHit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_dir": self.index_dir,
            "subproject_root": self.subproject_root,
            "sources": [item.to_dict() for item in self.sources],
        }


@dataclass(slots=True)
class AskResponse:
    answer: str
    sources: list[SearchHit]
    source_groups: list[SourceGroup] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "answer": self.answer,
            "sources": [item.to_dict() for item in self.sources],
        }
        if self.source_groups:
            payload["source_groups"] = [item.to_dict() for item in self.source_groups]
        if self.warnings:
            payload["warnings"] = self.warnings
        return payload


@dataclass(slots=True)
class ClassDescriptor:
    name: str
    methods: list[str]
    fields: list[str]
    decorators: list[str]
    bases: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModuleDescriptor:
    module_path: str
    imports: list[str]
    functions: list[str]
    classes: list[ClassDescriptor]
    constants: list[str]
    language: str = "python"
    parse_mode: str = "full"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["classes"] = [item.to_dict() for item in self.classes]
        return payload


@dataclass(slots=True)
class ExportDescriptor:
    source_module: str
    symbol: str
    mechanism: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SubprojectReport:
    root_path: str
    manifest_paths: list[str]
    package_managers: list[str]
    framework_hints: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolInvocationTrace:
    tool_name: str
    args_summary: str
    duration_ms: float
    status: str
    output_preview: str
    # Repo-relative paths a mutation tool (apply_patch/write_file/create_file/delete_file)
    # actually changed. Populated so strict mutation gates can recognize a
    # successful write; empty for non-mutating tools.
    changed_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AskResponseWithTrace(AskResponse):
    mode: str = "classic"
    trace: list[ToolInvocationTrace] = field(default_factory=list)
    route_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = AskResponse.to_dict(self)
        payload["mode"] = self.mode
        payload["trace"] = [item.to_dict() for item in self.trace]
        if self.route_trace:
            payload["route_trace"] = self.route_trace
        return payload


@dataclass(slots=True)
class ProjectStructureReport:
    project_root: str
    frameworks: list[str]
    runtime: str
    package_manager: str
    entrypoints: list[str]
    ci: list[str]
    tech_stack: list[str]
    dependencies_runtime: list[str]
    dependencies_dev: list[str]
    modules: list[ModuleDescriptor]
    exports: list[ExportDescriptor]
    data_structures: list[ClassDescriptor]
    commands: list[str]
    llm_capabilities: list[str]

    subprojects: list[SubprojectReport] = field(default_factory=list)
    directories: list[str] = field(default_factory=list)
    files_by_language: dict[str, list[str]] = field(default_factory=dict)
    language_counts: dict[str, int] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    file_counts: dict[str, int] = field(default_factory=dict)
    discovery_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def _convert(x: Any) -> Any:
            if isinstance(x, list):
                return [_convert(i) for i in x]
            if isinstance(x, dict):
                return {k: _convert(v) for k, v in x.items()}
            # Convert objects that expose to_dict()
            if hasattr(x, "to_dict") and callable(getattr(x, "to_dict")):
                return x.to_dict()
            return x

        return {
            "project_root": self.project_root,
            "frameworks": self.frameworks,
            "runtime": self.runtime,
            "package_manager": self.package_manager,
            "entrypoints": self.entrypoints,
            "ci": self.ci,
            "tech_stack": self.tech_stack,
            "dependencies_runtime": self.dependencies_runtime,
            "dependencies_dev": self.dependencies_dev,
            "modules": _convert(self.modules),
            "exports": _convert(self.exports),
            "data_structures": _convert(self.data_structures),
            "commands": self.commands,
            "llm_capabilities": self.llm_capabilities,
            "subprojects": _convert(self.subprojects),
            "directories": self.directories,
            "files_by_language": self.files_by_language,
            "language_counts": self.language_counts,
            "files": self.files,
            "file_counts": self.file_counts,
            "discovery_stats": self.discovery_stats,
        }

@dataclass(slots=True)
class DependencyEdge:
    source: str
    target: str
    kind: str
    file_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DependencyGraphReport:
    files: List[Path]  
    project_root: str
    package_managers: list[str]
    frameworks: list[str]
    technologies: list[str]
    runtime_dependencies: list[str]
    dev_dependencies: list[str]
    module_edges: list[DependencyEdge]
    dependency_edges: list[DependencyEdge]
    manifests: list[str]
    languages: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "package_managers": self.package_managers,
            "frameworks": self.frameworks,
            "technologies": self.technologies,
            "runtime_dependencies": self.runtime_dependencies,
            "dev_dependencies": self.dev_dependencies,
            "module_edges": [item.to_dict() for item in self.module_edges],
            "dependency_edges": [item.to_dict() for item in self.dependency_edges],
            "manifests": self.manifests,
            "languages": self.languages,
        }

    def to_dot(self) -> str:
        lines: list[str] = ["digraph mana_agent {"]
        lines.append('  rankdir="LR";')
        lines.append('  node [shape="box"];')
        for edge in self.module_edges:
            lines.append(
                f'  "{edge.source}" -> "{edge.target}" [label="{edge.kind}"];'
            )
        for edge in self.dependency_edges:
            lines.append(
                f'  "{edge.source}" -> "{edge.target}" [label="{edge.kind}"];'
            )
        lines.append("}")
        return "\n".join(lines)

    def to_graphml(self) -> str:
        nodes: set[str] = set()
        edges: list[tuple[str, str, str]] = []
        for item in self.module_edges + self.dependency_edges:
            nodes.add(item.source)
            nodes.add(item.target)
            edges.append((item.source, item.target, item.kind))

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            '  <key id="kind" for="edge" attr.name="kind" attr.type="string"/>',
            '  <graph id="G" edgedefault="directed">',
        ]
        for node in sorted(nodes):
            lines.append(f'    <node id="{node}"/>')
        for idx, (source, target, kind) in enumerate(edges):
            lines.append(f'    <edge id="e{idx}" source="{source}" target="{target}">')
            lines.append(f'      <data key="kind">{kind}</data>')
            lines.append("    </edge>")
        lines.append("  </graph>")
        lines.append("</graphml>")
        return "\n".join(lines)


@dataclass(slots=True)
class CodeDescription:
    file_path: str
    language: str
    symbols: list[str]
    summary: str
    entrypoint: bool = False
    symbol_docs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DescribeReport:
    project_root: str
    selected_files: list[str]
    descriptions: list[CodeDescription]
    architecture_summary: str
    tech_summary: str
    chain_steps: list[str]
    architecture_mermaid: str = ""
    architecture_data: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "selected_files": self.selected_files,
            "descriptions": [item.to_dict() for item in self.descriptions],
            "architecture_summary": self.architecture_summary,
            "tech_summary": self.tech_summary,
            "chain_steps": self.chain_steps,
            "architecture_mermaid": self.architecture_mermaid,
            "architecture_data": self.architecture_data,
            "metrics": self.metrics,
        }
