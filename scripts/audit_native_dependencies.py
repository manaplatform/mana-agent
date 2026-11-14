"""User-run source and installed dependency-graph audit for the native migration."""
from __future__ import annotations
import argparse
import ast
import importlib.metadata as metadata
import json
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

def forbidden(name: str) -> bool:
    normalized = canonicalize_name(name)
    return normalized.startswith(("langchain", "langgraph", "langsmith")) or normalized == "faiss-cpu"

def audit_sources(root: Path) -> list[str]:
    failures = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            modules = [node.module or ""] if isinstance(node, ast.ImportFrom) else [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            for module in modules:
                if forbidden(module.split(".")[0]):
                    failures.append(f"{path}:{node.lineno}: forbidden import {module}")
    return failures

def audit_graph(package: str, extras: set[str]) -> tuple[list[dict], list[str]]:
    pending = [(package, extras)]
    visited: dict[str, set[str]] = {}
    graph, failures = [], []
    while pending:
        name, requested = pending.pop()
        key = canonicalize_name(name)
        if key in visited and requested <= visited[key]:
            continue
        visited.setdefault(key, set()).update(requested)
        if forbidden(key):
            failures.append(f"Forbidden resolved dependency: {key}")
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            failures.append(f"Required distribution is not installed: {name}")
            continue
        graph.append({"name": key, "version": distribution.version, "extras": sorted(requested)})
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker is not None and not any(requirement.marker.evaluate({"extra": extra}) for extra in {"", *requested}):
                continue
            pending.append((requirement.name, set(requirement.extras)))
    return graph, failures

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extras", default="dev")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    graph, failures = audit_graph("mana-agent", set(filter(None, options.extras.split(","))))
    failures.extend(audit_sources(options.source))
    for distribution in metadata.distributions():
        if forbidden(distribution.metadata["Name"]):
            failures.append(f"Forbidden installed distribution: {distribution.metadata['Name']}")
    try:
        installed = metadata.distribution("mana-agent")
    except metadata.PackageNotFoundError:
        pass  # The dependency graph already reported the missing package.
    else:
        installed_root = Path(installed.locate_file("mana_agent"))
        if installed_root.is_dir():
            failures.extend(audit_sources(installed_root))
        else:
            failures.append("Installed package source is unavailable for artifact audit")
    result = {"verified": not failures, "graph": graph, "failures": failures}
    rendered = json.dumps(result, indent=2)
    if options.output:
        options.output.write_text(rendered + "\n")
    print(rendered)
    return int(bool(failures))

if __name__ == "__main__":
    raise SystemExit(main())
