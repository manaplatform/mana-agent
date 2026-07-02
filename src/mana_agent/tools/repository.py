"""Repository-local helper tools for coding agents."""

from __future__ import annotations

import ast
import fnmatch
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_SKIP_DIRS = {
    ".git",
    ".mana",
    ".mana_logs",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    command: list[str]
    status: str
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_rel(repo_root: Path, path: str) -> tuple[Path | None, str | None]:
    raw = str(path or "").strip().replace("\\", "/")
    if not raw or raw.startswith("/") or "\x00" in raw:
        return None, None
    parts = [item for item in raw.split("/") if item not in {"", "."}]
    if any(item == ".." for item in parts):
        return None, None
    target = (repo_root / Path(*parts)).resolve()
    try:
        rel = target.relative_to(repo_root)
    except ValueError:
        return None, None
    return target, rel.as_posix()


def _iter_files(repo_root: Path):
    for path in repo_root.rglob("*"):
        if any(part in _SKIP_DIRS for part in path.relative_to(repo_root).parts):
            continue
        if path.is_file():
            yield path


def _is_binary(path: Path) -> bool:
    try:
        return b"\x00" in path.read_bytes()[:4096]
    except Exception:
        return True


def _matches_file_glob(rel_path: str, pattern: str) -> bool:
    """Match repository file globs with predictable directory recursion.

    ``fnmatch`` treats ``**`` as ordinary ``*`` characters, so patterns like
    ``src/pkg/**/*`` can miss files directly under ``src/pkg``. For tool callers
    the useful contract is simpler: ``dir/**`` and ``dir/**/*`` both mean every
    file under ``dir``.
    """
    rel_path = rel_path.replace("\\", "/").lstrip("./")
    pattern = str(pattern or "**/*").replace("\\", "/").lstrip("./")
    if pattern in {"**", "**/*"}:
        return True
    for suffix in ("/**/*", "/**"):
        if pattern.endswith(suffix):
            prefix = pattern[: -len(suffix)].rstrip("/")
            return rel_path.startswith(f"{prefix}/") if prefix else True
    if "/" not in pattern and fnmatch.fnmatch(Path(rel_path).name, pattern):
        return True
    return PurePosixPath(rel_path).match(pattern)


def list_files(repo_root: Path, *, glob: str = "**/*", limit: int = 200) -> dict[str, Any]:
    """List repository files with deterministic ordering."""

    root = repo_root.resolve()
    pattern = str(glob or "**/*")
    max_items = max(1, min(int(limit or 200), 5000))
    files: list[str] = []
    for path in sorted(_iter_files(root), key=lambda item: item.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()
        if _matches_file_glob(rel, pattern):
            files.append(rel)
            if len(files) >= max_items:
                break
    return {"ok": True, "files": files, "limit": max_items, "truncated": len(files) >= max_items}


def repo_search(
    repo_root: Path,
    *,
    query: str,
    glob: str = "**/*",
    regex: bool = False,
    limit: int = 100,
) -> dict[str, Any]:
    """Search text files in the repository."""

    root = repo_root.resolve()
    needle = str(query or "")
    if not needle:
        return {"ok": False, "error": "query is required", "matches": []}
    max_items = max(1, min(int(limit or 100), 1000))
    pattern = re.compile(needle) if regex else None
    matches: list[dict[str, Any]] = []
    for path in sorted(_iter_files(root), key=lambda item: item.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()
        if not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(path.name, glob)):
            continue
        if _is_binary(path):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(lines, start=1):
            found = bool(pattern.search(line)) if pattern is not None else needle in line
            if found:
                matches.append({"file": rel, "line": line_no, "text": line[:500]})
                if len(matches) >= max_items:
                    return {"ok": True, "matches": matches, "limit": max_items, "truncated": True}
    return {"ok": True, "matches": matches, "limit": max_items, "truncated": False}


def find_symbols(repo_root: Path, *, query: str = "", limit: int = 100) -> dict[str, Any]:
    """Find Python classes/functions/methods using ast."""

    root = repo_root.resolve()
    needle = str(query or "").lower()
    max_items = max(1, min(int(limit or 100), 1000))
    symbols: list[dict[str, Any]] = []
    for path in sorted(_iter_files(root), key=lambda item: item.relative_to(root).as_posix()):
        if path.suffix != ".py" or _is_binary(path):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        parents: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                name = node.name
                kind = "class"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                kind = "function"
            else:
                continue
            if needle and needle not in name.lower():
                continue
            symbols.append({"file": rel, "line": int(getattr(node, "lineno", 1)), "name": name, "kind": kind, "parents": parents})
            if len(symbols) >= max_items:
                return {"ok": True, "symbols": symbols, "limit": max_items, "truncated": True}
    return {"ok": True, "symbols": symbols, "limit": max_items, "truncated": False}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return ""


class _CallGraphVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.scope: list[str] = []
        self.edges: list[dict[str, Any]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Call(self, node: ast.Call) -> Any:
        caller = ".".join(self.scope) if self.scope else "<module>"
        callee = _call_name(node.func)
        if callee:
            self.edges.append(
                {
                    "file": self.file_path,
                    "line": int(getattr(node, "lineno", 1)),
                    "caller": caller,
                    "callee": callee,
                }
            )
        self.generic_visit(node)


def call_graph(repo_root: Path, *, query: str = "", limit: int = 100) -> dict[str, Any]:
    """Build a lightweight Python AST call graph.

    This is a local static-inspection tool, not a semantic vector search. It is
    intentionally conservative: it records syntactic call edges as written and
    does not attempt dynamic dispatch resolution.
    """

    root = repo_root.resolve()
    needle = str(query or "").lower()
    max_items = max(1, min(int(limit or 100), 1000))
    edges: list[dict[str, Any]] = []
    for path in sorted(_iter_files(root), key=lambda item: item.relative_to(root).as_posix()):
        if path.suffix != ".py" or _is_binary(path):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        visitor = _CallGraphVisitor(rel)
        visitor.visit(tree)
        for edge in visitor.edges:
            searchable = f"{edge['file']} {edge['caller']} {edge['callee']}".lower()
            if needle and needle not in searchable:
                continue
            edges.append(edge)
            if len(edges) >= max_items:
                return {"ok": True, "edges": edges, "limit": max_items, "truncated": True}
    return {"ok": True, "edges": edges, "limit": max_items, "truncated": False}


def git_status(repo_root: Path) -> dict[str, Any]:
    completed = subprocess.run(["git", "status", "--short"], cwd=repo_root, capture_output=True, text=True, check=False)
    return {"ok": completed.returncode == 0, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def git_diff(repo_root: Path, *, path: str = "") -> dict[str, Any]:
    cmd = ["git", "diff", "--"]
    if path:
        target, rel = _safe_rel(repo_root.resolve(), path)
        if target is None or rel is None:
            return {
                "ok": False,
                "error_code": "path_outside_repo",
                "tool": "git_diff",
                "path": str(path),
                "error": "invalid path",
                "message": "Path resolves outside the repository root.",
            }
        cmd.append(rel)
    completed = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    return {"ok": completed.returncode == 0, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def _run_check(repo_root: Path, name: str, command: list[str], timeout: int = 120) -> VerificationCheck:
    exe = command[0]
    if shutil.which(exe) is None and not Path(exe).exists():
        return VerificationCheck(name=name, command=command, status="skipped", reason=f"{exe} not found")
    try:
        completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return VerificationCheck(name=name, command=command, status="failed", reason=f"timed out after {timeout}s")
    status = "passed" if completed.returncode == 0 else "failed"
    return VerificationCheck(
        name=name,
        command=command,
        status=status,
        returncode=completed.returncode,
        stdout=completed.stdout[:6000],
        stderr=completed.stderr[:6000],
    )


def verify_project(repo_root: Path, *, quick: bool = False) -> dict[str, Any]:
    """Run standard project verification checks and report skipped tools clearly."""

    root = repo_root.resolve()
    commands: list[tuple[str, list[str]]] = [
        ("pytest", ["pytest", "-q"]),
        ("ruff", ["ruff", "check", "src", "tests"]),
        ("mypy", ["mypy", "src", "tests"]),
        ("import", [sys.executable, "-c", "import mana_agent; print('ok')"]),
        ("cli_help", ["mana-agent", "--help"]),
        ("cli_ask_help", ["mana-agent", "ask", "--help"]),
        ("cli_chat_help", ["mana-agent", "chat", "--help"]),
    ]
    if quick:
        commands = [item for item in commands if item[0] in {"pytest", "import", "cli_help"}]
    checks = [_run_check(root, name, cmd) for name, cmd in commands]
    return {
        "ok": all(item.status in {"passed", "skipped"} for item in checks),
        "checks": [item.to_dict() for item in checks],
        "summary": {
            "passed": sum(1 for item in checks if item.status == "passed"),
            "failed": sum(1 for item in checks if item.status == "failed"),
            "skipped": sum(1 for item in checks if item.status == "skipped"),
        },
    }


def _top_level_dirs(repo_root: Path) -> list[str]:
    root = repo_root.resolve()
    dirs: list[str] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if child.is_dir() and child.name not in _SKIP_DIRS:
            dirs.append(child.name)
    return dirs


def inspect_project_structure(repo_root: Path, *, limit: int = 200) -> dict[str, Any]:
    """Deterministically inspect the project layout (top-level dirs + files).

    Reliable evidence-gathering for "inspect project structure" style steps so
    they never depend on an LLM choosing a valid tool. Mirrors ``ls`` +
    ``list_files`` so the model can summarize the result afterwards.
    """
    root = repo_root.resolve()
    listing = list_files(root, glob="**/*", limit=limit)
    return {
        "ok": True,
        "tool": "inspect_project_structure",
        "root": str(root),
        "directories": _top_level_dirs(root),
        "files": listing["files"],
        "truncated": listing["truncated"],
    }


def _list_under(repo_root: Path, prefix: str, *, tool: str, limit: int) -> dict[str, Any]:
    """List files whose repo-relative path is under ``prefix`` (e.g. ``src``)."""
    root = repo_root.resolve()
    max_items = max(1, min(int(limit or 500), 5000))
    needle = prefix.strip("/") + "/"
    files: list[str] = []
    for path in sorted(_iter_files(root), key=lambda item: item.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(needle):
            files.append(rel)
            if len(files) >= max_items:
                break
    return {"ok": True, "tool": tool, "files": files, "limit": max_items, "truncated": len(files) >= max_items}


def explore_src(repo_root: Path, *, limit: int = 500) -> dict[str, Any]:
    """List files under ``src/`` deterministically."""
    return _list_under(repo_root, "src", tool="explore_src", limit=limit)


def inspect_tests(repo_root: Path, *, limit: int = 500) -> dict[str, Any]:
    """List files under ``tests/`` deterministically."""
    return _list_under(repo_root, "tests", tool="inspect_tests", limit=limit)


def verify_file_created(repo_root: Path, *, path: str, max_lines: int = 20) -> dict[str, Any]:
    """Verify a file exists under the repo root and return its first lines.

    Returns structured errors (with ``error_code``) consistent with the rest of
    the toolset so verification steps can branch reliably on the outcome.
    """
    target, rel = _safe_rel(repo_root.resolve(), path)
    if target is None or rel is None:
        return {
            "ok": False,
            "error_code": "path_outside_repo",
            "tool": "verify_file_created",
            "path": str(path),
            "message": "Path resolves outside the repository root.",
        }
    if not target.exists() or not target.is_file():
        return {
            "ok": False,
            "error_code": "file_not_found",
            "tool": "verify_file_created",
            "path": rel,
            "resolved_path": str(target),
            "message": "File does not exist under repository root.",
        }
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError) as exc:
        return {
            "ok": False,
            "error_code": "read_failed",
            "tool": "verify_file_created",
            "path": rel,
            "resolved_path": str(target),
            "message": str(exc),
        }
    capped = max(1, int(max_lines or 20))
    return {
        "ok": True,
        "tool": "verify_file_created",
        "path": rel,
        "resolved_path": str(target),
        "exists": True,
        "line_count": len(lines),
        "preview": "\n".join(lines[:capped]),
    }


def dumps_tool_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)
