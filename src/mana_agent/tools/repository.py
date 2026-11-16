"""Repository-local helper tools for coding agents."""

from __future__ import annotations

import ast
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from mana_agent.tools.apply_patch import safe_apply_patch
from mana_agent.multi_agent.tools import git_tools

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
    selection_reason: str = ""
    duration_ms: float = 0.0
    timed_out: bool = False
    failure_code: str = ""
    affected_files: list[str] | None = None

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


def _is_skipped_dir(dirname: str) -> bool:
    return (
        dirname in _SKIP_DIRS
        or dirname.startswith(".venv")
        or dirname.startswith("venv")
        or (dirname.startswith("__") and dirname.endswith("__"))
    )


def _iter_files(repo_root: Path):
    for root, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if not _is_skipped_dir(d)]
        for filename in filenames:
            yield Path(root) / filename


def _scoped_files(root: Path, scope: tuple[str, ...] | None) -> Iterator[Path]:
    if scope is None:
        yield from _iter_files(root)
        return
    for relative in dict.fromkeys(scope):
        target, _ = _safe_rel(root, relative)
        if target is None or target != root / relative or any(
            _is_skipped_dir(part) for part in Path(relative).parts
        ):
            continue
        if target.is_file():
            yield target
        elif target.is_dir():
            yield from _iter_files(target)


def _glob_root(glob: str) -> str:
    parts = glob.replace("\\", "/").removeprefix("./").split("/")
    prefix: list[str] = []
    for part in parts[:-1]:
        if not part or part in {".", ".."} or any(token in part for token in "*?["):
            break
        prefix.append(part)
    return "/".join(prefix)


def _discovery_files(root: Path, glob: str, scope: tuple[str, ...] | None) -> list[Path]:
    prefix = _glob_root(glob)
    if prefix:
        if scope is None:
            scope = (prefix,)
        else:
            scope = tuple(prefix if prefix.startswith(f"{path}/") else path for path in scope
                          if prefix == path or prefix.startswith(f"{path}/") or path.startswith(f"{prefix}/"))
    return sorted(set(_scoped_files(root, scope)), key=lambda path: path.relative_to(root).as_posix())


def _searchable(root: Path, path: Path) -> bool:
    return (not path.is_symlink() and path.resolve().is_relative_to(root)
            and not any(part.startswith(".env") or part in {
                ".ssh", "secrets.toml", "credentials.json", "id_rsa", "id_ed25519",
            } for part in path.relative_to(root).parts))


def selected_file_versions(root: Path, paths: list[str]) -> dict[str, str] | None:
    """Version known files without reading their contents or enumerating a directory."""
    if not isinstance(paths, list) or not paths or not all(isinstance(path, str) for path in paths):
        return None
    versions: dict[str, str] = {}
    for path in paths:
        target, relative = _safe_rel(root, path)
        if target is None or relative is None or not _searchable(root, target) or not target.is_file():
            return None
        try:
            stat = target.stat()
        except OSError:
            return None
        versions[relative] = f"{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}"
    return versions


DISCOVERY_DESCRIPTION = (
    "Select relevant relative files/directories in paths, or a targeted glob (such as an owning "
    "directory or a distinctive filename). Start with the known file or symbol, then follow observed "
    "callers, tests, and dependencies as needed. An unrestricted search requires repository_reason "
    "identifying the requested audit, systemic evidence, or what a narrower search failed to find."
)


def _selected_discovery_scope(args: dict[str, Any], *, repository_search_allowed: bool = True) -> tuple[str, ...] | None:
    paths = args.get("paths") or []
    if not isinstance(paths, (list, tuple)):
        raise ValueError("Discovery paths must be an array of relative files or directories")
    normalized: list[str] = []
    for raw in paths:
        if not isinstance(raw, str):
            raise ValueError("Discovery paths must be relative strings")
        path = raw.strip().replace("\\", "/").removeprefix("./").rstrip("/")
        parsed = PurePosixPath(path)
        if (not parsed.parts or parsed.is_absolute() or ".." in parsed.parts
                or any(char in path for char in "\x00:*?[")):
            raise ValueError("Discovery paths must name files or directories inside the repository")
        normalized.append(parsed.as_posix())

    glob = str(args.get("glob") or "**/*").replace("\\", "/")
    if glob.startswith("/") or ".." in glob.split("/") or "\x00" in glob or ":" in glob:
        raise ValueError("Discovery globs must be repository-relative")
    if normalized:
        return tuple(dict.fromkeys(normalized))
    if _glob_root(glob):
        return None  # Traversal is anchored at the literal directory prefix.
    name = glob.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    if any(char.isalnum() for char in re.sub(r"\[[^]]*\]", "", stem)):
        return None  # Only matching candidates are opened; names may be visited.
    reason = args.get("repository_reason")
    if isinstance(reason, str) and reason.strip():
        if not repository_search_allowed:
            raise ValueError("Repository-wide search is outside the parent task's discovery scope")
        return None
    raise ValueError(DISCOVERY_DESCRIPTION)


def validate_repository_discovery(args: dict[str, Any]) -> None:
    """Check model-selected discovery arguments without reading the checkout."""
    _selected_discovery_scope(args)


def validate_delegated_discovery(tool: str, args: dict[str, Any]) -> None:
    """Fence scoped parent jobs before handing their arguments to a tool worker."""
    if tool in {"repo_search", "list_files", "find_symbols", "call_graph"}:
        _selected_discovery_scope(args, repository_search_allowed=False)
    elif tool in {"repo_batch_search", "batch_search_files"}:
        for entry in args.get("patterns") or []:
            _selected_discovery_scope(entry, repository_search_allowed=False)
    elif tool == "batch_list_files":
        for entry in args.get("queries") or ["**/*"]:
            spec = {"glob": entry} if isinstance(entry, str) else {**entry, "glob": entry.get("pattern") or entry.get("glob")}
            _selected_discovery_scope(spec, repository_search_allowed=False)


def list_repository_directories(root: Path) -> dict[str, Any]:
    return {"ok": True, "directories": sorted(
        path.name for path in root.iterdir()
        if not path.name.startswith(".") and not _is_skipped_dir(path.name)
        and not path.is_symlink() and path.is_dir()
    )}


def execute_repository_discovery(
    root: Path, tool: str, args: dict[str, Any], *, repository_search_allowed: bool = True,
) -> dict[str, Any]:
    """Execute a model-selected discovery operation; missing scope never triggers a scan."""
    if tool in {"repo_batch_search", "batch_search_files"}:
        result = repo_batch_search(root, patterns=args.get("patterns") or [], model_selected=True,
                                   repository_search_allowed=repository_search_allowed,
                                   default_limit=50 if tool == "batch_search_files" else 20)
        if tool == "batch_search_files":
            result["total_queries"] = len(result["results"])
        return result
    if tool == "batch_list_files":
        from mana_agent.tools.batch_file_ops import batch_list_files
        return batch_list_files(root, queries=args.get("queries") or [], limit=args.get("limit") or 200,
                                model_selected=True, repository_search_allowed=repository_search_allowed)
    try:
        options = {"glob": str(args.get("glob") or "**/*"),
                   "scope": _selected_discovery_scope(args, repository_search_allowed=repository_search_allowed)}
        if tool == "list_files":
            return list_files(root, limit=int(args.get("limit") or 200), **options)
        query = str(args.get("query") or "")
        limit = int(args.get("limit") or 100)
        if tool == "repo_search":
            return repo_search(root, query=query, regex=bool(args.get("regex")), limit=limit, **options)
        if tool == "find_symbols":
            return find_symbols(root, query=query, limit=limit, **options)
        if tool == "call_graph":
            return call_graph(root, query=query, limit=limit, **options)
        raise ValueError("Unknown repository discovery tool")
    except (ValueError, re.error) as exc:
        return {"ok": False, "error": str(exc), "error_code": "INVALID_DISCOVERY_SELECTION"}


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return b"\x00" in stream.read(4096)
    except Exception:
        return True


def _matches_file_glob(rel_path: str, pattern: str) -> bool:
    """Match repository file globs with predictable directory recursion.

    ``fnmatch`` treats ``**`` as ordinary ``*`` characters, so patterns like
    ``src/pkg/**/*`` can miss files directly under ``src/pkg``. For tool callers
    the useful contract is simpler: ``dir/**`` and ``dir/**/*`` both mean every
    file under ``dir``.
    """
    pattern = str(pattern or "**/*").replace("\\", "/").removeprefix("./")
    if pattern in {"**", "**/*"}:
        return True
    if "/" not in pattern:
        return fnmatch.fnmatch(PurePosixPath(rel_path).name, pattern)
    pattern_parts = pattern.split("/")
    if pattern_parts[-1] == "**":
        pattern_parts.append("*")
    path_parts = rel_path.split("/")

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def match(path_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)
        part = pattern_parts[pattern_index]
        if part == "**":
            return any(match(index, pattern_index + 1) for index in range(path_index, len(path_parts) + 1))
        return (path_index < len(path_parts) and fnmatch.fnmatchcase(path_parts[path_index], part)
                and match(path_index + 1, pattern_index + 1))

    return match(0, 0)


def list_files(repo_root: Path, *, glob: str = "**/*", limit: int = 200,
               scope: tuple[str, ...] | None = None) -> dict[str, Any]:
    """List repository files with deterministic ordering."""

    root = repo_root.resolve()
    pattern = str(glob or "**/*")
    max_items = max(1, min(int(limit or 200), 5000))
    files: list[str] = []
    for path in _discovery_files(root, pattern, scope):
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
    scope: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Search text files in the repository."""

    root = repo_root.resolve()
    needle = str(query or "")
    if not needle:
        return {"ok": False, "error": "query is required", "matches": []}
    max_items = max(1, min(int(limit or 100), 1000))
    pattern = re.compile(needle) if regex else None
    matches: list[dict[str, Any]] = []
    for path in _discovery_files(root, glob, scope):
        rel = path.relative_to(root).as_posix()
        if not _matches_file_glob(rel, glob):
            continue
        if not _searchable(root, path):
            continue
        if _is_binary(path):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for line_no, line in enumerate(lines, start=1):
            found = bool(pattern.search(line)) if pattern is not None else needle in line
            if found:
                from mana_agent.utils.redaction import redact_secrets
                matches.append({"file": rel, "line": line_no, "text": redact_secrets(line[:500])})
                if len(matches) >= max_items:
                    return {"ok": True, "matches": matches, "limit": max_items, "truncated": True}
    return {"ok": True, "matches": matches, "limit": max_items, "truncated": False}


def repo_batch_read(
    repo_root: Path,
    *,
    files: list[str],
    max_files: int = 20,
    max_bytes: int = 300_000,
    max_bytes_per_file: int = 80_000,
) -> dict[str, Any]:
    """Read multiple repository text files with per-file errors and caps."""

    root = repo_root.resolve()
    requested = list(files or [])[: max(1, min(int(max_files or 20), 100))]
    remaining = max(1, min(int(max_bytes or 300_000), 2_000_000))
    per_file_cap = max(1000, min(int(max_bytes_per_file or 80_000), remaining))
    out: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in requested:
        target, rel = _safe_rel(root, str(raw))
        if target is None or rel is None:
            err = {"path": str(raw), "ok": False, "error": "path_outside_repo"}
            errors.append(err)
            out.append(err)
            continue
        if rel in seen:
            continue
        seen.add(rel)
        if not target.exists() or not target.is_file():
            err = {"path": rel, "ok": False, "error": "file_not_found"}
            errors.append(err)
            out.append(err)
            continue
        if _is_binary(target):
            err = {"path": rel, "ok": False, "error": "binary_file"}
            errors.append(err)
            out.append(err)
            continue
        try:
            size = target.stat().st_size
            with target.open("rb") as stream:
                chunk = stream.read(min(per_file_cap, remaining))
            truncated = size > len(chunk)
            content = chunk.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
            out.append(
                {
                    "path": rel,
                    "ok": True,
                    "content": content,
                    "truncated": bool(truncated),
                    "bytes_read": len(chunk),
                    "file_size_bytes": size,
                }
            )
            remaining -= len(chunk)
            if remaining <= 0:
                break
        except UnicodeDecodeError:
            err = {"path": rel, "ok": False, "error": "decode_error"}
            errors.append(err)
            out.append(err)
        except Exception as exc:
            err = {"path": rel, "ok": False, "error": str(exc)}
            errors.append(err)
            out.append(err)
    return {"ok": not errors, "files": out, "errors": errors, "truncated": remaining <= 0}


def repo_batch_search(
    repo_root: Path, *, patterns: list[dict[str, Any]], model_selected: bool = False, default_limit: int = 20,
    repository_search_allowed: bool = True,
) -> dict[str, Any]:
    """Share traversal and reads between queries with the same selected roots."""
    from mana_agent.utils.redaction import redact_secrets

    root = repo_root.resolve()
    grouped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    candidates: dict[tuple[str, tuple[str, ...] | None], list[tuple[dict[str, Any], Any]]] = {}
    for index, item in enumerate((patterns or [])[:50]):
        item = item if isinstance(item, dict) else {}
        query = str(item.get("query") or "")
        glob = str(item.get("glob") or "**/*")
        regex = bool(item.get("regex", False))
        row = {"index": index, "query": query, "glob": glob, "regex": regex, "matches": [], "truncated": False}
        grouped.append(row)
        try:
            if not query:
                raise ValueError("query is required")
            scope = (_selected_discovery_scope(item, repository_search_allowed=repository_search_allowed)
                     if model_selected else None)
            row.update(ok=True, limit=max(1, min(int(item.get("limit") or default_limit), 1000)))
            matcher = re.compile(query).search if regex else lambda line, term=query: term in line
            candidates.setdefault((_glob_root(glob), scope), []).append((row, matcher))
        except (TypeError, ValueError, re.error) as exc:
            row.update(ok=False, error=str(exc))
            errors.append(row)
    for (prefix, scope), queries in candidates.items():
        for path in _discovery_files(root, f"{prefix}/**" if prefix else "**/*", scope):
            if all(row["truncated"] for row, _ in queries):
                break
            rel = path.relative_to(root).as_posix()
            active = [(row, matcher) for row, matcher in queries
                      if not row["truncated"] and _matches_file_glob(rel, row["glob"])]
            if not active or not _searchable(root, path) or _is_binary(path):
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for row, matcher in active:
                for number, line in enumerate(lines, 1):
                    if matcher(line):
                        row["matches"].append({"file": rel, "line": number, "text": redact_secrets(line[:500])})
                        if len(row["matches"]) >= row["limit"]:
                            row["truncated"] = True
                            break
    return {"ok": not errors, "results": grouped, "errors": errors}


_DESTRUCTIVE_SCRIPT_PATTERNS = (
    "rm -rf",
    "git reset --hard",
    "git checkout --",
    "mkfs",
    "shutdown",
    "reboot",
    "sudo ",
)


def run_script_once(repo_root: Path, *, script: str, cwd: str | None = None, timeout: int = 120) -> dict[str, Any]:
    """Execute one non-destructive shell script and return structured output."""

    text = str(script or "")
    lowered = text.lower()
    if any(pattern in lowered for pattern in _DESTRUCTIVE_SCRIPT_PATTERNS):
        return {"ok": False, "returncode": None, "stdout": "", "stderr": "script blocked by safety policy", "duration_ms": 0.0}
    root = repo_root.resolve()
    workdir = root
    if cwd:
        target, _rel = _safe_rel(root, cwd)
        if target is None or not target.exists() or not target.is_dir():
            return {"ok": False, "returncode": None, "stdout": "", "stderr": "cwd must be an existing repository directory", "duration_ms": 0.0}
        workdir = target
    started = time.perf_counter()
    try:
        from mana_agent.multi_agent.agents.tool_runner_agent import current_operation
        from mana_agent.utils.process_stream import run_streaming_process
        from mana_agent.utils.shell_argv import local_shell_argv
        native = current_operation() is not None
        runner = run_streaming_process if native else subprocess.run
        completed = runner(
            local_shell_argv(text) if native else text,
            cwd=workdir,
            shell=not native,
            check=False,
            timeout=max(1, min(int(timeout or 120), 900)),
            capture_output=True,
            text=True,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout[:12000],
            "stderr": completed.stderr[:12000],
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "cwd": str(workdir),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": (exc.stdout or "")[:12000] if isinstance(exc.stdout, str) else "",
            "stderr": f"script timed out after {timeout}s",
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "cwd": str(workdir),
        }


def apply_patch_batch(repo_root: Path, *, patches: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate and apply multiple Codex patches in one call."""

    patch_rows = list(patches or [])[:20]
    validation: list[dict[str, Any]] = []
    for index, item in enumerate(patch_rows):
        patch_text = str(item.get("patch", "") if isinstance(item, dict) else "")
        result = safe_apply_patch(repo_root=repo_root, patch=patch_text, check_only=True)
        validation.append({"index": index, **result})
        if not result.get("ok"):
            return {"ok": False, "stage": "validate", "results": validation, "changed_files": [], "error_index": index}
    results: list[dict[str, Any]] = []
    changed: list[str] = []
    for index, item in enumerate(patch_rows):
        patch_text = str(item.get("patch", "") if isinstance(item, dict) else "")
        result = safe_apply_patch(
            repo_root=repo_root,
            patch=patch_text,
            check_only=False,
            action_approval_id=str(item.get("action_approval_id") or "") if isinstance(item, dict) else "",
        )
        results.append({"index": index, **result})
        changed.extend(str(path) for path in result.get("files_changed") or [])
        if not result.get("ok"):
            return {"ok": False, "stage": "apply", "results": results, "changed_files": sorted(set(changed)), "error_index": index}
    return {"ok": True, "results": results, "changed_files": sorted(set(changed))}


def find_symbols(repo_root: Path, *, query: str = "", limit: int = 100, glob: str = "**/*",
                 scope: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Find Python classes/functions/methods using ast."""

    root = repo_root.resolve()
    needle = str(query or "").lower()
    max_items = max(1, min(int(limit or 100), 1000))
    symbols: list[dict[str, Any]] = []
    for path in _discovery_files(root, glob, scope):
        if (path.suffix != ".py" or not _matches_file_glob(path.relative_to(root).as_posix(), glob)
                or not _searchable(root, path) or _is_binary(path)):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError, UnicodeDecodeError):
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


def call_graph(repo_root: Path, *, query: str = "", limit: int = 100, glob: str = "**/*",
               scope: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Build a lightweight Python AST call graph.

    This is a local static-inspection tool, not a semantic vector search. It is
    intentionally conservative: it records syntactic call edges as written and
    does not attempt dynamic dispatch resolution.
    """

    root = repo_root.resolve()
    needle = str(query or "").lower()
    max_items = max(1, min(int(limit or 100), 1000))
    edges: list[dict[str, Any]] = []
    for path in _discovery_files(root, glob, scope):
        if (path.suffix != ".py" or not _matches_file_glob(path.relative_to(root).as_posix(), glob)
                or not _searchable(root, path) or _is_binary(path)):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError, UnicodeDecodeError):
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
    return git_tools.status(repo_path=repo_root, short=True)


def git_diff(repo_root: Path, *, path: str = "") -> dict[str, Any]:
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
        path = rel
    return git_tools.diff(repo_path=repo_root, path=path)


def _run_check(
    repo_root: Path,
    name: str,
    command: list[str],
    timeout: int = 120,
    *,
    selection_reason: str = "project verification policy",
    affected_files: list[str] | None = None,
) -> VerificationCheck:
    exe = command[0]
    if shutil.which(exe) is None and not Path(exe).exists():
        return VerificationCheck(
            name=name,
            command=command,
            status="skipped",
            reason=f"{exe} not found",
            selection_reason=selection_reason,
            failure_code="verification_tool_missing",
            affected_files=affected_files or [],
        )
    started = time.perf_counter()
    try:
        from mana_agent.multi_agent.agents.tool_runner_agent import current_operation
        from mana_agent.utils.process_stream import run_streaming_process
        runner = run_streaming_process if current_operation() is not None else subprocess.run
        completed = runner(command, cwd=repo_root, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return VerificationCheck(
            name=name,
            command=command,
            status="failed",
            reason=f"timed out after {timeout}s",
            stdout=str(exc.stdout or "")[:6000],
            stderr=str(exc.stderr or "")[:6000],
            selection_reason=selection_reason,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            timed_out=True,
            failure_code="verification_timeout",
            affected_files=affected_files or [],
        )
    status = "passed" if completed.returncode == 0 else "failed"
    return VerificationCheck(
        name=name,
        command=command,
        status=status,
        returncode=completed.returncode,
        stdout=completed.stdout[:6000],
        stderr=completed.stderr[:6000],
        selection_reason=selection_reason,
        duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
        failure_code="" if completed.returncode == 0 else "verification_command_failed",
        affected_files=affected_files or [],
    )


def verify_project(
    repo_root: Path,
    *,
    quick: bool = False,
    changed_files: list[str] | None = None,
) -> dict[str, Any]:
    """Run standard project verification checks and report skipped tools clearly."""

    root = repo_root.resolve()
    affected = [str(path).replace("\\", "/").lstrip("./") for path in (changed_files or []) if str(path).strip()]
    docs_only = bool(affected) and all(path.lower().endswith((".md", ".markdown", ".txt", ".rst")) for path in affected)
    if docs_only:
        from mana_agent.agent.verification_planner import verify_documentation_changes

        artifact_result = verify_documentation_changes(repo_root=root, changed_files=affected)
        return {
            "ok": artifact_result.ok,
            "verification_class": "documentation",
            "checks": list(artifact_result.checks),
            "selected_commands": ["verify_changed_artifacts"],
            "skipped_checks": list(artifact_result.skipped_checks),
            "affected_files": affected,
            "failure_code": artifact_result.failure_code,
            "summary": {"passed": int(artifact_result.ok), "failed": int(not artifact_result.ok), "skipped": len(artifact_result.skipped_checks)},
        }
    commands: list[tuple[str, list[str], str]] = [
        ("pytest", ["pytest", "-q"], "broad project behavior verification"),
        ("ruff", ["ruff", "check", "src", "tests"], "project lint verification"),
        ("mypy", ["mypy", "src", "tests"], "project type verification"),
        ("import", [sys.executable, "-c", "import mana_agent; print('ok')"], "package import smoke check"),
        ("cli_help", ["mana-agent", "--help"], "CLI startup smoke check"),
        ("cli_ask_help", ["mana-agent", "ask", "--help"], "ask command smoke check"),
        ("cli_chat_help", ["mana-agent", "chat", "--help"], "chat command smoke check"),
    ]
    if quick:
        commands = [item for item in commands if item[0] in {"import", "cli_help"}]
    checks = [
        _run_check(root, name, cmd, selection_reason=reason, affected_files=affected)
        for name, cmd, reason in commands
    ]
    failed = [item for item in checks if item.status == "failed"]
    return {
        "ok": all(item.status in {"passed", "skipped"} for item in checks),
        "checks": [item.to_dict() for item in checks],
        "verification_class": "quick_project" if quick else "project",
        "selected_commands": [item.command for item in checks],
        "affected_files": affected,
        "skipped_checks": [item.reason for item in checks if item.status == "skipped"],
        "failure_code": failed[0].failure_code if failed else "",
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
    from mana_agent.utils.tool_results import json_safe_dumps

    return json_safe_dumps(payload)
