"""
mana_agent.utils.project_search

Project-wide text search over live repository files.

Prefers ripgrep (``rg``) when available for speed and correctness; falls back
to a pure-Python recursive search when ``rg`` is not installed. Both paths
exclude noisy directories and cap output so large repositories never flood the
LLM context.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Directories that should never be searched: VCS, virtualenvs, caches, build
# output, and our own index/state directory.
EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".mana",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
        ".next",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
    }
)

DEFAULT_MAX_RESULTS = 50
# Hard cap on total characters returned so big repos do not flood the model.
DEFAULT_MAX_OUTPUT_CHARS = 12_000
# Skip individual files larger than this (bytes) in the Python fallback.
_MAX_FILE_BYTES = 2_000_000
_RG_TIMEOUT_SECONDS = 20


@dataclass(slots=True)
class ProjectSearchMatch:
    """A single matching line in a project file."""

    file_path: str
    line_number: int
    line_text: str

    def format(self, root: Path | None = None) -> str:
        path = self.file_path
        if root is not None:
            try:
                path = str(Path(self.file_path).relative_to(root))
            except ValueError:
                path = self.file_path
        return f"{path}:{self.line_number}: {self.line_text}".rstrip()


@dataclass(slots=True)
class ProjectSearchResult:
    matches: list[ProjectSearchMatch]
    backend: str  # "ripgrep" | "python" | "none"
    truncated: bool

    def format(self, root: Path | None = None) -> str:
        return "\n".join(match.format(root) for match in self.matches)


def ripgrep_available() -> bool:
    return shutil.which("rg") is not None


def _excluded(path_parts: Iterable[str]) -> bool:
    return any(part in EXCLUDED_DIRS for part in path_parts)


def _ripgrep_search(
    query: str,
    root: Path,
    *,
    max_results: int,
    fixed_strings: bool,
) -> list[ProjectSearchMatch]:
    cmd = [
        "rg",
        "--line-number",
        "--no-heading",
        "--color",
        "never",
        "--max-count",
        str(max_results),
    ]
    if fixed_strings:
        cmd.append("--fixed-strings")
    for directory in sorted(EXCLUDED_DIRS):
        cmd.extend(["--glob", f"!**/{directory}/**"])
    cmd.extend(["--", query, str(root)])

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=_RG_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ripgrep search failed (%s); falling back to python search", exc)
        raise

    # rg exit code 1 == no matches (not an error). >1 == real error.
    if completed.returncode not in (0, 1):
        logger.warning("ripgrep returned %s: %s", completed.returncode, completed.stderr.strip())
        raise RuntimeError(f"ripgrep error: {completed.stderr.strip()}")

    matches: list[ProjectSearchMatch] = []
    for raw in completed.stdout.splitlines():
        if not raw:
            continue
        # Format: <path>:<line>:<text>
        parts = raw.split(":", 2)
        if len(parts) < 3:
            continue
        file_path, line_str, text = parts
        try:
            line_number = int(line_str)
        except ValueError:
            continue
        matches.append(
            ProjectSearchMatch(
                file_path=file_path,
                line_number=line_number,
                line_text=text.strip()[:500],
            )
        )
        if len(matches) >= max_results:
            break
    return matches


def _python_search(
    query: str,
    root: Path,
    *,
    max_results: int,
) -> list[ProjectSearchMatch]:
    needle = query.casefold()
    matches: list[ProjectSearchMatch] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in-place so os.walk never descends into them.
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for filename in filenames:
            file_path = Path(dirpath) / filename
            try:
                if file_path.stat().st_size > _MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            try:
                with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if needle in line.casefold():
                            matches.append(
                                ProjectSearchMatch(
                                    file_path=str(file_path),
                                    line_number=line_number,
                                    line_text=line.strip()[:500],
                                )
                            )
                            if len(matches) >= max_results:
                                return matches
            except OSError:
                continue
    return matches


def project_search(
    query: str,
    root: str | Path,
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    fixed_strings: bool = True,
) -> ProjectSearchResult:
    """Search ``root`` for ``query`` using ripgrep, falling back to Python.

    Excludes noisy directories, limits result count, and truncates total
    output size so large repositories cannot flood downstream consumers.
    """
    query = (query or "").strip()
    resolved_root = Path(root).resolve()
    if not query or not resolved_root.exists():
        return ProjectSearchResult(matches=[], backend="none", truncated=False)

    backend = "python"
    matches: list[ProjectSearchMatch] = []
    if ripgrep_available():
        try:
            matches = _ripgrep_search(
                query,
                resolved_root,
                max_results=max_results,
                fixed_strings=fixed_strings,
            )
            backend = "ripgrep"
        except Exception:
            matches = _python_search(query, resolved_root, max_results=max_results)
            backend = "python"
    else:
        matches = _python_search(query, resolved_root, max_results=max_results)

    # Enforce the total output-size budget.
    truncated = False
    if max_output_chars and max_output_chars > 0:
        kept: list[ProjectSearchMatch] = []
        running = 0
        for match in matches:
            rendered = match.format(resolved_root)
            running += len(rendered) + 1
            if running > max_output_chars:
                truncated = True
                break
            kept.append(match)
        matches = kept

    if len(matches) >= max_results:
        truncated = True

    return ProjectSearchResult(matches=matches, backend=backend, truncated=truncated)
