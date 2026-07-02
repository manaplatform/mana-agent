"""Exact-string file editing tools for coding agents."""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .write_file import _atomic_write_bytes, _resolve_target_path

DEFAULT_ALLOWED_PREFIXES: Optional[tuple[str, ...]] = None


@dataclass(frozen=True)
class EditFileResult:
    ok: bool
    path: str
    files_changed: list[str] = field(default_factory=list)
    before_sha256: str = ""
    after_sha256: str = ""
    changed_ranges: list[dict[str, int]] = field(default_factory=list)
    error_code: str = ""
    error: str = ""
    match_lines: list[int] = field(default_factory=list)
    nearest_snippets: list[str] = field(default_factory=list)
    current_snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


def _changed_range(before: str, after: str) -> dict[str, int]:
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines)
    changed = [op for op in matcher.get_opcodes() if op[0] != "equal"]
    if not changed:
        return {"start": 0, "end": 0}
    start = min(item[3] for item in changed) + 1
    end = max(item[4] for item in changed)
    return {"start": start, "end": max(start, end)}


def _all_match_lines(text: str, needle: str) -> list[int]:
    lines: list[int] = []
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            return lines
        lines.append(_line_for_offset(text, idx))
        start = idx + max(1, len(needle))


def _nearest_snippets(text: str, needle: str, *, limit: int = 3) -> list[str]:
    if not needle:
        return []
    lines = text.splitlines()
    needle_lines = needle.splitlines() or [needle]
    first = needle_lines[0].strip()
    candidates = difflib.get_close_matches(first, [line.strip() for line in lines], n=limit, cutoff=0.25)
    snippets: list[str] = []
    for candidate in candidates:
        idx = next((i for i, line in enumerate(lines) if line.strip() == candidate), -1)
        if idx < 0:
            continue
        start = max(0, idx - 2)
        end = min(len(lines), idx + max(3, len(needle_lines) + 2))
        snippets.append("\n".join(f"{line_no}: {lines[line_no - 1]}" for line_no in range(start + 1, end + 1)))
    return snippets


def _current_snippet(text: str, needle: str) -> str:
    snippets = _nearest_snippets(text, needle, limit=1)
    return snippets[0] if snippets else "\n".join(
        f"{line_no}: {line}" for line_no, line in enumerate(text.splitlines()[:8], start=1)
    )


def safe_edit_file(
    *,
    repo_root: Path,
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES,
) -> dict[str, Any]:
    target, rel_posix, err = _resolve_target_path(repo_root=repo_root, path=path, allowed_prefixes=allowed_prefixes)
    if err or target is None or rel_posix is None:
        return EditFileResult(ok=False, path=path, error_code="invalid_path", error=err or "invalid path").to_dict()
    if not target.exists():
        return EditFileResult(ok=False, path=rel_posix, error_code="file_not_found", error="target file does not exist").to_dict()
    if old_string == "":
        return EditFileResult(ok=False, path=rel_posix, error_code="empty_old_string", error="old_string is required").to_dict()

    before = target.read_text(encoding="utf-8")
    match_lines = _all_match_lines(before, old_string)
    if not match_lines:
        return EditFileResult(
            ok=False,
            path=rel_posix,
            error_code="old_string_not_found",
            error="old_string was not found exactly in current file content",
            nearest_snippets=_nearest_snippets(before, old_string),
            current_snippet=_current_snippet(before, old_string),
        ).to_dict()
    if len(match_lines) > 1 and not replace_all:
        return EditFileResult(
            ok=False,
            path=rel_posix,
            error_code="ambiguous_old_string",
            error="old_string appears more than once; set replace_all=true or provide more context",
            match_lines=match_lines,
        ).to_dict()

    after = before.replace(old_string, new_string) if replace_all else before.replace(old_string, new_string, 1)
    if after == before:
        return EditFileResult(ok=False, path=rel_posix, error_code="no_change", error="replacement produced no change").to_dict()
    _atomic_write_bytes(target, after.encode("utf-8"))
    return EditFileResult(
        ok=True,
        path=rel_posix,
        files_changed=[rel_posix],
        before_sha256=_sha(before),
        after_sha256=_sha(after),
        changed_ranges=[_changed_range(before, after)],
        match_lines=match_lines,
    ).to_dict()


def safe_multi_edit_file(
    *,
    repo_root: Path,
    path: str,
    edits: list[dict[str, str]],
    allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES,
) -> dict[str, Any]:
    target, rel_posix, err = _resolve_target_path(repo_root=repo_root, path=path, allowed_prefixes=allowed_prefixes)
    if err or target is None or rel_posix is None:
        return EditFileResult(ok=False, path=path, error_code="invalid_path", error=err or "invalid path").to_dict()
    if not target.exists():
        return EditFileResult(ok=False, path=rel_posix, error_code="file_not_found", error="target file does not exist").to_dict()
    if not edits:
        return EditFileResult(ok=False, path=rel_posix, error_code="empty_edits", error="edits must not be empty").to_dict()

    before = target.read_text(encoding="utf-8")
    current = before
    ranges: list[dict[str, int]] = []
    for index, edit in enumerate(edits):
        old = str(edit.get("old_string") or "")
        new = str(edit.get("new_string") or "")
        if old == "":
            return EditFileResult(ok=False, path=rel_posix, error_code="empty_old_string", error=f"edit {index} old_string is required").to_dict()
        match_lines = _all_match_lines(current, old)
        if not match_lines:
            return EditFileResult(
                ok=False,
                path=rel_posix,
                error_code="old_string_not_found",
                error=f"edit {index} old_string was not found exactly in current in-memory content",
                nearest_snippets=_nearest_snippets(current, old),
                current_snippet=_current_snippet(current, old),
            ).to_dict()
        if len(match_lines) > 1:
            return EditFileResult(
                ok=False,
                path=rel_posix,
                error_code="ambiguous_old_string",
                error=f"edit {index} old_string appears more than once",
                match_lines=match_lines,
            ).to_dict()
        next_content = current.replace(old, new, 1)
        ranges.append(_changed_range(current, next_content))
        current = next_content

    if current == before:
        return EditFileResult(ok=False, path=rel_posix, error_code="no_change", error="edits produced no change").to_dict()
    _atomic_write_bytes(target, current.encode("utf-8"))
    return EditFileResult(
        ok=True,
        path=rel_posix,
        files_changed=[rel_posix],
        before_sha256=_sha(before),
        after_sha256=_sha(current),
        changed_ranges=ranges,
    ).to_dict()


def build_edit_file_tool(*, repo_root: Path, allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES):
    try:
        from langchain_core.tools import StructuredTool  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.tools import StructuredTool  # type: ignore

    def _tool(path: str, old_string: str, new_string: str, replace_all: bool = False) -> dict[str, Any]:
        return safe_edit_file(
            repo_root=repo_root,
            path=path,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
            allowed_prefixes=allowed_prefixes,
        )

    return StructuredTool.from_function(
        func=_tool,
        name="edit_file",
        description="Replace one exact old_string in a repository file. Re-reads the file before editing and never uses line numbers as truth.",
    )


def build_multi_edit_file_tool(*, repo_root: Path, allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES):
    try:
        from langchain_core.tools import StructuredTool  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.tools import StructuredTool  # type: ignore

    def _tool(path: str, edits: list[dict[str, str]]) -> dict[str, Any]:
        return safe_multi_edit_file(repo_root=repo_root, path=path, edits=edits, allowed_prefixes=allowed_prefixes)

    return StructuredTool.from_function(
        func=_tool,
        name="multi_edit_file",
        description="Apply several exact-string replacements to one file atomically. Re-reads once, writes once, and aborts on the first failed edit.",
    )
