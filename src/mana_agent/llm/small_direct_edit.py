"""Deterministic fast path for low-risk explicit file edits."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from mana_agent.tools import apply_patch as apply_patch_tool


_SIMPLE_EDIT_RE = re.compile(r"\b(update|change|replace|bump|set)\b", re.IGNORECASE)
_BROAD_RE = re.compile(
    r"\b(all project|whole project|entire project|everywhere|refactor|tests?|implement|add feature|analy[sz]e)\b",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(r"\bv?\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?\b")
_PATH_RE = re.compile(
    r"(?P<path>(?:[\w.()@+-]+/)*[\w.()@+-]+\.(?:md|markdown|rst|txt|toml|json|ya?ml|py|ts|tsx|js|jsx))",
    re.IGNORECASE,
)
_README_VERSION_LINE_RE = re.compile(
    r"^(?P<prefix>.*?\bCurrent documented version:\s+\*\*v?)(?P<version>\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)(?P<suffix>\*\*\..*)$"
)
_DOC_SUFFIXES = {".md", ".markdown", ".rst", ".txt"}
_DOC_BASENAMES = {"readme", "changelog", "license"}


@dataclass(frozen=True)
class EditIntent:
    kind: str
    explicit_path: str | None = None
    old_value: str | None = None
    new_value: str | None = None
    docs_only: bool = False
    requires_verification: bool = True
    reason: str = ""


@dataclass
class SmallEditResult:
    handled: bool
    ok: bool
    answer: str
    changed_files: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    changed_line: str = ""
    error: str = ""


def _normalize_user_path(path: str) -> str:
    return str(path or "").strip().strip("\"'`").replace("\\", "/").lstrip("./")


def _is_docs_only(path: str) -> bool:
    p = Path(path)
    stem = p.stem.lower()
    suffix = p.suffix.lower()
    rel = path.replace("\\", "/").lower()
    return suffix in _DOC_SUFFIXES or rel.startswith("docs/") or stem in _DOC_BASENAMES


def _extract_explicit_path(request: str) -> str | None:
    for match in _PATH_RE.finditer(str(request or "")):
        raw = _normalize_user_path(match.group("path"))
        if raw and not raw.startswith("-"):
            return raw
    return None


def _extract_new_value(request: str) -> str | None:
    text = str(request or "")
    to_match = re.search(r"\b(?:to|as|=)\s+(v?\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)\b", text, re.IGNORECASE)
    if to_match:
        return to_match.group(1)
    version_match = _VERSION_RE.search(text)
    return version_match.group(0) if version_match else None


def resolve_explicit_path(root: Path, user_path: str) -> str | None:
    """Resolve one explicit user path without repository-wide discovery."""

    repo_root = root.resolve()
    normalized = _normalize_user_path(user_path)
    if not normalized or "\x00" in normalized or normalized.startswith("/") or ".." in Path(normalized).parts:
        return None

    def _canonical_existing_rel(candidate: str) -> str | None:
        parts = Path(candidate).parts
        current = repo_root
        actual_parts: list[str] = []
        for part in parts:
            if not current.exists() or not current.is_dir():
                return None
            exact_matches = [child for child in current.iterdir() if child.name == part]
            if len(exact_matches) == 1:
                current = exact_matches[0]
                actual_parts.append(current.name)
                continue
            lowered = part.lower()
            matches = [child for child in current.iterdir() if child.name.lower() == lowered]
            if len(matches) != 1:
                return None
            current = matches[0]
            actual_parts.append(current.name)
        if not current.exists() or not current.is_file():
            return None
        try:
            current.resolve().relative_to(repo_root)
        except ValueError:
            return None
        return Path(*actual_parts).as_posix()

    candidates: list[str] = [normalized]
    p = Path(normalized)
    if len(p.parts) == 1:
        lowered = normalized.lower()
        common = {
            "readme.md": "README.md",
            "license": "LICENSE",
            "license.md": "LICENSE.md",
            "changelog.md": "CHANGELOG.md",
        }.get(lowered)
        if common and common not in candidates:
            candidates.append(common)
        if lowered in {"readme.md", "license", "license.md", "changelog.md"}:
            for child in repo_root.iterdir():
                if child.is_file() and child.name.lower() == lowered and child.name not in candidates:
                    candidates.append(child.name)

    seen_realpaths: set[Path] = set()
    for candidate in candidates:
        canonical = _canonical_existing_rel(candidate)
        if canonical is None:
            continue
        target = (repo_root / canonical).resolve()
        try:
            target.relative_to(repo_root)
        except ValueError:
            continue
        if target in seen_realpaths:
            continue
        seen_realpaths.add(target)
        return canonical
    return None


def classify_edit_intent(root: Path, request: str) -> EditIntent:
    text = str(request or "").strip()
    if not text or _BROAD_RE.search(text) or not _SIMPLE_EDIT_RE.search(text):
        return EditIntent(kind="normal", reason="not a narrow edit request")
    raw_path = _extract_explicit_path(text)
    new_value = _extract_new_value(text)
    if not raw_path or not new_value:
        return EditIntent(kind="normal", reason="missing explicit path or exact value")
    resolved = resolve_explicit_path(root, raw_path)
    if not resolved:
        return EditIntent(kind="normal", explicit_path=raw_path, new_value=new_value, reason="explicit path not found")
    docs_only = _is_docs_only(resolved)
    return EditIntent(
        kind="small_direct_edit",
        explicit_path=resolved,
        new_value=new_value,
        docs_only=docs_only,
        requires_verification=not docs_only,
        reason="explicit path and exact replacement value",
    )


def _read_range(path: Path, *, start: int, end: int) -> tuple[list[str], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start_idx = max(0, start - 1)
    end_idx = max(start_idx, min(len(lines), end))
    return lines, lines[start_idx:end_idx]


def handle_small_direct_edit(root: Path, request: str) -> SmallEditResult:
    intent = classify_edit_intent(root, request)
    if intent.kind != "small_direct_edit" or not intent.explicit_path:
        return SmallEditResult(handled=False, ok=False, answer="", error=intent.reason)

    path = intent.explicit_path
    if Path(path).name.lower() != "readme.md" or "version" not in request.lower():
        return SmallEditResult(handled=False, ok=False, answer="", error="no deterministic handler for this edit")
    new_value = str(intent.new_value or "").lstrip("v")
    if not _VERSION_RE.fullmatch(new_value):
        return SmallEditResult(handled=False, ok=False, answer="", error="new version is not exact semver")

    target = root.resolve() / path
    try:
        _all_lines, window = _read_range(target, start=1, end=40)
    except OSError as exc:
        return SmallEditResult(handled=True, ok=False, answer=f"Small direct edit failed: {exc}", error=str(exc))

    trace: list[dict[str, Any]] = [
        {"tool_name": "read_file", "status": "ok", "path": path, "start_line": 1, "end_line": 40}
    ]
    match_index: int | None = None
    replacement = ""
    for offset, line in enumerate(window):
        match = _README_VERSION_LINE_RE.match(line)
        if not match:
            continue
        match_index = offset
        replacement = f"{match.group('prefix')}{new_value}{match.group('suffix')}"
        if replacement == line:
            answer = (
                f"No change needed in {path}; documented version is already v{new_value}.\n\n"
                f"Minimal docs check passed: {line}"
            )
            return SmallEditResult(
                handled=True,
                ok=True,
                answer=answer,
                changed_files=[],
                trace=trace,
                changed_line=line,
            )
        break
    if match_index is None:
        answer = f"Small direct edit failed: could not find the README documented version line in {path}."
        return SmallEditResult(handled=True, ok=False, answer=answer, trace=trace, error="version_line_not_found")

    original_line = window[match_index]
    patch = (
        "*** Begin Patch\n"
        f"*** Update File: {path}\n"
        "@@\n"
        f"-{original_line}\n"
        f"+{replacement}\n"
        "*** End Patch\n"
    )
    patch_result = apply_patch_tool.safe_apply_patch(repo_root=root.resolve(), patch=patch)
    ok = bool(patch_result.get("ok"))
    trace.append(
        {
            "tool_name": "apply_patch",
            "status": "ok" if ok else "error",
            "path": path,
            "changed_files": list(patch_result.get("touched_files") or []),
            "error": "" if ok else str(patch_result.get("error") or ""),
        }
    )
    if not ok:
        answer = f"Small direct edit failed while patching {path}: {patch_result.get('error') or 'unknown error'}"
        return SmallEditResult(handled=True, ok=False, answer=answer, trace=trace, error=str(patch_result.get("error") or ""))

    _all_after, after_window = _read_range(target, start=max(1, (match_index + 1) - 2), end=(match_index + 1) + 2)
    confirmed_line = ""
    for line in after_window:
        if new_value in line and "Current documented version:" in line:
            confirmed_line = line
            break
    trace.append(
        {
            "tool_name": "read_file",
            "status": "ok" if confirmed_line else "error",
            "path": path,
            "start_line": max(1, (match_index + 1) - 2),
            "end_line": (match_index + 1) + 2,
        }
    )
    if not confirmed_line:
        answer = f"Patched {path}, but minimal docs check did not find v{new_value} afterward."
        return SmallEditResult(handled=True, ok=False, answer=answer, changed_files=[path], trace=trace, error="minimal_check_failed")

    answer = (
        f"Updated {path} to v{new_value}.\n\n"
        f"Verification skipped: docs-only one-line edit. Confirmed changed line in {path}:\n"
        f"{confirmed_line}"
    )
    return SmallEditResult(
        handled=True,
        ok=True,
        answer=answer,
        changed_files=[path],
        trace=trace,
        changed_line=confirmed_line,
    )
