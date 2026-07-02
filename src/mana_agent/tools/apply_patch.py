"""Codex-style patch application tool for coding agents."""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Sequence

from ..config.settings import default_logs_dir

DEFAULT_ALLOWED_PREFIXES: Optional[tuple[str, ...]] = None
_DRIVE_LETTER_RE = re.compile(r"^[a-zA-Z]:[\\/]")


@dataclass(frozen=True)
class ApplyPatchResult:
    ok: bool
    touched_files: list[str]
    check_only: bool = False
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    error_code: str = ""
    strategy: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)
    changed_ranges: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _CodexFilePatch:
    op: str
    path: str
    lines: list[str]


def _strip_markdown_fences(text: str) -> str:
    s = str(text or "").strip()
    if not s.startswith("```"):
        return str(text or "")
    lines = s.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip() + "\n"
    return str(text or "")


def _normalise_user_path(path: str) -> str:
    p = str(path or "").replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    while p.startswith("/"):
        p = p[1:]
    while "//" in p:
        p = p.replace("//", "/")
    return p


def _normalise_prefixes(prefixes: Optional[Sequence[str]]) -> Optional[tuple[str, ...]]:
    if not prefixes:
        return None
    out: list[str] = []
    for raw in prefixes:
        p = _normalise_user_path(raw)
        if p and not p.endswith("/"):
            p += "/"
        out.append(p)
    return tuple(out)


def _is_allowed_prefix(rel_posix: str, allowed_prefixes: Optional[Sequence[str]]) -> bool:
    if not allowed_prefixes:
        return True
    norm = _normalise_prefixes(allowed_prefixes)
    if not norm:
        return True
    rel_posix = _normalise_user_path(rel_posix)
    for prefix in norm:
        if prefix == "":
            return True
        if rel_posix == prefix[:-1] or rel_posix.startswith(prefix):
            return True
    return False


def _normalise_patch_payload(payload: Any) -> tuple[str, str]:
    if payload is None:
        return "", "Error: missing patch content (expected `patch` parameter)."
    if isinstance(payload, str):
        return payload, ""
    if isinstance(payload, bytes):
        try:
            return payload.decode("utf-8"), ""
        except UnicodeDecodeError as exc:
            return "", f"Error: invalid patch bytes: {exc}"
    if isinstance(payload, dict):
        for key in ("patch", "diff", "input"):
            if key in payload and payload[key] is not None:
                return _normalise_patch_payload(payload[key])
        return json.dumps(payload), ""
    if isinstance(payload, list):
        return json.dumps(payload), ""
    return "", f"Error: invalid patch content type {type(payload).__name__}"


def _parse_codex_patch(text: str) -> tuple[list[_CodexFilePatch], str]:
    patch = _strip_markdown_fences(text).strip("\n")
    lines = patch.splitlines()
    if not lines or lines[0].strip() != "*** Begin Patch":
        return [], "patch must start with *** Begin Patch"
    if lines[-1].strip() != "*** End Patch":
        return [], "patch must end with *** End Patch"

    idx = 1
    files: list[_CodexFilePatch] = []
    current_op = ""
    current_path = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_op, current_path, current_lines
        if current_op and current_path:
            files.append(_CodexFilePatch(op=current_op, path=current_path, lines=current_lines))
        current_op = ""
        current_path = ""
        current_lines = []

    while idx < len(lines) - 1:
        line = lines[idx]
        if line.startswith("*** Update File: "):
            flush()
            current_op = "update"
            current_path = _normalise_user_path(line.removeprefix("*** Update File: "))
        elif line.startswith("*** Add File: "):
            flush()
            current_op = "add"
            current_path = _normalise_user_path(line.removeprefix("*** Add File: "))
        elif line.startswith("*** Delete File: "):
            flush()
            current_op = "delete"
            current_path = _normalise_user_path(line.removeprefix("*** Delete File: "))
        elif not current_op:
            return [], f"unexpected patch line outside file block: {line}"
        else:
            current_lines.append(line)
        idx += 1
    flush()
    if not files:
        return [], "patch contains no file operations"
    return files, ""


def _validate_touched_paths(
    repo_root: Path,
    touched: set[str],
    allowed_prefixes: Optional[Sequence[str]],
) -> tuple[bool, list[str], str]:
    repo_root = repo_root.resolve()
    validated: list[str] = []
    for p in sorted(touched):
        raw = p.strip()
        if "\x00" in raw:
            return False, [], f"Blocked: NUL byte in patch path: {p}"
        if _DRIVE_LETTER_RE.match(raw):
            return False, [], f"Blocked: drive-letter path in patch: {p}"
        if raw.startswith("/"):
            return False, [], f"Blocked: absolute path in patch: {p}"
        parts = [seg for seg in raw.replace("\\", "/").split("/") if seg not in ("", ".")]
        if any(seg == ".." for seg in parts):
            return False, [], f"Blocked: traversal ('..') in patch path: {p}"
        rel_pp = PurePosixPath(_normalise_user_path(raw))
        if str(rel_pp) in ("", "."):
            return False, [], "Blocked: empty/invalid path in patch"
        target = (repo_root / Path(str(rel_pp))).resolve()
        try:
            rel = target.relative_to(repo_root)
        except ValueError:
            return False, [], f"Blocked: patch path escapes repository root: {p}"
        rel_posix = rel.as_posix()
        if not _is_allowed_prefix(rel_posix, allowed_prefixes):
            return False, [], f"Blocked: patch touches disallowed path: {rel_posix}"
        validated.append(rel_posix)
    return True, validated, ""


def _validate_patch_targets_are_read(
    repo_root: Path,
    touched_files: Sequence[str],
    read_files: Sequence[str] | None,
) -> tuple[bool, str]:
    read_normalized = {_normalise_user_path(item) for item in (read_files or [])}
    missing = [
        rel for rel in touched_files
        if (repo_root / rel).exists() and rel not in read_normalized
    ]
    if missing:
        return False, f"Blocked: patch targets unread files: {missing}. Re-read target files before patching."
    return True, ""


def _write_patch_history(
    *,
    repo_root: Path,
    patch: str,
    result: dict[str, Any],
    touched_files: Sequence[str],
    check_only: bool,
) -> None:
    try:
        logs_dir = default_logs_dir(repo_root)
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tool": "apply_patch",
            "check_only": bool(check_only),
            "touched_files": list(touched_files),
            "patch": patch,
            "result": result,
        }
        (logs_dir / f"apply_patch_{stamp}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        return


def _text_from_patch_lines(lines: Sequence[str], prefix: str) -> str:
    out = [line[1:] for line in lines if line.startswith(prefix)]
    return "\n".join(out) + ("\n" if out else "")


def _changed_range(before: str, after: str) -> dict[str, int]:
    matcher = difflib.SequenceMatcher(a=before.splitlines(), b=after.splitlines())
    changed = [item for item in matcher.get_opcodes() if item[0] != "equal"]
    if not changed:
        return {"start": 0, "end": 0}
    start = min(item[3] for item in changed) + 1
    end = max(item[4] for item in changed)
    return {"start": start, "end": max(start, end)}


def _nearby_snippet(content: str, patch_lines: Sequence[str]) -> str:
    anchors = [line[1:].strip() for line in patch_lines if line.startswith((" ", "-")) and line[1:].strip()]
    lines = content.splitlines()
    if not anchors or not lines:
        return "\n".join(f"{line_no}: {line}" for line_no, line in enumerate(lines[:8], start=1))
    matches = difflib.get_close_matches(anchors[0], [line.strip() for line in lines], n=1, cutoff=0.2)
    if not matches:
        return "\n".join(f"{line_no}: {line}" for line_no, line in enumerate(lines[:8], start=1))
    idx = next((i for i, line in enumerate(lines) if line.strip() == matches[0]), 0)
    start = max(0, idx - 3)
    end = min(len(lines), idx + 5)
    return "\n".join(f"{line_no}: {lines[line_no - 1]}" for line_no in range(start + 1, end + 1))


def _apply_codex_update(content: str, lines: Sequence[str], path: str) -> tuple[bool, str, str]:
    current = content
    hunk: list[str] = []

    def apply_hunk(hunk_lines: list[str], current_text: str) -> tuple[bool, str, str]:
        payload = [line for line in hunk_lines if line != "@@" and not line.startswith("@@ ")]
        if not payload:
            return True, current_text, ""
        invalid = [line for line in payload if not line.startswith((" ", "+", "-"))]
        if invalid:
            return False, current_text, f"invalid patch hunk line for {path}: {invalid[0]}"
        old_block = "\n".join(line[1:] for line in payload if line.startswith((" ", "-")))
        new_block = "\n".join(line[1:] for line in payload if line.startswith((" ", "+")))
        if old_block:
            old_block += "\n"
        if new_block:
            new_block += "\n"
        if old_block not in current_text:
            return False, current_text, "patch_context_not_found"
        return True, current_text.replace(old_block, new_block, 1), ""

    for raw in lines:
        if raw.startswith("@@"):
            ok, current, err = apply_hunk(hunk, current)
            if not ok:
                return ok, current, err
            hunk = [raw]
        else:
            hunk.append(raw)
    return apply_hunk(hunk, current)


def extract_patch_touched_files(patch: Any) -> dict[str, Any]:
    patch_text, normalise_err = _normalise_patch_payload(patch)
    if normalise_err:
        return {"ok": False, "touched_files": [], "error": normalise_err}
    parsed, error = _parse_codex_patch(patch_text)
    if error:
        return {"ok": False, "touched_files": [], "error": error}
    return {"ok": True, "touched_files": [item.path for item in parsed]}


def safe_apply_patch(
    *,
    repo_root: Path,
    patch: str,
    allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES,
    check_only: bool = False,
    read_files: Sequence[str] | None = None,
    require_read: bool = False,
) -> dict[str, Any]:
    patch_text = _strip_markdown_fences(str(patch or ""))
    repo_root = repo_root.resolve()
    parsed, parse_err = _parse_codex_patch(patch_text)
    if parse_err:
        result = ApplyPatchResult(
            ok=False,
            touched_files=[],
            error_code="invalid_patch_format",
            error=(
                f"Error: patch parse failed: {parse_err}. Expected Codex patch format "
                "with *** Begin Patch / *** Update File / *** Add File / *** Delete File / *** End Patch."
            ),
        ).to_dict()
        _write_patch_history(repo_root=repo_root, patch=patch_text, result=result, touched_files=[], check_only=check_only)
        return result

    touched = {item.path for item in parsed}
    ok, touched_files, err = _validate_touched_paths(repo_root, touched, allowed_prefixes)
    if not ok:
        result = ApplyPatchResult(ok=False, touched_files=[], error_code="invalid_path", error=err).to_dict()
        _write_patch_history(repo_root=repo_root, patch=patch_text, result=result, touched_files=[], check_only=check_only)
        return result

    if require_read:
        existing = [item.path for item in parsed if item.op in {"update", "delete"}]
        read_ok, read_err = _validate_patch_targets_are_read(repo_root, existing, read_files)
        if not read_ok:
            result = ApplyPatchResult(ok=False, touched_files=touched_files, error_code="unread_target", error=read_err).to_dict()
            _write_patch_history(repo_root=repo_root, patch=patch_text, result=result, touched_files=touched_files, check_only=check_only)
            return result

    computed: dict[str, str | None] = {}
    before_by_path: dict[str, str] = {}
    changed_ranges: list[dict[str, Any]] = []
    for item in parsed:
        target = (repo_root / Path(item.path)).resolve()
        exists = target.exists()
        before = target.read_text(encoding="utf-8") if exists else ""
        before_by_path.setdefault(item.path, before)
        if item.op == "add":
            if exists:
                result = ApplyPatchResult(ok=False, touched_files=touched_files, error_code="target_exists", error=f"Add File target already exists: {item.path}").to_dict()
                _write_patch_history(repo_root=repo_root, patch=patch_text, result=result, touched_files=touched_files, check_only=check_only)
                return result
            invalid = [line for line in item.lines if not line.startswith("+") and line.strip()]
            if invalid:
                result = ApplyPatchResult(ok=False, touched_files=touched_files, error_code="invalid_patch_format", error=f"Add File lines must start with '+': {invalid[0]}").to_dict()
                _write_patch_history(repo_root=repo_root, patch=patch_text, result=result, touched_files=touched_files, check_only=check_only)
                return result
            computed[item.path] = _text_from_patch_lines(item.lines, "+")
            changed_ranges.append({"path": item.path, **_changed_range("", str(computed[item.path] or ""))})
        elif item.op == "delete":
            if not exists:
                result = ApplyPatchResult(ok=False, touched_files=touched_files, error_code="target_missing", error=f"Delete File target does not exist: {item.path}").to_dict()
                _write_patch_history(repo_root=repo_root, patch=patch_text, result=result, touched_files=touched_files, check_only=check_only)
                return result
            computed[item.path] = None
            changed_ranges.append({"path": item.path, **_changed_range(before, "")})
        else:
            if not exists:
                result = ApplyPatchResult(ok=False, touched_files=touched_files, error_code="target_missing", error=f"Update File target does not exist: {item.path}").to_dict()
                _write_patch_history(repo_root=repo_root, patch=patch_text, result=result, touched_files=touched_files, check_only=check_only)
                return result
            current = str(computed.get(item.path, before) or "")
            patch_ok, after, patch_err = _apply_codex_update(current, item.lines, item.path)
            if not patch_ok:
                result = ApplyPatchResult(
                    ok=False,
                    touched_files=touched_files,
                    error_code="patch_context_not_found" if patch_err == "patch_context_not_found" else "invalid_patch_format",
                    error=f"{patch_err} in {item.path}. Re-read the target file and rebuild the patch against current exact context.",
                    stdout=_nearby_snippet(current, item.lines),
                ).to_dict()
                _write_patch_history(repo_root=repo_root, patch=patch_text, result=result, touched_files=touched_files, check_only=check_only)
                return result
            computed[item.path] = after
            changed_ranges.append({"path": item.path, **_changed_range(current, after)})

    changed_files = [path for path, after in computed.items() if after != before_by_path.get(path)]
    result = ApplyPatchResult(
        ok=True,
        touched_files=touched_files,
        check_only=check_only,
        strategy="codex",
        stdout="codex patch validated" if check_only else "codex patch applied",
        changed_ranges=changed_ranges,
    ).to_dict()
    result["files_changed"] = [] if check_only else changed_files
    if not check_only:
        for rel, after in computed.items():
            target = repo_root / rel
            if after is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(after, encoding="utf-8")
    _write_patch_history(repo_root=repo_root, patch=patch_text, result=result, touched_files=touched_files, check_only=check_only)
    return result


def build_apply_patch_tool(
    *,
    repo_root: Path,
    allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES,
):
    try:
        from langchain_core.tools import StructuredTool  # type: ignore[import-untyped]
    except Exception:  # pragma: no cover
        from langchain.tools import StructuredTool  # type: ignore[import-untyped]

    def _tool(
        patch: Any | None = None,
        diff: Any | None = None,
        input: Any | None = None,  # noqa: A002 - tool-call compatibility alias
        check_only: bool = False,
    ) -> dict[str, Any]:
        raw_patch = patch if patch is not None else (diff if diff is not None else input)
        patch_text, normalise_err = _normalise_patch_payload(raw_patch)
        if normalise_err:
            return ApplyPatchResult(ok=False, touched_files=[], check_only=check_only, error_code="missing_patch", error=normalise_err).to_dict()
        return safe_apply_patch(
            repo_root=repo_root,
            patch=patch_text,
            allowed_prefixes=allowed_prefixes,
            check_only=check_only,
        )

    return StructuredTool.from_function(
        func=_tool,
        name="apply_patch",
        description=(
            "Safely apply a Codex-style text patch inside the repository. "
            "Supports *** Update File, *** Add File, and *** Delete File blocks. "
            "Matches update hunks by surrounding text context, not line numbers. "
            "With check_only=true, validation runs without writing files."
        ),
    )
