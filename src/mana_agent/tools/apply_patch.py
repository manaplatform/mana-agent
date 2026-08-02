"""Codex-style patch application tool for coding agents.

Applies patches by surrounding text context (not line numbers). When context is
stale, a deterministic recovery workflow re-reads targets, matches unique
anchors, rebuilds hunks against current content, and retries within a strict
bound. Ambiguous matches never apply.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Sequence

from ..config.settings import default_logs_dir

DEFAULT_ALLOWED_PREFIXES: Optional[tuple[str, ...]] = None
_DRIVE_LETTER_RE = re.compile(r"^[a-zA-Z]:[\\/]")

# Bounded recovery: original exact → rebuilt exact context → minimal anchored.
MAX_PATCH_ATTEMPTS = 3

# Constrained single-line similarity (not unrestricted fuzzy replacement).
_SIMILAR_LINE_MIN_RATIO = 0.72
_SIMILAR_LINE_MARGIN = 0.08

_HEADING_RE = re.compile(r"^(#{1,6})\s+\S")
_DEF_CLASS_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+\w+")
_CONFIG_KEY_RE = re.compile(r"^\s*[A-Za-z_][\w.-]*\s*[:=]")
_IMPORT_RE = re.compile(r"^\s*(?:from\s+\S+\s+import|import\s+\S)")
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")


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
    matched_anchor: str = ""
    candidate_count: int = 0
    changed_ranges: list[dict[str, Any]] = field(default_factory=list)
    already_applied: bool = False
    recovery_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _CodexFilePatch:
    op: str
    path: str
    lines: list[str]


@dataclass(frozen=True)
class _HunkMatch:
    strategy: str
    line_index: int
    old_span: int
    new_lines: list[str]
    matched_anchor: str = ""
    candidate_count: int = 1
    already_applied: bool = False
    error: str = ""
    anchors_searched: list[str] = field(default_factory=list)
    candidates: list[int] = field(default_factory=list)


@dataclass
class _FileText:
    """Normalized line view that preserves original newline style on write-back."""

    lines: list[str]
    newline: str
    had_final_newline: bool

    @classmethod
    def from_text(cls, content: str) -> "_FileText":
        if "\r\n" in content:
            newline = "\r\n"
        elif "\r" in content and "\n" not in content.replace("\r\n", ""):
            newline = "\r"
        else:
            newline = "\n"
        had_final = bool(content) and content.endswith(("\n", "\r"))
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.splitlines()
        return cls(lines=lines, newline=newline, had_final_newline=had_final)

    def to_text(self) -> str:
        if not self.lines:
            return self.newline if self.had_final_newline and self.newline else ""
        body = self.newline.join(self.lines)
        if self.had_final_newline:
            body += self.newline
        return body


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


def _split_update_hunks(lines: Sequence[str]) -> list[list[str]]:
    hunks: list[list[str]] = []
    current: list[str] = []
    for raw in lines:
        if raw.startswith("@@"):
            if current:
                hunks.append(current)
            current = [raw]
        else:
            current.append(raw)
    if current:
        hunks.append(current)
    return hunks


def _hunk_payload(hunk_lines: Sequence[str]) -> list[str]:
    return [line for line in hunk_lines if line != "@@" and not line.startswith("@@ ")]


def _normalize_hunk_payload_lines(payload: Sequence[str]) -> tuple[list[str], str]:
    """Normalize hunk lines; bare empty lines are treated as empty context."""
    normalized: list[str] = []
    for line in payload:
        if line == "":
            normalized.append(" ")
            continue
        if not line.startswith((" ", "+", "-")):
            return [], f"invalid patch hunk line: {line}"
        normalized.append(line)
    return normalized, ""


def _hunk_signed_parts(payload: Sequence[str]) -> tuple[list[str], list[str], list[str], list[str], list[str], str]:
    """Return old_lines, new_lines, removed, added, context, error."""
    normalized, err = _normalize_hunk_payload_lines(payload)
    if err:
        return [], [], [], [], [], err
    old_lines = [line[1:] for line in normalized if line.startswith((" ", "-"))]
    new_lines = [line[1:] for line in normalized if line.startswith((" ", "+"))]
    removed = [line[1:] for line in normalized if line.startswith("-")]
    added = [line[1:] for line in normalized if line.startswith("+")]
    context = [line[1:] for line in normalized if line.startswith(" ")]
    return old_lines, new_lines, removed, added, context, ""


def _find_sequence(haystack: Sequence[str], needle: Sequence[str]) -> list[int]:
    if not needle:
        return []
    n = len(needle)
    if n > len(haystack):
        return []
    hits: list[int] = []
    for i in range(0, len(haystack) - n + 1):
        if list(haystack[i : i + n]) == list(needle):
            hits.append(i)
    return hits


def _find_sequence_ws(haystack: Sequence[str], needle: Sequence[str]) -> list[int]:
    if not needle:
        return []
    norm_h = [re.sub(r"[ \t]+", " ", line.rstrip()) for line in haystack]
    norm_n = [re.sub(r"[ \t]+", " ", line.rstrip()) for line in needle]
    return _find_sequence(norm_h, norm_n)


def _is_indent_significant(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(
        (
            ".py",
            ".pyi",
            ".yml",
            ".yaml",
            ".toml",
            ".json",
            ".jsonc",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".go",
            ".rs",
            ".java",
            ".kt",
            ".scala",
            ".rb",
            ".php",
            ".c",
            ".cc",
            ".cpp",
            ".h",
            ".hpp",
            ".cs",
            ".swift",
        )
    )


def _is_stable_anchor(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    if _HEADING_RE.match(line) or _DEF_CLASS_RE.match(line) or _IMPORT_RE.match(line):
        return True
    if _CONFIG_KEY_RE.match(line) or _TABLE_ROW_RE.match(line):
        return True
    # Unique-looking non-trivial lines (identifiers, links, table cells).
    if len(text) >= 8 and (text.startswith("|") or text.startswith("#") or text.startswith("-") or "://" in text):
        return True
    return len(text) >= 12


def _sequence_present(file_lines: Sequence[str], needle: Sequence[str]) -> bool:
    return bool(needle) and bool(_find_sequence(file_lines, list(needle)))


def _hunk_already_applied(
    file_lines: Sequence[str],
    *,
    old_lines: Sequence[str],
    new_lines: Sequence[str],
    removed: Sequence[str],
    added: Sequence[str],
) -> bool:
    if not new_lines and not added:
        return False
    # Pure insertion already present (prefer contiguous new block / added block).
    if added and not removed:
        if _sequence_present(file_lines, new_lines):
            return True
        if _sequence_present(file_lines, added):
            return True
        return False
    # Replacement: new block present, old block absent.
    if new_lines and _sequence_present(file_lines, new_lines):
        if old_lines and not _sequence_present(file_lines, old_lines):
            return True
        leftover_removed = [line for line in removed if line not in added]
        if leftover_removed and not any(_sequence_present(file_lines, [line]) for line in leftover_removed):
            return True
    if added and all(_sequence_present(file_lines, [line]) for line in added):
        leftover_removed = [line for line in removed if line not in added]
        if leftover_removed and not any(_sequence_present(file_lines, [line]) for line in leftover_removed):
            return True
    return False


def _reduce_old_variants(payload: Sequence[str]) -> list[tuple[list[str], list[str], str]]:
    """Build reduced-context old/new variants (full → less outer context)."""
    old_lines, new_lines, removed, added, context, err = _hunk_signed_parts(payload)
    if err:
        return []
    variants: list[tuple[list[str], list[str], str]] = []
    variants.append((list(old_lines), list(new_lines), "exact_context"))
    if not context:
        return variants

    # Drop leading then trailing context lines progressively.
    signed = list(payload)
    leading_ctx = 0
    for line in signed:
        if line.startswith(" "):
            leading_ctx += 1
        else:
            break
    trailing_ctx = 0
    for line in reversed(signed):
        if line.startswith(" "):
            trailing_ctx += 1
        else:
            break

    for drop_lead in range(0, leading_ctx + 1):
        for drop_trail in range(0, trailing_ctx + 1):
            if drop_lead == 0 and drop_trail == 0:
                continue
            trimmed = signed[drop_lead : len(signed) - drop_trail if drop_trail else None]
            if not trimmed:
                continue
            o, n, _r, _a, _c, e = _hunk_signed_parts(trimmed)
            if e or not o:
                continue
            variants.append((o, n, "reduced_context"))
    # Prefer more context first (already ordered by increasing drops).
    return variants


def _locate_hunk(file_lines: Sequence[str], payload: Sequence[str], *, path: str, minimal: bool = False) -> _HunkMatch:
    old_lines, new_lines, removed, added, context, err = _hunk_signed_parts(payload)
    if err:
        return _HunkMatch(strategy="", line_index=-1, old_span=0, new_lines=[], error=err, candidate_count=0)

    anchors_searched: list[str] = []
    if _hunk_already_applied(file_lines, old_lines=old_lines, new_lines=new_lines, removed=removed, added=added):
        return _HunkMatch(
            strategy="already_applied",
            line_index=0,
            old_span=0,
            new_lines=list(new_lines),
            already_applied=True,
            candidate_count=1,
            matched_anchor="content already matches intended result",
        )

    # 1–2. Exact full-hunk and reduced-context variants.
    variants = _reduce_old_variants(payload)
    if minimal:
        # Prefer shorter old spans first for minimal anchored rebuild.
        variants = sorted(variants, key=lambda item: len(item[0]))

    for old_var, new_var, strategy in variants:
        if not old_var:
            continue
        anchors_searched.append(f"{strategy}:{old_var[0][:80]}")
        hits = _find_sequence(file_lines, old_var)
        if len(hits) == 1:
            return _HunkMatch(
                strategy=strategy,
                line_index=hits[0],
                old_span=len(old_var),
                new_lines=list(new_var),
                matched_anchor=old_var[0] if old_var else "",
                candidate_count=1,
                anchors_searched=anchors_searched,
            )
        if len(hits) > 1:
            return _HunkMatch(
                strategy="ambiguous_context",
                line_index=-1,
                old_span=0,
                new_lines=list(new_var),
                candidate_count=len(hits),
                candidates=hits,
                anchors_searched=anchors_searched,
                matched_anchor=old_var[0] if old_var else "",
                error=(
                    f"ambiguous_context: {len(hits)} equally plausible locations for "
                    f"context starting with {old_var[0]!r}"
                ),
            )

    # 3. Unique removed-line sequence (exact).
    if removed:
        anchors_searched.append(f"unique_removed_lines:{removed[0][:80]}")
        hits = _find_sequence(file_lines, list(removed))
        if len(hits) == 1:
            # Expand with local context from the file when possible.
            start = hits[0]
            # Prefer replacing only the removed span with added (+ surrounding context from new_lines if available).
            if new_lines and context:
                # Rebuild new as: leading context from file if present + added + trailing
                return _HunkMatch(
                    strategy="unique_removed_lines",
                    line_index=start,
                    old_span=len(removed),
                    new_lines=list(added) if added else list(new_lines),
                    matched_anchor=removed[0],
                    candidate_count=1,
                    anchors_searched=anchors_searched,
                )
            return _HunkMatch(
                strategy="unique_removed_lines",
                line_index=start,
                old_span=len(removed),
                new_lines=list(added) if added else list(new_lines),
                matched_anchor=removed[0],
                candidate_count=1,
                anchors_searched=anchors_searched,
            )
        if len(hits) > 1:
            return _HunkMatch(
                strategy="ambiguous_context",
                line_index=-1,
                old_span=0,
                new_lines=list(new_lines),
                candidate_count=len(hits),
                candidates=hits,
                anchors_searched=anchors_searched,
                matched_anchor=removed[0],
                error=f"ambiguous_context: removed lines match {len(hits)} locations",
            )

        # Constrained single-line similarity for one drifted removed line.
        if len(removed) == 1 and len(added) == 1:
            needle = removed[0]
            scored = sorted(
                (
                    (difflib.SequenceMatcher(None, needle, line).ratio(), index, line)
                    for index, line in enumerate(file_lines)
                ),
                reverse=True,
            )
            anchors_searched.append(f"unique_similar_removed:{needle[:80]}")
            if scored and scored[0][0] >= _SIMILAR_LINE_MIN_RATIO:
                best_ratio, best_idx, best_line = scored[0]
                second = scored[1][0] if len(scored) > 1 else 0.0
                if best_ratio - second >= _SIMILAR_LINE_MARGIN:
                    return _HunkMatch(
                        strategy="unique_removed_lines",
                        line_index=best_idx,
                        old_span=1,
                        new_lines=list(added),
                        matched_anchor=best_line,
                        candidate_count=1,
                        anchors_searched=anchors_searched,
                    )
                return _HunkMatch(
                    strategy="ambiguous_context",
                    line_index=-1,
                    old_span=0,
                    new_lines=list(added),
                    candidate_count=2,
                    candidates=[scored[0][1], scored[1][1]] if len(scored) > 1 else [scored[0][1]],
                    anchors_searched=anchors_searched,
                    matched_anchor=best_line,
                    error="ambiguous_context: multiple similar lines for removed content",
                )

    # 4. Anchored insert / unique surrounding anchors.
    anchor_candidates: list[tuple[str, int, str]] = []  # kind, index, text
    search_lines = list(context) + list(removed) + list(old_lines)
    for line in search_lines:
        if not _is_stable_anchor(line):
            continue
        anchors_searched.append(f"anchor:{line[:80]}")
        hits = _find_sequence(file_lines, [line])
        if len(hits) == 1:
            kind = "heading" if _HEADING_RE.match(line) else (
                "table_row" if _TABLE_ROW_RE.match(line) else (
                    "import" if _IMPORT_RE.match(line) else (
                        "symbol" if _DEF_CLASS_RE.match(line) else "unique_line"
                    )
                )
            )
            anchor_candidates.append((kind, hits[0], line))

    # Pure insertion: place added lines after/before a unique context line.
    if added and not removed and anchor_candidates:
        # Prefer table rows / headings for docs, then other unique lines.
        priority = {"heading": 0, "table_row": 1, "symbol": 2, "import": 3, "unique_line": 4}
        anchor_candidates.sort(key=lambda item: (priority.get(item[0], 9), item[1]))
        kind, idx, text = anchor_candidates[0]
        # If context has lines after the anchor in the old_lines, try to find insertion point
        # between unique predecessor and successor when both exist.
        insert_at = idx + 1
        if context:
            # Find last context line that appears before added intent.
            for ctx in reversed(context):
                ctx_hits = _find_sequence(file_lines, [ctx])
                if len(ctx_hits) == 1:
                    insert_at = ctx_hits[0] + 1
                    text = ctx
                    break
        # If new_lines includes context around added, use only the insertion delta.
        return _HunkMatch(
            strategy="anchored_insert",
            line_index=insert_at,
            old_span=0,
            new_lines=list(added),
            matched_anchor=text,
            candidate_count=1,
            anchors_searched=anchors_searched,
        )

    # Replacement anchored by unique surrounding context when old block drifted.
    if removed and added and anchor_candidates:
        kind, idx, text = anchor_candidates[0]
        # Search near the anchor for best place to apply replacement.
        window_start = max(0, idx - 5)
        window_end = min(len(file_lines), idx + 8)
        window = list(file_lines[window_start:window_end])
        if len(removed) == 1:
            local = [
                (difflib.SequenceMatcher(None, removed[0], line).ratio(), window_start + i, line)
                for i, line in enumerate(window)
            ]
            local.sort(reverse=True)
            if local and local[0][0] >= _SIMILAR_LINE_MIN_RATIO:
                if len(local) == 1 or local[0][0] - local[1][0] >= _SIMILAR_LINE_MARGIN:
                    return _HunkMatch(
                        strategy="anchored_insert" if not removed else "unique_removed_lines",
                        line_index=local[0][1],
                        old_span=1,
                        new_lines=list(added),
                        matched_anchor=text,
                        candidate_count=1,
                        anchors_searched=anchors_searched,
                    )

    # 5. Whitespace-normalized matching when indentation is not significant.
    if not _is_indent_significant(path) and old_lines:
        anchors_searched.append(f"whitespace_normalized:{old_lines[0][:80]}")
        hits = _find_sequence_ws(file_lines, list(old_lines))
        if len(hits) == 1:
            # Map to exact current lines for rebuild (preserve file indentation).
            span = len(old_lines)
            return _HunkMatch(
                strategy="whitespace_normalized",
                line_index=hits[0],
                old_span=span,
                new_lines=_rebase_new_lines_ws(file_lines[hits[0] : hits[0] + span], list(old_lines), list(new_lines)),
                matched_anchor=file_lines[hits[0]],
                candidate_count=1,
                anchors_searched=anchors_searched,
            )
        if len(hits) > 1:
            return _HunkMatch(
                strategy="ambiguous_context",
                line_index=-1,
                old_span=0,
                new_lines=list(new_lines),
                candidate_count=len(hits),
                candidates=hits,
                anchors_searched=anchors_searched,
                error=f"ambiguous_context: whitespace-normalized match hit {len(hits)} locations",
            )

    # No reliable unique anchor.
    return _HunkMatch(
        strategy="ambiguous_context" if anchors_searched else "",
        line_index=-1,
        old_span=0,
        new_lines=list(new_lines),
        candidate_count=0,
        anchors_searched=anchors_searched,
        error=(
            "patch_context_not_found: no unique reliable anchor for intended edit. "
            f"anchors_searched={anchors_searched[:8]}"
        ),
    )


def _rebase_new_lines_ws(current_old: Sequence[str], patch_old: Sequence[str], patch_new: Sequence[str]) -> list[str]:
    """Preserve current indentation when only whitespace drifted in the old side."""
    if len(current_old) != len(patch_old):
        return list(patch_new)
    # Map leading whitespace from current old lines onto corresponding new lines by content.
    indent_by_stripped = {}
    for cur, old in zip(current_old, patch_old):
        indent_by_stripped[old.rstrip()] = cur[: len(cur) - len(cur.lstrip(" \t"))]
    rebuilt: list[str] = []
    for line in patch_new:
        key = line.rstrip()
        if key in indent_by_stripped:
            stripped = line.lstrip(" \t")
            rebuilt.append(indent_by_stripped[key] + stripped)
        else:
            rebuilt.append(line)
    return rebuilt


def _apply_match(file_text: _FileText, match: _HunkMatch) -> None:
    if match.already_applied:
        return
    if match.old_span == 0:
        # Pure insertion at line_index.
        idx = max(0, min(match.line_index, len(file_text.lines)))
        file_text.lines[idx:idx] = list(match.new_lines)
        return
    start = match.line_index
    end = start + match.old_span
    file_text.lines[start:end] = list(match.new_lines)


def _verify_expected_result(
    after_lines: Sequence[str],
    *,
    new_lines: Sequence[str],
    added: Sequence[str],
    strategy: str,
) -> bool:
    if strategy == "already_applied":
        return True
    if new_lines and _sequence_present(after_lines, new_lines):
        return True
    if added and all(_sequence_present(after_lines, [line]) for line in added):
        return True
    if not added and not new_lines:
        return True
    return False


def _apply_codex_update_exact(content: str, lines: Sequence[str], path: str) -> tuple[bool, str, str]:
    """Exact-context application requiring a unique old-block match (no recovery)."""
    file_text = _FileText.from_text(content)
    for hunk in _split_update_hunks(lines):
        payload = _hunk_payload(hunk)
        if not payload:
            continue
        old_lines, new_lines, _removed, _added, _context, err = _hunk_signed_parts(payload)
        if err:
            return False, content, f"invalid patch hunk line for {path}: {err.split(': ', 1)[-1]}"
        if not old_lines:
            # Empty old block: append new content when absent (rare).
            if new_lines and not _sequence_present(file_text.lines, new_lines):
                file_text.lines.extend(list(new_lines))
            continue
        hits = _find_sequence(file_text.lines, old_lines)
        if len(hits) == 0:
            return False, content, "patch_context_not_found"
        if len(hits) > 1:
            return False, content, "patch_context_not_found"
        start = hits[0]
        file_text.lines[start : start + len(old_lines)] = list(new_lines)
    return True, file_text.to_text(), ""


def _apply_codex_update_with_recovery(
    content: str,
    lines: Sequence[str],
    path: str,
) -> tuple[bool, str, str, dict[str, Any]]:
    """
    Apply update hunks with bounded deterministic recovery.

    Attempts:
      1. exact_context on the original patch
      2. rebuilt patch from freshly matched exact/reduced/removed/anchor context
      3. minimal anchored patch when the target is unique
    """
    meta: dict[str, Any] = {
        "strategy": "",
        "attempts": [],
        "matched_anchor": "",
        "candidate_count": 0,
        "already_applied": False,
        "recovery_error": "",
        "anchors_searched": [],
        "candidates": [],
        "failed_hunk": "",
    }

    # Attempt 1: exact original patch.
    ok, after, err = _apply_codex_update_exact(content, lines, path)
    meta["attempts"].append(
        {
            "attempt": 1,
            "strategy": "exact_context",
            "ok": ok,
            "error": err,
            "patch_fingerprint": _patch_fingerprint(lines),
        }
    )
    if ok:
        # Verify expected post-edit content for each hunk.
        verify_ok, verify_err = _verify_all_hunks(after, lines, path)
        if not verify_ok:
            meta["strategy"] = "exact_context"
            meta["recovery_error"] = verify_err
            meta["attempts"].append(
                {
                    "attempt": 1,
                    "strategy": "post_apply_verify",
                    "ok": False,
                    "error": verify_err,
                }
            )
            return False, content, verify_err, meta
        meta["strategy"] = "exact_context"
        return True, after, "", meta

    if err != "patch_context_not_found" and not err.startswith("patch_context_not_found"):
        meta["strategy"] = "exact_context"
        meta["recovery_error"] = err
        return False, content, err, meta

    # Attempt 2: rebuild using current file content and stable anchors.
    recovered, rec_meta = _recover_update_content(content, lines, path, minimal=False)
    meta["attempts"].append(
        {
            "attempt": 2,
            "strategy": rec_meta.get("strategy") or "rebuilt_context",
            "ok": recovered is not None,
            "error": rec_meta.get("error") or "",
            "matched_anchor": rec_meta.get("matched_anchor") or "",
            "candidate_count": rec_meta.get("candidate_count") or 0,
            "patch_fingerprint": rec_meta.get("patch_fingerprint") or "",
            "already_applied": bool(rec_meta.get("already_applied")),
        }
    )
    meta["matched_anchor"] = str(rec_meta.get("matched_anchor") or "")
    meta["candidate_count"] = int(rec_meta.get("candidate_count") or 0)
    meta["anchors_searched"] = list(rec_meta.get("anchors_searched") or [])
    meta["candidates"] = list(rec_meta.get("candidates") or [])
    meta["failed_hunk"] = str(rec_meta.get("failed_hunk") or "")
    if recovered is not None:
        if rec_meta.get("already_applied"):
            meta["strategy"] = "already_applied"
            meta["already_applied"] = True
            return True, content, "", meta
        verify_ok, verify_err = _verify_all_hunks(recovered, lines, path)
        if not verify_ok:
            meta["strategy"] = str(rec_meta.get("strategy") or "rebuilt_context")
            meta["recovery_error"] = verify_err
            return False, content, verify_err, meta
        meta["strategy"] = str(rec_meta.get("strategy") or "rebuilt_context")
        return True, recovered, "", meta

    # Attempt 3: minimal anchored patch only when unique.
    recovered3, rec_meta3 = _recover_update_content(content, lines, path, minimal=True)
    meta["attempts"].append(
        {
            "attempt": 3,
            "strategy": rec_meta3.get("strategy") or "anchored_insert",
            "ok": recovered3 is not None,
            "error": rec_meta3.get("error") or "",
            "matched_anchor": rec_meta3.get("matched_anchor") or "",
            "candidate_count": rec_meta3.get("candidate_count") or 0,
            "patch_fingerprint": rec_meta3.get("patch_fingerprint") or "",
            "already_applied": bool(rec_meta3.get("already_applied")),
        }
    )
    if rec_meta3.get("matched_anchor"):
        meta["matched_anchor"] = str(rec_meta3.get("matched_anchor") or "")
    meta["candidate_count"] = int(rec_meta3.get("candidate_count") or meta["candidate_count"] or 0)
    meta["anchors_searched"] = list(rec_meta3.get("anchors_searched") or meta["anchors_searched"])
    meta["candidates"] = list(rec_meta3.get("candidates") or meta["candidates"])
    meta["failed_hunk"] = str(rec_meta3.get("failed_hunk") or meta["failed_hunk"])
    if recovered3 is not None:
        if rec_meta3.get("already_applied"):
            meta["strategy"] = "already_applied"
            meta["already_applied"] = True
            return True, content, "", meta
        verify_ok, verify_err = _verify_all_hunks(recovered3, lines, path)
        if not verify_ok:
            meta["strategy"] = str(rec_meta3.get("strategy") or "anchored_insert")
            meta["recovery_error"] = verify_err
            return False, content, verify_err, meta
        meta["strategy"] = str(rec_meta3.get("strategy") or "anchored_insert")
        return True, recovered3, "", meta

    recovery_error = str(
        rec_meta3.get("error")
        or rec_meta.get("error")
        or err
        or "patch_context_not_found"
    )
    meta["strategy"] = str(rec_meta3.get("strategy") or rec_meta.get("strategy") or "ambiguous_context")
    meta["recovery_error"] = recovery_error
    return False, content, "patch_context_not_found", meta


def _patch_fingerprint(lines: Sequence[str]) -> str:
    body = "\n".join(lines)
    return f"len={len(body)}:sha1={__import__('hashlib').sha1(body.encode('utf-8')).hexdigest()[:12]}"


def _verify_all_hunks(after: str, lines: Sequence[str], path: str) -> tuple[bool, str]:
    after_lines = _FileText.from_text(after).lines
    for hunk in _split_update_hunks(lines):
        payload = _hunk_payload(hunk)
        if not payload:
            continue
        old_lines, new_lines, removed, added, _context, err = _hunk_signed_parts(payload)
        if err:
            return False, err
        # If already applied originally, new content must still be present.
        if not _verify_expected_result(after_lines, new_lines=new_lines, added=added, strategy=""):
            # Allow verification when the intended delta is present even if full new_lines
            # (with drifted context) are not an exact contiguous block.
            if added and all(_sequence_present(after_lines, [line]) for line in added):
                continue
            if not added and not removed:
                continue
            if new_lines and _sequence_present(after_lines, new_lines):
                continue
            # Substitution: removed gone, added present.
            if added and removed:
                if all(_sequence_present(after_lines, [line]) for line in added):
                    leftover = [line for line in removed if line not in added]
                    if not leftover or not any(_sequence_present(after_lines, [line]) for line in leftover):
                        continue
            return False, (
                f"post_apply_verify_failed for {path}: expected edit result not present after apply"
            )
    return True, ""


def _recover_update_content(
    content: str,
    lines: Sequence[str],
    path: str,
    *,
    minimal: bool,
) -> tuple[str | None, dict[str, Any]]:
    file_text = _FileText.from_text(content)
    meta: dict[str, Any] = {
        "strategy": "",
        "matched_anchor": "",
        "candidate_count": 0,
        "already_applied": False,
        "error": "",
        "anchors_searched": [],
        "candidates": [],
        "failed_hunk": "",
        "patch_fingerprint": "",
    }
    strategies: list[str] = []
    anchors: list[str] = []
    all_already = True
    rebuilt_hunks: list[list[str]] = []

    for hunk in _split_update_hunks(lines):
        payload = _hunk_payload(hunk)
        if not payload:
            rebuilt_hunks.append(list(hunk))
            continue
        match = _locate_hunk(file_text.lines, payload, path=path, minimal=minimal)
        meta["anchors_searched"].extend(match.anchors_searched)
        if match.candidates:
            meta["candidates"] = list(match.candidates)
            meta["candidate_count"] = match.candidate_count
        if match.error and match.line_index < 0 and not match.already_applied:
            meta["strategy"] = match.strategy or "ambiguous_context"
            meta["error"] = match.error
            meta["matched_anchor"] = match.matched_anchor
            meta["candidate_count"] = match.candidate_count
            meta["failed_hunk"] = "\n".join(payload[:12])
            meta["patch_fingerprint"] = _patch_fingerprint(lines)
            return None, meta

        strategies.append(match.strategy)
        if match.matched_anchor:
            anchors.append(match.matched_anchor)
        if not match.already_applied:
            all_already = False
            _apply_match(file_text, match)

        # Build a minimal rebuilt hunk against current (pre-apply snapshot is already mutated;
        # for fingerprinting we record the strategy only).
        rebuilt_hunks.append(["@@", *[f" {line}" for line in match.new_lines[:0]]])  # placeholder

    if all_already:
        meta["strategy"] = "already_applied"
        meta["already_applied"] = True
        meta["matched_anchor"] = anchors[0] if anchors else "content already matches intended result"
        meta["candidate_count"] = 1
        meta["patch_fingerprint"] = _patch_fingerprint(lines)
        return content, meta

    # Prefer the most specific non-exact strategy used.
    strategy = "rebuilt_context"
    for name in ("anchored_insert", "unique_removed_lines", "whitespace_normalized", "reduced_context", "exact_context"):
        if name in strategies:
            strategy = name
            break
    if strategies:
        # Use last non-empty strategy for multi-hunk summary if preferred not found.
        strategy = next((s for s in strategies if s and s != "already_applied"), strategies[-1]) or strategy

    after = file_text.to_text()
    meta["strategy"] = strategy
    meta["matched_anchor"] = anchors[0] if anchors else ""
    meta["candidate_count"] = 1
    meta["patch_fingerprint"] = _patch_fingerprint(lines)
    # Include a rebuilt minimal patch fingerprint distinct from the original when possible.
    meta["rebuilt"] = True
    return after, meta


def _apply_codex_update(content: str, lines: Sequence[str], path: str) -> tuple[bool, str, str]:
    """Backward-compatible exact-only apply used by tests that monkeypatch internals."""
    return _apply_codex_update_exact(content, lines, path)


def extract_patch_touched_files(patch: Any) -> dict[str, Any]:
    patch_text, normalise_err = _normalise_patch_payload(patch)
    if normalise_err:
        return {"ok": False, "touched_files": [], "error": normalise_err}
    parsed, error = _parse_codex_patch(patch_text)
    if error:
        return {"ok": False, "touched_files": [], "error": error}
    return {"ok": True, "touched_files": [item.path for item in parsed]}


def _safe_apply_patch_direct(
    *,
    repo_root: Path,
    patch: str,
    allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES,
    check_only: bool = False,
    read_files: Sequence[str] | None = None,
    require_read: bool = False,
    enable_recovery: bool = True,
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
    all_attempts: list[dict[str, Any]] = []
    strategies: list[str] = []
    matched_anchors: list[str] = []
    candidate_count = 0
    already_applied_all = True
    any_update = False
    recovery_error = ""
    file_results: list[dict[str, Any]] = []

    for item in parsed:
        target = (repo_root / Path(item.path)).resolve()
        exists = target.exists()
        # Always re-read from disk for each file (fresh content for recovery).
        before = target.read_bytes().decode("utf-8") if exists else ""
        before_by_path.setdefault(item.path, before)
        if item.op == "add":
            if exists:
                result = ApplyPatchResult(
                    ok=False,
                    touched_files=touched_files,
                    error_code="target_exists",
                    error=f"Add File target already exists: {item.path}",
                    attempts=all_attempts,
                ).to_dict()
                _write_patch_history(repo_root=repo_root, patch=patch_text, result=result, touched_files=touched_files, check_only=check_only)
                return result
            invalid = [line for line in item.lines if not line.startswith("+") and line.strip()]
            if invalid:
                result = ApplyPatchResult(
                    ok=False,
                    touched_files=touched_files,
                    error_code="invalid_patch_format",
                    error=f"Add File lines must start with '+': {invalid[0]}",
                    attempts=all_attempts,
                ).to_dict()
                _write_patch_history(repo_root=repo_root, patch=patch_text, result=result, touched_files=touched_files, check_only=check_only)
                return result
            computed[item.path] = _text_from_patch_lines(item.lines, "+")
            changed_ranges.append({"path": item.path, **_changed_range("", str(computed[item.path] or ""))})
            strategies.append("add_file")
            already_applied_all = False
            file_results.append({"path": item.path, "ok": True, "strategy": "add_file"})
        elif item.op == "delete":
            if not exists:
                result = ApplyPatchResult(
                    ok=False,
                    touched_files=touched_files,
                    error_code="target_missing",
                    error=f"Delete File target does not exist: {item.path}",
                    attempts=all_attempts,
                ).to_dict()
                _write_patch_history(repo_root=repo_root, patch=patch_text, result=result, touched_files=touched_files, check_only=check_only)
                return result
            computed[item.path] = None
            changed_ranges.append({"path": item.path, **_changed_range(before, "")})
            strategies.append("delete_file")
            already_applied_all = False
            file_results.append({"path": item.path, "ok": True, "strategy": "delete_file"})
        else:
            if not exists:
                result = ApplyPatchResult(
                    ok=False,
                    touched_files=touched_files,
                    error_code="target_missing",
                    error=f"Update File target does not exist: {item.path}",
                    attempts=all_attempts,
                ).to_dict()
                _write_patch_history(repo_root=repo_root, patch=patch_text, result=result, touched_files=touched_files, check_only=check_only)
                return result
            any_update = True
            current = str(computed.get(item.path, before) or "")
            if enable_recovery:
                patch_ok, after, patch_err, rec_meta = _apply_codex_update_with_recovery(current, item.lines, item.path)
            else:
                patch_ok, after, patch_err = _apply_codex_update_exact(current, item.lines, item.path)
                rec_meta = {
                    "strategy": "exact_context" if patch_ok else "",
                    "attempts": [{"attempt": 1, "strategy": "exact_context", "ok": patch_ok, "error": patch_err}],
                    "matched_anchor": "",
                    "candidate_count": 0,
                    "already_applied": False,
                    "recovery_error": "" if patch_ok else patch_err,
                }
            for attempt in rec_meta.get("attempts") or []:
                attempt_row = dict(attempt)
                attempt_row["path"] = item.path
                all_attempts.append(attempt_row)
            if rec_meta.get("matched_anchor"):
                matched_anchors.append(str(rec_meta["matched_anchor"]))
            candidate_count = max(candidate_count, int(rec_meta.get("candidate_count") or 0))
            if rec_meta.get("strategy"):
                strategies.append(str(rec_meta["strategy"]))
            if not rec_meta.get("already_applied"):
                already_applied_all = False
            if not patch_ok:
                recovery_error = str(rec_meta.get("recovery_error") or patch_err or "")
                snippet = _nearby_snippet(current, item.lines)
                error_detail = _format_recovery_error(
                    path=item.path,
                    patch_err=patch_err,
                    rec_meta=rec_meta,
                )
                file_results.append(
                    {
                        "path": item.path,
                        "ok": False,
                        "strategy": rec_meta.get("strategy") or "ambiguous_context",
                        "error": error_detail,
                        "candidate_count": rec_meta.get("candidate_count") or 0,
                        "matched_anchor": rec_meta.get("matched_anchor") or "",
                        "anchors_searched": rec_meta.get("anchors_searched") or [],
                        "candidates": rec_meta.get("candidates") or [],
                        "failed_hunk": rec_meta.get("failed_hunk") or "",
                    }
                )
                result = ApplyPatchResult(
                    ok=False,
                    touched_files=touched_files,
                    error_code="patch_context_not_found" if "patch_context_not_found" in (patch_err or "") or "ambiguous" in str(rec_meta.get("strategy") or "") or "post_apply_verify" in recovery_error else (
                        "invalid_patch_format" if patch_err and patch_err != "patch_context_not_found" and "post_apply" not in recovery_error else "patch_context_not_found"
                    ),
                    error=error_detail,
                    stdout=snippet,
                    strategy=str(rec_meta.get("strategy") or ""),
                    attempts=all_attempts,
                    matched_anchor=str(rec_meta.get("matched_anchor") or ""),
                    candidate_count=int(rec_meta.get("candidate_count") or 0),
                    already_applied=False,
                    recovery_error=recovery_error or error_detail,
                ).to_dict()
                result["file_results"] = file_results
                result["anchors_searched"] = list(rec_meta.get("anchors_searched") or [])
                result["candidates"] = list(rec_meta.get("candidates") or [])
                result["failed_hunk"] = str(rec_meta.get("failed_hunk") or "")
                _write_patch_history(repo_root=repo_root, patch=patch_text, result=result, touched_files=touched_files, check_only=check_only)
                return result
            computed[item.path] = after
            changed_ranges.append({"path": item.path, **_changed_range(current, after)})
            file_results.append(
                {
                    "path": item.path,
                    "ok": True,
                    "strategy": rec_meta.get("strategy") or "exact_context",
                    "already_applied": bool(rec_meta.get("already_applied")),
                    "matched_anchor": rec_meta.get("matched_anchor") or "",
                    "attempts": rec_meta.get("attempts") or [],
                }
            )

    if not any_update:
        already_applied_all = False

    changed_files = [path for path, after in computed.items() if after != before_by_path.get(path)]
    # If all updates were already applied, treat as idempotent success with no writes.
    if already_applied_all and any_update and not changed_files:
        strategy = "already_applied"
    elif strategies:
        strategy = strategies[0] if len(set(strategies)) == 1 else "multi"
    else:
        strategy = "codex"

    result = ApplyPatchResult(
        ok=True,
        touched_files=touched_files,
        check_only=check_only,
        strategy=strategy,
        stdout="codex patch validated" if check_only else (
            "codex patch already applied" if strategy == "already_applied" else "codex patch applied"
        ),
        attempts=all_attempts,
        matched_anchor=matched_anchors[0] if matched_anchors else "",
        candidate_count=candidate_count,
        changed_ranges=changed_ranges if not (strategy == "already_applied") else [],
        already_applied=strategy == "already_applied",
        recovery_error="",
    ).to_dict()
    result["files_changed"] = [] if check_only or strategy == "already_applied" else changed_files
    result["file_results"] = file_results
    if not check_only and strategy != "already_applied":
        for rel, after in computed.items():
            target = repo_root / rel
            if after is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Preserve newline style detected from the before content when possible.
                if rel in before_by_path and before_by_path[rel]:
                    before_ft = _FileText.from_text(before_by_path[rel])
                    after_ft = _FileText.from_text(after)
                    # Keep original newline character if after was normalized to \n.
                    if before_ft.newline != "\n":
                        normalized = after.replace("\r\n", "\n").replace("\r", "\n")
                        if after_ft.had_final_newline or normalized.endswith("\n"):
                            body = before_ft.newline.join(normalized.splitlines())
                            if after_ft.had_final_newline or normalized.endswith("\n"):
                                body += before_ft.newline
                            after = body
                target.write_text(after, encoding="utf-8", newline="")
    _write_patch_history(repo_root=repo_root, patch=patch_text, result=result, touched_files=touched_files, check_only=check_only)
    return result


def safe_apply_patch(
    *,
    repo_root: Path,
    patch: str,
    allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES,
    check_only: bool = False,
    read_files: Sequence[str] | None = None,
    require_read: bool = False,
    enable_recovery: bool = True,
    action_approval_id: str = "",
) -> dict[str, Any]:
    """Preview read-only, but gate every patch mutation through ActionGateway."""
    if check_only:
        return _safe_apply_patch_direct(
            repo_root=repo_root,
            patch=patch,
            allowed_prefixes=allowed_prefixes,
            check_only=True,
            read_files=read_files,
            require_read=require_read,
            enable_recovery=enable_recovery,
        )
    preview_result = _safe_apply_patch_direct(
        repo_root=repo_root,
        patch=patch,
        allowed_prefixes=allowed_prefixes,
        check_only=True,
        read_files=read_files,
        require_read=require_read,
        enable_recovery=enable_recovery,
    )
    if not preview_result.get("ok"):
        return preview_result
    parsed, parse_error = _parse_codex_patch(_strip_markdown_fences(str(patch or "")))
    if parse_error:
        return preview_result

    from mana_agent.transactional_actions.adapters import ActionAdapter, ActionInvalidatedError
    from mana_agent.transactional_actions.gateway import ApprovalRequired
    from mana_agent.transactional_actions.models import (
        ActionIntent,
        ActionPreview,
        BlastRadius,
        DataDisclosure,
        Reversibility,
        VerificationEvidence,
    )
    from mana_agent.transactional_actions.runtime import default_action_gateway

    root = repo_root.resolve()
    before_hashes = {
        item.path: hashlib.sha256((root / item.path).read_bytes()).hexdigest()
        if (root / item.path).is_file() else "missing"
        for item in parsed
    }
    expected_content: dict[str, str | None] = {}
    for item in parsed:
        if item.op == "add":
            expected_content[item.path] = _text_from_patch_lines(item.lines, "+")
        elif item.op == "delete":
            expected_content[item.path] = None
        else:
            current = expected_content.get(item.path)
            if current is None:
                current = (root / item.path).read_bytes().decode("utf-8")
            if enable_recovery:
                applied, after, _error, _metadata = _apply_codex_update_with_recovery(current, item.lines, item.path)
            else:
                applied, after, _error = _apply_codex_update_exact(current, item.lines, item.path)
            if not applied:
                return preview_result
            expected_content[item.path] = after
    patch_hash = hashlib.sha256(str(patch).encode("utf-8")).hexdigest()

    class PatchAdapter(ActionAdapter):
        def build_intent(self) -> ActionIntent:
            includes_delete = any(item.op == "delete" for item in parsed)
            return ActionIntent(
                parent_task_id="patch-tool",
                actor="model_tool",
                originating_agent="coding_agent",
                tool_name="file",
                operation_name="patch_delete" if includes_delete else "patch",
                target_resources=[str((root / item.path).resolve()) for item in parsed],
                normalized_arguments={"patch_sha256": patch_hash},
                requested_capabilities=sorted({f"file.{item.op}" for item in parsed}),
                expected_side_effects=[f"{item.op} {item.path}" for item in parsed],
                data_disclosure=DataDisclosure.NONE,
                blast_radius=BlastRadius.MULTIPLE_RESOURCES if len(parsed) > 1 else BlastRadius.SINGLE_RESOURCE,
                reversibility=Reversibility.PARTIALLY_REVERSIBLE,
                idempotency_key=f"patch:{hashlib.sha256(json.dumps({'patch': patch_hash, 'root': str(root)}, sort_keys=True).encode()).hexdigest()}",
                verification_plan=["re-run deterministic patch validation against resulting files", "require already-applied evidence"],
                compensation_strategy="Restore each pre-action file snapshot through a separately approved file transaction.",
            )

        def preview(self, action: ActionIntent) -> ActionPreview:
            return ActionPreview(
                summary="apply repository patch",
                resources=[{"path": str((root / item.path).resolve()), "change": item.op, "before_sha256": before_hashes[item.path]} for item in parsed],
                diff=str(patch),
                exact_invocation=action.normalized_arguments,
                expected_side_effects=action.expected_side_effects,
                risks=["patch includes deletion"] if any(item.op == "delete" for item in parsed) else [],
            )

        def execute(self, action: ActionIntent) -> dict[str, Any]:
            current = {
                item.path: hashlib.sha256((root / item.path).read_bytes()).hexdigest()
                if (root / item.path).is_file() else "missing"
                for item in parsed
            }
            if current != before_hashes:
                raise ActionInvalidatedError("patch targets changed after preview; policy and approval must be reevaluated")
            return _safe_apply_patch_direct(
                repo_root=root,
                patch=patch,
                allowed_prefixes=allowed_prefixes,
                check_only=False,
                read_files=read_files,
                require_read=require_read,
                enable_recovery=enable_recovery,
            )

        def verify(self, action: ActionIntent, result: dict[str, Any]) -> VerificationEvidence:
            checks: list[dict[str, Any]] = [{"check": "execution_result", "ok": bool(result.get("ok"))}]
            complete = bool(result.get("ok"))
            for path, expected in expected_content.items():
                target = root / path
                if expected is None:
                    observed = not target.exists()
                    checks.append({"check": "deleted", "path": path, "expected": True, "observed": observed})
                else:
                    expected_hash = hashlib.sha256(expected.encode("utf-8")).hexdigest()
                    observed_hash = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else "missing"
                    observed = observed_hash == expected_hash
                    checks.append({"check": "content_sha256", "path": path, "expected": expected_hash, "observed": observed_hash})
                complete = complete and observed
            return VerificationEvidence(
                complete=complete,
                summary="Every patched resource matched the independently computed expected state." if complete else "Patch verification failed or remained incomplete.",
                checks=checks,
            )

    gateway = default_action_gateway(root)
    adapter = PatchAdapter()
    try:
        outcome = gateway.execute(adapter, approval_id=action_approval_id)
    except ApprovalRequired as exc:
        action = exc.action
        return {
            "ok": False,
            "error_code": "approval_required",
            "error": str(exc),
            "permission_required": True,
            "permission_request_id": action.action_id,
            "action_id": action.action_id,
            "preview": action.preview.redacted() if action.preview else {},
            "preview_digest": action.preview_digest,
            "policy_decision": action.policy_decision.model_dump(mode="json") if action.policy_decision else {},
            "touched_files": [item.path for item in parsed],
        }
    result = dict(outcome.result)
    result.update({
        "ok": outcome.action.state.value == "committed",
        "action_id": outcome.action.action_id,
        "action_state": outcome.action.state.value,
        "verification": outcome.action.verification.model_dump(mode="json") if outcome.action.verification else {},
        "duplicate_suppressed": outcome.duplicate,
    })
    return result


def _format_recovery_error(*, path: str, patch_err: str, rec_meta: dict[str, Any]) -> str:
    strategy = str(rec_meta.get("strategy") or "")
    anchors = list(rec_meta.get("anchors_searched") or [])
    candidates = list(rec_meta.get("candidates") or [])
    failed_hunk = str(rec_meta.get("failed_hunk") or "")
    recovery_error = str(rec_meta.get("recovery_error") or rec_meta.get("error") or patch_err or "")
    parts = [
        f"patch_context_not_found in {path}.",
        "Automatic recovery stopped because the intended location is ambiguous or no reliable anchor exists.",
        f"strategy={strategy or 'none'}",
        f"attempts={len(rec_meta.get('attempts') or [])}",
        f"candidate_count={rec_meta.get('candidate_count') or 0}",
    ]
    if anchors:
        parts.append(f"anchors_searched={anchors[:12]}")
    if candidates:
        parts.append(f"candidate_locations={candidates[:12]}")
    if failed_hunk:
        parts.append(f"failed_hunk=\n{failed_hunk}")
    if recovery_error:
        parts.append(f"reason={recovery_error}")
    parts.append("Re-read the target file and rebuild a minimal unique-context patch.")
    return " ".join(parts)


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
        action_approval_id: str = "",
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
            action_approval_id=action_approval_id,
        )

    return StructuredTool.from_function(
        func=_tool,
        name="apply_patch",
        description=(
            "Safely apply a Codex-style text patch inside the repository. "
            "Supports *** Update File, *** Add File, and *** Delete File blocks. "
            "Matches update hunks by surrounding text context, not line numbers. "
            "When context is stale, re-reads the target, recovers via unique anchors, "
            "rebuilds a minimal patch, and retries within a strict bound. "
            "Ambiguous matches fail without writing. "
            "With check_only=true, validation runs without writing files."
        ),
    )
