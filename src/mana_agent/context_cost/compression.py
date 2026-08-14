"""Deterministic, type-aware and lossless-backed tool-result compression."""

from __future__ import annotations

import csv
import io
import json
import re
import uuid
from collections import Counter
from dataclasses import replace
from typing import Any

from mana_agent.context_cost.artifact_store import ContextArtifactStore
from mana_agent.context_cost.estimator import estimate_value_tokens
from mana_agent.context_cost.models import CompressionEnvelope, ToolResultEnvelope
from mana_agent.utils.redaction import redact_json_line, redact_secrets


def normalize_permitted_result(value: Any) -> Any:
    """Normalize and apply the repository redactor before storage or display."""
    normalized = value
    if not isinstance(value, (str, dict, list, tuple, int, float, bool, type(None))):
        normalized = str(value)
    if isinstance(normalized, tuple):
        normalized = list(normalized)
    if isinstance(normalized, str):
        return redact_json_line(normalized)
    return redact_secrets(normalized)


def detect_content_type(value: Any, *, tool_name: str = "") -> str:
    if isinstance(value, (dict, list)):
        return "json"
    text = str(value or "")
    try:
        if isinstance(json.loads(text), (dict, list)):
            return "json"
    except (json.JSONDecodeError, TypeError):
        pass
    lowered_name = tool_name.lower()
    if "diff" in lowered_name or re.search(r"(?m)^(diff --git|@@ )", text):
        return "diff"
    if "search" in lowered_name or re.search(r"(?m)^.+:\d+(?::\d+)?:", text):
        return "search"
    if "log" in lowered_name or re.search(r"(?im)\b(error|warning|exit code)\b", text):
        return "log"
    if "\t" in text or ("," in text and "\n" in text):
        try:
            rows = list(csv.reader(io.StringIO(text)))
            if len(rows) > 2 and len({len(row) for row in rows[:10]}) == 1:
                return "table"
        except csv.Error:
            pass
    return "text"


def create_tool_result_envelope(
    value: Any,
    *,
    tool_name: str,
    tool_call_id: str = "",
    store: ContextArtifactStore,
    session_id: str,
    repository_id: str,
    workspace_id: str,
    status: str = "success",
    source_refs: tuple[str, ...] = (),
    replayable: bool = True,
    sensitive: bool = False,
) -> ToolResultEnvelope:
    permitted = normalize_permitted_result(value)
    content_type = detect_content_type(permitted, tool_name=tool_name)
    reference = store.put(
        permitted,
        session_id=session_id,
        repository_id=repository_id,
        workspace_id=workspace_id,
        content_type=content_type,
    )
    summary_value = permitted
    if content_type == "json" and isinstance(permitted, str):
        try:
            summary_value = json.loads(permitted)
        except json.JSONDecodeError:
            summary_value = permitted
    summary, important, omitted = _compact(summary_value, content_type)
    original_tokens = estimate_value_tokens(permitted)

    inline_proj = {
        "summary": summary,
        "important_items": important,
        "omitted_counts": omitted,
        "content_type": content_type,
    }
    proj_tokens = estimate_value_tokens(inline_proj)
    has_omitted = bool(omitted and any(v > 0 for v in omitted.values()))

    return ToolResultEnvelope(
        tool_name=tool_name,
        tool_call_id=tool_call_id or f"tool-{uuid.uuid4().hex[:12]}",
        status=status,
        artifact_ref=reference.artifact_id,
        content_hash=reference.content_hash,
        original_tokens=original_tokens,
        projection_tokens=proj_tokens,
        inline_projection=inline_proj,
        truncated=has_omitted,
        more_available=has_omitted,
        source_refs=tuple(source_refs),
        content_type=content_type,
        replayable=replayable,
        sensitive=sensitive,
    )


def compress_tool_result(
    value: Any,
    *,
    tool_name: str,
    store: ContextArtifactStore,
    session_id: str,
    repository_id: str,
    workspace_id: str,
) -> CompressionEnvelope:
    permitted = normalize_permitted_result(value)
    content_type = detect_content_type(permitted, tool_name=tool_name)
    reference = store.put(
        permitted,
        session_id=session_id,
        repository_id=repository_id,
        workspace_id=workspace_id,
        content_type=content_type,
    )
    summary_value = permitted
    if content_type == "json" and isinstance(permitted, str):
        try:
            summary_value = json.loads(permitted)
        except json.JSONDecodeError:
            summary_value = permitted
    summary, important, omitted = _compact(summary_value, content_type)
    original_tokens = estimate_value_tokens(permitted)
    envelope = CompressionEnvelope(
        artifact_ref=reference,
        tool_name=tool_name,
        content_type=content_type,
        summary=summary,
        important_items=tuple(important),
        omitted_counts=omitted,
        original_token_estimate=original_tokens,
        compact_token_estimate=0,
        compression_ratio=0.0,
        content_hash=reference.content_hash,
    )
    compact_tokens = estimate_value_tokens(envelope.as_dict())
    envelope = replace(
        envelope,
        compact_token_estimate=compact_tokens,
        compression_ratio=round(compact_tokens / original_tokens if original_tokens else 1.0, 6),
    )
    corrected_tokens = estimate_value_tokens(envelope.as_dict())
    return replace(
        envelope,
        compact_token_estimate=corrected_tokens,
        compression_ratio=round(corrected_tokens / original_tokens if original_tokens else 1.0, 6),
    )


def _compact(value: Any, content_type: str) -> tuple[str, list[Any], dict[str, int]]:
    if content_type == "json":
        return _compact_json(value)
    text = str(value or "")
    if content_type == "diff":
        lines = text.splitlines()
        retained = [line for line in lines if line.startswith(("diff --git", "--- ", "+++ ", "@@", "+", "-"))][:120]
        return f"Diff with {len(lines)} lines; changed hunks retained.", retained, {"lines": max(0, len(lines) - len(retained))}
    if content_type == "search":
        lines = text.splitlines()
        located = [line for line in lines if re.search(r":\d+(?::\d+)?:", line)][:80]
        return f"Search output with {len(lines)} lines and {len(located)} retained locations.", located, {"lines": max(0, len(lines) - len(located))}
    if content_type == "log":
        lines = text.splitlines()
        significant = [line for line in lines if re.search(r"(?i)\b(error|warning|failed|exit code|traceback)\b", line)]
        retained = (lines[:20] + significant[:60] + lines[-20:])[:100]
        repeats = sum(count - 1 for count in Counter(lines).values() if count > 1)
        return f"Log with {len(lines)} lines; errors, warnings, head, and tail retained.", retained, {"lines": max(0, len(lines) - len(retained)), "repeated_lines": repeats}
    if content_type == "table":
        rows = list(csv.reader(io.StringIO(text)))
        retained = rows[:6] + (rows[-3:] if len(rows) > 9 else [])
        columns = rows[0] if rows else []
        return f"Table with {max(0, len(rows) - 1)} rows and {len(columns)} columns.", retained, {"rows": max(0, len(rows) - len(retained))}
    lines = text.splitlines()
    headings = [line for line in lines if re.match(r"^\s{0,3}#{1,6}\s", line)]
    retained = (headings[:30] + lines[:30] + (lines[-15:] if len(lines) > 45 else []))[:75]
    return f"Text with {len(lines)} lines; headings, head, and tail retained.", retained, {"lines": max(0, len(lines) - len(retained))}


def _compact_json(value: Any) -> tuple[str, list[Any], dict[str, int]]:
    if isinstance(value, dict):
        keys = list(value)
        important_keys = [key for key in keys if str(key).lower() in {"error", "errors", "status", "id", "ids", "ok", "success", "path", "paths", "count", "total", "exit_code", "decision"}]
        selected = important_keys + [key for key in keys if key not in important_keys][:12]
        important = [{str(key): _representative(value[key])} for key in selected[:20]]
        return f"JSON object with {len(keys)} top-level keys.", important, {"keys": max(0, len(keys) - len(selected[:20]))}
    if isinstance(value, list):
        important = [_representative(item) for item in value[:5]]
        if len(value) > 7:
            important.extend(_representative(item) for item in value[-2:])
        return f"JSON array with {len(value)} records.", important, {"records": max(0, len(value) - len(important))}
    return f"JSON scalar of type {type(value).__name__}.", [value], {}


def _representative(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _representative(item) for key, item in list(value.items())[:12]}
    if isinstance(value, list):
        return [_representative(item) for item in value[:3]] + ([f"... {len(value) - 3} omitted"] if len(value) > 3 else [])
    if isinstance(value, str) and len(value) > 500:
        return value[:350] + f" ... [{len(value) - 350} chars omitted]"
    return value


def render_envelope(envelope: Any) -> str:
    if hasattr(envelope, "as_dict"):
        return json.dumps(envelope.as_dict(), ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(envelope, dict):
        return json.dumps(envelope, ensure_ascii=False, sort_keys=True, default=str)
    return str(envelope)


__all__ = [
    "compress_tool_result",
    "create_tool_result_envelope",
    "detect_content_type",
    "normalize_permitted_result",
    "render_envelope",
]
