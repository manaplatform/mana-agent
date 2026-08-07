"""User-facing formatting for MCP provider results.

MCP tools (especially documentation providers such as Context7) return a
transport envelope with content blocks. Dumping that envelope as JSON is hard
to read. This module extracts text content and renders a compact, reviewable
message while still marking the payload as untrusted external data.
"""

from __future__ import annotations

import json
from typing import Any

from mana_agent.evals.redaction import redact_text
from mana_agent.utils.redaction import redact_secrets

_DEFAULT_BODY_LIMIT = 12_000
_DEFAULT_PREVIEW_LIMIT = 4_000

# Operation-name markers used only for presentation labels, not routing.
_DOC_OPERATION_MARKERS = (
    "doc",
    "docs",
    "library",
    "reference",
    "readme",
    "api-ref",
    "apiref",
)


def extract_mcp_text_content(result: dict[str, Any] | None) -> str:
    """Return concatenated human-readable text from an MCP tool result."""
    if not isinstance(result, dict):
        return ""
    parts: list[str] = []
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            text = _content_item_text(item)
            if text:
                parts.append(text)
    elif isinstance(content, str) and content.strip():
        parts.append(content.strip())
    structured = result.get("structured_content")
    if not parts and structured is not None:
        structured_text = _structured_content_text(structured)
        if structured_text:
            parts.append(structured_text)
    if not parts:
        for key in ("text", "message", "error", "result"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
                break
    return "\n\n".join(parts).strip()


def is_documentation_mcp_operation(operation_name: str) -> bool:
    """Return whether the MCP operation name looks documentation-oriented."""
    name = str(operation_name or "").casefold().replace("_", "-")
    if not name:
        return False
    return any(marker in name for marker in _DOC_OPERATION_MARKERS)


def format_mcp_result_preview(
    result: dict[str, Any] | None,
    *,
    limit: int = _DEFAULT_PREVIEW_LIMIT,
) -> str:
    """Compact activity preview: prefer extracted text over raw envelope JSON."""
    if not result:
        return ""
    safe = redact_secrets(result) if isinstance(result, dict) else result
    if not isinstance(safe, dict):
        return _truncate(redact_text(str(safe)), limit)
    text = extract_mcp_text_content(safe)
    if text:
        return _truncate(redact_text(text), limit)
    return _truncate(_compact_json(safe), limit)


def format_mcp_completion_message(
    *,
    provider_id: str,
    operation_name: str,
    result: dict[str, Any] | None,
    body_limit: int = _DEFAULT_BODY_LIMIT,
) -> str:
    """Create a deterministic user-visible receipt for an approved MCP action."""
    target = f"mcp.{provider_id}.{operation_name}".strip(".")
    header = f"Approved MCP action completed: `{target}`."
    if not result:
        return f"{header} The provider returned no displayable result."

    safe = redact_secrets(result) if isinstance(result, dict) else result
    if not isinstance(safe, dict):
        body = _truncate(redact_text(str(safe)), body_limit)
        return (
            f"{header}\n\n"
            "Provider result (untrusted data):\n"
            f"```\n{body}\n```"
        )

    text = extract_mcp_text_content(safe)
    is_error = bool(safe.get("is_error")) or safe.get("ok") is False
    if text:
        body = _truncate(redact_text(text), body_limit)
        label = _result_label(operation_name=operation_name, is_error=is_error, has_text=True)
        meta = _status_line(safe)
        parts = [header, "", f"{label} (untrusted data):", "", body]
        if meta:
            parts.extend(["", meta])
        return "\n".join(parts)

    body = _truncate(_compact_json(safe), body_limit)
    label = "Provider error" if is_error else "Provider result"
    return (
        f"{header}\n\n"
        f"{label} (untrusted data):\n"
        f"```json\n{body}\n```"
    )


def _result_label(*, operation_name: str, is_error: bool, has_text: bool) -> str:
    if is_error:
        return "Provider error"
    if has_text and is_documentation_mcp_operation(operation_name):
        return "Documentation"
    return "Provider result"


def _status_line(result: dict[str, Any]) -> str:
    bits: list[str] = []
    if "ok" in result:
        bits.append("ok" if result.get("ok") else "failed")
    if result.get("is_error"):
        bits.append("provider_error")
    duration = result.get("duration_ms")
    if isinstance(duration, (int, float)):
        bits.append(f"{duration:g} ms")
    transport = str(result.get("transport") or "").strip()
    if transport:
        bits.append(transport)
    if not bits:
        return ""
    return f"_Status: {' · '.join(bits)}_"


def _content_item_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return ""
    block_type = str(item.get("type") or "").strip().casefold()
    if block_type and block_type not in {"text", "output_text", "resource"}:
        # Keep non-text media out of the primary doc body.
        if block_type in {"image", "audio", "video", "blob", "resource_link"}:
            return ""
    for key in ("text", "markdown", "content"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            nested = value.get("text") or value.get("value")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    resource = item.get("resource")
    if isinstance(resource, dict):
        for key in ("text", "blob"):
            value = resource.get(key)
            if isinstance(value, str) and value.strip() and key == "text":
                return value.strip()
    return ""


def _structured_content_text(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "markdown", "content", "documentation", "docs"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
        # Prefer a short pretty form over losing structured docs entirely.
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, list):
        parts = [part for part in (_content_item_text(item) for item in value) if part]
        if parts:
            return "\n\n".join(parts)
    return ""


def _compact_json(value: dict[str, Any]) -> str:
    """JSON-encode a result while dropping null/empty noise fields."""
    cleaned = _drop_empty(value)
    try:
        return json.dumps(cleaned, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return redact_text(str(value))


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            cleaned = _drop_empty(item)
            if cleaned is None or cleaned == "" or cleaned == [] or cleaned == {}:
                continue
            out[str(key)] = cleaned
        return out
    if isinstance(value, list):
        return [item for item in (_drop_empty(item) for item in value) if item not in (None, "", [], {})]
    return value


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "… [truncated]"


__all__ = [
    "extract_mcp_text_content",
    "format_mcp_completion_message",
    "format_mcp_result_preview",
    "is_documentation_mcp_operation",
]
