"""Shared operation projection helpers. Observations only, never task control."""
from __future__ import annotations
from typing import Any

PREVIEW_LIMIT = 64_000


def merge_operation(previous: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    old_meta = previous.get("metadata") or previous.get("details") or {}
    new_meta = event.get("metadata") or event.get("details") or {}
    metadata = {**old_meta, **new_meta}
    if old_meta.get("trace_complete") is False:
        metadata["trace_complete"] = False
    metadata["omitted_bytes"] = max(int(old_meta.get("omitted_bytes") or 0), int(new_meta.get("omitted_bytes") or 0))
    merged = {**previous, **event, "metadata": metadata, "details": metadata,
              "started_at": previous.get("started_at") or event.get("started_at")}
    # Later stream envelopes must not remove command/path/actor provenance.
    for key in ("command", "path", "parent_agent_id", "parent_execution_id", "execution_level", "operation_attempt_id"):
        merged[key] = event.get(key) or previous.get(key) or metadata.get(key)
    channel = new_meta.get("stream_channel") or event.get("stream_channel")
    if channel:
        streams = dict(old_meta.get("streams") or {})
        offsets = dict(old_meta.get("stream_offsets") or {})
        offset = int(new_meta.get("stream_offset", event.get("stream_offset", 0)))
        chunk = str(event.get("output_preview") or new_meta.get("output_preview") or "")
        previous_end = int(offsets.get(channel, 0))
        if offset >= previous_end:
            gap = "\n[output omitted; open retained details]\n" if offset > previous_end else ""
            if len(chunk) < int(new_meta.get("content_length", len(chunk))):
                chunk += "\n[preview truncated; open retained details]\n"
                metadata["preview_truncated"] = True
            streams[channel] = (str(streams.get(channel, "")) + gap + chunk)[-PREVIEW_LIMIT:]
            offsets[channel] = offset + int(new_meta.get("content_length", len(chunk)))
        metadata["streams"], metadata["stream_offsets"] = streams, offsets
    return merged


def operation_details(event: dict[str, Any]) -> str:
    meta = event.get("metadata") or event.get("details") or {}
    command = event.get("command") or meta.get("command")
    lines = []
    if command:
        lines.append(f"$ {command}")
    for key, label in (("cwd", "cwd"), ("provenance", "Execution"), ("exit_code", "Exit code"),
                       ("termination_reason", "Termination"), ("timeout_seconds", "Timeout (s)")):
        value = meta.get(key, event.get(key))
        if value is not None and value != "":
            lines.append(f"{label}: {value}")
    for channel, body in (meta.get("streams") or {}).items():
        lines.extend((f"--- {channel} ---", str(body)))
    if meta.get("diff") and not (meta.get("streams") or {}).get("diff"):
        label = "observed turn diff" if meta.get("diff_scope") == "turn" else "provider-reported diff" if meta.get("provenance") == "codex_backend_owned" else "applied diff"
        lines.extend((f"--- {label} ---", str(meta["diff"])))
    if meta.get("trace_complete") is False:
        lines.append(f"Trace incomplete ({meta.get('omitted_bytes', 0)} bytes omitted)")
    return "\n".join(lines)


def execution_tree(events: list[dict[str, Any]]) -> str:
    nodes: dict[str, tuple[str, str]] = {}
    labels: dict[str, str] = {}
    for event in events:
        meta = event.get("metadata") or {}
        actor = str(event.get("agent_id") or meta.get("agent_id") or "")
        parent = str(event.get("parent_agent_id") or meta.get("parent_agent_id") or "")
        if actor:
            nodes[actor] = (parent, str(event.get("status") or "running"))
            level = event.get("execution_level") or meta.get("execution_level")
            labels[actor] = ("ToolRunnerAgent" if level == 3 else "CodingAgent" if level == 2 else actor) + (f" ({actor})" if level in {2, 3} else "")
    lines = []
    def draw(actor: str, depth: int, visited: set[str]):
        if actor in visited:
            return
        visited = visited | {actor}
        parent, status = nodes.get(actor, ("", ""))
        lines.append(f"{'  ' * depth}{labels.get(actor, actor)} · {status}".rstrip(" ·"))
        for child, (owner, _) in nodes.items():
            if owner == actor:
                draw(child, depth + 1, visited)
    for actor, (parent, _) in nodes.items():
        if not parent or parent not in nodes:
            if parent:
                lines.append(parent)
            draw(actor, 1 if parent else 0, set())
    return "\n".join(lines)
