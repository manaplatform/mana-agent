from __future__ import annotations

import json
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mana_agent.cli.events import ChatEvent, normalize_event_kind, normalize_event_status
from mana_agent.telemetry.tokens import TokenUsage, TokenUsageTracker


def format_token_usage(usage: TokenUsage) -> str:
    prefix = "~" if usage.estimated and usage.total_tokens else ""
    if usage.total_tokens <= 0:
        return "tokens: unavailable"
    parts = [
        f"in {prefix}{usage.input_tokens}",
        f"out {prefix}{usage.output_tokens}",
    ]
    if usage.cached_input_tokens:
        parts.append(f"cached {prefix}{usage.cached_input_tokens}")
    if usage.reasoning_tokens:
        parts.append(f"reasoning {prefix}{usage.reasoning_tokens}")
    if usage.tool_result_tokens:
        parts.append(f"tools {prefix}{usage.tool_result_tokens}")
    parts.append(f"total {prefix}{usage.total_tokens}")
    return "tokens: " + " · ".join(parts)


def _status_icon(status: str, *, plain: bool = False) -> str:
    normalized = normalize_event_status(status)
    if plain:
        return {
            "queued": "-",
            "running": ">",
            "success": "ok",
            "done": "ok",
            "failed": "x",
            "failure": "x",
            "skipped": "skip",
            "waiting": "?",
            "warning": "!",
        }.get(normalized, "-")
    return {
        "queued": "•",
        "running": "◌",
        "success": "✓",
        "done": "✓",
        "failed": "✗",
        "failure": "✗",
        "skipped": "↷",
        "waiting": "?",
        "warning": "!",
    }.get(normalized, "•")


def _event_actor_label(event: ChatEvent) -> str:
    if event.subagent_id:
        return str(event.subagent_id)
    if str(event.agent_id or "").startswith("subagent_"):
        return str(event.agent_id)
    role = str(event.metadata.get("agent_role") or event.metadata.get("role") or "").strip()
    return role or "main"


def _event_model_label(event: ChatEvent) -> str:
    level = str(event.metadata.get("model_level") or "").strip()
    model = str(event.metadata.get("resolved_model") or "").strip()
    if level and model:
        return f"{level} • {model}"
    return level or model


def _tool_name(event: ChatEvent) -> str:
    return str(event.metadata.get("tool_name") or event.title or "tool")


def _duration_label(event: ChatEvent) -> str:
    if event.duration_ms is None:
        return ""
    milliseconds = int(event.duration_ms)
    return f"{milliseconds}ms" if milliseconds < 1000 else f"{milliseconds / 1000:.1f}s"


def _compact_tool_line(event: ChatEvent) -> str:
    actor = _event_actor_label(event)
    model = _event_model_label(event)
    status = _status_icon(event.status, plain=False)
    duration = _duration_label(event)
    parts = [actor]
    if model:
        parts.append(model)
    parts.append(f"{_tool_name(event)} {status}{(' ' + duration) if duration else ''}".strip())
    return " • ".join(parts)


class EventRenderer:
    def __init__(self, *, mode: str = "rich", trace_mode: str = "compact") -> None:
        self.mode = self.normalize_mode(mode)
        self.trace_mode = self.normalize_trace_mode(trace_mode)

    @staticmethod
    def normalize_mode(mode: str) -> str:
        value = str(mode or "rich").strip().lower()
        return value if value in {"rich", "compact", "plain", "json"} else "rich"

    @staticmethod
    def normalize_trace_mode(mode: str) -> str:
        value = str(mode or "compact").strip().lower()
        return value if value in {"off", "compact", "full", "logs"} else "compact"

    def format_usage(self, usage: TokenUsage | None) -> str:
        return format_token_usage(usage or TokenUsage()).removeprefix("tokens: ")

    def render_event(self, event: ChatEvent) -> Any:
        if self.mode == "json":
            return json.dumps(event.as_dict(), ensure_ascii=False)
        if self.mode == "plain":
            return self._plain_event(event)
        if self.mode == "compact":
            return self._compact_event(event)
        return self._rich_event(event)

    def _plain_event(self, event: ChatEvent) -> str:
        duration = f"{event.duration_ms / 1000:.1f}s" if event.duration_ms else ""
        message = f" - {event.message}" if event.message else ""
        return f"[{_status_icon(event.status, plain=True)}] {event.step_id or event.event_id} {event.title}{message} {duration}".strip()

    def _compact_event(self, event: ChatEvent) -> Text:
        duration = f"{event.duration_ms / 1000:.1f}s" if event.duration_ms else ""
        text = Text()
        text.append(_status_icon(event.status), style="green" if event.status in {"success", "done"} else "cyan")
        text.append(f" {event.step_id or event.event_id[-6:]} ", style="dim")
        text.append(event.title or event.type, style="bold")
        if event.message:
            text.append(f" - {event.message}", style="dim")
        if duration:
            text.append(f" {duration}", style="dim")
        return text

    def _rich_event(self, event: ChatEvent) -> Panel:
        body = Table.grid(padding=(0, 1), expand=True)
        body.add_column(justify="right", no_wrap=True, style="bold")
        body.add_column(ratio=1, overflow="fold")
        body.add_row("status", f"{_status_icon(event.status)} {event.status}")
        if event.duration_ms:
            body.add_row("time", f"{event.duration_ms / 1000:.1f}s")
        token_text = self.format_usage(event.token_usage)
        if token_text != "unavailable":
            body.add_row("tokens", token_text)
        if event.message:
            label = "decision" if event.type == "agent.decision" else "summary"
            body.add_row(label, event.message)
        if self.trace_mode == "full" and event.metadata:
            body.add_row("metadata", json.dumps(event.metadata, ensure_ascii=False, default=str)[:1200])
        title = f"Step {event.step_id} · {event.title}" if event.step_id else event.title or event.type
        return Panel(body, title=title, title_align="left", border_style="cyan", box=box.ROUNDED)

    def render_events(self, events: list[ChatEvent], *, title: str = "Step timeline") -> Any:
        if self.mode == "json":
            return "\n".join(json.dumps(event.as_dict(), ensure_ascii=False) for event in events)
        if self.mode in {"plain", "compact"}:
            lines = [self.render_event(event) for event in events]
            return Group(*lines) if self.mode == "compact" else "\n".join(str(line) for line in lines)
        return Panel(Group(*(self._compact_event(event) for event in events)), title=title, box=box.ROUNDED)

    def render_inline_status(self, events: list[ChatEvent], tracker: TokenUsageTracker | None = None) -> Any:
        summary = _status_summary(events, tracker)
        if self.mode == "json":
            return json.dumps({"type": "inline_status", **summary}, ensure_ascii=False)
        rows = [
            f"Mana is working... elapsed {summary['elapsed']}",
            f"  {_status_icon('success')} files read {summary['files_read']}",
            f"  {_status_icon('success')} files changed {summary['files_changed']}",
            f"  {_status_icon('running') if summary['active_tool'] else '·'} tool {summary['active_tool'] or 'idle'}",
            f"  {_status_icon('running') if summary['active_subagent'] else '·'} subagent {summary['active_subagent'] or 'idle'}",
            f"  {_status_icon(summary['tests_status'])} tests {summary['tests_status']}",
        ]
        if summary["tokens"]:
            rows.append(f"  tok: {summary['tokens']}")
        if summary["waiting_approval"]:
            rows.append("  ? approval waiting")
        text = "\n".join(rows)
        if self.mode == "plain":
            return text
        return Panel(text, title="Inline status", title_align="left", border_style="cyan", box=box.ROUNDED)

    def render_timeline(self, events: list[ChatEvent], *, max_rows: int | None = None, scroll_offset: int = 0) -> Any:
        normalized_events = normalize_visible_events(events)
        if self.mode == "json":
            return "\n".join(json.dumps(event.as_dict(), ensure_ascii=False) for event in normalized_events)
        table = Table(show_header=True, header_style="bold", box=box.SIMPLE, expand=True)
        table.add_column("time", no_wrap=True)
        table.add_column("status", no_wrap=True)
        table.add_column("event", overflow="fold")
        table.add_column("summary", overflow="fold")
        visible_count = max(1, int(max_rows or 18))
        offset = max(0, int(scroll_offset or 0))
        end = len(normalized_events) - offset if offset else len(normalized_events)
        start = max(0, end - visible_count)
        for event in normalized_events[start:end]:
            table.add_row(
                _clock_label(event),
                _status_icon(event.status),
                timeline_event_label(event),
                timeline_summary(event),
            )
        return Panel(table, title="Timeline", border_style="cyan", box=box.ROUNDED)

    def render_tools_table(self, events: list[ChatEvent]) -> Any:
        tool_events = [event for event in events if event.type.startswith("tool.")]
        if self.mode == "json":
            return "\n".join(json.dumps(event.as_dict(), ensure_ascii=False) for event in tool_events)
        table = Table(show_header=True, header_style="bold", box=box.SIMPLE, expand=True)
        table.add_column("status", no_wrap=True)
        table.add_column("tool", no_wrap=True)
        table.add_column("target", overflow="fold")
        table.add_column("duration", no_wrap=True)
        table.add_column("summary", overflow="fold")
        for event in tool_events[-50:]:
            table.add_row(
                _status_icon(event.status),
                _tool_name(event),
                str(event.metadata.get("target") or event.metadata.get("path") or event.metadata.get("args_summary") or "-"),
                _duration_label(event) or "-",
                str(event.metadata.get("result_summary") or event.message or "-"),
            )
        return Panel(table, title="Tools", border_style="cyan", box=box.ROUNDED)

    def render_files(self, events: list[ChatEvent]) -> Any:
        file_events = [
            event
            for event in events
            if event.type.startswith(("file.", "patch."))
            or str(event.metadata.get("tool_name") or "") in {"read_file", "edit_file", "multi_edit_file", "apply_patch", "write_file", "create_file", "delete_file"}
            or event.metadata.get("path")
            or event.metadata.get("file_path")
        ]
        if self.mode == "json":
            return "\n".join(json.dumps(event.as_dict(), ensure_ascii=False) for event in file_events)
        rows: list[str] = []
        for event in file_events[-80:]:
            marker = "R"
            tool_name = str(event.metadata.get("tool_name") or "")
            if event.type.startswith(("file.changed", "patch.")) or tool_name in {"edit_file", "multi_edit_file", "apply_patch", "write_file", "create_file", "delete_file"}:
                marker = "M"
            if event.metadata.get("change_kind") == "added":
                marker = "A"
            if event.metadata.get("change_kind") == "deleted":
                marker = "D"
            path = str(event.metadata.get("path") or event.metadata.get("file_path") or event.title or "-")
            rows.append(f"{marker}  {path}")
        return Panel("\n".join(rows) if rows else "No file activity yet.", title="Files", border_style="green", box=box.ROUNDED)

    def render_diff(self, events: list[ChatEvent]) -> Any:
        diff_events = [
            event
            for event in events
            if event.type.startswith(("file.changed", "patch."))
            or str(event.metadata.get("tool_name") or "") in {"edit_file", "multi_edit_file", "apply_patch", "write_file", "create_file", "delete_file"}
        ]
        if self.mode == "json":
            return "\n".join(json.dumps(event.as_dict(), ensure_ascii=False) for event in diff_events)
        table = Table(show_header=True, header_style="bold", box=box.SIMPLE, expand=True)
        table.add_column("file", overflow="fold")
        table.add_column("+", justify="right", no_wrap=True)
        table.add_column("-", justify="right", no_wrap=True)
        table.add_column("summary", overflow="fold")
        for event in diff_events[-50:]:
            table.add_row(
                str(event.metadata.get("path") or event.metadata.get("file_path") or event.title or "-"),
                str(event.metadata.get("insertions") or event.metadata.get("added_lines") or 0),
                str(event.metadata.get("deletions") or event.metadata.get("deleted_lines") or 0),
                str(event.metadata.get("result_summary") or event.message or "-"),
            )
        return Panel(table if diff_events else Text("No diffs recorded yet."), title="Diff", border_style="magenta", box=box.ROUNDED)

    def render_tests(self, events: list[ChatEvent]) -> Any:
        test_events = [event for event in events if event.type.startswith("test.") or event.metadata.get("command")]
        if self.mode == "json":
            return "\n".join(json.dumps(event.as_dict(), ensure_ascii=False) for event in test_events)
        table = Table(show_header=True, header_style="bold", box=box.SIMPLE, expand=True)
        table.add_column("status", no_wrap=True)
        table.add_column("command", overflow="fold")
        table.add_column("duration", no_wrap=True)
        table.add_column("summary", overflow="fold")
        for event in test_events[-50:]:
            table.add_row(
                _status_icon(event.status),
                str(event.metadata.get("command") or event.title or "test"),
                _duration_label(event) or "-",
                str(event.metadata.get("result_summary") or event.message or "-"),
            )
        return Panel(table if test_events else Text("No verification runs yet."), title="Tests", border_style="yellow", box=box.ROUNDED)

    def render_tokens(self, tracker: TokenUsageTracker) -> Any:
        snapshot = tracker.snapshot()
        if self.mode == "json":
            return json.dumps({"type": "tokens", **snapshot}, ensure_ascii=False)
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(overflow="fold")
        current = tracker.by_turn.get(tracker.current_turn_id, TokenUsage())
        table.add_row("turn", self.format_usage(current))
        table.add_row("session", self.format_usage(tracker.session_total))
        table.add_row("cache", f"read {tracker.session_total.cached_input_tokens} · write {tracker.session_total.cache_creation_tokens}")
        table.add_row("subagents", str(sum(item.total_tokens for item in tracker.by_subagent.values())))
        table.add_row("tools injected", str(sum(item.tool_result_tokens for item in tracker.by_tool_result.values())))
        table.add_row("accounting", "estimated values are prefixed with ~; exact values require provider usage")
        return Panel(table, title="Token usage", border_style="magenta", box=box.ROUNDED)

    def render_tool_activity(self, events: list[ChatEvent]) -> Any:
        tool_events = [event for event in events if event.type.startswith("tool.")]
        if self.mode == "json":
            return "\n".join(json.dumps(event.as_dict(), ensure_ascii=False) for event in tool_events)
        if self.mode in {"plain", "compact"}:
            lines = [_compact_tool_line(event) for event in tool_events[-30:]]
            return Group(*(Text(line) for line in lines)) if self.mode == "compact" else "\n".join(lines)

        rows: list[Text] = []
        grouped: dict[str, list[ChatEvent]] = {}
        for event in tool_events[-30:]:
            grouped.setdefault(_event_actor_label(event), []).append(event)
        for actor, actor_events in grouped.items():
            header = Text(actor, style="bold cyan")
            model = next((label for label in (_event_model_label(item) for item in actor_events) if label), "")
            if model:
                header.append(f" • {model}", style="dim")
            rows.append(header)
            for index, event in enumerate(actor_events):
                branch = "└─" if index == len(actor_events) - 1 else "├─"
                purpose = str(event.metadata.get("args_summary") or event.message or "").strip()
                duration = _duration_label(event)
                line = Text(f"  {branch} ")
                line.append(_tool_name(event), style="bold")
                line.append(f" {_status_icon(event.status)}")
                if duration:
                    line.append(f" {duration}", style="dim")
                if purpose:
                    line.append(f": {purpose}", style="dim")
                rows.append(line)
        return Panel(Group(*rows) if rows else Text("No tool calls yet."), title="Tool activity", border_style="cyan", box=box.ROUNDED)

    def render_subagents(self, events: list[ChatEvent]) -> Any:
        subagent_events = [event for event in events if event.type.startswith("subagent.") or event.subagent_id]
        if self.mode == "json":
            return "\n".join(json.dumps(event.as_dict(), ensure_ascii=False) for event in subagent_events)
        table = Table(show_header=True, header_style="bold", box=box.SIMPLE, expand=True)
        table.add_column("ID", no_wrap=True)
        table.add_column("Role", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Current step", overflow="fold")
        table.add_column("Tokens", justify="right", no_wrap=True)
        table.add_column("Summary", overflow="fold")
        latest: dict[str, ChatEvent] = {}
        models: dict[str, str] = {}
        for event in subagent_events:
            key = str(event.subagent_id or event.agent_id or event.event_id)
            latest[key] = event
            if not models.get(key):
                models[key] = _event_model_label(event)
        for key, event in latest.items():
            model = models.get(key, "")
            role = str(event.metadata.get("role") or event.metadata.get("agent_role") or event.title or "subagent")
            if model:
                role = f"{role} • {model}"
            table.add_row(
                key,
                role,
                event.status,
                str(event.metadata.get("current_step") or event.step_id or "-"),
                str(event.token_usage.total_tokens if event.token_usage else 0),
                event.message or "-",
            )
        return Panel(table, title="Subagents", border_style="green", box=box.ROUNDED)

    def render_log_lines(self, lines: list[str]) -> Any:
        if self.mode == "json":
            return json.dumps({"type": "trace.logs", "lines": lines}, ensure_ascii=False)
        body = "\n".join(lines[-40:]) if lines else "No trace log lines available."
        if self.mode == "plain":
            return "Trace logs\n" + body
        return Panel(body, title="Trace logs", border_style="yellow", box=box.ROUNDED)


class InlineChatRenderer:
    """Append-only renderer for the default terminal chat experience."""

    def __init__(self, console: Console, *, mode: str = "rich") -> None:
        self.console = console
        self.mode = EventRenderer.normalize_mode(mode)
        self._rendered_signatures: set[tuple[str, str, str, str, str, str]] = set()
        # Track tool lifecycle by stable event_id so start/finish for the *same* activity
        # do not produce noisy duplicate transcript lines. Running tools are primarily
        # displayed via the LiveToolActivity region; only terminal state yields a line here.
        self._tool_event_ids: dict[str, str] = {}  # event_id -> last_status seen

    def render_event(self, event: ChatEvent) -> None:
        line = self.format_event(event)
        if not line:
            return
        if self.mode == "json":
            self.console.print(json.dumps({"type": "chat.event", "line": line, "event": event.as_dict()}, ensure_ascii=False))
            return
        self.console.print(line)

    def render_final(self, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        if self.mode == "json":
            self.console.print(json.dumps({"type": "assistant.final", "content": text}, ensure_ascii=False))
            return
        self.console.print(Markdown(text))

    def format_event(self, event: ChatEvent) -> str | None:
        kind = normalize_event_kind(event.kind)
        if kind == "session":
            return None
        if kind == "response" and event.type in {"turn.finished", "assistant.delta"}:
            return None

        if kind == "tool":
            eid = str(event.event_id or event.id or "")
            st = normalize_event_status(event.status)
            prev = self._tool_event_ids.get(eid)
            if prev == st:
                # identical status update for same id; skip duplicate
                return None
            self._tool_event_ids[eid] = st
            # In main chat transcript, only emit a line for *terminal* tool states.
            # In-progress is shown live via LiveToolActivity (spinner + summary).
            # This prevents noisy "→ running" followed by "✓ " duplicates for the same activity.
            if st == "running":
                return None
            # fallthrough to produce the final compact line for the (updated) item
            return _inline_tool_line(event)

        signature = _inline_signature(event)
        if signature in self._rendered_signatures:
            return None
        self._rendered_signatures.add(signature)

        if kind == "subagent":
            return _inline_subagent_line(event)
        if kind == "routing":
            return _inline_generic_line(event, noun="routing")
        if kind == "plan_step":
            return _inline_generic_line(event, noun="plan")
        if kind == "reasoning":
            return _inline_generic_line(event, noun="reasoning")
        if kind == "user_request":
            message = _clip_summary(event.message or "queued", 80)
            return f"{_status_icon(event.status)} request {message}"
        if kind == "error":
            return f"{_status_icon('failed')} {event.title or 'error'}: {_clip_summary(event.message, 96)}"
        return _inline_generic_line(event, noun=event.title or event.type)


class TimelineDebugRenderer(EventRenderer):
    """Verbose renderer for explicit timeline/debug views."""

    def __init__(self, *, mode: str = "rich", trace_mode: str = "full") -> None:
        super().__init__(mode=mode, trace_mode=trace_mode)


def select_chat_renderer(console: Console, *, mode: str, verbose: bool = False) -> InlineChatRenderer | TimelineDebugRenderer:
    if verbose:
        return TimelineDebugRenderer(mode=mode, trace_mode="full")
    return InlineChatRenderer(console, mode=mode)


def _inline_signature(event: ChatEvent) -> tuple[str, str, str, str, str, str]:
    metadata = event.metadata or {}
    return (
        event.type,
        normalize_event_status(event.status),
        str(event.title or ""),
        str(event.message or ""),
        str(metadata.get("tool_name") or ""),
        str(event.subagent_id or event.agent_id or metadata.get("agent_id") or ""),
    )


def _short_args(event: ChatEvent) -> str:
    metadata = event.metadata or {}
    for key in ("args_summary", "target", "path", "file_path", "command"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return _clip_summary(value, 72)
    return ""


def _inline_tool_line(event: ChatEvent) -> str:
    """Compact final-state line for a tool (used in transcript for terminal events).

    Prefers human-readable action summary from summary/metadata.
    Start (running) lines are suppressed in InlineChatRenderer to avoid noise;
    live in-progress uses LiveToolActivity which updates the same logical item.
    """
    tool = _tool_name(event)
    status = normalize_event_status(event.status)
    # Prefer explicit action summary or result summary for human readable display
    meta = event.metadata or {}
    summary_src = event.summary or meta.get("result_summary") or meta.get("args_summary") or event.message or ""
    detail = _clip_summary(str(summary_src).strip(), 80) if summary_src else _short_args(event)

    if status == "success":
        dur = _duration_label(event)
        base = f"✓ {tool}"
        if detail and detail.lower() != tool.lower():
            base += f" {detail}"
        if dur:
            base += f" ({dur})"
        return base
    if status == "failed":
        err = _clip_summary(event.message or meta.get("result_summary") or "failed", 80)
        return f"✕ {tool} {err}"
    # Fallback for any other non-running terminal-ish state
    icon = _status_icon(status)
    return f"{icon} {tool}{(' ' + detail) if detail else ''}"


def _inline_subagent_line(event: ChatEvent) -> str:
    metadata = event.metadata or {}
    agent_id = str(event.subagent_id or event.agent_id or metadata.get("agent_id") or "subagent").strip()
    role = str(metadata.get("role") or metadata.get("agent_role") or event.title or "").strip()
    status = normalize_event_status(event.status)
    if event.type in {"subagent.created"} or str(metadata.get("raw_kind") or "") == "subagent_created":
        suffix = f": {_clip_summary(role or event.message, 72)}" if (role or event.message) else ""
        return f"↳ subagent {agent_id} created{suffix}"
    if status == "running":
        return f"  → {agent_id} started"
    if status == "success":
        return f"  ✓ {agent_id} completed"
    if status == "failed":
        return f"  ✕ {agent_id} failed: {_clip_summary(event.message or 'failed', 72)}"
    return f"↳ subagent {agent_id} {status}"


def _inline_generic_line(event: ChatEvent, *, noun: str) -> str:
    status = normalize_event_status(event.status)
    label = str(event.title or noun).strip().lower()
    summary = _clip_summary(event.message or "", 88)
    if status == "running":
        return f"◌ {label}"
    if status == "success":
        return f"✓ {label}{(' ' + summary) if summary else ''}"
    if status == "failed":
        return f"✕ {label}: {summary or 'failed'}"
    return f"{_status_icon(status)} {label}{(' ' + summary) if summary else ''}"


def _clock_label(event: ChatEvent) -> str:
    raw = str(event.started_at or "")
    try:
        return raw.split("T", 1)[1].split(".", 1)[0]
    except Exception:
        return raw[-8:] or "-"


def normalize_visible_events(events: list[ChatEvent]) -> list[ChatEvent]:
    by_id: dict[str, ChatEvent] = {}
    order: list[str] = []
    for event in events:
        event.status = normalize_event_status(event.status)
        event.metadata["kind"] = normalize_event_kind(event.metadata.get("kind") or event.type)
        existing = by_id.get(event.event_id)
        if existing is None:
            by_id[event.event_id] = event
            order.append(event.event_id)
            continue
        existing.parent_event_id = event.parent_event_id or existing.parent_event_id
        existing.session_id = event.session_id or existing.session_id
        existing.turn_id = event.turn_id or existing.turn_id
        existing.agent_id = event.agent_id if event.agent_id is not None else existing.agent_id
        existing.subagent_id = event.subagent_id if event.subagent_id is not None else existing.subagent_id
        existing.step_id = event.step_id if event.step_id is not None else existing.step_id
        existing.type = event.type or existing.type
        existing.status = event.status or existing.status
        existing.title = event.title or existing.title
        existing.summary = event.summary if event.summary is not None else existing.summary
        existing.ended_at = event.ended_at or existing.ended_at
        existing.duration_ms = event.duration_ms if event.duration_ms is not None else existing.duration_ms
        existing.token_usage = event.token_usage or existing.token_usage
        existing.metadata.update(event.metadata or {})
    return [by_id[event_id] for event_id in order if event_id in by_id]


def timeline_event_label(event: ChatEvent) -> str:
    raw_kind = str(event.metadata.get("raw_kind") or event.metadata.get("kind") or "").strip()
    mapping = {
        "session_started": "Session",
        "session_ready": "Ready",
        "user_message": "User request",
        "plan_step_done": "Plan",
        "plan_step_started": "Plan",
        "thinking_summary": "Reasoning",
        "assistant_message_done": "Response",
        "assistant_message_start": "Response",
        "tool_done": "Tool",
        "tool_started": "Tool",
        "subagent_done": "Subagent",
        "subagent_started": "Subagent",
    }
    if raw_kind in mapping:
        return mapping[raw_kind]
    if event.type == "session.ready":
        return "Ready"
    if event.type == "session.started":
        return "Session"
    return {
        "session": "Session",
        "user_request": "User request",
        "routing": "Routing",
        "plan_step": "Plan",
        "reasoning": "Reasoning",
        "tool": "Tool",
        "subagent": "Subagent",
        "response": "Response",
        "error": "Error",
    }.get(normalize_event_kind(event.kind), "Event")


def timeline_summary(event: ChatEvent, *, limit: int = 72) -> str:
    metadata = event.metadata or {}
    explicit = str(metadata.get("display_summary") or metadata.get("result_summary") or "").strip()
    if explicit:
        return _clip_summary(explicit, limit)
    kind = normalize_event_kind(event.kind)
    if kind == "reasoning":
        return _safe_reasoning_summary(event)
    if kind == "user_request":
        return _clip_summary(event.message or event.title or "Queued", limit)
    if kind == "routing":
        return _clip_summary(event.message or event.title or "Complete", limit)
    if kind == "response":
        return _clip_summary(event.message or "Rendered", limit)
    if kind == "tool":
        tool = str(metadata.get("tool_name") or event.title or "Tool").strip()
        detail = str(metadata.get("path") or metadata.get("file_path") or metadata.get("command") or "").strip()
        return _clip_summary(f"{tool} {detail}".strip() or tool, limit)
    if kind == "session" and event.type == "session.started":
        return _clip_summary(event.message or event.title or "Started", limit)
    return _clip_summary(event.message or event.title or event.status.title(), limit)


def _safe_reasoning_summary(event: ChatEvent) -> str:
    metadata = event.metadata or {}
    for key in ("decision", "next_action", "display_summary"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return _clip_summary(value, 72)
    return "Reasoning updated"


def _clip_summary(value: Any, limit: int) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text or "-"
    return text[: max(0, limit - 1)].rstrip() + "…"


def _status_summary(events: list[ChatEvent], tracker: TokenUsageTracker | None = None) -> dict[str, Any]:
    files_read = len(
        {
            str(event.metadata.get("path") or event.metadata.get("file_path") or event.event_id)
            for event in events
            if event.type == "file.read" or event.metadata.get("tool_name") == "read_file"
        }
    )
    files_changed = len(
        {
            str(event.metadata.get("path") or event.metadata.get("file_path") or event.event_id)
            for event in events
            if event.type.startswith(("file.changed", "patch."))
            or str(event.metadata.get("tool_name") or "") in {"edit_file", "multi_edit_file", "apply_patch", "write_file", "create_file", "delete_file"}
        }
    )
    running_tools = [event for event in events if event.type.startswith("tool.") and event.status == "running"]
    running_subagents = [event for event in events if (event.type.startswith("subagent.") or event.subagent_id) and event.status == "running"]
    test_events = [event for event in events if event.type.startswith("test.") or event.metadata.get("command")]
    failed_tests = any(event.status in {"failed", "failure"} for event in test_events)
    running_tests = any(event.status == "running" for event in test_events)
    tests_status = "failed" if failed_tests else ("running" if running_tests else ("success" if test_events else "queued"))
    elapsed = "-"
    if events:
        try:
            started = events[0].started_at
            ended = events[-1].ended_at or events[-1].started_at
            from datetime import datetime

            start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(ended.replace("Z", "+00:00"))
            seconds = max(0.0, (end_dt - start_dt).total_seconds())
            elapsed = f"{seconds:.1f}s" if seconds < 60 else f"{seconds / 60:.1f}m"
        except Exception:
            elapsed = "-"
    token_text = ""
    if tracker is not None and tracker.session_total.total_tokens:
        token_text = f"{tracker.session_total.input_tokens} in / {tracker.session_total.output_tokens} out"
    return {
        "files_read": files_read,
        "files_changed": files_changed,
        "active_tool": _tool_name(running_tools[-1]) if running_tools else "",
        "active_subagent": _event_actor_label(running_subagents[-1]) if running_subagents else "",
        "tests_status": tests_status,
        "tokens": token_text,
        "elapsed": elapsed,
        "waiting_approval": any(event.type == "approval.required" and event.status == "waiting" for event in events),
    }
