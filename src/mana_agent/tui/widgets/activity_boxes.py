"""Unified persistent activity boxes for TUI chat log.

Provides:
- MainActivityBox: unified, chronological stream for all general agent activities
  (tool calls, tool results, searches, reasoning summaries, delegation, retries, errors).
- CodingActivityBox: dedicated persistent container for all coding operations
  (file reads/edits/patches, terminal commands, test execution, git operations, changed files).
"""

from __future__ import annotations

import json
from typing import Any

from textual.containers import Vertical
from textual.widgets import Collapsible, Static

from mana_agent.tui.widgets.selectable_text import SelectableText
from mana_agent.tui.widgets.tool_card import ToolCard, _safe_textual_id


def _safe_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        val_str = value.strip()
        if (val_str.startswith("{") and val_str.endswith("}")) or (val_str.startswith("[") and val_str.endswith("]")):
            try:
                parsed = json.loads(val_str)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
    return {}


class MainActivityBox(Vertical):
    """Unified, chronological stream for all general agent activities.

    Exactly one instance per active turn. All non-coding tool calls, results,
    searches, reasoning summaries, and execution events are appended here
    instead of creating separate cards on the root chat timeline.
    """

    DEFAULT_CSS = """
    MainActivityBox {
        height: auto;
        margin: 0 0 1 0;
        border: round $primary 50%;
        padding: 0 1;
        background: $surface 40%;
    }
    .main-activity-header {
        text-style: bold;
        color: $primary;
        padding: 0;
        margin: 0;
    }
    .main-activity-stream {
        height: auto;
        max-height: 18;
    }
    .main-activity-cards {
        height: auto;
    }
    .main-activity-summary {
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, *, turn_id: str = "") -> None:
        super().__init__(id=_safe_textual_id(f"main-activity-{turn_id}" if turn_id else "main-activity"))
        self.turn_id = turn_id
        self.events: list[dict[str, Any]] = []
        self._event_ids: set[str] = set()
        self._tool_cards: dict[str, ToolCard] = {}
        self._pending_cards: list[ToolCard] = []
        self._active_status = "running"

        self.header = Static("◌ agent activity · running", classes="main-activity-header")
        self.details = Collapsible(title="activity stream", collapsed=False)
        self.activity_log = SelectableText("Waiting for activity…", classes="main-activity-stream")
        self.cards_container = Vertical(classes="main-activity-cards")
        self.footer = Static("0 activities", classes="main-activity-summary")
        self._user_collapsed: bool | None = None

    def on_collapsible_toggled(self, event: Collapsible.Toggled) -> None:
        if event.collapsible == self.details:
            self._user_collapsed = event.collapsible.collapsed

    def compose(self):
        yield self.header
        with Collapsible(title="activity stream", collapsed=False) as details:
            self.details = details
            yield self.activity_log
            yield self.cards_container
        yield self.footer

    def on_mount(self) -> None:
        for card in list(self._pending_cards):
            try:
                self.cards_container.mount(card)
            except Exception:
                pass
        self._pending_cards.clear()
        if self.events:
            self._render_stream()

    def add_tool_call(self, call_event: Any) -> ToolCard:
        """Add a general tool call as an expandable card inside this activity box."""
        card = ToolCard(call_event)
        self._tool_cards[call_event.call_id] = card
        # Bounded virtualization: mount active card into internal container
        if self.is_mounted and getattr(self.cards_container, "is_mounted", False):
            try:
                self.cards_container.mount(card)
            except Exception:
                self._pending_cards.append(card)
        else:
            self._pending_cards.append(card)

        self.update_event({
            "event_id": getattr(call_event, "event_id", None) or f"call-{call_event.call_id}",
            "call_id": call_event.call_id,
            "tool_name": call_event.tool_name,
            "title": f"Tool: {call_event.tool_name}",
            "status": "running",
            "summary": getattr(call_event, "summary", "") or "running…",
            "args": getattr(call_event, "args", {}),
        })
        return card

    def add_tool_result(self, result_event: Any) -> ToolCard | None:
        """Update a general tool call with its execution result."""
        card = self._tool_cards.get(result_event.call_id)
        if card is not None:
            card.set_result(result_event)

        self.update_event({
            "event_id": getattr(result_event, "event_id", None) or f"res-{result_event.call_id}",
            "call_id": result_event.call_id,
            "tool_name": result_event.tool_name,
            "title": f"Tool: {result_event.tool_name}",
            "status": "success" if result_event.success else "failed",
            "summary": getattr(result_event, "summary", "") or (result_event.tool_name + (" completed" if result_event.success else " failed")),
            "duration_ms": getattr(result_event, "duration_ms", None),
            "error": getattr(result_event, "error", None),
        })
        return card

    def update_event(self, event: dict[str, Any]) -> None:
        """Append or update a general agent activity event in the stream."""
        ev = dict(event)
        event_id = str(ev.get("event_id") or ev.get("id") or ev.get("call_id") or "").strip()
        if event_id and event_id in self._event_ids:
            for i, existing in enumerate(self.events):
                ex_id = str(existing.get("event_id") or existing.get("id") or existing.get("call_id") or "").strip()
                if ex_id == event_id:
                    self.events[i] = {**existing, **ev}
                    break
        else:
            if event_id:
                self._event_ids.add(event_id)
            self.events.append(ev)

        self._render_stream()

    def _render_stream(self) -> None:
        if not self.activity_log:
            return

        total_count = len(self.events)
        if total_count == 0:
            return

        has_failed = any(str(e.get("status") or "").lower() in {"failed", "error"} for e in self.events)
        all_completed = all(str(e.get("status") or "").lower() in {"success", "completed", "done"} for e in self.events)

        if has_failed:
            self._active_status = "failed"
            icon = "✗"
        elif all_completed and len(self.events) > 0:
            self._active_status = "success"
            icon = "✓"
        else:
            self._active_status = "running"
            icon = "◌"

        status_label = "completed" if self._active_status == "success" else self._active_status
        self.header.update(f"{icon} agent activity · {status_label} ({total_count} operations)")

        # Render rows with bounded rendering (display up to 40 recent rows, full history preserved in self.events)
        visible_events = self.events[-40:]
        lines: list[str] = []
        if total_count > 40:
            lines.append(f"… ({total_count - 40} earlier operations preserved)")

        # Consecutive operation grouping
        grouped: list[dict[str, Any]] = []
        for ev in visible_events:
            tool_name = str(ev.get("tool_name") or ev.get("title") or ev.get("type") or "operation").strip()
            summary = str(ev.get("summary") or ev.get("detail") or ev.get("message") or "").strip()
            status = str(ev.get("status") or "running").lower()
            sig = f"{tool_name}:{status}:{summary[:40]}"
            if grouped and grouped[-1].get("signature") == sig:
                grouped[-1]["count"] += 1
                grouped[-1]["latest_ev"] = ev
            else:
                grouped.append({
                    "signature": sig,
                    "tool_name": tool_name,
                    "summary": summary,
                    "status": status,
                    "count": 1,
                    "latest_ev": ev,
                })

        for item in grouped:
            st = item["status"]
            st_icon = {"success": "✓", "completed": "✓", "failed": "✗", "error": "✗", "cancelled": "■"}.get(st, "◌")
            count_suffix = f" (x{item['count']})" if item["count"] > 1 else ""
            dur = item["latest_ev"].get("duration_ms")
            dur_str = f" ({dur}ms)" if dur is not None else ""
            summary_part = f" — {item['summary']}" if item["summary"] else ""
            lines.append(f"{st_icon} {item['tool_name']}{count_suffix}{dur_str}{summary_part}")

            err = item["latest_ev"].get("error")
            if err:
                lines.append(f"  Error: {str(err)[:180]}")

        self.activity_log.load_text("\n".join(lines))
        self.footer.update(f"{total_count} total operations")

        # Do not close if already open when new activity arrives.
        # If open, advance to latest activity.
        if self.details and not self.details.collapsed:
            try:
                self.activity_log.scroll_end(animate=False)
            except Exception:
                pass
        self.refresh(layout=True)


class CodingActivityBox(Vertical):
    """Dedicated persistent activity box for all coding operations.

    Captures file reads, edits, writes, patches, terminal commands, test executions,
    builds, Git operations, and validation results. Groups related operations
    under their execution task.
    """

    DEFAULT_CSS = """
    CodingActivityBox {
        height: auto;
        margin: 0 0 1 0;
        border: round $accent;
        padding: 0 1;
        background: $surface 40%;
    }
    .coding-activity-header {
        text-style: bold;
        color: $accent;
        padding: 0;
        margin: 0;
    }
    .coding-changed-files {
        color: $success;
        text-style: bold;
        padding: 0 1;
        margin-bottom: 0;
    }
    .coding-activity-log {
        height: auto;
        max-height: 20;
    }
    .coding-activity-footer {
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, *, turn_id: str = "") -> None:
        super().__init__(id=_safe_textual_id(f"coding-activity-{turn_id}" if turn_id else "coding-activity"))
        self.turn_id = turn_id
        self.backend = ""
        self.model = ""
        self.phase = "queued"
        self.active_group = ""
        self.active_concurrency = 0
        self.batch_progress = ""
        self.events: list[dict[str, Any]] = []
        self.changed_files: list[str] = []
        self._event_ids: set[str] = set()
        self.context_meter: dict[str, Any] = {}

        self.header = Static("◌ coding · queued", classes="coding-activity-header")
        self.changed_files_view = Static("", classes="coding-changed-files")
        self.details = Collapsible(title="coding activity", collapsed=False)
        self.activity_log = SelectableText("Waiting for coding engine…", classes="coding-activity-log")
        self.footer = Static("0 coding events", classes="coding-activity-footer")
        self._user_collapsed: bool | None = None

    def on_collapsible_toggled(self, event: Collapsible.Toggled) -> None:
        if event.collapsible == self.details:
            self._user_collapsed = event.collapsible.collapsed

    def compose(self):
        yield self.header
        yield self.changed_files_view
        with Collapsible(title="coding activity", collapsed=False) as details:
            self.details = details
            yield self.activity_log
        yield self.footer

    def update_event(self, event: dict[str, Any]) -> None:
        """Append or update a coding operation in the box."""
        ev = dict(event)
        event_id = str(ev.get("event_id") or ev.get("id") or "").strip()
        if event_id and event_id in self._event_ids:
            for i, existing in enumerate(self.events):
                ex_id = str(existing.get("event_id") or existing.get("id") or "").strip()
                if ex_id == event_id:
                    self.events[i] = {**existing, **ev}
                    break
        else:
            if event_id:
                self._event_ids.add(event_id)
            self.events.append(ev)

        # Track phase and concurrency metadata
        payload_dict = _safe_dict(ev.get("payload"))
        meta = dict(ev.get("metadata") or ev.get("details") or payload_dict)
        if meta.get("phase"):
            self.phase = str(meta["phase"])
        if "active_concurrency" in meta or "active_concurrency" in payload_dict:
            self.active_concurrency = int(meta.get("active_concurrency") or payload_dict.get("active_concurrency") or 0)
        if meta.get("group") or meta.get("operation_group"):
            self.active_group = str(meta.get("group") or meta.get("operation_group"))

        ev_type = str(ev.get("event_type") or ev.get("type") or "")
        if ev_type.startswith("parallel.group"):
            count = payload_dict.get("count") or payload_dict.get("total_operations") or ""
            self.active_group = f"parallel group ({count} ops)" if count else "parallel group"

        # Track changed files
        args_dict = _safe_dict(payload_dict.get("args")) or _safe_dict(meta.get("args"))
        raw_path = (
            ev.get("path")
            or meta.get("path")
            or meta.get("filePath")
            or meta.get("file_path")
            or payload_dict.get("path")
            or payload_dict.get("filePath")
            or payload_dict.get("file_path")
            or args_dict.get("path")
            or args_dict.get("filePath")
            or args_dict.get("file_path")
        )
        if raw_path and str(raw_path).strip() and str(raw_path).strip() not in self.changed_files:
            self.changed_files.append(str(raw_path).strip())
        for cf in (ev.get("changed_files") or meta.get("changed_files") or payload_dict.get("changed_files") or []):
            cf_s = str(cf).strip()
            if cf_s and cf_s not in self.changed_files:
                self.changed_files.append(cf_s)

        # Context meter
        if ev_type.startswith(("context.", "cost.", "budget.")):
            self.context_meter.update(ev.get("payload") or ev.get("metadata") or {})

        backend_val = ev.get("backend")
        if backend_val:
            self.backend = str(backend_val)
        if ev.get("model"):
            self.model = str(ev.get("model"))

        self._render_state(ev)

    def _render_state(self, event: dict[str, Any]) -> None:
        status = str(event.get("status") or "running").lower()
        title = str(event.get("title") or event.get("event_type") or event.get("type") or "Coding")

        icon = {"success": "✓", "completed": "✓", "failed": "✗", "error": "✗", "cancelled": "■"}.get(status, "◌")
        backend_tag = f"[{self.backend.upper()}] " if self.backend and self.backend.lower() != "coding" else ""
        phase_tag = f" · Phase: {self.phase.capitalize()}" if self.phase and self.phase != "queued" else ""
        model_tag = f" · {self.model}" if self.model else ""
        self.header.update(f"{icon} {backend_tag}{title}{phase_tag}{model_tag}")

        # Update changed files list
        if self.changed_files:
            files_preview = ", ".join(self.changed_files[:5])
            if len(self.changed_files) > 5:
                files_preview += f" (+{len(self.changed_files) - 5} more)"
            self.changed_files_view.update(f"📁 changed files ({len(self.changed_files)}): {files_preview}")
        else:
            self.changed_files_view.update("")

        # Activity log (terminal commands, patches, outputs, parallel events)
        lines: list[str] = []
        for row in self.events[-30:]:
            row_status = str(row.get("status") or "running").lower()
            row_type = str(row.get("event_type") or row.get("type") or "")
            row_icon = {"success": "✓", "completed": "✓", "failed": "✗", "error": "✗", "cancelled": "■"}.get(row_status, "›")
            r_title = str(row.get("title") or row_type or "code")
            dur = row.get("duration_ms")
            dur_str = f" ({dur}ms)" if dur is not None else ""
            detail = str(row.get("output_preview") or row.get("summary") or row.get("error") or row.get("message") or "")
            cmd = str(row.get("command") or (row.get("payload") or {}).get("command") or "").strip()

            if row_type.startswith("parallel.group"):
                lines.append(f"⚡ {r_title}" + (f" — {detail}" if detail else ""))
            elif cmd:
                lines.append(f"{row_icon} $ {cmd}{dur_str}")
            else:
                lines.append(f"{row_icon} {r_title}{dur_str}" + (f" — {detail}" if detail else ""))

            if row.get("error"):
                lines.append(f"  Error: {str(row.get('error'))[:180]}")

        if self.activity_log:
            self.activity_log.load_text("\n".join(lines))

        # Footer stats
        usage = event.get("token_usage") or {}
        tokens = usage.get("total_tokens") or usage.get("totalTokens")
        duration = event.get("duration_ms")
        stats = [f"{len(self.events)} events"]
        if self.active_concurrency > 0:
            stats.append(f"{self.active_concurrency} concurrent")
        if tokens is not None:
            stats.append(f"{tokens} tokens")
        if duration is not None:
            stats.append(f"{duration} ms")
        if self.context_meter:
            used = self.context_meter.get("used_tokens", 0)
            maximum = self.context_meter.get("context_window", 0)
            ratio = float(self.context_meter.get("utilization_ratio", 0) or 0) * 100
            stats.append(f"ctx {used}/{maximum} ({ratio:.0f}%)")
        self.footer.update(" · ".join(stats))

        event_type = str(event.get("event_type") or event.get("type") or "")
        # Do not close if already open when new activity arrives.
        # Only turn completion collapses if not explicitly kept open by user.
        if self.details:
            if event_type in {"turn.completed", "turn.finished"} and self._user_collapsed is not False:
                self.details.collapsed = True
            elif not self.details.collapsed:
                # Advance to latest activity
                try:
                    self.activity_log.scroll_end(animate=False)
                except Exception:
                    pass

        self.refresh(layout=True)


__all__ = [
    "CodingActivityBox",
    "MainActivityBox",
]
