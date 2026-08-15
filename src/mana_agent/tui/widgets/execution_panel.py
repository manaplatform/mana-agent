"""Compact live coding execution panel shared by every coding backend."""

from __future__ import annotations

from textual.containers import Vertical
from textual.widgets import Collapsible, Static

from mana_agent.tui.widgets.selectable_text import SelectableText


class ExecutionPanel(Vertical):
    DEFAULT_CSS = """
    ExecutionPanel { height: auto; margin: 0 0 1 0; border: round $accent; padding: 0 1; }
    .execution-header { text-style: bold; color: $accent; }
    .execution-footer { color: $text-muted; }
    .execution-log { height: auto; max-height: 18; }
    """

    def __init__(self, *, turn_id: str) -> None:
        super().__init__()
        self.turn_id = turn_id
        self.backend = "coding"
        self.model = ""
        self.phase = "queued"
        self.events: list[dict] = []
        self.header: Static | None = None
        self.activity_log: SelectableText | None = None
        self.footer: Static | None = None
        self.details: Collapsible | None = None
        self.context_meter: dict = {}

    def compose(self):
        self.header = Static("◌ coding · queued", classes="execution-header")
        yield self.header
        with Collapsible(title="activity", collapsed=False) as details:
            self.details = details
            self.activity_log = SelectableText("Waiting for backend…", classes="execution-log")
            yield self.activity_log
        self.footer = Static("0 events", classes="execution-footer")
        yield self.footer

    def update_event(self, event: dict) -> None:
        self.events.append(dict(event))
        self.events = self.events[-80:]
        self._render_state(event)

    def on_mount(self) -> None:
        if self.events:
            self._render_state(self.events[-1])

    def _render_state(self, event: dict) -> None:
        self.backend = str(event.get("backend") or self.backend)
        self.model = str(event.get("model") or self.model)
        event_type = str(event.get("event_type") or "activity")
        if event_type.startswith(("context.", "cost.", "budget.")):
            self.context_meter.update(event.get("payload") or event.get("metadata") or {})
        self.phase = event_type
        status = str(event.get("status") or "running")
        icon = {"success": "✓", "failed": "✗", "cancelled": "■"}.get(status, "●")
        if self.header:
            model = f" · {self.model}" if self.model else ""
            self.header.update(f"{icon} {self.backend}{model} · {event_type.replace('.', ' ')}")
        lines: list[str] = []
        for row in self.events[-14:]:
            row_status = str(row.get("status") or "running")
            row_icon = {"success": "✓", "failed": "✗", "cancelled": "■"}.get(row_status, "›")
            title = str(row.get("title") or row.get("event_type") or "activity")
            detail = str(row.get("output_preview") or row.get("summary") or row.get("error") or "")
            lines.append(f"{row_icon} {title}" + (f" — {detail}" if detail else ""))
        if self.activity_log:
            self.activity_log.load_text("\n".join(lines))
        usage = event.get("token_usage") or {}
        tokens = usage.get("total_tokens") or usage.get("totalTokens")
        duration = event.get("duration_ms")
        stats = [f"{len(self.events)} events"]
        if tokens is not None:
            stats.append(f"{tokens} tokens")
        if duration is not None:
            stats.append(f"{duration} ms")
        if self.context_meter:
            used = self.context_meter.get("used_tokens", 0)
            maximum = self.context_meter.get("context_window", 0)
            task_used = self.context_meter.get("task_used_tokens", 0)
            task_budget = self.context_meter.get("task_budget_tokens", 0)
            ratio = float(self.context_meter.get("utilization_ratio", 0) or 0) * 100
            schema = (self.context_meter.get("breakdown") or {}).get("schema_tokens", self.context_meter.get("schema_tokens", 0))
            cost = float(self.context_meter.get("cumulative_cost", 0) or 0)
            saved = self.context_meter.get("compression_tokens_saved") or self.context_meter.get("tokens_saved", 0)
            marker = "~" if self.context_meter.get("estimated", True) else ""
            stat_str = f"ctx {used}/{maximum} ({ratio:.0f}%) · schema {schema}"
            if task_budget:
                stat_str += f" · task {task_used}/{task_budget}"
            if saved:
                stat_str += f" · saved {saved}"
            stat_str += f" · {marker}${cost:.4f}"
            stats.append(stat_str)
        if self.footer:
            self.footer.update(" · ".join(stats))
        if self.details and event_type in {"turn.completed", "turn.cancelled", "error"}:
            self.details.collapsed = True
        self.refresh(layout=True)


__all__ = ["ExecutionPanel"]
