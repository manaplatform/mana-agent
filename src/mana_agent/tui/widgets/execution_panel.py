"""Compact live coding execution panel shared by every coding backend."""

from __future__ import annotations

from textual.containers import Vertical
from textual.widgets import Collapsible, Static

from mana_agent.chat.execution_trace import ExecutionTrace
from mana_agent.tui.widgets.selectable_text import SelectableText


class ExecutionPanel(Vertical):
    DEFAULT_CSS = """
    ExecutionPanel { height: auto; margin: 0 0 1 0; border: round $accent; padding: 0 1; }
    .execution-header { text-style: bold; color: $accent; margin: 0 0 0 0; }
    .execution-steps { height: auto; margin: 0 0 0 0; }
    .execution-footer { color: $text-muted; margin-top: 0; }
    .execution-log { height: auto; max-height: 18; }
    """

    def __init__(self, *, turn_id: str) -> None:
        super().__init__()
        self.turn_id = turn_id
        self.trace = ExecutionTrace(turn_id=turn_id)
        self.backend = "coding"
        self.model = ""
        self.phase = "queued"
        self.events: list[dict] = []
        self._event_ids: set[str] = set()
        self.header = Static("◌ execution · queued", classes="execution-header")
        self.steps_view = Static("", classes="execution-steps")
        self.activity_log = SelectableText("Waiting for backend…", classes="execution-log")
        self.footer = Static("0 events", classes="execution-footer")
        self.details = Collapsible(title="coding activity", collapsed=False)
        self.context_meter: dict = {}

    def compose(self):
        yield self.header
        yield self.steps_view
        with Collapsible(title="coding activity", collapsed=False) as details:
            self.details = details
            yield self.activity_log
        yield self.footer

    def update_trace_event(self, event: dict) -> None:
        self.update_event(event)

    def update_event(self, event: dict) -> None:
        event_dict = dict(event)
        event_id = str(event_dict.get("event_id") or event_dict.get("id") or "").strip()
        if event_id:
            if event_id in self._event_ids:
                # Still pass to trace for idempotent updates/completion
                self.trace.apply_event(event_dict)
                self._render_state(event_dict)
                return
            self._event_ids.add(event_id)
        self.events.append(event_dict)
        if len(self.events) > 80:
            self.events = self.events[-80:]
            self._event_ids = {
                str(e.get("event_id") or e.get("id") or "").strip()
                for e in self.events
                if str(e.get("event_id") or e.get("id") or "").strip()
            }
        self.trace.apply_event(event_dict)
        self._render_state(event_dict)

    def on_mount(self) -> None:
        if self.events:
            self._render_state(self.events[-1])
        else:
            self._render_steps()

    def _render_steps(self) -> None:
        if not self.steps_view:
            return
        lines: list[str] = []
        for step in self.trace.steps:
            duration_str = f" ({step.duration_ms}ms)" if step.duration_ms is not None and step.status != "running" else ""
            detail_str = f" — {step.detail}" if step.detail else ""
            if step.status == "running":
                # Codex-like subtle / low-opacity visual treatment for running steps
                lines.append(f"[dim]◌ {step.title.lower()} · running{detail_str}[/dim]")
            elif step.status in {"completed", "success"}:
                lines.append(f"[bold $success]✓[/] {step.title.lower()}{detail_str}{duration_str}")
            elif step.status in {"failed", "error"}:
                lines.append(f"[bold $error]✗[/] {step.title.lower()}{detail_str}{duration_str}")
            elif step.status in {"cancelled", "interrupted"}:
                lines.append(f"[bold $warning]■[/] {step.title.lower()} · cancelled{detail_str}")
            else:
                lines.append(f"● {step.title.lower()} · {step.status}{detail_str}")
        self.steps_view.update("\n".join(lines))

    def _render_state(self, event: dict) -> None:
        self.backend = str(event.get("backend") or self.backend)
        self.model = str(event.get("model") or self.model)
        event_type = str(event.get("event_type") or event.get("type") or "activity")
        if event_type.startswith(("context.", "cost.", "budget.")):
            self.context_meter.update(event.get("payload") or event.get("metadata") or {})
        self.phase = event_type
        status = str(event.get("status") or "running")
        icon = {"success": "✓", "failed": "✗", "cancelled": "■"}.get(status, "●")
        if self.header:
            model = f" · {self.model}" if self.model else ""
            self.header.update(f"{icon} {self.backend}{model} · {event_type.replace('.', ' ')}")

        self._render_steps()

        # Activity log (for coding / command / progress items)
        lines: list[str] = []
        for row in self.events[-14:]:
            row_status = str(row.get("status") or "running")
            row_icon = {"success": "✓", "failed": "✗", "cancelled": "■"}.get(row_status, "›")
            title = str(row.get("title") or row.get("event_type") or row.get("type") or "activity")
            detail = str(row.get("output_preview") or row.get("summary") or row.get("error") or row.get("message") or "")
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
        if self.details and event_type in {"turn.completed", "turn.finished", "turn.cancelled", "error", "coding.terminal"}:
            self.details.collapsed = True
        self.refresh(layout=True)


__all__ = ["ExecutionPanel"]
