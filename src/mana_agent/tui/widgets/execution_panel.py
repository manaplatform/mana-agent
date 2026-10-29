"""Compact live execution coordinator panel hosting persistent activity boxes."""

from __future__ import annotations

from typing import Any

from textual.containers import Vertical
from textual.widgets import Collapsible, Static

from mana_agent.chat.execution_trace import (
    ExecutionTrace,
    classify_activity_category,
    event_classification_metadata,
)
from mana_agent.tui.widgets.activity_boxes import CodingActivityBox, MainActivityBox
from mana_agent.tui.widgets.selectable_text import SelectableText


class ExecutionPanel(Vertical):
    DEFAULT_CSS = """
    ExecutionPanel { height: auto; margin: 0 0 1 0; padding: 0; }
    .execution-header { text-style: bold; color: $accent; margin: 0 0 0 0; padding: 0 1; }
    .execution-steps { height: auto; margin: 0 0 0 0; padding: 0 1; display: none; }
    .execution-summary { height: auto; margin: 0 0 1 0; padding: 0 1; color: $success; }
    .execution-footer { color: $text-muted; margin-top: 0; padding: 0 1; }
    .execution-log { height: auto; max-height: 18; }
    """

    def __init__(self, *, turn_id: str) -> None:
        super().__init__()
        self.turn_id = turn_id
        self.trace = ExecutionTrace(turn_id=turn_id)
        self.backend = ""
        self.model = ""
        self.phase = "queued"
        self.events: list[dict[str, Any]] = []
        self._event_ids: set[str] = set()

        # Two persistent activity boxes
        self.main_box = MainActivityBox(turn_id=turn_id)
        self.coding_box = CodingActivityBox(turn_id=turn_id)

        # Header, steps, summary and footer widgets for overall turn
        self.header = Static("◌ execution · queued", classes="execution-header")
        self.steps_view = Static("", classes="execution-steps")
        self.steps_view.display = False
        self.summary_card = Static("", classes="execution-summary")
        self.footer = Static("0 events", classes="execution-footer")
        self.context_meter: dict[str, Any] = {}

        # Aliases for backward compatibility with existing tests
        self.details = self.coding_box.details
        self.activity_log = self.coding_box.activity_log

    def compose(self):
        yield self.header
        yield self.steps_view
        yield self.main_box
        yield self.coding_box
        yield self.summary_card
        yield self.footer

    def update_trace_event(self, event: dict[str, Any]) -> None:
        self.update_event(event)

    def update_event(self, event: dict[str, Any]) -> None:
        event_dict = dict(event)
        event_id = str(event_dict.get("event_id") or event_dict.get("id") or "").strip()
        if event_id:
            if event_id in self._event_ids:
                # Still pass to trace and activity boxes for idempotent updates/completion
                self.trace.apply_event(event_dict)
                category = classify_activity_category(
                    event_dict.get("event_type") or event_dict.get("type") or "",
                    event_dict.get("tool_name"),
                    event_classification_metadata(event_dict),
                )
                if category == "coding":
                    self.coding_box.update_event(event_dict)
                else:
                    self.main_box.update_event(event_dict)
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

        # Route to appropriate activity box
        category = classify_activity_category(
            event_dict.get("event_type") or event_dict.get("type") or "",
            event_dict.get("tool_name"),
            event_classification_metadata(event_dict),
        )
        if category == "coding":
            self.coding_box.update_event(event_dict)
        else:
            self.main_box.update_event(event_dict)

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
            if step.status in {"waiting_approval", "pending_approval"}:
                lines.append(f"[bold #f59e0b]⚠️ {step.title.lower()} · waiting for approval{detail_str}[/]")
            elif step.status == "running":
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

    def _render_state(self, event: dict[str, Any]) -> None:
        backend_val = event.get("backend")
        if backend_val:
            self.backend = str(backend_val)
        self.model = str(event.get("model") or self.model)
        event_type = str(event.get("event_type") or event.get("type") or "activity")
        if event_type.startswith(("context.", "cost.", "budget.")):
            self.context_meter.update(event.get("payload") or event.get("metadata") or {})
        self.phase = event_type
        status = str(event.get("status") or "running")

        # Determine current display phase & title from trace active/latest step
        current_step = self.trace.active_step or (self.trace.steps[-1] if self.trace.steps else None)
        if current_step is not None:
            current_title = current_step.title
            current_status = current_step.status
        else:
            current_title = str(event.get("title") or event_type.replace(".", " ").replace("_", " ").title())
            current_status = status

        if current_status in {"waiting_approval", "pending_approval"}:
            icon = "⚠️"
            approval_suffix = " · waiting for approval"
        else:
            icon = {"success": "✓", "completed": "✓", "failed": "✗", "error": "✗", "cancelled": "■"}.get(current_status, "◌")
            approval_suffix = ""

        if self.header:
            model = f" · {self.model}" if self.model else ""
            if self.backend and self.backend.lower() not in {"coding", "local"}:
                self.header.update(f"{icon} {self.backend}{model} · {current_title.lower()}{approval_suffix}")
            else:
                self.header.update(f"{icon} {current_title.lower()}{model}{approval_suffix}")

        self._render_steps()

        # Update details title: only call it "coding activity" if a coding step is present
        has_coding = any(s.phase == "coding" for s in self.trace.steps) or len(self.coding_box.events) > 0
        if self.details:
            self.details.title = "coding activity" if has_coding else "activity"

        # Synchronize backward compatible activity log
        lines: list[str] = []
        for row in self.events[-14:]:
            row_status = str(row.get("status") or "running")
            row_icon = {"success": "✓", "failed": "✗", "cancelled": "■"}.get(row_status, "›")
            title = str(row.get("title") or row.get("event_type") or row.get("type") or "activity")
            detail = str(row.get("output_preview") or row.get("summary") or row.get("error") or row.get("message") or "")
            lines.append(f"{row_icon} {title}" + (f" — {detail}" if detail else ""))
        if self.activity_log:
            self.activity_log.load_text("\n".join(lines))

        # Overall footer statistics
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

        # Only display boxes when they have events
        self.coding_box.display = len(self.coding_box.events) > 0
        self.main_box.display = len(self.main_box.events) > 0

        is_turn_terminal = event_type in {"turn.completed", "turn.finished", "turn.cancelled"}
        if is_turn_terminal:
            if self.details and getattr(self.coding_box, "_user_collapsed", None) is not False:
                self.details.collapsed = True
            if self.coding_box.details and getattr(self.coding_box, "_user_collapsed", None) is not False:
                self.coding_box.details.collapsed = True
            if self.main_box.details and getattr(self.main_box, "_user_collapsed", None) is not False:
                self.main_box.details.collapsed = True

            has_coding = len(self.coding_box.events) > 0 or len(self.coding_box.changed_files) > 0
            if has_coding:
                files_count = len(self.coding_box.changed_files)
                files_text = f"{files_count} file{'s' if files_count != 1 else ''} modified" if files_count > 0 else "Code inspected"
                ops_count = len(self.coding_box.events)
                summary_lines = [f"✓ Coding complete · {files_text} ({ops_count} operations)"]
                if self.coding_box.changed_files:
                    cfs = self.coding_box.changed_files
                    preview = ", ".join(cfs[:3]) + ("..." if len(cfs) > 3 else "")
                    summary_lines.append(f"  Files: {preview}")
                self.summary_card.update("\n".join(summary_lines))
                self.summary_card.display = True
            else:
                self.summary_card.update("")
                self.summary_card.display = False
        else:
            # During active execution (new activities arriving):
            # Do NOT close if already open; scroll to latest activity.
            if self.coding_box.details and not self.coding_box.details.collapsed:
                try:
                    self.coding_box.activity_log.scroll_end(animate=False)
                except Exception:
                    pass
            if self.main_box.details and not self.main_box.details.collapsed:
                try:
                    self.main_box.activity_log.scroll_end(animate=False)
                except Exception:
                    pass
            self.summary_card.update("")
            self.summary_card.display = False

        self.refresh(layout=True)


__all__ = ["ExecutionPanel"]
