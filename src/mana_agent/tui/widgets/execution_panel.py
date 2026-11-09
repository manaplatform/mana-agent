"""Compact live execution coordinator panel hosting persistent activity boxes."""

from __future__ import annotations

from typing import Any

from textual.containers import Vertical
from textual.widgets import Button, Collapsible, Static

from mana_agent.chat.execution_trace import (
    ExecutionTrace,
    classify_activity_category,
    event_classification_metadata,
)
from mana_agent.tui.widgets.activity_boxes import CodingActivityBox, MainActivityBox, _ActivityRendering
from mana_agent.tui.widgets.selectable_text import SelectableText


class ExecutionPanel(_ActivityRendering, Vertical):
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
        self.header = Static("◌ execution · queued", classes="execution-header", markup=False)
        self.steps_view = Static("", classes="execution-steps", markup=False)
        self.steps_view.display = False
        self.summary_card = Static("", classes="execution-summary", markup=False)
        self.footer = Static("0 events", classes="execution-footer", markup=False)
        self.context_meter: dict[str, Any] = {}
        self._history_before = 0
        self.history_view = SelectableText("", classes="execution-log")
        self.history_details = Collapsible(self.history_view, title="Earlier activity", collapsed=True)
        self.history_button = Button("Earlier activity", variant="default")

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
        yield self.history_button
        yield self.history_details

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button is not self.history_button:
            return
        event.stop()
        import asyncio
        from mana_agent.services.execution_event_hub import get_execution_event_hub
        scope = next((row for row in reversed(self.events) if row.get("conversation_id") and row.get("repository_id")), None)
        if scope is None:
            self.history_view.load_text("No durable activity history is available for this turn.")
        else:
            cursor = self._history_before or max(int(row.get("sequence") or 0) for row in self.events) + 1
            rows = await asyncio.to_thread(get_execution_event_hub().history,
                conversation_id=scope["conversation_id"], repository_id=scope["repository_id"],
                execution_id=scope.get("execution_id") or self.turn_id, before_sequence=cursor, limit=200)
            if rows:
                self._history_before = min(int(row.get("sequence") or 0) for row in rows)
                self.history_view.load_text("\n".join(
                    f"{row.get('agent_id') or 'MainAgent'} · {row.get('title') or row.get('type')} · {row.get('status')}"
                    + (f" — {row['summary']}" if row.get("summary") else "")
                    for row in rows if row.get("visibility") != "internal"))
            else:
                self.history_view.load_text("No earlier activity.")
                self.history_button.disabled = True
        self.history_details.collapsed = False

    def update_trace_event(self, event: dict[str, Any]) -> None:
        self.update_event(event)

    def update_event(self, event: dict[str, Any]) -> None:
        from mana_agent.chat.activity import project_activity, delivery_key, bound_operations, operation_key, accepts_activity_update
        event_dict = project_activity(event)
        if event_dict is None:
            return
        key = delivery_key(event_dict)
        if key in self._event_ids:
            return
        self._event_ids.add(key)
        if len(self._event_ids) > 4000:
            self._event_ids = {key}
        op = operation_key(event_dict)
        for i, row in enumerate(self.events):
            if operation_key(row) == op:
                if not accepts_activity_update(row, event_dict):
                    return
                self.events[i] = {**row, **event_dict}
                break
        else:
            self.events.append(event_dict)
        self.events[:] = bound_operations(self.events)
        self.trace.apply_event(event_dict)
        if self.trace.is_completed or self.trace.is_failed or self.trace.is_cancelled:
            outcome = "cancelled" if self.trace.is_cancelled else "failed" if self.trace.is_failed else "success"
            for box in (self.main_box, self.coding_box):
                box._root_status = outcome
                for operation in box.events:
                    if str(operation.get("status")) not in {"success", "completed", "done", "failed", "error", "cancelled", "unverified"}:
                        operation["status"] = "unverified"
        else:
            for box in (self.main_box, self.coding_box):
                box._root_status = None
        category = classify_activity_category(event_dict["event_type"], event_dict.get("tool_name"), event_classification_metadata(event_dict))
        if category == "coding":
            self.coding_box.update_event(event_dict)
        else:
            self.main_box.update_event(event_dict)
        self._schedule_activity_render(lambda: self._render_state(event_dict))

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
                if step.detail and "auto-approved" in step.detail.lower():
                    lines.append(f"[bold $success]✓[/] {step.title.lower()} · auto-approved{detail_str}")
                else:
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

        is_turn_terminal = self.trace.is_completed or self.trace.is_failed or self.trace.is_cancelled
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
                outcome = "cancelled" if self.trace.is_cancelled else "failed" if self.trace.is_failed else "completed"
                marker = "✓" if self.trace.is_completed else "■" if self.trace.is_cancelled else "✗"
                evidence = " · verified" if (event.get("metadata") or {}).get("evidence_status") == "verified" else " · observed; verification recorded separately"
                summary_lines = [f"{marker} Coding activity {outcome} · {files_text} ({ops_count} operations){evidence}"]
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
            self.summary_card.update("")
            self.summary_card.display = False

        self.refresh(layout=True)


__all__ = ["ExecutionPanel"]
