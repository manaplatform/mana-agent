from __future__ import annotations

import os
import json
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from mana_agent import __version__
from mana_agent.cli.events import ChatEvent, make_event
from mana_agent.cli.renderers import EventRenderer
from mana_agent.telemetry.session_trace import SessionTrace
from mana_agent.telemetry.tokens import TokenUsageTracker
from mana_agent.observability import ObservabilityStore
from mana_agent.services.chat_session_history import ChatSessionHistory
from mana_agent.tools.catalog import (
    ToolCatalogEntry,
    format_tool_catalog_summary,
    list_auto_chat_tools,
)
from mana_agent.workspaces.paths import session_dir
from mana_agent.workspaces.service import WorkspaceService


def git_branch(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except Exception:
        return "unknown"
    return (proc.stdout or "").strip() or "detached/unknown"


@dataclass(slots=True)
class ChatUIState:
    repo_root: Path
    provider: str
    model: str
    mode: str = "chat"
    tools_enabled: bool = True
    approvals: str = "auto"
    memory_enabled: bool = True
    skills_status: str = "unknown"
    token_status: str = "exact when provider usage is returned; estimated otherwise"
    ui_mode: str = "rich"
    trace_mode: str = "compact"
    index_path: str = ""
    k_value: int | None = None
    ephemeral_index: bool = False
    coding_agent: bool = True
    coding_memory: bool = True
    flow_memory: bool = True
    auto_execute: bool = True
    max_passes: int = 0
    auto_continue: bool = True
    execution_profile: str = "balanced"
    diagram_rendering: str = "on"
    tool_worker_backend: str = "local"
    log_path: Path | None = None
    trace_path: Path | None = None
    session_path: Path | None = None
    session_id: str = field(default_factory=lambda: f"sess-{uuid.uuid4().hex}")
    workspace_id: str = ""
    repository_id: str = ""
    tracker: TokenUsageTracker = field(default_factory=TokenUsageTracker)
    trace: SessionTrace | None = None
    observability: ObservabilityStore | None = None
    events: list[ChatEvent] = field(default_factory=list)
    events_by_id: dict[str, ChatEvent] = field(default_factory=dict)
    event_order: list[str] = field(default_factory=list)
    tool_runs: list[ChatEvent] = field(default_factory=list)
    subagent_events: list[ChatEvent] = field(default_factory=list)
    file_events: list[ChatEvent] = field(default_factory=list)
    test_runs: list[ChatEvent] = field(default_factory=list)
    log_events: list[ChatEvent] = field(default_factory=list)
    conversation: list[dict[str, str]] = field(default_factory=list)
    pending_conversation_turns: dict[str, str] = field(default_factory=dict)
    conversation_history_store: ChatSessionHistory = field(default_factory=ChatSessionHistory)
    # Auto-chat tool catalog (name + description) for TUI visibility.
    available_tools: list[ToolCatalogEntry] = field(default_factory=list)
    active_panel: str = "chat"
    verbose_logs: bool = False
    compact_mode: bool = False
    timeline_scroll_offset: int = 0

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve()
        if not self.available_tools:
            # Lazy, side-effect-light discovery (no MCP process start).
            self.available_tools = list_auto_chat_tools(include_mcp_discovery=False)
        service = WorkspaceService()
        try:
            context = service.context_for_session(self.session_id)
            if context.primary_root != self.repo_root:
                raise ValueError(
                    f"session {self.session_id} belongs to {context.primary_root}; relaunch chat with that --root-dir"
                )
            session = context.session
        except FileNotFoundError:
            session = service.create_session(self.repo_root, session_id=self.session_id)
        self.workspace_id = session.workspace_id
        self.repository_id = session.primary_repository_id
        if not self.conversation:
            self.conversation = [
                {"role": item.role, "content": item.content}
                for item in self.conversation_history_store.list(self.session_id)
                if item.role in {"user", "assistant", "tool"}
            ][-40:]
        self.ui_mode = EventRenderer.normalize_mode(self.ui_mode)
        self.trace_mode = EventRenderer.normalize_trace_mode(self.trace_mode)
        if self.trace_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.trace_path = session_dir(self.session_id) / "traces" / f"session_{stamp}.jsonl"
        if self.session_path is None:
            self.session_path = session_dir(self.session_id) / "events.jsonl"
        if self.trace is None:
            self.trace = SessionTrace(
                session_id=self.session_id,
                trace_mode=self.trace_mode,
                path=self.trace_path,
            )
        if self.observability is None:
            self.observability = ObservabilityStore(self.repo_root)

    @property
    def renderer(self) -> EventRenderer:
        return EventRenderer(mode=self.ui_mode, trace_mode=self.trace_mode)

    def record_event(self, event: ChatEvent) -> ChatEvent:
        if not event.session_id:
            event.session_id = self.session_id
        stored = self._upsert_event(event)
        self._persist_session_event(event)
        if self.trace is not None:
            self.trace.record(event)
        if self.observability is not None:
            self.observability.record_event(stored)
        self._sync_event_collections(stored)
        return stored

    def activate_session(self, session_id: str) -> None:
        """Switch the UI to another isolated session for the same repository."""

        context = WorkspaceService().context_for_session(session_id)
        if context.primary_root != self.repo_root:
            raise ValueError(
                f"session {session_id} belongs to {context.primary_root}; relaunch chat with that --root-dir"
            )
        self.session_id = context.session.session_id
        self.workspace_id = context.session.workspace_id
        self.repository_id = context.session.primary_repository_id
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trace_path = session_dir(self.session_id) / "traces" / f"session_{stamp}.jsonl"
        self.session_path = session_dir(self.session_id) / "events.jsonl"
        self.trace = SessionTrace(session_id=self.session_id, trace_mode=self.trace_mode, path=self.trace_path)
        self.tracker = TokenUsageTracker()
        self.events.clear()
        self.events_by_id.clear()
        self.event_order.clear()
        self.tool_runs.clear()
        self.subagent_events.clear()
        self.file_events.clear()
        self.test_runs.clear()
        self.log_events.clear()
        self.conversation.clear()
        self.pending_conversation_turns.clear()
        self.conversation.extend(
            {"role": item.role, "content": item.content}
            for item in self.conversation_history_store.list(self.session_id)
            if item.role in {"user", "assistant", "tool"}
        )

    def begin_conversation_turn(self, question: str, turn_id: str) -> None:
        """Persist user input before model execution without adding it twice to prompts."""
        question_text = str(question or "").strip()
        if not question_text:
            return
        self.conversation_history_store.append(
            self.session_id,
            role="user",
            content=question_text,
            turn_id=turn_id,
        )
        self.pending_conversation_turns[question_text] = turn_id

    def conversation_prompt(self, question: str) -> str:
        question_text = str(question or "").strip()
        durable = self.conversation_history_store.list(self.session_id)
        if durable and durable[-1].role == "user" and durable[-1].content == question_text:
            durable = durable[:-1]
        prior = [
            {"role": item.role, "content": item.content}
            for item in durable
            if item.role in {"user", "assistant", "tool"}
        ][-40:]
        if not prior:
            return question_text
        labels = {"user": "User", "assistant": "Assistant", "tool": "Tool result"}
        lines = ["Active conversation history (chronological):"]
        for item in prior:
            role = item.get("role", "user")
            lines.append(f"{labels.get(role, role.title())}: {item.get('content', '')}")
        lines.extend(["", "Current user message:", question_text])
        return "\n".join(lines)[-40000:]

    @property
    def normalized_events(self) -> list[ChatEvent]:
        return [self.events_by_id[event_id] for event_id in self.event_order if event_id in self.events_by_id]

    def _upsert_event(self, event: ChatEvent) -> ChatEvent:
        existing = self.events_by_id.get(event.event_id)
        if existing is None:
            self.events_by_id[event.event_id] = event
            self.event_order.append(event.event_id)
            self.events.append(event)
            return event
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
        return existing

    def _sync_event_collections(self, event: ChatEvent) -> None:
        if event.type.startswith("tool."):
            _upsert_collection(self.tool_runs, event)
        if event.type.startswith("subagent.") or event.subagent_id:
            _upsert_collection(self.subagent_events, event)
        if event.type.startswith("file.") or event.type.startswith("patch.") or event.metadata.get("path") or event.metadata.get("file_path"):
            _upsert_collection(self.file_events, event)
        if event.type.startswith("test.") or event.metadata.get("command"):
            _upsert_collection(self.test_runs, event)
        if event.type.startswith("log.") or event.type in {"tool.stdout", "tool.stderr"}:
            _upsert_collection(self.log_events, event)

    def update_event_status(
        self,
        event_id: str,
        *,
        status: str,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> ChatEvent:
        original = self.events_by_id.get(event_id)
        if original is None:
            original = next((item for item in reversed(self.events) if item.event_id == event_id), None)
        if original is None:
            raise ValueError(f"Cannot update unknown event id: {event_id}")
        updated = make_event(
            original.type,
            title=original.title,
            message=message if message is not None else original.message,
            status=status,
            session_id=self.session_id,
            turn_id=original.turn_id,
            agent_id=original.agent_id,
            subagent_id=original.subagent_id,
            step_id=original.step_id,
            parent_event_id=original.parent_event_id,
            token_usage=original.token_usage,
            metadata={**original.metadata, **(details or {}), "updates_event_id": event_id},
        )
        updated.event_id = event_id
        updated.started_at = original.started_at
        return self.record_event(updated.finish(status=status, message=updated.message))

    def _persist_session_event(self, event: ChatEvent) -> None:
        if self.session_path is None:
            return
        try:
            self.session_path.parent.mkdir(parents=True, exist_ok=True)
            with self.session_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.as_dict(), ensure_ascii=False, default=str) + "\n")
        except Exception:
            return

    def start_turn(self, turn_id: str) -> ChatEvent:
        self.tracker.start_turn(turn_id)
        first = self.record_event(
            make_event(
                "turn.started",
                title="User input received",
                message="Request queued for chat routing.",
                status="success",
                session_id=self.session_id,
                turn_id=turn_id,
                step_id="01",
            ).finish(status="success")
        )
        for step_id, title, message in (
            ("02", "Context loading", f"Repository context: {self.repo_root}"),
            ("03", "Memory retrieval", "Session and coding memory status checked."),
            ("04", "Skill selection", f"Skill index status: {self.skills_status}."),
            ("05", "Decision routing", "Routing decision pending."),
        ):
            self.record_event(
                make_event(
                    "step.finished" if step_id != "05" else "step.started",
                    title=title,
                    message=message,
                    status="success" if step_id != "05" else "running",
                    session_id=self.session_id,
                    turn_id=turn_id,
                    step_id=step_id,
                ).finish(status="success" if step_id != "05" else "running")
            )
        return first

    def finish_turn(self, turn_id: str, message: str = "Final response rendered.") -> ChatEvent:
        return self.record_event(
            make_event(
                "turn.finished",
                title="Final response",
                message=message,
                status="success",
                session_id=self.session_id,
                turn_id=turn_id,
                step_id="10",
                token_usage=self.tracker.by_turn.get(turn_id, self.tracker.session_total),
            ).finish(status="success")
        )

    def add_conversation_turn(self, question: str, answer: str) -> None:
        question_text = str(question or "").strip()
        answer_text = str(answer or "").strip()
        if question_text:
            self.conversation.append({"role": "user", "content": question_text})
        if answer_text:
            self.conversation.append({"role": "assistant", "content": answer_text})
            turn_id = self.pending_conversation_turns.pop(question_text, "")
            if not turn_id and self.pending_conversation_turns:
                pending_question = next(reversed(self.pending_conversation_turns))
                turn_id = self.pending_conversation_turns.pop(pending_question)
            self.conversation_history_store.append(
                self.session_id,
                role="assistant",
                content=answer_text,
                turn_id=turn_id,
                metadata={"model": self.model, "provider": self.provider},
            )
        if len(self.conversation) > 40:
            self.conversation = self.conversation[-40:]

    def agents_used(self, *, turn_id: str | None = None) -> list[str]:
        agents: list[str] = []
        for event in self.events:
            if turn_id and event.turn_id != turn_id:
                continue
            label = str(event.subagent_id or event.agent_id or "").strip()
            if not label:
                label = str(event.metadata.get("agent_role") or "").strip()
            if not label:
                continue
            if label.startswith("agent_main_") or label == "main":
                label = "main"
            if label not in agents:
                agents.append(label)
        if "main" not in agents:
            agents.insert(0, "main")
        return agents

    def execution_summary(self, *, turn_id: str | None = None) -> str:
        agents = self.agents_used(turn_id=turn_id)
        if not agents:
            return ""
        return "Agents used:\n" + "\n".join(f"- {agent}" for agent in agents)


def detect_skills_status(root: Path) -> str:
    candidates = [root / ".mana" / "skills", root / "src" / "mana_agent" / "default_skills"]
    for candidate in candidates:
        if candidate.exists():
            return "indexed" if any(candidate.iterdir()) else "empty"
    return "not found"


def _upsert_collection(collection: list[ChatEvent], event: ChatEvent) -> None:
    for index, existing in enumerate(collection):
        if existing.event_id == event.event_id:
            collection[index] = event
            return
    collection.append(event)


def default_ui_mode(console: Console, *, as_json: bool = False) -> str:
    if as_json:
        return "json"
    env_mode = str(os.getenv("MANA_CHAT_UI", "") or "").strip().lower()
    if env_mode in {"rich", "compact", "plain", "json"}:
        return env_mode
    # Treat record (capture/test) consoles and CI as plain for deterministic output.
    # Fall back to is_terminal for other non-tty cases. Width-based choice only for
    # real terminal-like consoles. This keeps tests stable across rich versions.
    if getattr(console, "record", False) or os.getenv("CI"):
        return "plain"
    if not bool(getattr(console, "is_terminal", False)):
        return "plain"
    width = int(getattr(console, "width", 100) or 100)
    if width < 80:
        return "plain"
    if width < 100:
        return "compact"
    return "rich"


def compact_path(path: str | Path, *, width: int = 72) -> str:
    text = str(path).replace("\\", "/")
    if len(text) <= width:
        return text
    parts = PurePosixPath(text).parts
    if len(parts) >= 4:
        candidate = (
            str(PurePosixPath(parts[0], parts[1], "...", parts[-2], parts[-1]))
            if parts[0] == "/"
            else str(PurePosixPath(parts[0], "...", parts[-2], parts[-1]))
        )
        if len(candidate) <= width:
            return candidate
    return text[: max(0, width - 1)] + "…"


def render_startup_header(console: Console, state: ChatUIState) -> None:
    tools_summary = format_tool_catalog_summary(state.available_tools)
    if state.ui_mode == "json":
        event = make_event(
            "session.started",
            title="Mana-Agent session started",
            status="success",
            session_id=state.session_id,
            message=str(state.repo_root),
            metadata=startup_metadata(state),
        ).finish(status="success")
        state.record_event(event)
        console.print(state.renderer.render_event(event))
        tools_event = make_event(
            "session.tools",
            title="Auto-chat tools",
            status="success",
            session_id=state.session_id,
            message=tools_summary,
            metadata={
                "tools": [entry.to_dict() for entry in state.available_tools],
                "count": len(state.available_tools),
            },
        ).finish(status="success")
        state.record_event(tools_event)
        console.print(state.renderer.render_event(tools_event))
        ready = make_event(
            "session.ready",
            title="Ready",
            status="success",
            session_id=state.session_id,
            message="Ready for chat input.",
            metadata={"prompt": "mana ❯"},
        ).finish(status="success")
        state.record_event(ready)
        console.print(state.renderer.render_event(ready))
        return
    branch = git_branch(state.repo_root)
    width = max(60, int(getattr(console, "width", 100) or 100))
    cwd_text = compact_path(state.repo_root, width=max(36, width - 4))
    header = Text()
    if state.ui_mode == "plain":
        lines = [
            f"Mana-Agent v{__version__}  repo: {state.repo_root.name}  branch: {branch}",
            f"model {state.model}   mode {state.mode}   tools {'auto' if state.tools_enabled else 'off'}   memory {'on' if state.memory_enabled else 'off'}   skills {state.skills_status}   tokens exact/~",
            f"cwd {cwd_text}",
            f"auto tools  {tools_summary}",
            "",
            "Ready. Ask for code changes, repo analysis, debugging, or planning.",
            "/help  /status  /tokens  /tools  /agents  /trace  /ui  /exit",
            "",
            "Try: \"explain this repo\"  \"fix failing tests\"  \"add a CLI command\"  \"/analyze src/mana_agent\"",
            "Enter send · Shift+Enter newline · Ctrl+C cancel · Ctrl+D exit",
        ]
        console.print("\n".join(lines))
        # Full catalog on start so tools are visible without running /tools.
        console.print(
            state.renderer.render_available_tools(
                state.available_tools,
                title="Auto-chat tools",
            )
        )
    else:
        header.append("Mana-Agent", style="bold bright_cyan")
        header.append(f" v{__version__}", style="dim")
        header.append(f"  repo: {state.repo_root.name}  branch: {branch}", style="default")
        console.print(header)
        console.print(
            f"[bold]model[/bold] {state.model}   [bold]mode[/bold] {state.mode}   "
            f"[bold]tools[/bold] {'auto' if state.tools_enabled else 'off'}   "
            f"[bold]memory[/bold] {'on' if state.memory_enabled else 'off'}   "
            f"[bold]skills[/bold] {state.skills_status}   [bold]tokens[/bold] exact/~"
        )
        console.print(f"[bold]cwd[/bold] {cwd_text}")
        console.print(f"[bold]auto tools[/bold] {tools_summary}")
        console.print()
        # Always print the full name+description catalog at session start.
        console.print(
            state.renderer.render_available_tools(
                state.available_tools,
                title="Auto-chat tools",
            )
        )
        console.print()
        console.print("Ready. Ask for code changes, repo analysis, debugging, or planning.")
        console.print("[dim]/help  /status  /tokens  /tools  /agents  /trace  /ui  /exit[/dim]")
        if width >= 100:
            console.print('[dim]Try: "explain this repo"  "fix failing tests"  "add a CLI command"  "/analyze src/mana_agent"[/dim]')
        console.print("[dim]Enter send · Shift+Enter newline · Ctrl+C cancel · Ctrl+D exit[/dim]")
    state.record_event(
        make_event(
            "session.started",
            title="Mana-Agent session started",
            status="success",
            session_id=state.session_id,
            message=str(state.repo_root),
            metadata=startup_metadata(state),
        ).finish(status="success")
    )
    state.record_event(
        make_event(
            "session.tools",
            title="Auto-chat tools",
            status="success",
            session_id=state.session_id,
            message=tools_summary,
            metadata={
                "tools": [entry.to_dict() for entry in state.available_tools],
                "count": len(state.available_tools),
            },
        ).finish(status="success")
    )
    state.record_event(
        make_event(
            "session.ready",
            title="Ready",
            status="success",
            session_id=state.session_id,
            message="Ready for chat input.",
            metadata={"prompt": "mana ❯"},
        ).finish(status="success")
    )


def render_status(state: ChatUIState, *, full: bool = False) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("repo", state.repo_root.name)
    table.add_row("branch", git_branch(state.repo_root))
    table.add_row("model", state.model)
    table.add_row("provider", state.provider)
    table.add_row("mode", state.mode)
    table.add_row("tools", "auto" if state.tools_enabled else "off")
    table.add_row("approvals", state.approvals)
    table.add_row("memory", "on" if state.memory_enabled else "off")
    table.add_row("skills", state.skills_status)
    table.add_row("active agents", str(len([e for e in state.subagent_events if e.status == "running"])))
    table.add_row("tokens", state.renderer.format_usage(state.tracker.session_total))
    if full:
        table.add_row("index path", state.index_path or "default")
        table.add_row("k", str(state.k_value or "default"))
        table.add_row("ephemeral index", "on" if state.ephemeral_index else "off")
        table.add_row("coding agent", "on" if state.coding_agent else "off")
        table.add_row("coding memory", "on" if state.coding_memory else "off")
        table.add_row("flow memory", "on" if state.flow_memory else "off")
        table.add_row("auto execute", "on" if state.auto_execute else "off")
        table.add_row("max passes", str(state.max_passes))
        table.add_row("auto continue", "on" if state.auto_continue else "off")
        table.add_row("execution profile", state.execution_profile)
        table.add_row("diagram rendering", state.diagram_rendering)
        table.add_row("tool worker backend", state.tool_worker_backend)
        table.add_row("trace path", str(state.trace_path or ""))
        table.add_row("log path", str(state.log_path or ""))
    return table


def startup_metadata(state: ChatUIState) -> dict[str, Any]:
    return {
        "repo": str(state.repo_root),
        "branch": git_branch(state.repo_root),
        "provider": state.provider,
        "model": state.model,
        "mode": state.mode,
        "tools_enabled": state.tools_enabled,
        "available_tools_count": len(state.available_tools),
        "available_tool_names": [entry.name for entry in state.available_tools[:40]],
        "approvals": state.approvals,
        "memory_enabled": state.memory_enabled,
        "skills_status": state.skills_status,
        "token_status": state.token_status,
        "ui_mode": state.ui_mode,
        "trace_mode": state.trace_mode,
    }
