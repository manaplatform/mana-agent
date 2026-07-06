from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from mana_agent import __version__
from mana_agent.cli.events import ChatEvent, make_event
from mana_agent.cli.renderers import EventRenderer
from mana_agent.telemetry.session_trace import SessionTrace
from mana_agent.telemetry.tokens import TokenUsageTracker


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
    session_id: str = field(default_factory=lambda: f"sess-{uuid.uuid4().hex}")
    tracker: TokenUsageTracker = field(default_factory=TokenUsageTracker)
    trace: SessionTrace | None = None
    events: list[ChatEvent] = field(default_factory=list)
    tool_runs: list[ChatEvent] = field(default_factory=list)
    subagent_events: list[ChatEvent] = field(default_factory=list)
    conversation: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve()
        self.ui_mode = EventRenderer.normalize_mode(self.ui_mode)
        self.trace_mode = EventRenderer.normalize_trace_mode(self.trace_mode)
        if self.trace_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.trace_path = self.repo_root / ".mana" / "traces" / f"session_{stamp}.jsonl"
        if self.trace is None:
            self.trace = SessionTrace(
                session_id=self.session_id,
                trace_mode=self.trace_mode,
                path=self.trace_path,
            )

    @property
    def renderer(self) -> EventRenderer:
        return EventRenderer(mode=self.ui_mode, trace_mode=self.trace_mode)

    def record_event(self, event: ChatEvent) -> ChatEvent:
        if not event.session_id:
            event.session_id = self.session_id
        self.events.append(event)
        if self.trace is not None:
            self.trace.record(event)
        if event.type.startswith("tool."):
            self.tool_runs.append(event)
        if event.type.startswith("subagent."):
            self.subagent_events.append(event)
        return event

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
        if len(self.conversation) > 40:
            self.conversation = self.conversation[-40:]


def detect_skills_status(root: Path) -> str:
    candidates = [root / ".mana" / "skills", root / "src" / "mana_agent" / "default_skills"]
    for candidate in candidates:
        if candidate.exists():
            return "indexed" if any(candidate.iterdir()) else "empty"
    return "not found"


def default_ui_mode(console: Console, *, as_json: bool = False) -> str:
    if as_json:
        return "json"
    env_mode = str(os.getenv("MANA_CHAT_UI", "") or "").strip().lower()
    if env_mode in {"fullscreen", "rich", "compact", "plain", "json"}:
        return env_mode
    if not bool(getattr(console, "is_terminal", False)) or os.getenv("CI"):
        return "plain"
    width = int(getattr(console, "width", 100) or 100)
    if width < 80:
        return "plain"
    if width < 100:
        return "compact"
    try:
        from mana_agent.cli.fullscreen_chat import fullscreen_available

        if fullscreen_available():
            return "fullscreen"
    except Exception:
        pass
    return "rich"


def compact_path(path: str | Path, *, width: int = 72) -> str:
    text = str(path)
    if len(text) <= width:
        return text
    parts = Path(text).parts
    if len(parts) >= 4:
        candidate = str(Path(parts[0], parts[1], "...", parts[-2], parts[-1])) if parts[0] == "/" else str(Path(parts[0], "...", parts[-2], parts[-1]))
        if len(candidate) <= width:
            return candidate
    return text[: max(0, width - 1)] + "…"


def render_startup_header(console: Console, state: ChatUIState) -> None:
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
    if state.ui_mode == "fullscreen":
        try:
            from mana_agent.cli.fullscreen_chat import show_startup_pet_animation

            show_startup_pet_animation()
        except Exception:
            pass
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
                "session.ready",
                title="Ready",
                status="success",
                session_id=state.session_id,
                message="Ready for chat input.",
                metadata={"prompt": "mana >"},
            ).finish(status="success")
        )
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
            "",
            "Ready. Ask for code changes, repo analysis, debugging, or planning.",
            "/help  /status  /tokens  /tools  /agents  /trace  /ui  /exit",
            "",
            "Try: \"explain this repo\"  \"fix failing tests\"  \"add a CLI command\"  \"/analyze src/mana_agent\"",
            "Enter send · Shift+Enter newline · Ctrl+C cancel · Ctrl+D exit",
        ]
        console.print("\n".join(lines))
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
        "approvals": state.approvals,
        "memory_enabled": state.memory_enabled,
        "skills_status": state.skills_status,
        "token_status": state.token_status,
        "ui_mode": state.ui_mode,
        "trace_mode": state.trace_mode,
    }
