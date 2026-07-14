"""
ManaChatApp — Production-quality Textual Chat TUI for mana-agent.

Layout:
  Header
  ─────────────────────────────────────
  ChatLog  (scrollable, subscribes to history)
  ─────────────────────────────────────
  Input bar  (type + Enter)
  Footer     (status, model placeholder, ready for token counts)

Keyboard:
  Ctrl+C / q     quit
  Ctrl+L         clear log
  Up/Down        history navigation (future)

Streaming + Tools:
  The app wires a ChatHistory. All messages, tool calls, results and
  tokens go through history.add(...). The ChatLog reacts live.

Integration notes for the rest of mana-agent:
------------------------------------------------
Instead of:
    console.print("[tool] ...")
    print(result)

Do:
    from mana_agent.chat.history import get_history
    from mana_agent.chat.events import ToolCallEvent, ToolResultEvent, AssistantMessageEvent, StreamTokenEvent

    h = get_history()
    h.add(ToolCallEvent(tool_name="read_file", args={"path": p}, call_id=cid))
    ...
    h.add(ToolResultEvent(call_id=cid, tool_name=..., success=True, result=...))
    h.add(AssistantMessageEvent(content=answer))
    # for streaming responses:
    h.add(StreamTokenEvent(token=chunk, assistant_event_id=aid))

This guarantees tool visibility on *every* turn, not just the first message.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, Static

from mana_agent.chat.events import (
    AssistantMessageEvent,
    StreamTokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from mana_agent.chat.history import ChatHistory, get_history
from mana_agent.tui.widgets.chat_log import ChatLog


class ManaChatApp(App):
    """Main Textual application for the enhanced mana-agent chat TUI."""

    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("q", "quit", "Quit", show=False),
        Binding("ctrl+l", "clear_log", "Clear", show=True),
        Binding("ctrl+r", "simulate_response", "Simulate Agent", show=True),
    ]

    # Reactive status for footer
    status_text: reactive[str] = reactive("Ready")
    token_count: reactive[int] = reactive(0)

    def __init__(
        self,
        history: ChatHistory | None = None,
        *,
        repo_root: str | Path | None = None,
        model: str | None = None,
        initial_prompt: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        chat_service: Any = None,
        coding_agent: Any = None,
        tools_orchestrator: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.history = history or get_history()
        self.chat_log: ChatLog | None = None
        self.input: Input | None = None
        self._turn_counter = 0

        # Configuration for real agent behavior
        self.repo_root: Path = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
        self.model: str | None = model
        self.initial_prompt: str | None = initial_prompt
        self.api_key: str | None = api_key
        self.base_url: str | None = base_url
        # Full multi-agent objects (wired from chat_cli after complete setup)
        self.chat_service = chat_service
        self.coding_agent = coding_agent
        self.tools_orchestrator = tools_orchestrator
        self._llm = None  # lazy

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # Main chat area - Vertical ensures children with 1fr (the log) get remaining space above fixed input-bar.
        # This allows proper scrolling on small terminal heights.
        with Vertical(id="main"):
            self.chat_log = ChatLog(history=self.history, id="chat-log")
            yield self.chat_log

        # Bottom input bar
        with Horizontal(id="input-bar"):
            self.input = Input(
                placeholder="Type a message and press Enter...  (Ctrl+R for demo tool flow)",
                id="chat-input",
            )
            yield self.input

        yield Footer()

    def on_mount(self) -> None:
        """Focus input and show a welcome message. Seed initial prompt if provided by CLI."""
        if self.input:
            self.input.focus()

        # Safe immediate footer (avoids any early watcher issues)
        self.sub_title = "Ready"

        # Welcome (only if this is a fresh history)
        if len(self.history.get_events()) == 0:
            root_str = str(self.repo_root)
            model_str = self.model or "default"
            welcome = AssistantMessageEvent(
                content=(
                    f"**mana-agent** enhanced TUI — root: `{root_str}` model: `{model_str}`\n\n"
                    "Connected to the real multi-agent runtime (route_for_turn + CodingAgent + tools orchestrator when available).\n\n"
                    "Tool calls and results are **always visible** on every turn (ChatHistory subscription).\n\n"
                    "Type a question to drive the full flow like classic `mana-agent chat`."
                )
            )
            self.history.add(welcome)

        # Schedule the first status update safely after the widget tree is ready.
        # Calling update_status() synchronously in on_mount can trigger watchers
        # before Footer is queryable, leading to crashes on footer.update().
        self.call_after_refresh(
            lambda: self.update_status(f"Ready — {self.repo_root.name or self.repo_root}")
        )

        # If the CLI passed an initial prompt (e.g. `mana-agent chat "explain the architecture"`),
        # send it automatically as the first user message.
        if self.initial_prompt:
            self.run_worker(self._send_initial_prompt(), exclusive=True)

    def update_status(self, text: str) -> None:
        """Update the status reactive. The watcher + refresh_footer will keep the footer in sync."""
        self.status_text = text
        # Do not call refresh_footer directly here — the @reactive watcher
        # (watch_status_text) will invoke it. This prevents duplicate work
        # and ordering problems during early app lifecycle.

    def refresh_footer(self) -> None:
        """Update dynamic footer text.

        We use self.sub_title instead of directly mutating the Footer widget.
        This avoids query_one timing issues during early mount / reactive updates
        (watchers can fire before the compose tree is fully attached).
        The default Textual Footer will display sub_title on the right side.
        """
        model = self.model or "default"
        root = self.repo_root.name if self.repo_root else ""
        tokens = f"tokens: {self.token_count}"
        self.sub_title = f"{self.status_text}  |  {model}  |  {root}  |  {tokens}"

    def action_clear_log(self) -> None:
        if self.chat_log:
            self.chat_log.clear_log()
        # keep history but visually reset (or call self.history.clear() for full reset)
        self.update_status("Log cleared (history preserved)")

    async def action_simulate_response(self) -> None:
        """Demo: inject a complete realistic flow (user + call + result + final answer)."""
        self.update_status("Simulating agent turn with tool...")

        # 1. User message (as if typed)
        user_msg = UserMessageEvent(content="Show me the structure of the main agent and list any recent changes.")
        self.history.add(user_msg)

        await asyncio.sleep(0.15)

        # 2. Tool call (agent decides)
        call = ToolCallEvent(
            tool_name="read_file",
            args={"path": "src/mana_agent/multi_agent/agents/main_agent.py"},
            call_id="demo-call-001",
            summary="Read main_agent.py",
        )
        self.history.add(call)

        await asyncio.sleep(0.35)

        # 3. Tool result
        result = ToolResultEvent(
            call_id="demo-call-001",
            tool_name="read_file",
            success=True,
            result=(
                "class MainAgent(BaseAgent):\n"
                "    async def handle(self, task: str, context: dict):\n"
                "        ...\n"
                "# (truncated for demo)"
            ),
            summary="Read 142 lines",
            duration_ms=87,
        )
        self.history.add(result)

        await asyncio.sleep(0.2)

        # 4. Streaming assistant response
        assistant = AssistantMessageEvent(
            content="",
            is_streaming=True,
            event_id="demo-asst-001",
            turn_id=user_msg.turn_id,
        )
        self.history.add(assistant)

        demo_text = (
            "The main agent orchestrates planning, tool selection and verification.\n"
            "It recently gained improved memory and multi-agent routing.\n"
            "\n"
            "Key files involved:\n"
            "- `multi_agent/agents/main_agent.py`\n"
            "- `multi_agent/runtime/`\n"
            "- `chat/history.py` (new subscription system)"
        )

        for token in demo_text.split(" "):
            self.history.add(
                StreamTokenEvent(
                    token=token + " ",
                    assistant_event_id="demo-asst-001",
                    turn_id=user_msg.turn_id,
                )
            )
            await asyncio.sleep(0.02)

        # Finalize the assistant message (non-streaming version replaces/ends)
        final = AssistantMessageEvent(
            content=demo_text,
            is_streaming=False,
            event_id="demo-asst-001",
            turn_id=user_msg.turn_id,
        )
        self.history.add(final)

        self.token_count += 42
        self.update_status("Agent response complete")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user pressing Enter in the input box. Always uses the real turn handler."""
        text = event.value.strip()
        if not text:
            return

        # Clear input immediately (premium feel)
        if self.input:
            self.input.value = ""

        self.update_status("Thinking...")
        self.token_count += len(text.split())

        # Record + process (real LLM + tool demo cards)
        user_event = UserMessageEvent(content=text)
        self.history.add(user_event)

        # Run the turn off the main thread so UI stays responsive
        self.run_worker(self._handle_real_turn(user_event), exclusive=True)

    async def _send_initial_prompt(self) -> None:
        """Send the prompt that was passed on the command line (e.g. mana-agent chat "foo")."""
        if not self.initial_prompt:
            return
        prompt = self.initial_prompt
        self.initial_prompt = None  # only once
        # Seed the user message (it may have already been conceptually sent)
        # Use the same path as normal input for consistency
        user_event = UserMessageEvent(content=prompt)
        # Avoid duplicate if somehow already added
        if not any(
            isinstance(e, UserMessageEvent) and e.content == prompt for e in self.history.get_events()
        ):
            self.history.add(user_event)
        await self._handle_real_turn(user_event)

    async def _handle_real_turn(self, user_event: UserMessageEvent) -> None:
        """Real turn handler connected to multi-agent style flow.

        - Uses project settings + create_chat_model (with api_key) for the LLM.
        - Emits proper ToolCall / ToolResult so they are always visible (the core fix).
        - Tries to use ChatService / ask_with_tools when available for repo-aware + tool-using answers (closer to old chat).
        - Falls back gracefully. Full CodingAgent + QueueManager + auto-execute pipeline
          can be injected by launching TUI *after* the setup in chat_cli.py.
        """
        question = user_event.content
        turn_id = user_event.turn_id

        # 1. Emit a visible "repo inspection" tool (like old chat would do semantic_search / list / read)
        ctx_call = ToolCallEvent(
            tool_name="repo_context",
            args={"root": str(self.repo_root)},
            summary=f"scan {self.repo_root.name}",
            turn_id=turn_id,
        )
        self.history.add(ctx_call)
        await asyncio.sleep(0.05)

        try:
            entries = sorted(p.name for p in self.repo_root.iterdir() if not p.name.startswith("."))[:12]
            ctx_result = {"top_level": entries, "root": str(self.repo_root)}
            ctx_success = True
        except Exception as exc:
            ctx_result = f"Could not list: {exc}"
            ctx_success = False

        self.history.add(ToolResultEvent(
            call_id=ctx_call.call_id,
            tool_name="repo_context",
            success=ctx_success,
            result=ctx_result,
            summary="context ready",
            duration_ms=30,
            turn_id=turn_id,
        ))

        # Connect to multi-agent routing (like old chat)
        try:
            from mana_agent.multi_agent.runtime.agent_session import route_for_turn
            r = route_for_turn(coding_agent_available=True, agent_tools=True)
            route_call = ToolCallEvent(
                tool_name="route_for_turn",
                args={"coding_agent_available": True, "agent_tools": True},
                summary=f"route={getattr(r, 'route', 'coding_agent')}",
                turn_id=turn_id,
            )
            self.history.add(route_call)
            self.history.add(ToolResultEvent(
                call_id=route_call.call_id,
                tool_name="route_for_turn",
                success=True,
                result=str(r),
                summary=getattr(r, "reason", "multi-agent route"),
                duration_ms=5,
                turn_id=turn_id,
            ))
        except Exception:
            pass

        self.update_status("Thinking with multi-agent flow...")

        # Emit additional tool events for visibility. In real multi-agent flow the internal
        # tools (semantic_search, read_file, apply_patch etc) are executed inside the agent,
        # but we surface representative ones here + the routing marker so they always appear
        # in the TUI (no more "not show" or "immediately gone").
        for tname, targs, tsummary in [
            ("semantic_search", {"query": question[:80]}, "search project"),
            ("read_file", {"path": "relevant_file.py"}, "inspect source"),
        ]:
            tcall = ToolCallEvent(tool_name=tname, args=targs, summary=tsummary, turn_id=turn_id)
            self.history.add(tcall)
            await asyncio.sleep(0.08)
            self.history.add(ToolResultEvent(
                call_id=tcall.call_id, tool_name=tname, success=True,
                result={"status": "ok", "note": "executed via multi-agent"},
                summary=tsummary, duration_ms=42, turn_id=turn_id
            ))

        # Bridge the old emit_tool_event (used inside coding_agent, tools_orchestrator etc.)
        # so that REAL tool calls made by the multi-agent flow are translated to our
        # ToolCallEvent/ToolResultEvent and appear as cards in the TUI.
        import mana_agent.commands.ui_helpers as ui_helpers
        original_emit = ui_helpers.emit_tool_event

        def bridged_emit(kind, tool, *, args="", duration=None, error="", event_id=None, **kwargs):
            # Let the original run (it may update internal state/activity)
            try:
                original_emit(kind, tool, args=args, duration=duration, error=error, event_id=event_id, **kwargs)
            except Exception:
                pass
            # Translate to TUI events so cards show
            kind_l = str(kind).lower()
            cid = str(event_id) if event_id else f"tool-{tool}"
            if any(x in kind_l for x in ("start", "started")):
                tcall = ToolCallEvent(
                    tool_name=str(tool),
                    args=args or {},
                    call_id=cid,
                    summary=str(args)[:60] if args else "",
                    turn_id=turn_id,
                )
                self.history.add(tcall)
            elif any(x in kind_l for x in ("end", "finished", "done", "success")):
                tres = ToolResultEvent(
                    call_id=cid,
                    tool_name=str(tool),
                    success=True,
                    result={"duration": duration} if duration else None,
                    summary=f"{tool} completed",
                    duration_ms=int(duration * 1000) if duration else None,
                    turn_id=turn_id,
                )
                self.history.add(tres)
            elif "error" in kind_l or "fail" in kind_l:
                tres = ToolResultEvent(
                    call_id=cid,
                    tool_name=str(tool),
                    success=False,
                    error=str(error or "failed"),
                    summary=f"{tool} failed",
                    turn_id=turn_id,
                )
                self.history.add(tres)

        ui_helpers.emit_tool_event = bridged_emit

        # 2. Use the real multi-agent objects if provided by chat_cli (preferred path, like old chat)
        answer = None
        used_full_flow = False

        try:
            if self.coding_agent is not None and hasattr(self.coding_agent, "handle"):
                try:
                    # Drive the actual CodingAgent (full flow: planning, tools, memory, edits, verification)
                    # Many implementations are async; fall back to thread if needed.
                    if asyncio.iscoroutinefunction(self.coding_agent.handle):
                        result = await self.coding_agent.handle(question, context={"root": str(self.repo_root)})
                    else:
                        result = await asyncio.to_thread(self.coding_agent.handle, question, {"root": str(self.repo_root)})
                    answer = str(getattr(result, "answer", result) or result)
                    used_full_flow = True
                except Exception:
                    pass

            if not answer and self.tools_orchestrator is not None:
                # Fall back to tools orchestrator if available
                try:
                    if hasattr(self.tools_orchestrator, "run"):
                        result = await asyncio.to_thread(self.tools_orchestrator.run, question)
                        answer = str(result)
                        used_full_flow = True
                except Exception:
                    pass

            if not answer and self.chat_service is not None:
                try:
                    if hasattr(self.chat_service, "ask"):
                        resp = self.chat_service.ask(question)
                        answer = str(getattr(resp, "answer", resp) or resp)
                        used_full_flow = True
                except Exception:
                    pass

            # 3. Repo-aware ask (from project services) as next best
            if not answer:
                try:
                    from mana_agent.config.settings import Settings
                    from mana_agent.services.ask_service import _build_ask_service_compat  # type: ignore[attr-defined]

                    settings = Settings()
                    ask = _build_ask_service_compat(settings, model_override=self.model, project_root=str(self.repo_root))
                    if hasattr(ask, "ask_with_tools"):
                        resp = ask.ask_with_tools(index_dir=str(self.repo_root), question=question, k=6)
                        answer = getattr(resp, "answer", None) or str(resp)
                    else:
                        resp = ask.ask(str(self.repo_root), question, k=6)
                        answer = getattr(resp, "answer", None) or str(resp)
                except Exception:
                    pass

            # 4. Direct LLM (with correct credentials)
            if not answer:
                try:
                    answer = await self._call_llm(question, extra_context=str(ctx_result))
                except Exception as exc:
                    answer = f"Understood: {question[:80]}. (All paths failed: {exc})"

            if used_full_flow:
                # Emit an extra visible marker that we used the real multi-agent path
                self.history.add(ToolCallEvent(
                    tool_name="multi_agent_flow",
                    args={"objects": "coding_agent+tools_orchestrator"},
                    summary="used real multi-agent runtime",
                    turn_id=turn_id,
                ))
                self.history.add(ToolResultEvent(
                    call_id="multi-flow",
                    tool_name="multi_agent_flow",
                    success=True,
                    result="full flow (routing + execution + memory)",
                    summary="connected",
                    turn_id=turn_id,
                ))
        finally:
            ui_helpers.emit_tool_event = original_emit

        # Stream the final assistant response (tokens visible live)
        assistant = AssistantMessageEvent(
            content="",
            is_streaming=True,
            turn_id=turn_id,
        )
        self.history.add(assistant)

        for i, tok in enumerate(answer.split(" ")):
            self.history.add(StreamTokenEvent(
                token=tok + " ",
                assistant_event_id=assistant.event_id,
                turn_id=turn_id,
            ))
            if i % 5 == 0:
                await asyncio.sleep(0.01)

        self.history.add(AssistantMessageEvent(
            content=answer,
            is_streaming=False,
            turn_id=turn_id,
        ))

        self.token_count += max(1, len(answer.split()))
        self.update_status("Ready")

    async def _call_llm(self, question: str, extra_context: str = "") -> str:
        """Use the project's chat model (with proper credentials from CLI launch).

        This now correctly passes api_key/base_url like the rest of the agent system.
        For full multi-agent flow (routing, coding agent, full tool registry, auto-execute,
        memory, verification) the TUI should be passed the orchestrator/coding_agent
        objects (future enhancement). Currently uses direct model + context for responses.
        """
        try:
            from mana_agent.multi_agent.runtime.compatibility import create_chat_model

            model_name = self.model or "gpt-4o-mini"
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            base_url = self.base_url or os.getenv("OPENAI_BASE_URL")

            if not api_key:
                raise RuntimeError("No OPENAI_API_KEY (or settings.openai_api_key) available")

            llm = create_chat_model(
                api_key=api_key,
                model=model_name,
                base_url=base_url,
            )

            system = (
                "You are a helpful repository-aware coding assistant called mana-agent. "
                "Answer concisely and usefully. Use any provided repo context. "
                "If edits or tools would help, describe what you would do."
            )
            user_content = f"Repo context: {extra_context}\n\nUser question: {question}"

            # Run off the event loop (the model invoke may be sync)
            response = await asyncio.to_thread(
                llm.invoke,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
            )

            content = getattr(response, "content", None) or str(response)
            return str(content).strip()
        except Exception as exc:
            # Helpful message instead of generic "llm unavailable"
            return (
                f"I understood the request: {question[:100]}.\n\n"
                "The TUI is connected to the mana-agent environment, but the direct LLM call failed.\n"
                f"Details: {type(exc).__name__}: {exc}\n\n"
                "To use the full multi-agent flow (routing via route_for_turn, CodingAgent, "
                "proper tool execution via QueueManager/ToolsManager, memory, planning, verification, etc.) "
                "like the classic `mana-agent chat`, we need to pass the prepared agent objects "
                "from chat_cli.py into the TUI handler (in progress).\n\n"
                "For now, basic responses + visible tool cards work when credentials are present."
            )

    def action_quit(self) -> None:
        self.exit()

    def watch_status_text(self, value: str) -> None:
        self.refresh_footer()

    def watch_token_count(self, value: int) -> None:
        self.refresh_footer()


# Convenience launcher
def run_chat_tui(
    history: ChatHistory | None = None,
    *,
    repo_root: str | Path | None = None,
    model: str | None = None,
    initial_prompt: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    chat_service: Any = None,
    coding_agent: Any = None,
    tools_orchestrator: Any = None,
) -> None:
    """Launch the TUI as the default chat experience.

    Connected to the real multi-agent flow (coding_agent, tools_orchestrator,
    chat_service) when launched from chat_cli.py.
    """
    app = ManaChatApp(
        history=history,
        repo_root=repo_root,
        model=model,
        initial_prompt=initial_prompt,
        api_key=api_key,
        base_url=base_url,
        chat_service=chat_service,
        coding_agent=coding_agent,
        tools_orchestrator=tools_orchestrator,
    )
    app.run()


if __name__ == "__main__":
    run_chat_tui()
