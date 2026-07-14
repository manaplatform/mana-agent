"""Central AgentChatGateway for mana-agent.

Responsibilities:
- Own construction of the chat / coding-agent stack for a repository.
- Provide a single point that TUI, Telegram, and Dashboard use to reach agents.
- Simple path (send/ask) for connectors that do not need the full toolbox.
- Rich path (get_rich_context) for TUI and full console chat so they receive
  pre-wired objects (chat_service, coding_agent, tools_orchestrator) plus the
  parity flags that the current TUI expects.

The gateway re-uses builders from cli_internal / services and can be driven
by the same configuration that the chat command collects ("use chat-cli function").

This is the in-process gateway. All frontends should go through an instance
of this (or a thin adapter) rather than building AskService/CodingAgent directly.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mana_agent.config.settings import Settings
from mana_agent.services.chat_service import ChatService
from mana_agent.workspaces.service import WorkspaceService


@dataclass
class RichChatContext:
    """Objects and flags needed by rich clients (TUI, full console loop).

    Mirrors the objects that chat_cli currently builds and forwards to run_chat_tui.
    """

    chat_service: ChatService | Any
    coding_agent: Any | None = None
    tools_orchestrator: Any | None = None
    # Parity / control context (used to construct identical generate* calls)
    dir_mode: bool = False
    index_dir: str | None = None
    index_dirs: list[str] | None = None
    auto_execute_plan: bool = False
    auto_execute_max_passes: int = 3
    coding_agent_max_steps: int = 200
    resolved_k: int = 6
    agent_timeout_seconds: int = 600
    # Additional useful handles
    root: Path | None = None
    session_id: str | None = None
    # Optional event sink bridge (dashboard / TUI)
    event_sink: Callable[..., None] | None = None


class AgentChatGateway:
    """Gateway for all agent (multi-agent) chat connections.

    Typical usage from a frontend or from chat_cli:

        gw = AgentChatGateway(root=repo_root, coding_agent=True, ...)
        sid = gw.create_session(frontend="tui")
        # simple
        answer = gw.send(sid, "explain the architecture")
        # rich (for TUI / full console)
        ctx = gw.get_rich_context(sid)
        # pass ctx.coding_agent, ctx.chat_service, and the parity fields to TUI
    """

    def __init__(
        self,
        root: str | Path,
        *,
        # Core model / index config (subset of chat() flags for now)
        model: str | None = None,
        index_dir: str | Path | None = None,
        dir_mode: bool = False,
        max_indexes: int = 0,
        auto_index_missing: bool = True,
        # Agent behavior
        agent_tools: bool = True,
        coding_agent: bool = True,
        # Execution / worker
        tool_worker_process: bool = True,
        tool_worker_strict: bool = True,
        tool_exec_backend: str = "local",
        redis_url: str | None = None,
        toolsmanager_parallel_requests: int = 3,
        redis_queue_name: str = "mana-tools",
        redis_ttl_seconds: int = 86_400,
        # Memory & flow
        coding_memory: bool = True,
        flow_id: str | None = None,
        # Coding budgets
        coding_plan_max_steps: int = 8,
        coding_search_budget: int = 4,
        coding_read_budget: int = 6,
        coding_require_read_files: int = 2,
        # Auto / execution profile
        auto_execute_plan: bool = True,
        auto_execute_max_passes: int = 4,
        auto_continue: bool = True,
        execution_profile: str = "balanced",
        full_auto: bool = False,
        full_auto_status_every: int = 10,
        agent_max_steps: int = 6,
        agent_unlimited: bool = False,
        agent_timeout_seconds: int = 30,
        # Misc
        session_id: str | None = None,
        event_sink: Callable[..., None] | None = None,
        # Allow passing pre-built objects (for transition / tests)
        chat_service: Any = None,
        coding_agent_instance: Any = None,
        tools_orchestrator: Any = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.settings = Settings()
        self._event_sink = event_sink
        self._workspaces = WorkspaceService()

        # Store key flags for rich context
        self._dir_mode = bool(dir_mode)
        self._index_dir = str(index_dir) if index_dir else None
        self._max_indexes = int(max_indexes)
        self._auto_index_missing = bool(auto_index_missing)
        self._auto_execute_plan = bool(auto_execute_plan)
        self._auto_execute_max_passes = int(auto_execute_max_passes)
        self._coding_agent_max_steps = int(agent_max_steps) if coding_agent else 6
        if full_auto:
            self._auto_execute_max_passes = max(self._auto_execute_max_passes, 10)
        self._resolved_k = 6  # default; can be made configurable
        self._agent_timeout_seconds = int(agent_timeout_seconds)

        # If caller supplied pre-built objects (transition from chat_cli), use them.
        self._chat_service = chat_service
        self._coding_agent = coding_agent_instance
        self._tools_orchestrator = tools_orchestrator

        # Simple per-session state (frontend conversation -> internal id + history)
        self._sessions: dict[str, dict[str, Any]] = {}
        self._active: set[str] = set()

        # Build the stack if not supplied
        if self._chat_service is None:
            self._build_stack(
                model=model,
                agent_tools=agent_tools,
                coding_agent=coding_agent,
                tool_worker_process=tool_worker_process,
                tool_worker_strict=tool_worker_strict,
                tool_exec_backend=tool_exec_backend,
                redis_url=redis_url,
                toolsmanager_parallel_requests=toolsmanager_parallel_requests,
                redis_queue_name=redis_queue_name,
                redis_ttl_seconds=redis_ttl_seconds,
                coding_memory=coding_memory,
                flow_id=flow_id,
                coding_plan_max_steps=coding_plan_max_steps,
                coding_search_budget=coding_search_budget,
                coding_read_budget=coding_read_budget,
                coding_require_read_files=coding_require_read_files,
                execution_profile=execution_profile,
                full_auto=full_auto,
                agent_max_steps=agent_max_steps,
                agent_unlimited=agent_unlimited,
                agent_timeout_seconds=agent_timeout_seconds,
                session_id=session_id,
            )

    def _build_stack(self, **kwargs: Any) -> None:
        """Build ask/chat + optional full coding stack.

        This re-uses the same builders that chat_cli uses today.
        Future work: move the large construction blocks from chat_cli into
        reusable functions here (or in cli_internal) so gateway is the owner.
        """
        from mana_agent.commands.cli_internal import build_ask_service

        # Basic ask service (used by simple path and as base for coding)
        ask_service = build_ask_service(
            self.settings,
            kwargs.get("model"),
            project_root=self.root,
        )

        # ChatService wrapper (headless-friendly, like Telegram's ManaChatGateway)
        self._chat_service = ChatService(
            ask_service=ask_service,
            settings=self.settings,
            root_dir=self.root,
            index_dir=self._index_dir,
            dir_mode=self._dir_mode,
            agent_tools=kwargs.get("agent_tools", True),
            agent_max_steps=kwargs.get("agent_max_steps", 6),
            agent_timeout_seconds=kwargs.get("agent_timeout_seconds", 30),
            max_indexes=self._max_indexes,
            auto_index_missing=self._auto_index_missing,
        )

        # Full coding agent stack only when requested (expensive)
        if kwargs.get("coding_agent", True):
            # We deliberately keep construction light here.
            # The full rich construction (CodingAgent + ToolWorkerClient +
            # QueueManager + memory + skill context) lives in chat_cli today
            # because of the large number of parity flags.
            #
            # For the gateway to truly "own" it we will extract the
            # construction helpers (per plan). For now, if the caller
            # (chat_cli) has already built the instances, they are accepted
            # in __init__. When chat_cli calls the gateway first we can
            # move the code here.
            #
            # To satisfy immediate requirements we leave the heavy objects
            # as optional injection for the rich TUI path.
            pass

        # Record that we used the main multi-agent entry point (if possible)
        # This preserves the existing "multi agents path" recording.
        try:
            from mana_agent.commands.cli_internal import _record_multi_agent_request

            _record_multi_agent_request(
                self.root,
                "gateway:init",
                entrypoint="gateway",
                command_scope=False,
            )
        except Exception:
            # Best-effort; do not break chat because of recording
            pass

    # ------------------------------------------------------------------
    # Session management (shared by all frontends)
    # ------------------------------------------------------------------

    def create_session(self, *, frontend: str = "cli") -> str:
        """Create a logical session id for a frontend conversation."""
        # Reuse workspace session when possible for consistency with current code
        try:
            ws = self._workspaces.create_session(self.root)
            sid = ws.session_id
        except Exception:
            sid = f"gw-{frontend}-{id(self)}-{len(self._sessions)}"

        self._sessions[sid] = {
            "frontend": frontend,
            "history": [],  # (user, assistant) pairs for simple path
            "root": str(self.root),
        }
        return sid

    # ------------------------------------------------------------------
    # Simple path (Telegram, basic dashboard, API)
    # ------------------------------------------------------------------

    def send(self, session_id: str, text: str) -> str:
        """Synchronous send (used by most current callers)."""
        return asyncio.run(self.send_async(session_id, text))

    async def send_async(self, session_id: str, text: str) -> str:
        """Primary simple-path entry used by gateway-connected frontends."""
        self._active.add(session_id)
        try:
            # Basic history stitching (same pattern as current ManaChatGateway)
            hist = self._sessions.get(session_id, {}).get("history", [])[-12:]
            question = text
            if hist:
                transcript = "\n\n".join(
                    f"User: {q}\nMana-Agent: {a}" for q, a in hist
                )
                question = f"Conversation history for continuity:\n{transcript[-20000:]}\n\nCurrent user message:\n{text}"

            # Delegate to the ChatService (which uses the ask stack)
            # ChatService.ask is sync in current implementation.
            resp = self._chat_service.ask(
                self._index_dir or str(self.root),
                question,
                k=getattr(self._chat_service, "_k", 6),
            )
            answer = getattr(resp, "answer", resp)
            if not isinstance(answer, str):
                answer = str(answer or "").strip()

            result = (answer or "").strip()
            if not result:
                result = "(No response from agent)"

            self._sessions.setdefault(session_id, {}).setdefault("history", []).append((text, result))
            h = self._sessions[session_id]["history"]
            self._sessions[session_id]["history"] = h[-12:]

            return result
        finally:
            self._active.discard(session_id)

    def status(self, session_id: str) -> str:
        return "running" if session_id in self._active else "ready"

    def cancel(self, session_id: str) -> bool:
        # No cooperative cancel in current ChatService / CodingAgent.
        return False

    # ------------------------------------------------------------------
    # Rich path (TUI + full console chat from chat_cli)
    # ------------------------------------------------------------------

    def get_rich_context(self, session_id: str | None = None) -> RichChatContext:
        """Return the objects + parity flags expected by the current TUI.

        chat_cli and run_chat_tui can obtain the exact same objects they
        currently build, but sourced from the gateway. This makes the TUI
        "connect with gateway to agent".
        """
        # If we have injected rich objects from chat_cli (current transition path),
        # return them together with the stored parity flags.
        return RichChatContext(
            chat_service=self._chat_service,
            coding_agent=self._coding_agent,
            tools_orchestrator=self._tools_orchestrator,
            dir_mode=self._dir_mode,
            index_dir=self._index_dir,
            index_dirs=None,
            auto_execute_plan=self._auto_execute_plan,
            auto_execute_max_passes=self._auto_execute_max_passes,
            coding_agent_max_steps=self._coding_agent_max_steps,
            resolved_k=self._resolved_k,
            agent_timeout_seconds=self._agent_timeout_seconds,
            root=self.root,
            session_id=session_id,
            event_sink=self._event_sink,
        )

    # ------------------------------------------------------------------
    # Introspection / future multi-agent surface
    # ------------------------------------------------------------------

    def get_ask_service(self) -> Any:
        return getattr(self._chat_service, "_ask_service", None)

    def owns_coding_stack(self) -> bool:
        return self._coding_agent is not None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AgentChatGateway root={self.root} coding={self.owns_coding_stack()}>"


# Convenience alias for the Telegram protocol expectation
ChatGateway = AgentChatGateway
