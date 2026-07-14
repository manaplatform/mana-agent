from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Protocol

from mana_agent.config.settings import Settings
from mana_agent.services.chat_service import ChatService
from mana_agent.workspaces.service import WorkspaceService

# Central gateway (the single connection point for all frontends to agents).
# When provided we delegate; otherwise fall back to the original local implementation.
try:
    from mana_agent.gateway.chat_gateway import AgentChatGateway as _CoreGateway
except Exception:  # pragma: no cover
    _CoreGateway = None  # type: ignore[assignment]


class TelegramChatGateway(Protocol):
    async def send(self, session_id: str, text: str) -> str: ...
    async def status(self, session_id: str) -> str: ...
    async def cancel(self, session_id: str) -> bool: ...


class ManaChatGateway:
    """Headless adapter around the same ChatService used by CLI chat.

    When a central AgentChatGateway (from mana_agent.gateway) is supplied
    (via the core_gateway= kwarg or by the TelegramConnector), send/create
    operations are delegated to it. This makes Telegram go through the
    single gateway to the agents, exactly as requested.

    The original local implementation is kept as fallback for backward compat.
    """

    def __init__(
        self,
        repository: Path | str,
        *,
        core_gateway: Any | None = None,
    ) -> None:
        self.repository = Path(repository).expanduser().resolve()
        self.workspaces = WorkspaceService()
        self._services: dict[str, ChatService] = {}
        self._history: dict[str, list[tuple[str, str]]] = {}
        self._history_store: Any | None = None
        self._active: set[str] = set()

        # Central gateway (preferred path)
        self._core: Any | None = None
        if core_gateway is not None:
            self._core = core_gateway
        elif _CoreGateway is not None:
            # Auto-create a central gateway for this repository (simple path).
            # Rich coding features can still be enabled by passing one explicitly
            # from higher layers (e.g. via TelegramConnector(gateway=...)).
            try:
                self._core = _CoreGateway(self.repository, coding_agent=False)
            except Exception:
                self._core = None

    def create_session(self) -> str:
        if self._core is not None and hasattr(self._core, "create_session"):
            try:
                return self._core.create_session(frontend="telegram")
            except Exception:
                pass
        return self.workspaces.create_session(self.repository).session_id

    def bind_store(self, store: Any) -> None:
        self._history_store = store

    def _service(self, session_id: str) -> ChatService:
        existing = self._services.get(session_id)
        if existing is not None:
            return existing
        context = self.workspaces.context_for_session(session_id)
        root = Path(context.session.cwd).resolve()
        settings = Settings()
        from mana_agent.commands.cli_internal import build_ask_service
        ask_service = build_ask_service(settings, None, project_root=root)
        service = ChatService(
            ask_service=ask_service, settings=settings, root_dir=root,
            index_dir=None, agent_tools=True,
        )
        self._services[session_id] = service
        return service

    async def send(self, session_id: str, text: str) -> str:
        # Prefer the central gateway (this is the "all connections via gateway" path).
        if self._core is not None and hasattr(self._core, "send"):
            try:
                # Core gateway send may be sync or async; normalize.
                send_fn = self._core.send
                if asyncio.iscoroutinefunction(send_fn):
                    return await send_fn(session_id, text)
                return await asyncio.to_thread(send_fn, session_id, text)
            except Exception:
                # Fall through to local implementation on any error
                pass

        self._active.add(session_id)
        try:
            history = (
                self._history_store.history(session_id, limit=12)
                if self._history_store is not None
                else self._history.get(session_id, [])[-12:]
            )
            question = text
            if history:
                transcript = "\n\n".join(
                    f"User: {question_text}\nMana-Agent: {answer_text}"
                    for question_text, answer_text in history
                )
                question = f"Conversation history for continuity:\n{transcript[-20_000:]}\n\nCurrent user message:\n{text}"
            response = await asyncio.to_thread(self._service(session_id).ask, question)
            answer = getattr(response, "answer", response)
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("Mana-Agent returned no final response.")
            result = answer.strip()
            self._history.setdefault(session_id, []).append((text, result))
            self._history[session_id] = self._history[session_id][-12:]
            if self._history_store is not None:
                self._history_store.append_history(session_id, text, result)
            return result
        finally:
            self._active.discard(session_id)

    async def status(self, session_id: str) -> str:
        if self._core is not None and hasattr(self._core, "status"):
            try:
                st = self._core.status(session_id)
                return await st if asyncio.iscoroutine(st) else st
            except Exception:
                pass
        return "running" if session_id in self._active else "ready"

    async def cancel(self, session_id: str) -> bool:
        if self._core is not None and hasattr(self._core, "cancel"):
            try:
                res = self._core.cancel(session_id)
                return await res if asyncio.iscoroutine(res) else bool(res)
            except Exception:
                pass
        # ChatService currently has no cooperative cancellation contract.
        return False


class TelegramConversationRouter:
    def __init__(self, store: Any, gateway: ManaChatGateway) -> None:
        self.store = store
        self.gateway = gateway
        bind_store = getattr(gateway, "bind_store", None)
        if callable(bind_store):
            bind_store(store)

    def session(self, conversation_key: str) -> str:
        session_id = self.store.session_id(conversation_key)
        if session_id:
            return session_id
        session_id = self.gateway.create_session()
        self.store.bind_session(conversation_key, session_id)
        return session_id

    def reset(self, conversation_key: str) -> str:
        session_id = self.gateway.create_session()
        self.store.bind_session(conversation_key, session_id)
        return session_id
