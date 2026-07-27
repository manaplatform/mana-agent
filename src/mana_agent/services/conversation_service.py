"""Persistent dashboard/API conversations integrated with Mana-Agent chat.

Conversations are stored under:
  ~/.mana/repositories/<repository_id>/dashboard/conversations/

Each conversation keeps:
  meta.json          — conversation metadata
  messages.jsonl     — user/assistant/system/tool timeline entries
  events.jsonl       — runtime ChatEvent stream (via ExecutionEventHub)
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from mana_agent.services.execution_event_hub import (
    conversations_root,
    get_execution_event_hub,
    repository_id_for_root,
)
from mana_agent.workspaces.paths import repository_id_for_path
from mana_agent.sessions import SessionService
from mana_agent.sessions.migration import DashboardConversationMigration


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass
class ConversationMessage:
    message_id: str
    role: str  # user | assistant | system | tool | agent
    content: str
    created_at: str = field(default_factory=_utc_now)
    execution_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationRecord:
    conversation_id: str
    repository_id: str
    root: str
    title: str = "New conversation"
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    status: str = "idle"  # idle | running | failed
    message_count: int = 0
    last_execution_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConversationService:
    """CRUD + message log + chat execution for dashboard/API conversations."""

    def __init__(self, root: str | Path | None = None, *, repository_id: str | None = None) -> None:
        if root is not None:
            self.root = Path(root).expanduser().resolve()
            self.repository_id = repository_id or repository_id_for_path(self.root)
        elif repository_id:
            self.repository_id = repository_id
            self.root = Path(".")
        else:
            raise ValueError("root or repository_id is required")
        self._lock = threading.RLock()
        self._hub = get_execution_event_hub()
        self._sessions = SessionService()
        DashboardConversationMigration(self._sessions.workspaces, self._sessions.history).run()

    @property
    def base_dir(self) -> Path:
        return conversations_root(self.repository_id)

    def _conversation_dir(self, conversation_id: str) -> Path:
        safe = str(conversation_id or "").strip()
        if not safe.startswith("conv_") or not safe.replace("_", "").isalnum():
            raise ValueError("invalid conversation id")
        return self.base_dir / safe

    def _meta_path(self, conversation_id: str) -> Path:
        return self._conversation_dir(conversation_id) / "meta.json"

    def _messages_path(self, conversation_id: str) -> Path:
        return self._conversation_dir(conversation_id) / "messages.jsonl"

    def create(self, *, title: str = "", metadata: dict[str, Any] | None = None) -> ConversationRecord:
        session = self._sessions.create(self.root, frontend="dashboard")
        if title:
            session = self._sessions.rename(session.session_id, title)
        return self._from_session(session)

    def list(self, *, limit: int = 100) -> list[ConversationRecord]:
        limit = max(1, min(int(limit or 100), 500))
        return [
            self._from_session(row)
            for row in self._sessions.workspaces.store.list_sessions()
            if row.primary_repository_id == self.repository_id
        ][:limit]

    def get(self, conversation_id: str) -> ConversationRecord:
        record = self._sessions.workspaces.store.get_session(conversation_id)
        if record.primary_repository_id != self.repository_id:
            raise FileNotFoundError(conversation_id)
        return self._from_session(record)

    def rename(self, conversation_id: str, title: str) -> ConversationRecord:
        self.get_or_raise(conversation_id)
        return self._from_session(self._sessions.rename(conversation_id, title))

    def delete(self, conversation_id: str) -> None:
        self.get_or_raise(conversation_id)
        self._sessions.delete(conversation_id)

    def _save(self, record: ConversationRecord) -> ConversationRecord:
        session = self._sessions.rename(record.conversation_id, record.title)
        return self._from_session(session, execution_id=record.last_execution_id)

    def get_or_raise(self, conversation_id: str) -> ConversationRecord:
        try:
            return self.get(conversation_id)
        except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise FileNotFoundError(conversation_id) from exc

    def list_messages(self, conversation_id: str, *, limit: int = 500) -> list[ConversationMessage]:
        self.get_or_raise(conversation_id)
        return [ConversationMessage(
            message_id=row.message_id, role=row.role, content=row.content,
            created_at=row.created_at, execution_id=row.turn_id, metadata=row.metadata,
        ) for row in self._sessions.history.list(conversation_id, limit=limit)]

    def append_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        execution_id: str = "",
        metadata: dict[str, Any] | None = None,
        message_id: str | None = None,
    ) -> ConversationMessage:
        self.get_or_raise(conversation_id)
        row = self._sessions.history.append(
            conversation_id, role=role, content=content,
            turn_id=execution_id, metadata=metadata, message_id=message_id,
        )
        if role == "user":
            self._sessions.maybe_title_from_message(conversation_id, content)
        self._sessions.workspaces.touch_session(conversation_id)
        return ConversationMessage(
            message_id=row.message_id, role=row.role, content=row.content,
            created_at=row.created_at, execution_id=row.turn_id, metadata=row.metadata,
        )

    def list_events(
        self,
        conversation_id: str,
        *,
        execution_id: str = "",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        self.get_or_raise(conversation_id)
        return self._hub.history(
            conversation_id=conversation_id,
            execution_id=execution_id,
            limit=limit,
            repository_id=self.repository_id,
        )

    def set_status(self, conversation_id: str, status: str, *, execution_id: str = "") -> ConversationRecord:
        record = self.get_or_raise(conversation_id)
        record.status = "running" if status == "running" else str(status or "idle")
        if execution_id:
            record.last_execution_id = execution_id
        return self._save(record)

    def _from_session(self, session: Any, *, execution_id: str = "") -> ConversationRecord:
        messages = self._sessions.history.list(session.session_id, limit=5000)
        count = len(messages)
        last_execution_id = execution_id or (messages[-1].turn_id if messages else "")
        return ConversationRecord(
            conversation_id=session.session_id, repository_id=session.primary_repository_id,
            root=session.cwd, title=session.title or "New chat", created_at=session.created_at,
            updated_at=session.updated_at, status="idle" if session.status == "active" else session.status,
            message_count=count, last_execution_id=last_execution_id,
        )

    def get_full(self, conversation_id: str, *, message_limit: int = 500, event_limit: int = 200) -> dict[str, Any]:
        record = self.get_or_raise(conversation_id)
        return {
            "conversation": record.to_dict(),
            "messages": [item.to_dict() for item in self.list_messages(conversation_id, limit=message_limit)],
            "events": self.list_events(conversation_id, limit=event_limit),
        }

    def send_message(
        self,
        conversation_id: str,
        content: str,
        *,
        chat_runner: Callable[..., dict[str, Any]] | None = None,
        emit_events: bool = True,
        client_message_id: str = "",
    ) -> dict[str, Any]:
        """Append a user message, run chat, append assistant reply, emit runtime events.

        ``chat_runner`` defaults to ``run_dashboard_chat``. When provided, it must accept
        ``(prompt, root=..., conversation_id=..., execution_id=..., event_sink=...)``.
        """
        prompt = str(content or "").strip()
        if not prompt:
            raise ValueError("message content is required")
        self.get_or_raise(conversation_id)
        requested_message_id = str(client_message_id or "").strip()
        if requested_message_id and (
            len(requested_message_id) > 128
            or re.fullmatch(r"[A-Za-z0-9_-]+", requested_message_id) is None
        ):
            raise ValueError("client_message_id must contain only letters, numbers, '_' or '-'")
        if requested_message_id:
            existing = next(
                (
                    item
                    for item in self.list_messages(conversation_id, limit=5000)
                    if item.message_id == requested_message_id
                ),
                None,
            )
            if existing is not None:
                if existing.role != "user" or existing.content != prompt:
                    raise ValueError("client_message_id already belongs to a different message")
                replies = [
                    item
                    for item in self.list_messages(conversation_id, limit=5000)
                    if item.execution_id == existing.execution_id and item.role == "assistant"
                ]
                return {
                    "ok": True,
                    "duplicate": True,
                    "conversation_id": conversation_id,
                    "execution_id": existing.execution_id,
                    "user_message": existing.to_dict(),
                    "assistant_message": replies[-1].to_dict() if replies else None,
                    "events": self.list_events(
                        conversation_id,
                        execution_id=existing.execution_id,
                        limit=200,
                    ),
                }
        if prompt.startswith("/"):
            from mana_agent.chat_commands import (
                CommandContext,
                CommandDispatcher,
                build_default_registry,
            )

            record = self._sessions.workspaces.store.get_session(conversation_id)
            context = CommandContext(
                frontend="dashboard",
                session_id=conversation_id,
                workspace_id=record.workspace_id,
                repository_id=record.primary_repository_id,
                capabilities={"chat", "sessions"},
                sessions=self._sessions,
            )
            command = CommandDispatcher(build_default_registry()).dispatch(prompt, context)
            if command is not None:
                active_id = str(command.data.get("session_id") or conversation_id)
                return {
                    "ok": command.status == "success",
                    "conversation_id": active_id,
                    "execution_id": "",
                    "command_result": command.model_dump(mode="json"),
                    "events": command.events,
                }
        execution_id = _new_id("exec")
        self.set_status(conversation_id, "running", execution_id=execution_id)
        user_message = self.append_message(
            conversation_id,
            role="user",
            content=prompt,
            execution_id=execution_id,
            message_id=requested_message_id or None,
            metadata={"client_message_id": requested_message_id} if requested_message_id else None,
        )

        if emit_events:
            self._hub.emit(
                "message.accepted",
                title="User message accepted",
                conversation_id=conversation_id,
                execution_id=execution_id,
                repository_id=self.repository_id,
                message=prompt[:240],
                status="running",
                metadata={
                    "role": "user",
                    "message_id": user_message.message_id,
                    "client_message_id": requested_message_id,
                    "content": prompt,
                },
            )
            self._hub.emit(
                "turn.started",
                title="Run started",
                conversation_id=conversation_id,
                execution_id=execution_id,
                repository_id=self.repository_id,
                message=prompt[:240],
                status="running",
                metadata={
                    "role": "user",
                    "message_id": user_message.message_id,
                    "client_message_id": requested_message_id,
                    "content": prompt,
                },
            )
            self._hub.emit(
                "agent.routing",
                title="Routing",
                conversation_id=conversation_id,
                execution_id=execution_id,
                repository_id=self.repository_id,
                message="Model decision routing requested",
                status="running",
            )
            self._hub.emit(
                "assistant.started",
                title="Assistant response",
                conversation_id=conversation_id,
                execution_id=execution_id,
                repository_id=self.repository_id,
                status="running",
                metadata={"message_id": f"assistant_{execution_id}"},
            )

        def event_sink(event_type: str, title: str, **kwargs: Any) -> None:
            if not emit_events:
                return
            supplied = dict(kwargs)
            payload = dict(title) if isinstance(title, dict) else {}
            payload.update(supplied)
            resolved_title = str(
                payload.pop("title", "")
                or (event_type.replace(".", " ").title() if isinstance(title, dict) else title)
            )
            message = str(
                payload.pop("message", "")
                or payload.pop("summary", "")
                or payload.pop("output_preview", "")
            )
            metadata = dict(payload.pop("metadata", {}) or payload.pop("payload", {}) or {})
            for key in (
                "tool_name",
                "command",
                "path",
                "error",
                "model",
                "thread_id",
                "task_id",
            ):
                value = payload.pop(key, None)
                if value not in (None, ""):
                    metadata[key] = value
            supplied_execution_id = str(payload.pop("execution_id", "") or "")
            payload.pop("conversation_id", None)
            payload.pop("repository_id", None)
            allowed = {
                key: payload.pop(key)
                for key in (
                    "status",
                    "agent_id",
                    "subagent_id",
                    "step_id",
                    "session_id",
                    "event_id",
                    "parent_event_id",
                    "duration_ms",
                    "ended_at",
                    "persist",
                )
                if key in payload
            }
            metadata.update(payload)
            self._hub.emit(
                event_type,
                title=resolved_title,
                conversation_id=conversation_id,
                execution_id=supplied_execution_id or execution_id,
                repository_id=self.repository_id,
                message=message,
                metadata=metadata,
                **allowed,
            )

        runner = chat_runner
        if runner is None:
            from mana_agent.ui.streamlit_helpers import run_dashboard_chat

            def runner(prompt_text: str, **kwargs: Any) -> dict[str, Any]:  # type: ignore[misc]
                return run_dashboard_chat(
                    prompt_text,
                    root=self.root,
                    conversation_id=conversation_id,
                    execution_id=execution_id,
                    event_sink=event_sink,
                )

        try:
            result = runner(
                prompt,
                root=self.root,
                conversation_id=conversation_id,
                execution_id=execution_id,
                event_sink=event_sink,
            )
        except Exception as exc:
            self.set_status(conversation_id, "failed", execution_id=execution_id)
            if emit_events:
                self._hub.emit(
                    "error",
                    title="Chat execution failed",
                    conversation_id=conversation_id,
                    execution_id=execution_id,
                    repository_id=self.repository_id,
                    message=str(exc),
                    status="failed",
                    metadata={"message_id": user_message.message_id},
                )
            raise

        answer = str((result or {}).get("answer") or "")
        assistant = self.append_message(
            conversation_id,
            role="assistant",
            content=answer,
            execution_id=execution_id,
            metadata={
                "mode": (result or {}).get("mode"),
                "sources": (result or {}).get("sources") or [],
            },
        )
        if emit_events:
            self._hub.emit(
                "turn.finished",
                title="Assistant response",
                conversation_id=conversation_id,
                execution_id=execution_id,
                repository_id=self.repository_id,
                message=answer[:240],
                status="success" if answer else "failed",
                metadata={
                    "mode": (result or {}).get("mode"),
                    "message_id": assistant.message_id,
                    "content": answer,
                },
            )
        self.set_status(conversation_id, "idle", execution_id=execution_id)
        return {
            "ok": True,
            "conversation_id": conversation_id,
            "execution_id": execution_id,
            "user_message": user_message.to_dict(),
            "assistant_message": assistant.to_dict(),
            "result": result or {},
            "events": self.list_events(conversation_id, execution_id=execution_id, limit=200),
        }


def conversation_service_for_root(root: str | Path) -> ConversationService:
    root_path = Path(root).expanduser().resolve()
    return ConversationService(root=root_path, repository_id=repository_id_for_root(root_path))
