from __future__ import annotations

import threading
import logging
from pathlib import Path
from typing import Any, Callable

from mana_agent.memory.models import MemoryScope
from mana_agent.services.chat_session_history import ChatSessionHistory
from mana_agent.sessions.models import SessionActivation, SessionSummary
from mana_agent.workspaces.models import SessionRecord
from mana_agent.workspaces.service import WorkspaceService
from mana_agent.workspaces.paths import mana_home
from mana_agent.workspaces.store import atomic_write_json

logger = logging.getLogger(__name__)


class SessionBusyError(RuntimeError):
    pass


class SessionService:
    """Authority for lifecycle, durable history, and active-session binding."""

    def __init__(
        self,
        workspace_service: WorkspaceService | None = None,
        *,
        history: ChatSessionHistory | None = None,
        memory_service: Any | None = None,
        process_manager: Any | None = None,
        browser_closer: Callable[[str], None] | None = None,
    ) -> None:
        self.workspaces = workspace_service or WorkspaceService()
        self.history = history or ChatSessionHistory()
        self.memory_service = memory_service
        if process_manager is None:
            try:
                from mana_agent.background import BackgroundProcessManager

                self.process_manager = BackgroundProcessManager()
            except Exception:
                self.process_manager = None
        else:
            self.process_manager = process_manager
        self.browser_closer = browser_closer
        self._lock = threading.RLock()
        self._active_by_frontend: dict[str, str] = {}

    def create(self, cwd: str | Path, *, workspace_id: str | None = None, frontend: str = "cli") -> SessionRecord:
        with self._lock:
            record = self.workspaces.create_session(cwd, workspace_id=workspace_id)
            self._active_by_frontend[frontend] = record.session_id
            return record

    def current(self, frontend: str = "cli") -> SessionRecord | None:
        sid = self._active_by_frontend.get(frontend)
        if not sid:
            return None
        try:
            return self.workspaces.store.get_session(sid)
        except FileNotFoundError:
            self._active_by_frontend.pop(frontend, None)
            return None

    def bind(
        self,
        session_id: str,
        *,
        frontend: str = "cli",
        workspace_id: str | None = None,
        repository_id: str | None = None,
    ) -> SessionActivation:
        with self._lock:
            record = self.workspaces.store.get_session(session_id)
            if workspace_id and record.workspace_id != workspace_id:
                raise ValueError("session does not belong to the selected workspace")
            if repository_id and repository_id not in record.attached_repository_ids:
                raise ValueError("session does not include the active repository")
            if record.status == "archived":
                raise ValueError("archived sessions cannot be activated")
            if record.status != "active":
                record = self.workspaces.reopen_session(session_id)
            self._active_by_frontend[frontend] = session_id
            return SessionActivation(
                session=self.summary(record, current_id=session_id),
                messages=[row.to_dict() for row in self.history.list(session_id, limit=5000)],
            )

    def list(
        self,
        *,
        workspace_id: str | None = None,
        repository_id: str | None = None,
        current_id: str = "",
    ) -> list[SessionSummary]:
        rows = self.workspaces.store.list_sessions()
        return [
            self.summary(row, current_id=current_id)
            for row in rows
            if (not workspace_id or row.workspace_id == workspace_id)
            and (not repository_id or repository_id in row.attached_repository_ids)
        ]

    def summary(self, record: SessionRecord, *, current_id: str = "") -> SessionSummary:
        messages = self.history.list(record.session_id, limit=5000)
        attached = False
        if self.process_manager is not None:
            attached = bool(self.process_manager.list(session_id=record.session_id))
        try:
            repository = self.workspaces.store.get_repository(record.primary_repository_id).name
        except FileNotFoundError:
            repository = record.primary_repository_id
        return SessionSummary(
            session_id=record.session_id,
            short_id=record.session_id[-8:],
            title=record.title or "New chat",
            current=record.session_id == current_id,
            status=record.status,
            repository=repository,
            workspace_id=record.workspace_id,
            created_at=record.created_at,
            updated_at=record.updated_at,
            message_count=len(messages),
            has_background_processes=attached,
        )

    def rename(self, session_id: str, title: str) -> SessionRecord:
        with self._lock:
            return self.workspaces.rename_session(session_id, title)

    def maybe_title_from_message(self, session_id: str, content: str) -> SessionRecord:
        record = self.workspaces.store.get_session(session_id)
        if record.title not in {"", "New chat", "New conversation"}:
            return record
        meaningful = " ".join(str(content or "").split())
        if not meaningful or meaningful.startswith("/"):
            return record
        return self.workspaces.rename_session(session_id, meaningful[:72])

    def _fence_session(self, session_id: str) -> None:
        tombstone = mana_home() / "runtime" / "session-tombstones" / f"{session_id}.json"
        atomic_write_json(tombstone, {"session_id": session_id, "deleted": True})

    def delete(self, session_id: str, *, gateway: Any | None = None) -> None:
        record = self.workspaces.store.get_session(session_id)
        self._fence_session(session_id)
        if gateway is not None and gateway.status(session_id) == "running":
            if not gateway.cancel(session_id, reason="session deleted", source="delete"):
                raise SessionBusyError("session has an active turn that could not be cancelled")
        if self.process_manager is not None:
            self.process_manager.stop_session_processes(session_id, transient_only=False)
        if self.browser_closer is not None:
            try:
                self.browser_closer(session_id)
            except Exception:
                pass
        with self._lock:
            for frontend, active in list(self._active_by_frontend.items()):
                if active == session_id:
                    self._active_by_frontend.pop(frontend, None)
        self._clear_memory(record)
        from mana_agent.canvas.store import CanvasStore

        try:
            CanvasStore().delete_session(session_id)
        except Exception:
            pass
        self.workspaces.delete_session(session_id)

    def replace(
        self, session_id: str, *, gateway: Any | None, frontend: str
    ) -> SessionRecord:
        record = self.workspaces.store.get_session(session_id)
        self._fence_session(session_id)
        if gateway is not None and gateway.status(session_id) == "running":
            gateway.cancel(session_id, reason="session replaced by /new", source="new")
        if self.process_manager is not None:
            self.process_manager.stop_session_processes(session_id, transient_only=False)
        if self.browser_closer is not None:
            try:
                self.browser_closer(session_id)
            except Exception:
                pass
        with self._lock:
            new_record = self.workspaces.create_session(
                record.cwd, workspace_id=record.workspace_id
            )
            self._active_by_frontend[frontend] = new_record.session_id
            for f_name, active in list(self._active_by_frontend.items()):
                if active == session_id:
                    self._active_by_frontend.pop(f_name, None)
        self._clear_memory(record)
        from mana_agent.canvas.store import CanvasStore

        try:
            CanvasStore().delete_session(session_id)
        except Exception:
            pass
        self.workspaces.delete_session(session_id)
        return new_record

    def _clear_memory(self, record: SessionRecord) -> None:
        self._fence_session(record.session_id)
        memory = self.memory_service
        if memory is None or not hasattr(memory, "clear"):
            return
        scope = MemoryScope(session_id=record.session_id)
        from mana_agent.memory.compatibility import run_sync
        try:
            run_sync(memory.clear(scope))
        except Exception as exc:
            # The content-free tombstone is authoritative when a provider cannot
            # physically delete.  Recall checks it before contacting the backend.
            logger.warning(
                "Session memory deletion failed for %s; recall tombstone remains active (%s).",
                record.session_id,
                type(exc).__name__,
            )
            return
