from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mana_agent.services.chat_session_history import ChatSessionHistory
from mana_agent.workspaces.paths import mana_home
from mana_agent.workspaces.service import WorkspaceService
from mana_agent.workspaces.store import atomic_write_json


class DashboardConversationMigration:
    """One-way, idempotent import into canonical workspace sessions."""

    def __init__(self, workspaces: WorkspaceService | None = None, history: ChatSessionHistory | None = None) -> None:
        self.workspaces = workspaces or WorkspaceService()
        self.history = history or ChatSessionHistory()

    def run(self) -> list[str]:
        migrated: list[str] = []
        for meta_path in sorted((mana_home() / "repositories").glob("*/dashboard/conversations/*/meta.json")):
            source = meta_path.parent
            marker = source / ".migrated-to-session.json"
            if marker.exists():
                try:
                    sid = str(json.loads(marker.read_text(encoding="utf-8")).get("session_id") or "")
                    if sid:
                        self.workspaces.store.get_session(sid)
                        continue
                except (OSError, ValueError, FileNotFoundError, json.JSONDecodeError):
                    pass
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            root = Path(str(payload.get("root") or "")).expanduser().resolve()
            if not root.is_dir():
                continue
            conversation_id = str(payload.get("conversation_id") or source.name)
            sid = "session_" + hashlib.sha256(f"dashboard:{conversation_id}".encode()).hexdigest()[:20]
            try:
                self.workspaces.store.get_session(sid)
            except FileNotFoundError:
                pass
            else:
                self.workspaces.store.delete_session(sid)
            record = self.workspaces.create_session(root, session_id=sid)
            record.title = str(payload.get("title") or "New chat")[:120]
            record.created_at = str(payload.get("created_at") or record.created_at)
            record.opened_at = record.created_at
            record.updated_at = str(payload.get("updated_at") or record.updated_at)
            self.workspaces.store.save_session(record)
            messages_path = source / "messages.jsonl"
            if messages_path.exists():
                for index, line in enumerate(messages_path.read_text(encoding="utf-8").splitlines()):
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.history.append(
                        sid, role=str(item.get("role") or "system"),
                        content=str(item.get("content") or ""),
                        turn_id=str(item.get("execution_id") or f"migration-{index}"),
                        message_id=str(item.get("message_id") or "") or None,
                        created_at=str(item.get("created_at") or record.created_at),
                        metadata=dict(item.get("metadata") or {}),
                    )
            atomic_write_json(marker, {"session_id": sid, "schema_version": 1})
            migrated.append(sid)
        return migrated
