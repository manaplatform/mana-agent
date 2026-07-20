"""Map user-safe Mana progress into official A2A events."""

from __future__ import annotations

from typing import Any

from mana_agent.protocols.common.security import redact_protocol_value


class A2AEventMapper:
    def progress(self, *, task_id: str, context_id: str, event: Any) -> Any | None:
        from a2a.helpers.proto_helpers import new_text_status_update_event
        from a2a.types.a2a_pb2 import TaskState

        text = ""
        if isinstance(event, dict):
            text = str(event.get("summary") or event.get("message") or event.get("title") or "")
        else:
            text = str(getattr(event, "summary", "") or "")
        if not text:
            return None
        safe = str(redact_protocol_value(text))[:2000]
        return new_text_status_update_event(task_id, context_id, TaskState.TASK_STATE_WORKING, safe)
