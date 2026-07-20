"""Durable, caller-scoped A2A task store using official protobuf models."""

from __future__ import annotations

import threading
from typing import Any

from google.protobuf.json_format import MessageToDict, ParseDict

from mana_agent.protocols.common.lifecycle import DurableProtocolStore


def caller_id(context: Any) -> str:
    user = getattr(context, "user", None)
    if user is None or not bool(getattr(user, "is_authenticated", False)):
        return ""
    return str(getattr(user, "user_name", "") or "")


class ManaA2ATaskStore:
    """SDK TaskStore implementation that never leaks cross-caller task existence."""

    def __init__(self) -> None:
        self.store = DurableProtocolStore("a2a", "tasks.json")
        self._lock = threading.RLock()

    async def save(self, task: Any, context: Any) -> None:
        owner = caller_id(context)
        if not owner:
            raise PermissionError("Authenticated A2A caller required.")
        with self._lock:
            payload = self.store.load()
            tasks = dict(payload.get("tasks") or {})
            existing = tasks.get(task.id)
            if existing and existing.get("owner") != owner:
                raise PermissionError("A2A task is not available.")
            if existing:
                from a2a.types.a2a_pb2 import Task, TaskState

                prior = ParseDict(existing.get("task") or {}, Task())
                terminal = {
                    TaskState.TASK_STATE_COMPLETED,
                    TaskState.TASK_STATE_FAILED,
                    TaskState.TASK_STATE_CANCELLED,
                    TaskState.TASK_STATE_REJECTED,
                }
                if prior.status.state in terminal and task.status.state != prior.status.state:
                    raise ValueError("Terminal A2A tasks cannot transition to another state.")
            tasks[task.id] = {
                "owner": owner,
                "task": MessageToDict(task, preserving_proto_field_name=True),
            }
            self.store.save({"tasks": tasks})

    async def get(self, task_id: str, context: Any) -> Any | None:
        from a2a.types.a2a_pb2 import Task

        owner = caller_id(context)
        row = dict(self.store.load().get("tasks") or {}).get(task_id)
        if not owner or not isinstance(row, dict) or row.get("owner") != owner:
            return None
        return ParseDict(row.get("task") or {}, Task())

    async def list(self, params: Any, context: Any) -> Any:
        from a2a.types.a2a_pb2 import ListTasksResponse, Task

        owner = caller_id(context)
        if not owner:
            return ListTasksResponse()
        rows = [
            ParseDict(row.get("task") or {}, Task())
            for row in dict(self.store.load().get("tasks") or {}).values()
            if isinstance(row, dict) and row.get("owner") == owner
        ]
        rows.sort(key=lambda item: item.id)
        context_id = str(getattr(params, "context_id", "") or "")
        if context_id:
            rows = [item for item in rows if item.context_id == context_id]
        status = int(getattr(params, "status", 0) or 0)
        if status:
            rows = [item for item in rows if item.status.state == status]
        page_size = int(getattr(params, "page_size", 0) or 100)
        token = str(getattr(params, "page_token", "") or "0")
        try:
            start = max(0, int(token))
        except ValueError:
            start = 0
        page = rows[start : start + page_size]
        next_token = str(start + len(page)) if start + len(page) < len(rows) else ""
        return ListTasksResponse(tasks=page, next_page_token=next_token, page_size=len(page), total_size=len(rows))

    async def delete(self, task_id: str, context: Any) -> None:
        owner = caller_id(context)
        with self._lock:
            payload = self.store.load()
            tasks = dict(payload.get("tasks") or {})
            row = tasks.get(task_id)
            if owner and isinstance(row, dict) and row.get("owner") == owner:
                tasks.pop(task_id, None)
                self.store.save({"tasks": tasks})


INTERNAL_TO_A2A = {
    "new": "submitted",
    "planning": "working",
    "discussing": "working",
    "routed": "working",
    "waiting_for_tools": "working",
    "queued": "submitted",
    "in_progress": "working",
    "needs_review": "input_required",
    "verifying": "working",
    "done": "completed",
    "blocked": "input_required",
    "failed": "failed",
    "cancelled": "canceled",
    "skipped": "rejected",
}


def map_internal_task_state(value: str) -> str:
    try:
        return INTERNAL_TO_A2A[str(value)]
    except KeyError as exc:
        raise ValueError(f"Unknown internal task state: {value}") from exc
