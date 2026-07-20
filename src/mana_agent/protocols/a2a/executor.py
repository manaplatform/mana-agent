"""Official A2A AgentExecutor around AgentChatGateway and TaskBoard."""

from __future__ import annotations

import asyncio
from typing import Any

from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.evals.recorder import record_current
from mana_agent.multi_agent.core.types import TaskStatus
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.protocols.common.lifecycle import DurableProtocolStore

from .event_mapper import A2AEventMapper


class ManaA2AExecutor:
    def __init__(self, gateway: AgentChatGateway, *, max_concurrent_tasks: int = 4) -> None:
        self.gateway = gateway
        self.taskboard: TaskBoard = gateway.get_lane_coordinator().taskboard
        self.mapper = A2AEventMapper()
        self.metadata = DurableProtocolStore("a2a", "mappings.json")
        self.semaphore = asyncio.Semaphore(max(1, int(max_concurrent_tasks)))
        self._active: dict[str, asyncio.Task[Any]] = {}

    async def execute(self, context: Any, event_queue: Any) -> None:
        from a2a.helpers.proto_helpers import (
            new_text_artifact_update_event,
            new_text_status_update_event,
        )
        from a2a.types.a2a_pb2 import Task, TaskState, TaskStatus as A2ATaskStatus

        task_id = str(context.task_id or "")
        context_id = str(context.context_id or "")
        prompt = context.get_user_input()
        if not task_id or not context_id or not prompt.strip():
            raise ValueError("A2A task, context, and text message are required.")
        record_current("protocol.a2a.task.started", {"task_id": task_id, "context_id": context_id})
        mapping = self._mapping()
        row = dict(mapping.get("tasks", {}).get(task_id) or {})
        if row and row.get("terminal"):
            raise ValueError("Terminal A2A tasks cannot be resumed.")
        if not row:
            session_id = self._session_for_context(context_id, mapping)
            local = self.taskboard.create_task(
                title=f"A2A task {task_id}",
                user_request=prompt,
                normalized_goal=prompt,
                action_type="a2a",
                session_id=session_id,
            )
            self.taskboard.update_status(local.task_id, TaskStatus.ROUTED)
            row = {"local_task_id": local.task_id, "session_id": session_id, "context_id": context_id, "terminal": False}
            mapping.setdefault("tasks", {})[task_id] = row
            self.metadata.save(mapping)
        local_id = row["local_task_id"]
        session_id = row["session_id"]
        await event_queue.enqueue_event(Task(id=task_id, context_id=context_id, status=A2ATaskStatus(state=TaskState.TASK_STATE_SUBMITTED)))
        self.taskboard.update_status(local_id, TaskStatus.IN_PROGRESS)
        await event_queue.enqueue_event(new_text_status_update_event(task_id, context_id, TaskState.TASK_STATE_WORKING, "Mana-Agent accepted the task."))
        loop = asyncio.get_running_loop()

        def sink(event: Any) -> None:
            update = self.mapper.progress(task_id=task_id, context_id=context_id, event=event)
            if update is not None:
                loop.call_soon_threadsafe(asyncio.create_task, event_queue.enqueue_event(update))

        async with self.semaphore:
            current = asyncio.current_task()
            if current is not None:
                self._active[task_id] = current
            try:
                result = await self.gateway.process_turn_async(session_id, prompt, event_sink=sink)
            except asyncio.CancelledError:
                self._transition(local_id, TaskStatus.CANCELLED)
                row["terminal"] = True
                mapping["tasks"][task_id] = row
                self.metadata.save(mapping)
                await event_queue.enqueue_event(new_text_status_update_event(task_id, context_id, TaskState.TASK_STATE_CANCELLED, "Task canceled."))
                return
            except Exception as exc:
                self._transition(local_id, TaskStatus.FAILED, reason="A2A gateway execution failed.")
                row["terminal"] = True
                mapping["tasks"][task_id] = row
                self.metadata.save(mapping)
                await event_queue.enqueue_event(new_text_status_update_event(task_id, context_id, TaskState.TASK_STATE_FAILED, "Mana-Agent could not complete the task."))
                raise RuntimeError("A2A gateway execution failed.") from exc
            finally:
                self._active.pop(task_id, None)
        if result.error:
            self._transition(local_id, TaskStatus.FAILED, reason=str(result.error)[:500])
            final_state = TaskState.TASK_STATE_FAILED
            final_text = "Mana-Agent could not complete the task."
        else:
            self._transition(local_id, TaskStatus.DONE)
            final_state = TaskState.TASK_STATE_COMPLETED
            final_text = "Task completed."
            await event_queue.enqueue_event(
                new_text_artifact_update_event(
                    task_id,
                    context_id,
                    "answer",
                    result.answer,
                    last_chunk=True,
                    artifact_id=f"artifact-{task_id}-answer",
                )
            )
        row["terminal"] = True
        mapping["tasks"][task_id] = row
        self.metadata.save(mapping)
        record_current("protocol.a2a.task.finished", {"task_id": task_id, "state": int(final_state)})
        await event_queue.enqueue_event(new_text_status_update_event(task_id, context_id, final_state, final_text))

    async def cancel(self, context: Any, event_queue: Any) -> None:
        from a2a.helpers.proto_helpers import new_text_status_update_event
        from a2a.types.a2a_pb2 import TaskState

        task_id = str(context.task_id or "")
        active = self._active.get(task_id)
        if active is not None:
            active.cancel()
        mapping = self._mapping()
        row = dict(mapping.get("tasks", {}).get(task_id) or {})
        if row:
            self.gateway.cancel(row.get("session_id", ""))
            self._transition(row["local_task_id"], TaskStatus.CANCELLED)
            row["terminal"] = True
            mapping["tasks"][task_id] = row
            self.metadata.save(mapping)
        await event_queue.enqueue_event(new_text_status_update_event(task_id, str(context.context_id or ""), TaskState.TASK_STATE_CANCELLED, "Task canceled."))

    def _session_for_context(self, context_id: str, mapping: dict[str, Any]) -> str:
        existing = str(dict(mapping.get("contexts") or {}).get(context_id) or "")
        if existing:
            try:
                self.gateway.create_session(frontend="a2a", session_id=existing)
            except ValueError:
                workspaces = getattr(self.gateway, "_workspaces", None)
                if workspaces is None:
                    raise
                record = workspaces.store.get_session(existing)
                record.status = "active"
                record.closed_at = None
                workspaces.store.save_session(record)
                self.gateway.create_session(frontend="a2a", session_id=existing)
            return existing
        session_id = self.gateway.create_new_session(frontend="a2a")
        mapping.setdefault("contexts", {})[context_id] = session_id
        return session_id

    def _mapping(self) -> dict[str, Any]:
        payload = self.metadata.load()
        payload.setdefault("tasks", {})
        payload.setdefault("contexts", {})
        return payload

    def _transition(self, task_id: str, status: TaskStatus, *, reason: str | None = None) -> None:
        task = self.taskboard.get_task(task_id)
        if task.status in {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.SKIPPED}:
            return
        self.taskboard.update_status(task_id, status, reason=reason)
