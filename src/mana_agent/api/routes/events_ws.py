"""WebSocket transport for conversation execution events.

Subscribes to the shared ExecutionEventHub (ChatEvent envelope). Does not
introduce a second event model. Disconnects cleanly and never block producers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from mana_agent.services.conversation_service import ConversationService
from mana_agent.services.execution_event_hub import get_execution_event_hub
from mana_agent.ui.streamlit_helpers import find_mana_root
from mana_agent.workspaces.paths import repository_id_for_path

logger = logging.getLogger(__name__)

router = APIRouter(tags=["events"])


@router.websocket("/api/v1/ws/conversations/{conversation_id}")
async def conversation_events_ws(
    websocket: WebSocket,
    conversation_id: str,
    root: str | None = Query(default=None),
    repository_id: str | None = Query(default=None),
    execution_id: str | None = Query(default=None),
    replay_limit: int = Query(default=100, ge=0, le=1000),
    after_sequence: int = Query(default=0, ge=0),
) -> None:
    await websocket.accept()
    root_path = find_mana_root(None if not root else __import__("pathlib").Path(root))
    repo_id = repository_id or repository_id_for_path(root_path)
    hub = get_execution_event_hub()
    service = ConversationService(root=root_path, repository_id=repo_id)

    # Validate conversation exists; still allow subscribe after create races.
    try:
        service.get_or_raise(conversation_id)
    except (FileNotFoundError, ValueError):
        await websocket.send_json(
            {
                "type": "error",
                "status": "failed",
                "conversation_id": conversation_id,
                "message": "Conversation not found.",
            }
        )
        await websocket.close(code=4404)
        return

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _enqueue(payload: dict[str, Any]) -> None:
        queue.put_nowait(payload)

    def _on_event(payload: dict[str, Any]) -> None:
        if str(payload.get("conversation_id") or "") != conversation_id:
            return
        if execution_id and str(payload.get("execution_id") or payload.get("turn_id") or "") != execution_id:
            return
        try:
            loop.call_soon_threadsafe(_enqueue, payload)
        except RuntimeError:
            logger.debug("websocket loop closed for %s", conversation_id)

    unsubscribe = hub.subscribe(conversation_id, _on_event)
    try:
        await websocket.send_json(
            {
                "type": "socket.ready",
                "status": "success",
                "conversation_id": conversation_id,
                "execution_id": execution_id,
                "repository_id": repo_id,
                "after_sequence": after_sequence,
            }
        )
        # Replay durable history so reconnecting clients recover state.
        if replay_limit > 0:
            history = hub.history(
                conversation_id=conversation_id,
                execution_id=execution_id or "",
                limit=replay_limit,
                repository_id=repo_id,
                after_sequence=after_sequence,
            )
            for item in history:
                await websocket.send_json({"type": "event.replay", "event": item})
            await websocket.send_json(
                {
                    "type": "socket.replay_complete",
                    "status": "success",
                    "conversation_id": conversation_id,
                    "count": len(history),
                    "last_sequence": max(
                        [
                            after_sequence,
                            *[
                                (
                                    int(item.get("sequence") or 0)
                                    if str(item.get("sequence") or "0").isdigit()
                                    else 0
                                )
                                for item in history
                            ],
                        ]
                    ),
                }
            )

        while True:
            # Multiplex: receive client pings and push events.
            receive_task = asyncio.create_task(websocket.receive_text())
            event_task = asyncio.create_task(queue.get())
            done, pending = await asyncio.wait(
                {receive_task, event_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            if receive_task in done:
                try:
                    raw = receive_task.result()
                except WebSocketDisconnect:
                    break
                except Exception:
                    break
                text = str(raw or "").strip().lower()
                if text in {"ping", '{"type":"ping"}', '{"type": "ping"}'}:
                    await websocket.send_json({"type": "pong", "conversation_id": conversation_id})
                elif text in {"close", "bye"}:
                    break
            if event_task in done:
                try:
                    payload = event_task.result()
                except Exception:
                    continue
                await websocket.send_json({"type": "event", "event": payload})
    except WebSocketDisconnect:
        logger.debug("websocket disconnected conversation=%s", conversation_id)
    except Exception:
        logger.debug("websocket error conversation=%s", conversation_id, exc_info=True)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        unsubscribe()
