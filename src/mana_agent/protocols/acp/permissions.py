"""Fail-closed ACP permission broker for gateway approval events."""

from __future__ import annotations

import asyncio
from typing import Any


class AcpPermissionBroker:
    def __init__(self, client: Any) -> None:
        self.client = client
        self._session_grants: set[tuple[str, str]] = set()
        self._pending: dict[str, set[asyncio.Task[Any]]] = {}

    async def request(self, *, session_id: str, call_id: str, title: str, tool_name: str, raw_input: Any = None) -> bool:
        from acp.helpers import start_tool_call
        from acp.schema import PermissionOption

        grant_key = (session_id, tool_name)
        if grant_key in self._session_grants:
            return True
        tool_call = start_tool_call(
            call_id,
            title,
            kind="other",
            status="pending",
            raw_input=raw_input,
        )
        options = [
            PermissionOption(option_id="allow-once", name="Allow once", kind="allow_once"),
            PermissionOption(option_id="allow-session", name="Allow for this session", kind="allow_always"),
            PermissionOption(option_id="deny", name="Deny", kind="reject_once"),
        ]
        current = asyncio.current_task()
        if current is not None:
            self._pending.setdefault(session_id, set()).add(current)
        try:
            response = await self.client.request_permission(
                session_id=session_id,
                tool_call=tool_call,
                options=options,
            )
        except BaseException:
            return False
        finally:
            if current is not None:
                self._pending.get(session_id, set()).discard(current)
        outcome = getattr(response, "outcome", None)
        if str(getattr(outcome, "outcome", "")) != "selected":
            return False
        option_id = str(getattr(outcome, "option_id", ""))
        if option_id == "allow-session":
            self._session_grants.add(grant_key)
        return option_id in {"allow-once", "allow-session"}

    def clear_session(self, session_id: str) -> None:
        for task in self._pending.pop(session_id, set()):
            task.cancel()
        self._session_grants = {item for item in self._session_grants if item[0] != session_id}
