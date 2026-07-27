"""Bridge synchronous gateway cancellation to the async desktop service."""

from __future__ import annotations

import asyncio
import threading

from mana_agent.integrations.computer_control.models import PermissionDecision
from mana_agent.integrations.computer_control.service import default_computer_control_service


def cancel_computer_session(session_id: str) -> bool:
    service = default_computer_control_service()
    coroutine = service.cancel_session(session_id)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    result: list[bool] = []

    def runner() -> None:
        result.append(asyncio.run(coroutine))

    thread = threading.Thread(target=runner, name="mana-computer-cancel", daemon=True)
    thread.start()
    thread.join()
    return bool(result and result[0])


def approve_computer_action(request_id: str, *, client_type: str):
    service = default_computer_control_service()
    coroutine = service.approve_and_execute(
        request_id,
        client_type=client_type,
        explicitly_confirmed=True,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    result: list[object] = []
    failure: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(coroutine))
        except BaseException as exc:
            failure.append(exc)

    thread = threading.Thread(target=runner, name="mana-computer-approve", daemon=True)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    return result[0]


def decide_computer_permission(
    request_id: str,
    *,
    decision: PermissionDecision,
    client_type: str,
):
    service = default_computer_control_service()
    coroutine = service.approve_permission_and_execute(
        request_id,
        decision=decision,
        client_type=client_type,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    result: list[object] = []
    failure: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(coroutine))
        except BaseException as exc:
            failure.append(exc)

    thread = threading.Thread(target=runner, name="mana-computer-permission", daemon=True)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    return result[0]


def deny_computer_permission(request_id: str, *, client_type: str) -> None:
    default_computer_control_service().deny_permission_request(
        request_id,
        client_type=client_type,
    )
