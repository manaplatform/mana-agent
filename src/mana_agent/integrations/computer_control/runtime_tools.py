"""Narrow model tools backed by the shared computer-control service."""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from mana_agent.integrations.computer_control.context import current_computer_client
from mana_agent.integrations.computer_control.errors import (
    ComputerControlError,
    OperatingSystemPermissionDenied,
    RemoteControlDenied,
)
from mana_agent.integrations.computer_control.models import (
    CalendarEvent,
    ComputerAction,
    ComputerTarget,
    NoteDocument,
    ScreenRecordingRequest,
)
from mana_agent.integrations.computer_control.policy import ACTION_SPECS
from mana_agent.integrations.computer_control.service import ComputerControlService, default_computer_control_service
from mana_agent.runtime_context import DurableExecutionContext
from mana_agent.transactional_actions.models import ActionState, TransactionalRequestState
from mana_agent.utils.durable_diagnostics import resource_digest


class _Decision(BaseModel):
    source_decision_id: str = Field(description="ID of the validated model decision selecting this exact tool and arguments.")
    execution_id: str | None = Field(default=None, description="Reuse the execution ID returned with an exact-action confirmation.")
    confirmation_token: str | None = Field(default=None, description="Short-lived exact-action token from a trusted confirmation UI.")


class _NoTarget(_Decision):
    pass


class _Application(_Decision):
    application_id: str


class _OpenUrl(_Decision):
    url: str


class _CalendarList(_Decision):
    starts_after: datetime | None = None
    ends_before: datetime | None = None
    calendar_id: str | None = None
    limit: int = Field(default=20, ge=1, le=100)


class _CalendarWrite(_Decision):
    event: CalendarEvent


class _Resource(_Decision):
    resource_id: str


class _MediaPlay(_Decision):
    application_id: str | None = None
    query: str | None = None
    kind: Literal["song", "album", "artist", "playlist"] | None = None


class _MediaVolume(_Decision):
    volume: float = Field(ge=0, le=1)
    application_id: str | None = None


class _NotesSearch(_Decision):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=20, ge=1, le=100)


class _NoteWrite(_Decision):
    note: NoteDocument


class _ClipboardWrite(_Decision):
    text: str = Field(max_length=1_000_000)


class _PathTarget(_Decision):
    path: str


class _PathDestination(_PathTarget):
    destination: str


class _Screenshot(_Decision):
    mode: Literal["full_screen", "active_window", "display"] = "full_screen"
    display_id: str | None = None


class _ScreenRecording(_Decision):
    mode: Literal["display"] = "display"
    display_id: str | None = None
    output_path: str | None = None
    container: Literal["mov"] = "mov"
    microphone_audio: bool = False
    system_audio: bool = False
    maximum_duration_seconds: int | None = Field(default=None, ge=1, le=1800)


class _Notification(_Decision):
    title: str = Field(max_length=200)
    body: str = Field(max_length=1000)


class _SystemVolume(_Decision):
    volume: float | None = Field(default=None, ge=0, le=1)
    muted: bool | None = None


class _SystemControl(_Decision):
    operation: Literal["settings", "lock", "sleep", "restart", "shutdown"]
    pane: str | None = None


class _PermissionInput(BaseModel):
    scope: Literal[
        "computer.apps.read", "computer.apps.control", "computer.calendar.read", "computer.calendar.write",
        "computer.media.read", "computer.media.control", "computer.notes.read", "computer.notes.write",
        "computer.browser.tabs.read", "computer.browser.page.read", "computer.browser.control",
        "computer.clipboard.read", "computer.clipboard.write", "computer.files.read", "computer.files.write",
        "computer.screenshot.capture", "computer.screen_recording.capture", "computer.notifications.send", "computer.system.read", "computer.system.control",
    ]


def _run(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result: list[Any] = []
    failure: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:
            failure.append(exc)

    thread = threading.Thread(target=runner, name="mana-computer-tool-sync", daemon=True)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    return result[0]


def _response(call) -> str:
    try:
        value = call()
        if isinstance(value, dict) and value.get("ok") is False:
            return json.dumps(value, ensure_ascii=False, default=str)
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        return json.dumps({"ok": True, "result": value}, ensure_ascii=False, default=str)
    except ComputerControlError as exc:
        payload = exc.payload()
        return json.dumps({"ok": False, **payload}, ensure_ascii=False)
    except ValueError as exc:
        return json.dumps({"ok": False, "error_code": "invalid_tool_input", "message": str(exc)})


def _action(operation: str, payload: _Decision, *, capability: str, target: ComputerTarget | None = None, arguments: dict[str, Any] | None = None) -> ComputerAction:
    spec = ACTION_SPECS[operation]
    values: dict[str, Any] = {
        "capability": capability,
        "operation": operation,
        "permission_scope": spec.permission_scope,
        "risk": spec.risk,
        "target": target or ComputerTarget(),
        "arguments": arguments or {},
        "source_decision_id": payload.source_decision_id,
        "confirmation_token": payload.confirmation_token,
    }
    if payload.execution_id:
        values["execution_id"] = payload.execution_id
    client = current_computer_client()
    if client is not None:
        values["execution_context"] = client.execution_context or DurableExecutionContext(
            session_id=client.session_id, source_decision_id=payload.source_decision_id,
        )
    return ComputerAction.model_validate(values)


def build_computer_langchain_tools(service: ComputerControlService | None = None) -> list[Any]:
    """Build tools without touching the desktop until a selected tool executes."""
    control = service or default_computer_control_service()

    def authenticated_client():
        client = current_computer_client()
        if client is None:
            raise RemoteControlDenied("Computer tool execution has no authenticated gateway client context.")
        probe = _action(
            "capabilities.discover",
            _Decision(source_decision_id="system:capability-read"),
            capability="applications",
        )
        control.policy.validate_client(
            probe,
            client_type=client.client_type,
            locally_confirmed=client.locally_confirmed,
        )
        return client

    def record_terminal_request(
        *,
        payload: _Decision,
        operation: str,
        outcome_code: str,
        state: TransactionalRequestState,
        resource: str = "",
    ) -> None:
        client = authenticated_client()
        from mana_agent.transactional_actions.runtime import create_transactional_runtime

        context = client.execution_context
        workspace_root = Path(client.workspace_root).resolve() if client.workspace_root else Path.cwd()
        runtime = client.transactional_runtime or create_transactional_runtime(workspace_root)
        runtime.record_request(
            state=state,
            source_decision_id=payload.source_decision_id,
            session_id=client.session_id,
            conversation_id=context.conversation_id if context else "",
            turn_id=context.turn_id if context else "",
            task_id=context.task_id if context else "",
            branch_id=context.branch_id if context else "",
            client_type=client.client_type,
            tool_name="computer",
            operation_name=operation,
            resource_digest=resource_digest(resource) if resource else "",
            outcome_code=outcome_code,
            create_notice=True,
        )

    def execute(action: ComputerAction) -> Any:
        client = authenticated_client()
        if action.source_decision_id not in client.allowed_decision_ids:
            raise RemoteControlDenied(
                "The computer action is not bound to the gateway's validated model decision."
            )
        from mana_agent.transactional_actions.computer import ComputerActionAdapter
        from mana_agent.transactional_actions.gateway import ApprovalRequired
        from mana_agent.transactional_actions.runtime import create_transactional_runtime

        adapter = ComputerActionAdapter(
            action=action,
            session_id=client.session_id,
            client_type=client.client_type,
            service=control,
        )
        workspace_root = Path(client.workspace_root).resolve() if client.workspace_root else Path.cwd()
        runtime = client.transactional_runtime or create_transactional_runtime(workspace_root)
        execution_context = action.execution_context
        target_material = str(
            action.arguments.get("output_path")
            or action.target.display_id
            or action.target.resource_id
            or action.target.path
            or action.capability
        )
        request = runtime.record_request(
            state=TransactionalRequestState.RECEIVED,
            source_decision_id=action.source_decision_id,
            session_id=client.session_id,
            conversation_id=execution_context.conversation_id if execution_context else "",
            turn_id=execution_context.turn_id if execution_context else "",
            task_id=execution_context.task_id if execution_context else "",
            branch_id=execution_context.branch_id if execution_context else "",
            client_type=client.client_type,
            tool_name="computer",
            operation_name=action.operation,
            resource_digest=resource_digest(target_material),
        )
        try:
            capability_report = _run(control.capabilities())
        except ComputerControlError as exc:
            runtime.update_request(
                request,
                TransactionalRequestState.ROUTE_UNAVAILABLE,
                outcome_code=exc.code,
                create_notice=True,
            )
            raise
        if not capability_report.supports(action.capability, action.operation):
            runtime.update_request(
                request,
                TransactionalRequestState.CAPABILITY_UNAVAILABLE,
                outcome_code="capability_unavailable",
                create_notice=True,
            )
            return {
                "ok": False,
                "error_code": "capability_unavailable",
                "message": f"{action.operation!r} is not implemented by {capability_report.provider}.",
                "inbox_item_id": request.inbox_item_id,
            }
        os_permission = _run(control.provider.check_permission(action.capability))
        if not os_permission.granted:
            runtime.update_request(
                request,
                TransactionalRequestState.PERMISSION_DENIED,
                outcome_code="os_permission_denied",
                create_notice=True,
            )
            raise OperatingSystemPermissionDenied(
                os_permission.reason or "Operating-system privacy authorization is unavailable.",
                corrective_action="Enable Screen Recording in macOS Privacy & Security settings.",
            )
        try:
            outcome = runtime.gateway.execute(adapter)
        except ApprovalRequired as exc:
            runtime.update_request(
                request,
                TransactionalRequestState.APPROVAL_REQUIRED,
                outcome_code="approval_required",
                action_id=exc.action.action_id,
                inbox_item_id=exc.inbox_item_id,
            )
            decision = exc.action.policy_decision.model_dump(mode="json") if exc.action.policy_decision else {}
            return {
                "ok": False,
                "error_code": "transactional_approval_required",
                "message": "This exact computer action is waiting for approval in the active local TUI or dashboard.",
                "permission_request_id": exc.inbox_item_id,
                "permission_scope": "transactional_action.once",
                "action_id": exc.action.action_id,
                "transaction_id": exc.action.transaction_id,
                "inbox_item_id": exc.inbox_item_id,
                "preview_digest": exc.action.preview_digest,
                "preview": exc.action.preview.redacted() if exc.action.preview else {},
                "policy_decision": decision,
                "risk_effect_labels": exc.action.approval_effect_labels(),
                "transactional_action_approval": True,
            }
        except ComputerControlError as exc:
            stored = runtime.store.action_for_idempotency_key(adapter.build_intent().idempotency_key)
            runtime.update_request(
                request,
                TransactionalRequestState.PERMISSION_DENIED if exc.code in {"os_permission_denied", "mana_permission_denied", "permission_required"} else TransactionalRequestState.FAILED,
                outcome_code=exc.code,
                action_id=stored.action_id if stored else "",
                inbox_item_id=stored.inbox_item_id if stored else "",
                create_notice=stored is None,
            )
            raise
        if outcome.action.state is ActionState.FAILED:
            runtime.update_request(
                request,
                TransactionalRequestState.POLICY_DENIED if outcome.action.policy_decision and outcome.action.policy_decision.outcome.value == "deny" else TransactionalRequestState.FAILED,
                outcome_code=(outcome.action.policy_decision.reason_codes[0] if outcome.action.policy_decision else "action_failed"),
                action_id=outcome.action.action_id,
                create_notice=True,
            )
            return {
                "ok": False,
                "error_code": (
                    outcome.action.policy_decision.reason_codes[0]
                    if outcome.action.policy_decision
                    else "action_failed"
                ),
                "message": outcome.action.error or (
                    outcome.action.policy_decision.explanation
                    if outcome.action.policy_decision
                    else "The transactional action did not reach execution."
                ),
                "action_id": outcome.action.action_id,
                "inbox_item_id": request.inbox_item_id,
            }
        else:
            runtime.update_request(
                request,
                TransactionalRequestState.VERIFIED if outcome.action.state is ActionState.COMMITTED else TransactionalRequestState.EXECUTING,
                action_id=outcome.action.action_id,
                inbox_item_id=outcome.action.inbox_item_id,
            )
        return {"ok": True, "result": outcome.result, "action_id": outcome.action.action_id}

    def capabilities() -> str:
        return _response(lambda: (authenticated_client(), _run(control.capabilities()))[1])

    def permission_status(**kwargs: object) -> str:
        payload = _PermissionInput.model_validate(kwargs)

        def read_status() -> dict[str, Any]:
            authenticated_client()
            status = _run(control.permission_status(payload.scope))
            result = status.model_dump(mode="json")
            result["request_created"] = False
            if status.decision.value == "ask":
                result["next_step"] = (
                    "No approval prompt exists yet. If the user requested a concrete action, "
                    "invoke that exact computer action tool now; it will create the bound in-chat request."
                )
            elif not status.granted:
                result["next_step"] = "Stop because this permission is denied or unavailable."
            else:
                result["next_step"] = "Continue with the exact model-selected computer action."
            return result

        return _response(read_status)

    def no_target(operation: str, capability: str, kwargs: dict[str, object]) -> str:
        payload = _NoTarget.model_validate(kwargs)
        return _response(lambda: execute(_action(operation, payload, capability=capability)))

    def application(operation: str, kwargs: dict[str, object]) -> str:
        payload = _Application.model_validate(kwargs)
        return _response(lambda: execute(_action(operation, payload, capability="applications", target=ComputerTarget(application_id=payload.application_id))))

    def resource(operation: str, capability: str, kwargs: dict[str, object]) -> str:
        payload = _Resource.model_validate(kwargs)
        return _response(lambda: execute(_action(operation, payload, capability=capability, target=ComputerTarget(resource_id=payload.resource_id))))

    def calendar_list(**kwargs: object) -> str:
        payload = _CalendarList.model_validate(kwargs)
        args = payload.model_dump(exclude={"source_decision_id", "execution_id", "confirmation_token"}, mode="json", exclude_none=True)
        return _response(lambda: execute(_action("calendar.list", payload, capability="calendar", arguments=args)))

    def calendar_write(operation: str, kwargs: dict[str, object]) -> str:
        payload = _CalendarWrite.model_validate(kwargs)
        target = ComputerTarget(resource_id=payload.event.event_id) if payload.event.event_id else ComputerTarget()
        return _response(lambda: execute(_action(operation, payload, capability="calendar", target=target, arguments={"event": payload.event.model_dump(mode="json")})))

    def media_play(**kwargs: object) -> str:
        payload = _MediaPlay.model_validate(kwargs)
        args = payload.model_dump(exclude={"source_decision_id", "execution_id", "confirmation_token", "application_id"}, exclude_none=True)
        return _response(lambda: execute(_action("media.play", payload, capability="media", target=ComputerTarget(application_id=payload.application_id), arguments=args)))

    def media_volume(**kwargs: object) -> str:
        payload = _MediaVolume.model_validate(kwargs)
        return _response(lambda: execute(_action("media.volume", payload, capability="media", target=ComputerTarget(application_id=payload.application_id), arguments={"volume": payload.volume})))

    def notes_search(**kwargs: object) -> str:
        payload = _NotesSearch.model_validate(kwargs)
        return _response(lambda: execute(_action("notes.search", payload, capability="notes", arguments={"query": payload.query, "limit": payload.limit})))

    def note_write(operation: str, kwargs: dict[str, object]) -> str:
        payload = _NoteWrite.model_validate(kwargs)
        target = ComputerTarget(resource_id=payload.note.note_id) if payload.note.note_id else ComputerTarget()
        return _response(lambda: execute(_action(operation, payload, capability="notes", target=target, arguments={"note": payload.note.model_dump(mode="json")})))

    def browser_open(**kwargs: object) -> str:
        payload = _OpenUrl.model_validate(kwargs)
        return _response(lambda: execute(_action("browser.open_url", payload, capability="browser", target=ComputerTarget(url=payload.url))))

    def clipboard_write(**kwargs: object) -> str:
        payload = _ClipboardWrite.model_validate(kwargs)
        return _response(lambda: execute(_action("clipboard.write", payload, capability="clipboard", arguments={"text": payload.text})))

    def path_target(operation: str, kwargs: dict[str, object]) -> str:
        payload = _PathTarget.model_validate(kwargs)
        return _response(lambda: execute(_action(operation, payload, capability="filesystem", target=ComputerTarget(path=payload.path))))

    def path_destination(operation: str, kwargs: dict[str, object]) -> str:
        payload = _PathDestination.model_validate(kwargs)
        return _response(lambda: execute(_action(operation, payload, capability="filesystem", target=ComputerTarget(path=payload.path), arguments={"destination": payload.destination})))

    def screenshot(**kwargs: object) -> str:
        payload = _Screenshot.model_validate(kwargs)
        return _response(lambda: execute(_action("screenshots.capture", payload, capability="screenshots", target=ComputerTarget(display_id=payload.display_id), arguments={"mode": payload.mode})))

    def screen_recording(**kwargs: object) -> str:
        payload = _ScreenRecording.model_validate(kwargs)
        request = ScreenRecordingRequest.model_validate(payload.model_dump(
            exclude={"source_decision_id", "execution_id", "confirmation_token"}
        ))
        missing = [field for field, value in (
            ("display_id", request.display_id),
            ("output_path", request.output_path),
            ("maximum_duration_seconds", request.maximum_duration_seconds),
        ) if value in {None, ""}]
        if missing:
            record_terminal_request(
                payload=payload,
                operation="screen_recording.capture",
                outcome_code="screen_recording_clarification_required",
                state=TransactionalRequestState.CLARIFICATION_REQUIRED,
                resource=str(request.output_path or ""),
            )
            return json.dumps({"ok": False, "error_code": "screen_recording_clarification_required", "message": "A bounded recording requires the missing material parameters before an action can be proposed.", "clarification_fields": missing, "capability": "screen_recording"}, ensure_ascii=False)
        if request.system_audio:
            record_terminal_request(
                payload=payload,
                operation="screen_recording.capture",
                outcome_code="capability_unavailable",
                state=TransactionalRequestState.CAPABILITY_UNAVAILABLE,
                resource=str(request.output_path or ""),
            )
            return json.dumps({"ok": False, "error_code": "capability_unavailable", "message": "System-audio recording is not implemented by the bounded native provider."}, ensure_ascii=False)
        return _response(lambda: execute(_action("screen_recording.capture", payload, capability="screen_recording", target=ComputerTarget(display_id=request.display_id), arguments=request.model_dump(mode="json"))))

    def notification(**kwargs: object) -> str:
        payload = _Notification.model_validate(kwargs)
        return _response(lambda: execute(_action("notifications.send", payload, capability="notifications", arguments={"title": payload.title, "body": payload.body})))

    def system_volume(**kwargs: object) -> str:
        payload = _SystemVolume.model_validate(kwargs)
        args = payload.model_dump(exclude={"source_decision_id", "execution_id", "confirmation_token"}, exclude_none=True)
        return _response(lambda: execute(_action("system.volume", payload, capability="system", arguments=args)))

    def system_control(**kwargs: object) -> str:
        payload = _SystemControl.model_validate(kwargs)
        operation = f"system.{payload.operation}"
        args = {"pane": payload.pane} if payload.pane else {}
        return _response(lambda: execute(_action(operation, payload, capability="system", arguments=args)))

    def tool(func, name: str, description: str, schema=None):
        return StructuredTool.from_function(func=func, name=name, description=description, args_schema=schema)

    decision = "Requires a validated model decision ID. "
    tools = [
        tool(capabilities, "computer_capabilities", "Discover available desktop capability groups; installed application details are redacted unless apps-read permission exists."),
        tool(
            permission_status,
            "computer_permission_status",
            "Read one Mana-Agent computer permission decision. This never creates a prompt: an `ask` "
            "result means invoke the exact requested action tool so its bound in-chat prompt can be shown.",
            _PermissionInput,
        ),
        tool(lambda **kw: no_target("applications.list", "applications", kw), "computer_list_apps", decision + "Lists discovered application metadata; read-only.", _NoTarget),
        tool(lambda **kw: application("applications.open", kw), "computer_open_app", decision + "Opens one validated application identifier; modifies desktop state.", _Application),
        tool(lambda **kw: application("applications.close", kw), "computer_close_app", decision + "Closes one application and may affect unsaved work; permission/preview required.", _Application),
        tool(lambda **kw: no_target("applications.active", "applications", kw), "computer_active_app", decision + "Reads the active application identifier.", _NoTarget),
        tool(calendar_list, "calendar_list_events", decision + "Reads private calendar metadata and requires calendar-read permission.", _CalendarList),
        tool(lambda **kw: calendar_write("calendar.create", kw), "calendar_create_event", decision + "Creates an event; requires calendar-write permission and preview.", _CalendarWrite),
        tool(lambda **kw: calendar_write("calendar.update", kw), "calendar_update_event", decision + "Updates an event; requires calendar-write permission and preview.", _CalendarWrite),
        tool(lambda **kw: resource("calendar.delete", "calendar", kw), "calendar_delete_event", decision + "Deletes one event; requires exact-action confirmation.", _Resource),
        tool(lambda **kw: no_target("media.status", "media", kw), "media_get_status", decision + "Reads current playback metadata.", _NoTarget),
        tool(media_play, "media_play", decision + "Starts model-selected media; modifies playback state.", _MediaPlay),
        *[tool(lambda _op=op, **kw: no_target(_op, "media", kw), f"media_{op.split('.')[1]}", decision + f"Performs the {_label} playback control.", _NoTarget) for op, _label in (("media.pause", "pause"), ("media.next", "next"), ("media.previous", "previous"))],
        tool(media_volume, "media_set_volume", decision + "Changes media volume between 0 and 1.", _MediaVolume),
        tool(notes_search, "notes_search", decision + "Searches private note titles/content; explicit notes-read permission required.", _NotesSearch),
        tool(lambda **kw: resource("notes.read", "notes", kw), "notes_read", decision + "Reads one private note; explicit notes-read permission required.", _Resource),
        tool(lambda **kw: note_write("notes.create", kw), "notes_create", decision + "Creates a note; notes-write permission and preview required.", _NoteWrite),
        tool(lambda **kw: note_write("notes.update", kw), "notes_update", decision + "Updates a note; notes-write permission and preview required.", _NoteWrite),
        tool(lambda **kw: resource("notes.delete", "notes", kw), "notes_delete", decision + "Deletes one note; exact-action confirmation required.", _Resource),
        tool(lambda **kw: no_target("browser.active_page", "browser", kw), "browser_get_active_page", decision + "Reads active tab URL/title, never cookies, passwords, tokens, or private storage.", _NoTarget),
        tool(lambda **kw: no_target("browser.read_page", "browser", kw), "browser_read_page", decision + "Reads accessible page text; explicit page-read permission required.", _NoTarget),
        tool(lambda **kw: no_target("browser.list_tabs", "browser", kw), "browser_list_tabs", decision + "Lists tab titles/URLs; explicit tabs-read permission required.", _NoTarget),
        tool(browser_open, "browser_open_url", decision + "Opens one validated absolute HTTP(S) URL.", _OpenUrl),
        tool(lambda **kw: resource("browser.activate_tab", "browser", kw), "browser_activate_tab", decision + "Activates one validated tab identifier.", _Resource),
        tool(lambda **kw: resource("browser.close_tab", "browser", kw), "browser_close_tab", decision + "Closes one validated tab and changes browser state.", _Resource),
        tool(lambda **kw: no_target("clipboard.read", "clipboard", kw), "clipboard_read", decision + "Reads sensitive clipboard text; explicit clipboard-read permission required.", _NoTarget),
        tool(clipboard_write, "clipboard_write", decision + "Replaces clipboard text; explicit clipboard-write permission required.", _ClipboardWrite),
        tool(lambda **kw: path_target("filesystem.open", kw), "computer_open_path", decision + "Opens one file/folder inside configured allowed roots.", _PathTarget),
        tool(lambda **kw: path_target("filesystem.reveal", kw), "computer_reveal_path", decision + "Reveals one allowed path in the native file manager.", _PathTarget),
        tool(lambda **kw: path_target("filesystem.metadata", kw), "computer_file_metadata", decision + "Reads metadata for one path inside configured allowed roots.", _PathTarget),
        tool(lambda **kw: path_destination("filesystem.copy", kw), "computer_copy_path", decision + "Copies one allowed path to another allowed path after preview.", _PathDestination),
        tool(lambda **kw: path_destination("filesystem.move", kw), "computer_move_path", decision + "Moves one allowed path to another allowed path after preview.", _PathDestination),
        tool(lambda **kw: path_destination("filesystem.rename", kw), "computer_rename_path", decision + "Renames one allowed path after preview.", _PathDestination),
        tool(lambda **kw: path_target("filesystem.mkdir", kw), "computer_create_directory", decision + "Creates one directory inside configured allowed roots.", _PathTarget),
        tool(lambda **kw: path_target("filesystem.trash", kw), "computer_trash_path", decision + "Moves one allowed file/folder to OS Trash/Recycle Bin; exact-action confirmation required.", _PathTarget),
        tool(screenshot, "computer_take_screenshot", decision + "Captures visible screen content; first-use OS and Mana permissions required.", _Screenshot),
        tool(screen_recording, "computer_record_screen", decision + "Captures one bounded display recording after typed material parameters and exact approval; never records indefinitely.", _ScreenRecording),
        tool(notification, "computer_send_notification", decision + "Displays a local notification; notification permission required.", _Notification),
        tool(lambda **kw: no_target("system.status", "system", kw), "computer_get_system_status", decision + "Reads non-content system status such as volume and battery.", _NoTarget),
        tool(system_volume, "computer_set_system_volume", decision + "Changes system volume/mute state.", _SystemVolume),
        tool(system_control, "computer_control_system", decision + "Opens settings or performs lock/power actions; high/critical actions require exact-action local confirmation.", _SystemControl),
    ]
    return tools
