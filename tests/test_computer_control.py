from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mana_agent.api.app import create_app
from mana_agent.chat.events import CodingActivityEvent
from mana_agent.chat.history import ChatHistory
from mana_agent.integrations.computer_control.audit import AuditLogger
from mana_agent.integrations.computer_control.config import ComputerControlSettings
from mana_agent.integrations.computer_control.context import computer_client_scope
from mana_agent.integrations.computer_control.discovery import detect_platform
from mana_agent.integrations.computer_control.errors import (
    ActionCancelled,
    ActionTimedOut,
    ApplicationNotInstalled,
    ApplicationNotResponding,
    ConfirmationRequired,
    CapabilityUnavailable,
    InvalidActionDecision,
    InvalidConfirmation,
    OperatingSystemPermissionDenied,
    PermissionApprovalRequired,
    RemoteControlDenied,
)
from mana_agent.integrations.computer_control.models import (
    ApplicationDescriptor,
    ComputerAction,
    ComputerTarget,
    ExecutionRisk,
    PermissionDecision,
    SupportedPlatform,
)
from mana_agent.integrations.computer_control.permissions import PermissionService
from mana_agent.integrations.computer_control.policy import ACTION_SPECS, ConfirmationService, ExecutionPolicy
from mana_agent.integrations.computer_control.providers.fake import FakeComputerControlProvider
from mana_agent.integrations.computer_control.providers.linux.provider import LinuxProvider
from mana_agent.integrations.computer_control.providers.macos.provider import MacOSProvider
from mana_agent.integrations.computer_control.providers.windows.provider import WindowsProvider
from mana_agent.integrations.computer_control.registry import ApplicationAdapterRegistry
from mana_agent.integrations.computer_control.runtime_tools import build_computer_langchain_tools
from mana_agent.integrations.computer_control.service import ComputerControlService
from mana_agent.config.user_config import load_user_config, save_user_config
from mana_agent.chat_commands.builtins import definitions as command_definitions
from mana_agent.chat_commands.models import CommandContext
from mana_agent.gateway.entry_routing import (
    EntryRouteContext,
    EntryRouteRegistry,
    EntryRouter,
    RouteAvailability,
    RouteRegistration,
)
from mana_agent.gateway.chat_gateway import _computer_permission_requests_from_trace
from mana_agent.services.execution_event_hub import (
    get_execution_event_hub,
    reset_execution_event_hub_for_tests,
)
from mana_agent.tui.app import ManaChatApp
from mana_agent.tui.computer_permission import ComputerPermissionScreen


def run(coro):
    return asyncio.run(coro)


def settings(tmp_path: Path, **updates) -> ComputerControlSettings:
    values = {
        "enabled": True,
        "allowed_clients": {"local_cli", "tui", "dashboard"},
        "permissions": {scope: "always" for scope in {
            spec.permission_scope for spec in ACTION_SPECS.values()
        }},
        "allowed_paths": [tmp_path],
    }
    values.update(updates)
    return ComputerControlSettings.model_validate(values)


def action(operation: str, *, target: ComputerTarget | None = None, arguments=None, timeout=2) -> ComputerAction:
    spec = ACTION_SPECS[operation]
    capability = operation.split(".", 1)[0]
    if capability == "screenshots":
        capability = "screenshots"
    return ComputerAction(
        capability=capability,
        operation=operation,
        permission_scope=spec.permission_scope,
        risk=spec.risk,
        target=target or ComputerTarget(),
        arguments=arguments or {},
        timeout_seconds=timeout,
        source_decision_id=f"decision:{operation}",
    )


def service(tmp_path: Path, *, provider=None, config=None, confirmations=None, events=None):
    config = config or settings(tmp_path)
    permissions = PermissionService(config, store_path=tmp_path / "permissions.json")
    return ComputerControlService(
        settings=config,
        provider=provider or FakeComputerControlProvider(),
        permissions=permissions,
        policy=ExecutionPolicy(config, confirmations),
        audit=AuditLogger(path=tmp_path / "audit.jsonl"),
        event_sink=events.append if events is not None else (lambda _event: None),
    )


def test_platform_auto_detection_is_explicit() -> None:
    assert detect_platform("darwin") is SupportedPlatform.MACOS
    assert detect_platform("win32") is SupportedPlatform.WINDOWS
    assert detect_platform("linux") is SupportedPlatform.LINUX


def test_capability_discovery_and_unavailable_os_permission(tmp_path: Path) -> None:
    fake = FakeComputerControlProvider(capabilities={"applications"}, os_permissions={"applications": False})
    control = service(tmp_path, provider=fake)
    report = run(control.capabilities())
    assert report.supports("applications")
    with pytest.raises(OperatingSystemPermissionDenied):
        run(control.execute(action("applications.list"), session_id="s", client_type="local_cli"))
    assert fake.executed == []


def test_unimplemented_operation_stops_before_permission_or_confirmation(tmp_path: Path) -> None:
    fake = FakeComputerControlProvider(
        capabilities={"system"},
        operations={"system.volume"},
    )
    control = service(tmp_path, provider=fake)
    with pytest.raises(CapabilityUnavailable):
        run(control.execute(action("system.shutdown"), session_id="s", client_type="local_cli"))
    assert fake.executed == []
    assert control.pending_confirmations() == []


def test_ask_permission_waits_for_local_choice_and_allow_once_is_consumed(tmp_path: Path) -> None:
    config = settings(tmp_path, permissions={spec.permission_scope: "ask" for spec in ACTION_SPECS.values()})
    fake = FakeComputerControlProvider()
    events = []
    control = service(tmp_path, provider=fake, config=config, events=events)
    request = action("media.pause")
    with pytest.raises(PermissionApprovalRequired) as raised:
        run(control.execute(request, session_id="s", client_type="local_cli"))
    assert fake.executed == []
    pending = control.pending_permissions()
    assert pending[0]["permission_request_id"] == raised.value.permission_request_id
    assert raised.value.payload()["execution_id"] == request.execution_id
    waiting = next(event for event in events if event.event_type == "waiting_permission")
    assert waiting.metadata["permission_scope"] == "computer.media.control"
    assert waiting.metadata["permission_request_id"] == raised.value.permission_request_id
    result = run(control.approve_permission_and_execute(
        raised.value.permission_request_id,
        decision=PermissionDecision.ALLOW_ONCE,
        client_type="tui",
    ))
    assert result.state.value == "completed"
    assert fake.executed == [request]
    assert control.pending_permissions() == []
    with pytest.raises(PermissionApprovalRequired):
        run(control.execute(request.model_copy(update={"execution_id": "second"}), session_id="s", client_type="local_cli"))


def test_remote_client_cannot_approve_pending_permission(tmp_path: Path) -> None:
    config = settings(
        tmp_path,
        permissions={spec.permission_scope: "ask" for spec in ACTION_SPECS.values()},
        allowed_clients={"local_cli", "tui", "dashboard", "telegram"},
        allow_remote_control=True,
    )
    fake = FakeComputerControlProvider()
    control = service(tmp_path, provider=fake, config=config)
    request = action("media.pause")
    with pytest.raises(PermissionApprovalRequired) as raised:
        run(control.execute(request, session_id="remote", client_type="telegram"))
    with pytest.raises(OperatingSystemPermissionDenied, match="trusted local"):
        run(control.approve_permission_and_execute(
            raised.value.permission_request_id,
            decision=PermissionDecision.ALLOW_SESSION,
            client_type="telegram",
        ))
    assert len(control.pending_permissions()) == 1
    assert fake.executed == []


def test_permission_approval_executes_stored_screenshot_action(tmp_path: Path) -> None:
    config = settings(
        tmp_path,
        permissions={spec.permission_scope: "ask" for spec in ACTION_SPECS.values()},
    )
    fake = FakeComputerControlProvider()
    control = service(tmp_path, provider=fake, config=config)
    request = action("screenshots.capture", arguments={"mode": "full_screen"})
    with pytest.raises(PermissionApprovalRequired) as raised:
        run(control.execute(request, session_id="screen", client_type="tui"))
    result = run(control.approve_permission_and_execute(
        raised.value.permission_request_id,
        decision=PermissionDecision.ALLOW_SESSION,
        client_type="dashboard",
    ))
    assert result.operation == "screenshots.capture"
    assert fake.executed == [request]


def test_session_permission_survives_multiple_actions(tmp_path: Path) -> None:
    config = settings(tmp_path, permissions={spec.permission_scope: "ask" for spec in ACTION_SPECS.values()})
    control = service(tmp_path, config=config)
    control.permissions.grant("computer.media.control", PermissionDecision.ALLOW_SESSION)
    for execution_id in ("one", "two"):
        request = action("media.pause").model_copy(update={"execution_id": execution_id})
        assert run(control.execute(request, session_id="s", client_type="local_cli")).state.value == "completed"


def test_high_risk_requires_exact_expiring_confirmation(tmp_path: Path) -> None:
    confirmations = ConfirmationService(ttl_seconds=120)
    fake = FakeComputerControlProvider()
    control = service(tmp_path, provider=fake, confirmations=confirmations)
    request = action("notes.delete", target=ComputerTarget(resource_id="note-1"))
    with pytest.raises(ConfirmationRequired) as raised:
        run(control.execute(request, session_id="s", client_type="local_cli"))
    assert fake.executed == []
    request_id = raised.value.confirmation_request_id
    premature = request.model_copy(update={"confirmation_token": request_id})
    with pytest.raises(InvalidConfirmation, match="waiting for approval"):
        run(control.execute(premature, session_id="s", client_type="local_cli"))
    control.approve_confirmation(request_id, client_type="local_cli", explicitly_confirmed=True)
    confirmed = request.model_copy(update={"confirmation_token": request_id})
    assert run(control.execute(confirmed, session_id="s", client_type="local_cli")).state.value == "completed"
    with pytest.raises(InvalidConfirmation):
        run(control.execute(confirmed, session_id="s", client_type="local_cli"))

    expired = ConfirmationService(ttl_seconds=1)
    token = expired.issue(request, preview="Delete note")
    digest, _, approved, preview = expired._tokens[token]
    expired._tokens[token] = (digest, datetime.now(timezone.utc) - timedelta(seconds=1), approved, preview)
    with pytest.raises(InvalidConfirmation):
        expired.consume(request.model_copy(update={"confirmation_token": token}))


def test_confirmation_does_not_authorize_changed_target(tmp_path: Path) -> None:
    control = service(tmp_path)
    request = action("calendar.delete", target=ComputerTarget(resource_id="event-1"))
    with pytest.raises(ConfirmationRequired) as raised:
        run(control.execute(request, session_id="s", client_type="local_cli"))
    control.approve_confirmation(
        raised.value.confirmation_request_id,
        client_type="local_cli",
        explicitly_confirmed=True,
    )
    changed = request.model_copy(update={
        "target": ComputerTarget(resource_id="event-2"),
        "confirmation_token": raised.value.confirmation_request_id,
    })
    with pytest.raises(InvalidConfirmation):
        run(control.execute(changed, session_id="s", client_type="local_cli"))


def test_trusted_approval_executes_stored_exact_action_without_model_retry(tmp_path: Path) -> None:
    fake = FakeComputerControlProvider()
    control = service(tmp_path, provider=fake)
    request = action("notes.delete", target=ComputerTarget(resource_id="note-exact"))
    with pytest.raises(ConfirmationRequired) as raised:
        run(control.execute(request, session_id="session", client_type="local_cli"))
    result = run(control.approve_and_execute(
        raised.value.confirmation_request_id,
        client_type="tui",
        explicitly_confirmed=True,
    ))
    assert result.state.value == "completed"
    assert [item.target.resource_id for item in fake.executed] == ["note-exact"]
    assert control.pending_confirmations() == []


def test_untrusted_approval_cannot_consume_pending_request(tmp_path: Path) -> None:
    control = service(tmp_path)
    request = action("notes.delete", target=ComputerTarget(resource_id="note-pending"))
    with pytest.raises(ConfirmationRequired) as raised:
        run(control.execute(request, session_id="session", client_type="local_cli"))
    with pytest.raises(OperatingSystemPermissionDenied):
        run(control.approve_and_execute(
            raised.value.confirmation_request_id,
            client_type="telegram",
            explicitly_confirmed=True,
        ))
    assert control.pending_confirmations()[0]["confirmation_request_id"] == raised.value.confirmation_request_id


def test_remote_control_and_sensitive_scope_are_separately_restricted(tmp_path: Path) -> None:
    control = service(tmp_path)
    with pytest.raises(RemoteControlDenied):
        run(control.execute(action("media.pause"), session_id="s", client_type="telegram"))
    config = settings(tmp_path, allow_remote_control=True)
    still_not_allowlisted = service(tmp_path, config=config)
    with pytest.raises(RemoteControlDenied):
        run(still_not_allowlisted.execute(action("media.pause"), session_id="s", client_type="telegram"))
    config = settings(
        tmp_path,
        allow_remote_control=True,
        allowed_clients={"local_cli", "tui", "dashboard", "telegram"},
    )
    remote = service(tmp_path, config=config)
    with pytest.raises(RemoteControlDenied):
        run(remote.execute(action("notes.read", target=ComputerTarget(resource_id="n")), session_id="s", client_type="telegram"))


def test_allowlisted_remote_high_risk_action_waits_for_local_approval(tmp_path: Path) -> None:
    config = settings(
        tmp_path,
        allow_remote_control=True,
        allowed_clients={"local_cli", "tui", "dashboard", "telegram"},
    )
    fake = FakeComputerControlProvider()
    control = service(tmp_path, provider=fake, config=config)
    request = action("calendar.delete", target=ComputerTarget(resource_id="remote-event"))
    with pytest.raises(ConfirmationRequired) as raised:
        run(control.execute(request, session_id="remote", client_type="telegram"))
    assert fake.executed == []
    result = run(control.approve_and_execute(
        raised.value.confirmation_request_id,
        client_type="dashboard",
        explicitly_confirmed=True,
    ))
    assert result.state.value == "completed"


def test_invalid_risk_and_unknown_arguments_stop_before_provider(tmp_path: Path) -> None:
    fake = FakeComputerControlProvider()
    control = service(tmp_path, provider=fake)
    request = action("clipboard.write", arguments={"text": "safe", "command": "rm -rf"})
    with pytest.raises(InvalidActionDecision):
        run(control.execute(request, session_id="s", client_type="local_cli"))
    wrong_risk = action("media.pause").model_copy(update={"risk": ExecutionRisk.HIGH})
    with pytest.raises(InvalidActionDecision):
        run(control.execute(wrong_risk, session_id="s", client_type="local_cli"))
    assert fake.executed == []


def test_timeout_cancels_provider_and_records_event(tmp_path: Path) -> None:
    fake = FakeComputerControlProvider(delay_seconds=0.2)
    events = []
    control = service(tmp_path, provider=fake, events=events)
    request = action("media.pause", timeout=0.1)
    with pytest.raises(ActionTimedOut):
        run(control.execute(request, session_id="s", client_type="local_cli"))
    assert request.execution_id in fake.cancelled
    assert any(event.event_type == "action_failed" for event in events)


def test_cooperative_cancellation_stops_action(tmp_path: Path) -> None:
    async def scenario() -> None:
        fake = FakeComputerControlProvider(delay_seconds=1)
        control = service(tmp_path, provider=fake)
        request = action("media.pause")
        pending = asyncio.create_task(control.execute(request, session_id="s", client_type="local_cli"))
        await asyncio.sleep(0.01)
        assert await control.cancel(request.execution_id)
        with pytest.raises(ActionCancelled):
            await pending
        assert request.execution_id in fake.cancelled

    run(scenario())


def test_audit_and_events_never_duplicate_sensitive_content(tmp_path: Path) -> None:
    events = []
    control = service(tmp_path, events=events)
    result = run(control.execute(action("clipboard.read"), session_id="session", client_type="local_cli"))
    assert result.data["content"] == "private fake clipboard"
    audit_text = (tmp_path / "audit.jsonl").read_text()
    assert "private fake clipboard" not in audit_text
    assert "content" not in json.dumps([event.model_dump(mode="json") for event in events])
    row = json.loads(audit_text)
    assert row["sensitive_content_accessed"] is True


def test_runtime_tools_require_authenticated_gateway_context(tmp_path: Path) -> None:
    control = service(tmp_path)
    tools = {tool.name: tool for tool in build_computer_langchain_tools(control)}
    payload = {"source_decision_id": "decision-1"}
    denied = json.loads(tools["media_pause"].invoke(payload))
    assert denied["error_code"] == "remote_control_denied"
    with computer_client_scope("session", "local_cli", allowed_decision_ids=frozenset({"decision-1"})):
        allowed = json.loads(tools["media_pause"].invoke(payload))
    assert allowed["ok"] is True
    with computer_client_scope("session", "local_cli", allowed_decision_ids=frozenset({"another-decision"})):
        wrong_decision = json.loads(tools["media_pause"].invoke(payload))
    assert wrong_decision["error_code"] == "remote_control_denied"


def test_permission_status_ask_says_no_prompt_exists_until_exact_action(tmp_path: Path) -> None:
    config = settings(
        tmp_path,
        permissions={spec.permission_scope: "ask" for spec in ACTION_SPECS.values()},
    )
    control = service(tmp_path, config=config)
    tools = {tool.name: tool for tool in build_computer_langchain_tools(control)}
    with computer_client_scope("session", "tui"):
        response = json.loads(tools["computer_permission_status"].invoke({
            "scope": "computer.apps.control",
        }))
    assert response["ok"] is True
    assert response["result"]["decision"] == "ask"
    assert response["result"]["request_created"] is False
    assert "invoke that exact computer action tool" in response["result"]["next_step"]
    assert control.pending_permissions() == []


def test_worker_permission_result_is_recovered_for_frontend_modal() -> None:
    response = SimpleNamespace(trace=[{
        "tool_name": "media_play",
        "status": "ok",
        "output_preview": json.dumps({
            "ok": False,
            "error_code": "permission_required",
            "permission_request_id": "permission-worker",
            "permission_scope": "computer.media.control",
            "execution_id": "computer-execution",
            "preview": "Play music.",
        }),
    }])
    assert _computer_permission_requests_from_trace(response) == [{
        "permission_request_id": "permission-worker",
        "permission_scope": "computer.media.control",
        "execution_id": "computer-execution",
        "preview": "Play music.",
    }]


@pytest.mark.parametrize(
    ("operation", "target", "arguments"),
    [
        ("calendar.create", ComputerTarget(), {"event": {"title": "Review"}}),
        ("calendar.update", ComputerTarget(resource_id="event-1"), {"event": {"title": "Updated"}}),
        ("media.pause", ComputerTarget(), {}),
        ("notes.search", ComputerTarget(), {"query": "roadmap", "limit": 10}),
        ("notes.read", ComputerTarget(resource_id="note-1"), {}),
        ("browser.active_page", ComputerTarget(), {}),
        ("browser.read_page", ComputerTarget(), {}),
    ],
)
def test_fake_personal_data_and_media_flows(
    tmp_path: Path,
    operation: str,
    target: ComputerTarget,
    arguments: dict[str, object],
) -> None:
    result = run(service(tmp_path).execute(
        action(operation, target=target, arguments=arguments),
        session_id="s",
        client_type="local_cli",
    ))
    assert result.state.value == "completed"


def test_screenshot_requires_its_own_permission_scope(tmp_path: Path) -> None:
    permission_values = {spec.permission_scope: "always" for spec in ACTION_SPECS.values()}
    permission_values["computer.screenshot.capture"] = "ask"
    config = settings(tmp_path, permissions=permission_values)
    fake = FakeComputerControlProvider()
    control = service(tmp_path, provider=fake, config=config)
    with pytest.raises(PermissionApprovalRequired):
        run(control.execute(
            action("screenshots.capture", arguments={"mode": "full_screen"}),
            session_id="s",
            client_type="local_cli",
        ))
    assert fake.executed == []


class _FakeAdapter:
    def __init__(self, application_id: str, available: bool) -> None:
        self.application_id = application_id
        self.supported_platforms = {SupportedPlatform.LINUX}
        self.capabilities = {"media"}
        self.available = available

    async def is_available(self) -> bool:
        return self.available

    async def get_status(self):
        raise NotImplementedError

    async def execute(self, action):
        raise NotImplementedError


def test_adapter_selection_honors_preference_then_availability() -> None:
    registry = ApplicationAdapterRegistry()
    registry.register(_FakeAdapter("preferred", False))
    registry.register(_FakeAdapter("active", True))
    selected = run(registry.select(
        action("media.pause"),
        platform=SupportedPlatform.LINUX,
        preference="preferred",
        active_application="active",
    ))
    assert selected is not None
    assert selected.application_id == "active"


def test_filesystem_move_is_restricted_to_allowed_roots(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("content")
    provider = LinuxProvider(settings(tmp_path))
    result = run(provider._filesystem(action(
        "filesystem.move",
        target=ComputerTarget(path=str(source)),
        arguments={"destination": str(destination)},
    )))
    assert result.data["path"] == str(destination)
    assert destination.read_text() == "content"
    with pytest.raises(InvalidActionDecision):
        run(provider._filesystem(action(
            "filesystem.metadata",
            target=ComputerTarget(path=str(tmp_path.parent / "outside.txt")),
        )))


def test_linux_trash_constructs_gio_argv_not_delete_command(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "trash-me.txt"
    target.write_text("data")
    monkeypatch.setenv("DISPLAY", ":99")
    provider = LinuxProvider(settings(tmp_path))
    captured = []

    async def fake_run(_action, argv, **_kwargs):
        captured.append(list(argv))
        return "", ""

    monkeypatch.setattr(provider, "_run", fake_run)
    monkeypatch.setattr("mana_agent.integrations.computer_control.providers.linux.provider.command_available", lambda command: command == "gio")
    run(provider.execute_platform_action(action("filesystem.trash", target=ComputerTarget(path=str(target)))))
    assert captured == [["gio", "trash", str(target)]]
    assert target.exists()  # Mocked Trash proves the provider itself never unlinks.


def test_macos_application_identifier_blocks_argument_injection(tmp_path: Path, monkeypatch) -> None:
    provider = MacOSProvider(settings(tmp_path))
    captured = []

    async def fake_run(_action, argv, **_kwargs):
        captured.append(list(argv))
        return "", ""

    monkeypatch.setattr(provider, "_run", fake_run)
    provider._report = run(provider.discover_capabilities())
    provider._report.applications.append(
        ApplicationDescriptor(
            application_id="com.microsoft.VSCode",
            name="Visual Studio Code",
            capabilities={"applications"},
        )
    )
    safe = action("applications.open", target=ComputerTarget(application_id="com.microsoft.VSCode"))
    run(provider.execute_platform_action(safe))
    assert captured == [["open", "-b", "com.microsoft.VSCode"]]
    unsafe = action("applications.open", target=ComputerTarget(application_id="Bad; touch /tmp/pwned"))
    with pytest.raises(InvalidActionDecision):
        run(provider.execute_platform_action(unsafe))
    missing = action("applications.open", target=ComputerTarget(application_id="com.example.Missing"))
    with pytest.raises(ApplicationNotInstalled):
        run(provider.execute_platform_action(missing))


def test_macos_random_music_play_is_verified_and_query_is_argv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provider = MacOSProvider(settings(tmp_path))
    captured = []

    async def playing(_action, argv, **_kwargs):
        captured.append(list(argv))
        return "playing\n", ""

    monkeypatch.setattr(provider, "_run", playing)
    request = action("media.play", arguments={"query": "safe; not script"})
    result = run(provider.execute_platform_action(request))
    assert result.message == "Music playback started and was verified."
    assert result.data["playback_state"] == "playing"
    assert captured[0][-1] == "safe; not script"
    assert "safe; not script" not in captured[0][2]

    async def stopped(_action, _argv, **_kwargs):
        return "stopped\n", ""

    monkeypatch.setattr(provider, "_run", stopped)
    with pytest.raises(ApplicationNotResponding, match="accepted the play command"):
        run(provider.execute_platform_action(action("media.play")))


def test_windows_recycle_bin_uses_path_as_separate_argument(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "file with spaces.txt"
    target.write_text("data")
    provider = WindowsProvider(settings(tmp_path))
    captured = []

    async def fake_run(_action, argv, **_kwargs):
        captured.append(list(argv))
        return "", ""

    monkeypatch.setattr(provider, "_run", fake_run)
    monkeypatch.setattr("mana_agent.integrations.computer_control.providers.windows.provider.command_available", lambda _command: True)
    run(provider.execute_platform_action(action("filesystem.trash", target=ComputerTarget(path=str(target)))))
    assert captured[0][-1] == str(target)
    assert "SendToRecycleBin" in captured[0][-2]
    assert target.exists()


def test_linux_headless_report_does_not_claim_desktop_capabilities(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    report = run(LinuxProvider(settings(tmp_path)).discover_capabilities())
    assert report.headless is True
    assert not report.supports("applications")
    assert report.supports("filesystem")


def test_computer_permission_scopes_round_trip_as_literal_toml_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana-home"))
    save_user_config({
        "computer_control": {
            "enabled": True,
        "permissions": {"computer.notes.read": "always"},
        }
    }, merge=False)
    loaded = load_user_config()
    assert loaded["computer_control"]["permissions"]["computer.notes.read"] == "always"


def test_legacy_flat_enable_switch_is_honored_without_explicit_table(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana-home"))
    save_user_config({"MANA_COMPUTER_CONTROL_ENABLED": True}, merge=False)
    assert ComputerControlSettings.load().enabled is True


def test_invalid_permission_and_relative_allowed_path_fail_configuration() -> None:
    with pytest.raises(ValueError, match="invalid computer permission"):
        ComputerControlSettings(permissions={"computer.apps.read": "grant_everything"})
    with pytest.raises(ValueError, match="absolute paths"):
        ComputerControlSettings(allowed_paths=[Path("relative")])


def test_entry_router_accepts_explicit_computer_model_decision() -> None:
    registry = EntryRouteRegistry()
    registry.register(RouteRegistration(
        "computer",
        "Desktop control",
        lambda: RouteAvailability(available=True),
        ("computer_capabilities",),
    ))

    class Llm:
        def invoke(self, _messages):
            return SimpleNamespace(content=json.dumps({
                "route": "computer",
                "confidence": 0.99,
                "reason": "The request targets the installed desktop.",
                "required_sources": ["computer"],
                "target_urls": [],
                "requires_live_data": True,
            }))

    decision = EntryRouter(llm=Llm(), registry=registry).route(
        user_prompt="Pause the active desktop media player.",
        context=EntryRouteContext(session_id="s", conversation_id="c", turn_id="t"),
    )
    assert decision.route == "computer"
    assert decision.required_sources == ("computer",)


def test_local_confirmation_command_executes_pending_action(tmp_path: Path, monkeypatch) -> None:
    control = service(tmp_path)
    monkeypatch.setattr(
        "mana_agent.integrations.computer_control.service._default_service",
        control,
    )
    request = action("calendar.delete", target=ComputerTarget(resource_id="event-command"))
    with pytest.raises(ConfirmationRequired) as raised:
        run(control.execute(request, session_id="session", client_type="tui"))
    command = next(item for item in command_definitions() if item.canonical_name == "computer-confirm")
    result = command.handler(
        CommandContext(frontend="tui", session_id="session", capabilities={"gateway"}),
        [raised.value.confirmation_request_id],
    )
    assert result.status == "success"
    assert result.data["result"]["state"] == "completed"


def test_local_permission_command_executes_pending_action(tmp_path: Path, monkeypatch) -> None:
    config = settings(
        tmp_path,
        permissions={spec.permission_scope: "ask" for spec in ACTION_SPECS.values()},
    )
    control = service(tmp_path, config=config)
    monkeypatch.setattr(
        "mana_agent.integrations.computer_control.service._default_service",
        control,
    )
    with pytest.raises(PermissionApprovalRequired) as raised:
        run(control.execute(action("screenshots.capture"), session_id="session", client_type="tui"))
    command = next(
        item for item in command_definitions()
        if item.canonical_name == "computer-permission"
    )
    result = command.handler(
        CommandContext(frontend="tui", session_id="session", capabilities={"gateway"}),
        [raised.value.permission_request_id, "once"],
    )
    assert result.status == "success"
    assert result.data["result"]["operation"] == "screenshots.capture"


def test_dashboard_chat_permission_decision_executes_and_emits_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana-home"))
    monkeypatch.delenv("MANA_API_TOKEN", raising=False)
    reset_execution_event_hub_for_tests()
    root = tmp_path / "repo"
    root.mkdir()
    config = settings(
        tmp_path,
        permissions={spec.permission_scope: "ask" for spec in ACTION_SPECS.values()},
    )
    control = service(tmp_path, config=config)
    monkeypatch.setattr(
        "mana_agent.integrations.computer_control.service._default_service",
        control,
    )
    client = TestClient(create_app())
    created = client.post(
        "/api/v1/conversations",
        json={"title": "Permission chat", "root": str(root)},
    )
    conversation_id = created.json()["conversation"]["conversation_id"]
    with pytest.raises(PermissionApprovalRequired) as raised:
        run(control.execute(
            action("screenshots.capture"),
            session_id=conversation_id,
            client_type="dashboard",
        ))
    response = client.post(
        f"/api/v1/conversations/{conversation_id}/computer-permissions/"
        f"{raised.value.permission_request_id}",
        json={"decision": "allow_once", "root": str(root)},
    )
    assert response.status_code == 200
    assert response.json()["executed"] is True
    assert response.json()["result"]["operation"] == "screenshots.capture"
    events = get_execution_event_hub().history(conversation_id=conversation_id)
    assert any(event["type"] == "computer.permission_decided" for event in events)


def test_tui_chat_opens_permission_decision_screen() -> None:
    history = ChatHistory()
    app = ManaChatApp(history=history)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            history.add(CodingActivityEvent(activity={
                "event_id": "computer-permission-test",
                "event_type": "computer.waiting_permission",
                "status": "waiting_permission",
                "title": "Capture the full screen.",
                "metadata": {
                    "permission_request_id": "permission-test",
                    "permission_scope": "computer.screenshot.capture",
                    "preview": "Capture the full screen.",
                },
            }))
            await pilot.pause()
            assert isinstance(app.screen, ComputerPermissionScreen)
            assert app.screen.scope == "computer.screenshot.capture"

    run(exercise())
