"""End-to-end regression coverage for the local Teach Mode foundation."""

from __future__ import annotations

import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mana_agent.automations.service import AutomationValidationError, ScheduleDefinition, schedule_command
from mana_agent.commands.cli import app
from mana_agent.teach.config import TeachSettings
from mana_agent.teach.correction import TargetedSelectorRepair
from mana_agent.teach.desktop_recorder import NativeDesktopRecorder
from mana_agent.teach.models import (
    EventApplication,
    EventSource,
    EventTarget,
    FlowInput,
    FlowStep,
    ManaFlow,
    RecordedEvent,
    SelectorCandidate,
    SessionState,
    TeachError,
    TeachSession,
    VerificationRule,
)
from mana_agent.teach.normalizer import SemanticNormalizer, rank_selectors
from mana_agent.teach.packaging import ManaFlowPackager
from mana_agent.teach.permissions import DESKTOP_GRANTS, GrantStatus, TeachGrantStore, grant_status
from mana_agent.teach.replay import SafeReplayExecutor
from mana_agent.teach.service import TeachService
from mana_agent.teach.storage import LocalTeachStorage


def _service(tmp_path: Path, **kwargs) -> TeachService:
    settings = TeachSettings(storage_path=tmp_path / "teach")
    return TeachService(settings=settings, storage=LocalTeachStorage(settings.storage_path), **kwargs)


def _event(session_id: str, *, action: str = "click", value: str | None = None) -> RecordedEvent:
    data = {} if value is None else {"value": value}
    return RecordedEvent(
        session_id=session_id,
        source=EventSource.BROWSER,
        action=action,
        application=EventApplication(id="browser", name="Browser"),
        target=EventTarget(
            role="button",
            name="Export",
            selectors=[
                SelectorCandidate(
                    type="css", value=".volatile > button:nth-child(2)", confidence=0.8
                ),
                SelectorCandidate(
                    type="playwright_role",
                    value={"role": "button", "name": "Export"},
                    confidence=0.9,
                ),
            ],
        ),
        data=data,
    )


def test_session_state_transitions_and_pause_resume(tmp_path: Path) -> None:
    service = _service(tmp_path)
    session = service.start("Safe report export")
    assert session.state == SessionState.RECORDING
    assert service.pause(session.id).state == SessionState.PAUSED
    assert service.resume(session.id).state == SessionState.RECORDING
    with pytest.raises(TeachError, match="Invalid Teach Mode transition"):
        service.resume(session.id)


def test_crash_recovery_and_recording_redaction(tmp_path: Path) -> None:
    service = _service(tmp_path)
    session = service.start("Fill a safe form")
    event = _event(session.id, action="fill", value="sk-abcdefghijklmnopqrstuvwxyz")
    service.record_event(event)
    recovered = _service(tmp_path).active_session()
    persisted = service.storage.load_events(session.id)[0]
    assert recovered.raw_event_count == 1
    assert persisted.sensitive is True
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in persisted.model_dump_json()


def test_normalizer_deduplicates_and_ranks_semantic_selectors() -> None:
    session_id = "teach_test"
    event = _event(session_id)
    duplicate = event.model_copy(update={"event_id": "evt_other"})
    normalized = SemanticNormalizer().normalize([event, duplicate])
    assert len(normalized) == 1
    assert normalized[0].target.selectors[0].type == "playwright_role"
    ranked = rank_selectors([SelectorCandidate(type="coordinate", value=[1, 2], confidence=1)])
    assert ranked[0].confidence < 0.6


def test_compiler_parameterizes_date_preserves_provenance_and_confirmation(tmp_path: Path) -> None:
    service = _service(tmp_path)
    session = service.start("Publish weekly report")
    service.explain("This date changes every week", session.id)
    service.record_event(_event(session.id, action="fill", value="2026-07-27"))
    service.record_event(_event(session.id, action="publish"))
    stopped, flow = service.stop(session.id)
    assert stopped.state == SessionState.AWAITING_REVIEW
    assert flow.inputs["date"].type == "date"
    assert flow.steps[0].with_["value"] == "{{ date }}"
    assert flow.steps[0].provenance
    assert flow.steps[1].requires_confirmation is True


def test_replay_dry_run_never_commits_or_claims_verified() -> None:
    calls: list[str] = []
    flow = ManaFlow(
        id="email",
        name="Email",
        steps=[FlowStep(id="send", action="email.send", requires_confirmation=True)],
        verify=[VerificationRule(id="sent", type="email.sent", arguments={})],
    )
    replay = SafeReplayExecutor(action_executor=lambda action, arguments: calls.append(action) or {"ok": True})
    result = replay.replay(flow, mode="dry_run", inputs={})
    assert calls == []
    assert result.verification_status == "unverified"
    assert result.steps[0].status == "planned"


def test_replay_requires_permission_confirmation_and_observable_verification(tmp_path: Path) -> None:
    output = tmp_path / "report.pdf"
    flow = ManaFlow(
        id="export",
        name="Export",
        steps=[
            FlowStep(
                id="write",
                action="filesystem.export",
                permissions=["computer.files.write"],
                requires_confirmation=True,
            )
        ],
        verify=[VerificationRule(id="exists", type="file.exists", arguments={"path": str(output)})],
    )
    denied = SafeReplayExecutor(
        action_executor=lambda *_: {"ok": True},
        permission_checker=lambda _: False,
        confirmation_checker=lambda _: True,
    ).replay(flow, mode="normal", inputs={})
    assert denied.verification_status == "failed"
    waiting = SafeReplayExecutor(
        action_executor=lambda *_: {"ok": True},
        permission_checker=lambda _: True,
        confirmation_checker=lambda _: False,
    ).replay(flow, mode="normal", inputs={})
    assert waiting.steps[-1].status == "waiting_confirmation"
    verified = SafeReplayExecutor(
        action_executor=lambda *_: (output.write_bytes(b"%PDF"), {"ok": True})[1],
        permission_checker=lambda _: True,
        confirmation_checker=lambda _: True,
    ).replay(flow, mode="normal", inputs={})
    assert verified.verification_status == "verified"


def test_secret_inputs_require_credential_reference() -> None:
    flow = ManaFlow(
        id="secret",
        name="Secret",
        inputs={"credential": FlowInput(secret=True)},
        steps=[],
        verify=[],
    )
    with pytest.raises(TeachError, match="secret://"):
        SafeReplayExecutor().replay(flow, mode="dry_run", inputs={"credential": "plaintext"})


def test_targeted_selector_repair_only_changes_selected_step() -> None:
    original = SelectorCandidate(type="css", value=".old", confidence=0.7)
    flow = ManaFlow(
        id="repair",
        name="Repair",
        steps=[
            FlowStep(id="one", action="browser.click", selectors=[original]),
            FlowStep(id="two", action="browser.click", selectors=[original.model_copy()]),
        ],
    )
    repaired = TargetedSelectorRepair().repair(
        flow, "one", SelectorCandidate(type="playwright_role", value={"role": "button"}, confidence=0.9)
    )
    assert repaired.version == 2
    assert repaired.steps[0].selectors[0].type == "playwright_role"
    assert repaired.steps[1].selectors[0].type == "css"


def test_package_determinism_import_and_secret_block(tmp_path: Path) -> None:
    fixed = datetime(2026, 7, 27, tzinfo=timezone.utc)
    flow = ManaFlow(id="safe", name="Safe", created_at=fixed, updated_at=fixed)
    packager = ManaFlowPackager()
    first = packager.export(flow, tmp_path / "first.mana-flow")
    second = packager.export(flow, tmp_path / "second.mana-flow")
    assert first.read_bytes() == second.read_bytes()
    imported = packager.import_package(first)
    assert imported.status == "imported_pending"
    unsafe = flow.model_copy(update={"description": "token sk-abcdefghijklmnopqrstuvwxyz"})
    with pytest.raises(TeachError, match="blocked"):
        packager.export(unsafe, tmp_path / "unsafe.mana-flow")


def test_package_rejects_path_traversal(tmp_path: Path) -> None:
    package = tmp_path / "bad.mana-flow"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("../flow.yaml", "{}")
    with pytest.raises(TeachError, match="structure|Unsafe"):
        ManaFlowPackager().import_package(package)


def test_scheduler_accepts_versioned_verified_flow_contract() -> None:
    schedule = ScheduleDefinition.create(
        name="Weekly flow",
        action="teach-flow",
        cron="0 16 * * 5",
        targets=["local"],
        action_config={
            "flow_id": "weekly-report",
            "flow_version": 2,
            "version_policy": "pinned",
            "inputs": {},
        },
    )
    assert "teach replay weekly-report" in schedule_command(schedule, Path("/tmp/repo"))
    with pytest.raises(AutomationValidationError):
        ScheduleDefinition.create(
            name="Bad",
            action="teach-flow",
            cron="0 16 * * 5",
            targets=["local"],
            action_config={"flow_id": "x", "version_policy": "pinned", "inputs": {}},
        )


def test_cli_doctor_and_start_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    runner = CliRunner()
    doctor = runner.invoke(app, ["teach", "doctor", "--json"])
    assert doctor.exit_code == 0
    assert "manual_semantic" in doctor.stdout
    start = runner.invoke(app, ["teach", "start", "Safe local task"])
    assert start.exit_code == 0
    assert "REC" in start.stdout
    status = runner.invoke(app, ["teach", "status", "--json"])
    assert status.exit_code == 0
    assert '"state": "recording"' in status.stdout


def test_teach_storage_and_grants_work_without_posix_fchmod(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Windows does not expose os.fchmod for temporary-file permissions."""
    monkeypatch.delattr(os, "fchmod", raising=False)

    service = _service(tmp_path)
    session = service.start("Portable Teach session")
    assert service.storage.load_session(session.id).id == session.id

    store = TeachGrantStore(tmp_path / "teach" / "grants.json")
    store.grant(list(DESKTOP_GRANTS))
    assert all(store.is_granted(scope) for scope in DESKTOP_GRANTS)


def test_desktop_grants_are_explicit_owner_only_and_reported(tmp_path: Path) -> None:
    store = TeachGrantStore(tmp_path / "teach" / "grants.json")
    assert all(not item.mana_granted for item in grant_status(store))
    store.grant(list(DESKTOP_GRANTS))
    assert all(item.mana_granted for item in grant_status(store))
    if hasattr(os, "fchmod"):
        assert store.path.stat().st_mode & 0o777 == 0o600
    store.revoke(["teach.record.keyboard"])
    assert store.is_granted("teach.record.keyboard") is False


def test_native_keyboard_recorder_aggregates_printable_keys_and_edits() -> None:
    session = TeachSession(task_name="Desktop", state=SessionState.RECORDING)
    captured: list[RecordedEvent] = []
    recorder = NativeDesktopRecorder()
    recorder._session = session
    recorder._emit = captured.append

    class Printable:
        def __init__(self, char: str):
            self.char = char

        def __str__(self) -> str:
            return self.char

    class Enter:
        char = None

        def __str__(self) -> str:
            return "Key.enter"

    class Shift:
        char = None

        def __str__(self) -> str:
            return "Key.shift"

    class Space:
        char = None

        def __str__(self) -> str:
            return "Key.space"

    recorder._on_press(Printable("s"))
    recorder._on_press(Printable("e"))
    recorder._on_press(Printable("c"))
    recorder._on_press(Space())
    recorder._on_press(Printable("r"))
    recorder._on_press(Printable("x"))
    recorder._on_press(type("Backspace", (), {"char": None, "__str__": lambda self: "Key.backspace"})())
    recorder._on_press(Enter())
    serialized = "".join(item.model_dump_json() for item in captured)
    assert '"character_count":5' in serialized
    assert '"content_captured":true' in serialized
    assert captured[0].data["value"] == "sec r"
    assert captured[0].sensitive is False


def test_native_keyboard_recorder_never_captures_secure_field_text() -> None:
    session = TeachSession(task_name="Desktop", state=SessionState.RECORDING)
    captured: list[RecordedEvent] = []
    recorder = NativeDesktopRecorder()
    recorder._session = session
    recorder._emit = captured.append
    recorder._typing_target = EventTarget(role="AXSecureTextField", name="Password")
    recorder._typing_application = EventApplication(name="Login")
    recorder._typed_count = 3
    recorder._typed_characters = list("abc")
    recorder._flush_typing()
    assert captured[0].data == {"character_count": 3, "content_captured": False}
    assert captured[0].sensitive is True


def test_cli_grant_records_local_consent_without_claiming_os_grant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    result = CliRunner().invoke(
        app,
        ["teach", "grant", "--scope", "keyboard", "--allow"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    keyboard = next(
        item for item in payload["grants"] if item["scope"] == "teach.record.keyboard"
    )
    assert keyboard["mana_granted"] is True
    assert "Local consent does not grant OS permission" in payload["notice"]


def test_desktop_start_fails_closed_without_explicit_grants(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(TeachError, match="explicit Mana grants"):
        service.start("Monitor desktop", desktop=True)
    assert service.storage.list_sessions() == []


def test_start_fails_instead_of_silently_dropping_to_semantic_capture_after_grants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    service = _service(tmp_path)
    service.grants.grant(list(DESKTOP_GRANTS))
    monkeypatch.setattr(
        "mana_agent.teach.service.require_desktop_grants",
        lambda _store: (_ for _ in ()).throw(TeachError("OS desktop permission is missing.")),
    )
    with pytest.raises(TeachError, match="OS desktop permission"):
        service.start("Monitor desktop")
    assert service.storage.list_sessions() == []


def test_desktop_start_persists_grants_and_monitor_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    service = _service(tmp_path)
    service.grants.grant(list(DESKTOP_GRANTS))
    monkeypatch.setattr("mana_agent.teach.service.require_desktop_grants", lambda _store: None)

    def attach(session: TeachSession) -> int:
        session.monitor_pid = 4321
        service.storage.save_session(session)
        return 4321

    monkeypatch.setattr(service.monitor, "start", attach)
    session = service.start("Monitor desktop", desktop=True)
    assert session.monitor_pid == 4321
    assert set(DESKTOP_GRANTS).issubset(session.permission_grants)


def test_start_auto_attaches_desktop_after_all_grants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    service = _service(tmp_path)
    service.grants.grant(list(DESKTOP_GRANTS))
    monkeypatch.setattr("mana_agent.teach.service.require_desktop_grants", lambda _store: None)
    monkeypatch.setattr(
        "mana_agent.teach.service.grant_status",
        lambda _store: [
            GrantStatus(scope=scope, mana_granted=True, available=True, os_granted=True)
            for scope in DESKTOP_GRANTS
        ],
    )

    def attach(session: TeachSession) -> int:
        session.monitor_pid = 9876
        service.storage.save_session(session)
        return 9876

    monkeypatch.setattr(service.monitor, "start", attach)
    session = service.start("Desktop by default")
    assert session.monitor_pid == 9876


def test_redacted_typing_compiles_to_mandatory_review(tmp_path: Path) -> None:
    service = _service(tmp_path)
    session = service.start("Type into form")
    service.record_event(
        RecordedEvent(
            session_id=session.id,
            source=EventSource.KEYBOARD,
            action="type",
            data={
                "value": "sk-live-abcdefghijklmnopqrstuvwxyz",
                "character_count": 8,
                "content_captured": False,
            },
            sensitive=True,
        )
    )
    _, flow = service.stop(session.id)
    assert flow.steps[0].requires_review is True
    assert flow.steps[0].with_["value"] == "{{ redacted_token }}"
