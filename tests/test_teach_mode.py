"""End-to-end regression coverage for the local Teach Mode foundation."""

from __future__ import annotations

import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mana_agent.automations.service import AutomationValidationError, ScheduleDefinition, schedule_command
from mana_agent.commands.cli import app
from mana_agent.teach.config import TeachSettings
from mana_agent.teach.correction import TargetedSelectorRepair
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
    VerificationRule,
)
from mana_agent.teach.normalizer import SemanticNormalizer, rank_selectors
from mana_agent.teach.packaging import ManaFlowPackager
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
