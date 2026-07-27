"""Application service coordinating recording, compilation and replay."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .compiler import FlowCompiler
from .config import TeachSettings
from .correction import TargetedSelectorRepair
from .events import publish_teach_event
from .models import (
    Explanation,
    AuditEntry,
    FlowCard,
    ManaFlow,
    RecordedEvent,
    ReplayResult,
    SelectorCandidate,
    SessionState,
    TeachError,
    TeachSession,
)
from .normalizer import SemanticNormalizer
from .packaging import ManaFlowPackager
from .permissions import TeachGrantStore, grant_status, require_desktop_grants
from .platform import doctor_report, platform_name
from .recorder import SemanticEventRecorder
from .redaction import Redactor
from .replay import SafeReplayExecutor
from .storage import LocalTeachStorage
from .monitor_process import DesktopMonitorProcess


class TeachService:
    def __init__(
        self,
        *,
        settings: TeachSettings | None = None,
        storage: LocalTeachStorage | None = None,
        replay_executor: SafeReplayExecutor | None = None,
    ):
        self.settings = settings or TeachSettings.load()
        self.storage = storage or LocalTeachStorage(self.settings.storage_path)
        self.redactor = Redactor()
        self.normalizer = SemanticNormalizer()
        self.compiler = FlowCompiler()
        self.packager = ManaFlowPackager(self.redactor)
        self.replay_executor = replay_executor or SafeReplayExecutor()
        self.recorder = SemanticEventRecorder()
        self.grants = TeachGrantStore(self.storage.root / "grants.json")
        self.monitor = DesktopMonitorProcess(self.storage)

    def start(
        self,
        task_name: str,
        *,
        permissions: list[str] | None = None,
        desktop: bool | None = None,
    ) -> TeachSession:
        if not self.settings.enabled:
            raise TeachError("Teach Mode is disabled in ~/.mana/config.toml.")
        if desktop is None:
            statuses = grant_status(self.grants)
            # Once the user has granted every local desktop scope, `teach
            # start` means desktop recording. Do not silently create an empty
            # semantic-only session when macOS/Windows/Linux approval is still
            # missing: require_desktop_grants below emits the corrective error.
            desktop = self.settings.desktop_capture or all(item.mana_granted for item in statuses)
        report = self.doctor()
        active = [name for name, item in report["recorders"].items() if item["available"]]
        if not active:
            raise TeachError("No Teach Mode recorder is available.")
        if desktop:
            require_desktop_grants(self.grants)
        granted_permissions = list(permissions or [])
        if desktop:
            granted_permissions.extend(
                item.scope for item in grant_status(self.grants) if item.mana_granted
            )
        session = TeachSession(
            task_name=task_name.strip(),
            permission_grants=list(dict.fromkeys(granted_permissions)),
            recorder_capabilities=active,
            platform_metadata={"platform": platform_name(), "doctor": report},
        )
        session.transition(SessionState.RECORDING, "Recording explicitly started by the user.")
        self.storage.save_session(session)
        self.recorder.start(session, self.record_event)
        if desktop:
            self.monitor.start(session)
            publish_teach_event(
                "recorder_attached",
                session_id=session.id,
                title="Native desktop recorder attached",
                metadata={"sources": ["accessibility", "application", "keyboard", "pointer"]},
            )
        publish_teach_event("session_started", session_id=session.id, title=f"Recording: {session.task_name}")
        return session

    def active_session(self) -> TeachSession:
        candidates = [
            session
            for session in self.storage.list_sessions()
            if session.state in {SessionState.RECORDING, SessionState.PAUSED}
        ]
        if not candidates:
            raise TeachError("No active Teach Mode session.")
        if len(candidates) > 1:
            raise TeachError("Multiple recoverable sessions exist; specify a session ID.")
        return candidates[0]

    def pause(self, session_id: str | None = None) -> TeachSession:
        session = self._session(session_id)
        session.transition(SessionState.PAUSED)
        self.recorder.pause()
        self.storage.save_session(session)
        publish_teach_event("recording_paused", session_id=session.id, title="Teach recording paused", status="paused")
        return session

    def resume(self, session_id: str | None = None) -> TeachSession:
        session = self._session(session_id)
        session.transition(SessionState.RECORDING)
        self.recorder.resume()
        self.storage.save_session(session)
        publish_teach_event("recording_resumed", session_id=session.id, title="Teach recording resumed")
        return session

    def explain(self, text: str, session_id: str | None = None) -> TeachSession:
        session = self._session(session_id)
        if session.state not in {SessionState.RECORDING, SessionState.PAUSED}:
            raise TeachError("Explanations can only be added to an active recording.")
        session.explanations.append(Explanation(text=text))
        session.audit_trail.append(AuditEntry(action="explanation.added", detail="Typed explanation recorded."))
        self.storage.save_session(session)
        return session

    def record_event(self, event: RecordedEvent) -> None:
        session = self.storage.load_session(event.session_id)
        if session.state != SessionState.RECORDING:
            raise TeachError("Events may only be captured while recording.")
        if event.source.value not in self.settings.event_sources:
            raise TeachError(f"Event source is disabled: {event.source.value}")
        if event.application.id in self.settings.excluded_applications:
            raise TeachError("The active application is excluded from Teach Mode.")
        if (
            self.settings.allowed_applications
            and event.application.id not in self.settings.allowed_applications
        ):
            raise TeachError("The active application is outside the Teach Mode application allowlist.")
        domain = str(event.context.get("domain", "")).lower()
        if domain and any(domain == item or domain.endswith(f".{item}") for item in self.settings.excluded_domains):
            raise TeachError("The active browser domain is excluded from Teach Mode.")
        if event.source.value == "voice" and not self.settings.voice_enabled:
            raise TeachError("Voice recording is disabled.")
        if event.source.value == "filesystem":
            raw_path = event.data.get("path")
            selected = bool(event.context.get("user_selected", False))
            allowed = False
            if isinstance(raw_path, str):
                candidate = Path(raw_path).expanduser().resolve()
                allowed = any(
                    candidate == root.expanduser().resolve()
                    or candidate.is_relative_to(root.expanduser().resolve())
                    for root in self.settings.recording_allowed_paths
                )
            if not selected and not allowed:
                raise TeachError(
                    "Filesystem event is outside recording allowlists and was not explicitly user-selected."
                )
        payload, findings = self.redactor.redact(event.data)
        event.data = payload
        if findings or event.sensitive:
            event.sensitive = True
            event.redactions = sorted(set(event.redactions + findings))
            session.sensitive_fields.extend(item for item in findings if item not in session.sensitive_fields)
            publish_teach_event(
                "sensitive_value_redacted",
                session_id=session.id,
                title="Sensitive value redacted",
                metadata={"finding_types": findings},
            )
        self.storage.append_raw_event(event)
        session.raw_event_count += 1
        if event.application.id and event.application.id not in session.active_applications:
            session.active_applications.append(event.application.id)
        self.storage.save_session(session)
        publish_teach_event("event_captured", session_id=session.id, title="Semantic event captured")

    def stop(self, session_id: str | None = None) -> tuple[TeachSession, ManaFlow]:
        session = self._session(session_id)
        session = self.monitor.stop(session)
        session.transition(SessionState.COMPILING)
        session.compilation_status = "running"
        self.storage.save_session(session)
        self.recorder.stop()
        publish_teach_event("compilation_started", session_id=session.id, title="Compiling demonstrated workflow")
        events = self.normalizer.normalize(self.storage.load_events(session.id))
        self.storage.save_normalized_events(session.id, events)
        flow = self.compiler.compile(session, events)
        session.normalized_event_count = len(events)
        session.detected_inputs = sorted(flow.inputs)
        session.proposed_verification_rules = [item.model_dump(mode="json") for item in flow.verify]
        session.generated_flow_id = flow.id
        session.compilation_status = "draft"
        session.transition(SessionState.AWAITING_REVIEW)
        self.storage.save_session(session)
        # Draft persistence supports crash recovery; activation still requires review/acceptance.
        self.storage.save_flow(flow)
        publish_teach_event(
            "review_required",
            session_id=session.id,
            flow_id=flow.id,
            title="Draft workflow is ready for review",
            status="waiting",
            metadata={"steps": len(flow.steps), "inputs": sorted(flow.inputs), "verification_rules": len(flow.verify)},
        )
        return session, flow

    def accept(self, flow_id: str, *, explicit_unverified_acceptance: bool = False) -> ManaFlow:
        flow = self.storage.load_flow(flow_id)
        if flow.status != "verified" and not explicit_unverified_acceptance:
            raise TeachError("Flow has not passed verification; explicit unverified acceptance is required.")
        flow.status = "active"
        flow.version += 1
        flow.updated_at = datetime.now(timezone.utc)
        self.storage.save_flow(flow)
        if flow.source_session_id:
            session = self.storage.load_session(flow.source_session_id)
            if session.state in {SessionState.AWAITING_REVIEW, SessionState.VERIFIED}:
                session.transition(SessionState.SAVED)
                self.storage.save_session(session)
        publish_teach_event("flow_saved", flow_id=flow.id, title="Teach workflow saved", status="completed")
        return flow

    def replay(
        self,
        flow_id: str,
        *,
        version: int | None = None,
        mode: str = "dry_run",
        inputs: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ReplayResult:
        flow = self.storage.load_flow(flow_id, version)
        if flow.status == "imported_pending" and mode != "dry_run":
            raise TeachError("Imported flows must pass a dry run and be explicitly activated first.")
        publish_teach_event("replay_started", flow_id=flow.id, title=f"Replaying {flow.name}", metadata={"mode": mode})
        result = self.replay_executor.replay(flow, mode=mode, inputs=inputs, context=context)
        if result.verification_status == "verified":
            flow.status = "verified"
            flow.statistics.verified_replays += 1
            flow.statistics.successful_replays += 1
            flow.statistics.last_verified_at = result.finished_at
            event = "verification_passed"
            status = "completed"
        elif result.verification_status == "failed":
            flow.statistics.failed_replays += 1
            event = "verification_failed"
            status = "failed"
        else:
            event = "verification_failed"
            status = "waiting"
        self.storage.save_flow(flow)
        publish_teach_event(event, flow_id=flow.id, title=f"Replay result: {result.verification_status}", status=status)
        return result

    def repair(self, flow_id: str, step_id: str, candidate: SelectorCandidate) -> ManaFlow:
        flow = TargetedSelectorRepair().repair(self.storage.load_flow(flow_id), step_id, candidate)
        self.storage.save_flow(flow)
        publish_teach_event("selector_repaired", flow_id=flow.id, title=f"Repaired selector for {step_id}")
        return flow

    def cancel(self, session_id: str | None = None) -> TeachSession:
        session = self._session(session_id)
        session = self.monitor.stop(session)
        session.transition(SessionState.CANCELLED)
        self.recorder.stop()
        self.storage.save_session(session)
        publish_teach_event("recording_cancelled", session_id=session.id, title="Teach recording cancelled", status="cancelled")
        return session

    def export(self, flow_id: str, destination: str | Path) -> Path:
        path = self.packager.export(self.storage.load_flow(flow_id), destination)
        publish_teach_event("package_exported", flow_id=flow_id, title="Flow package exported", status="completed")
        return path

    def import_package(self, package: str | Path) -> ManaFlow:
        flow = self.packager.import_package(package)
        try:
            self.storage.load_flow(flow.id)
        except TeachError:
            pass
        else:
            raise TeachError(
                f"Import would overwrite existing flow {flow.id}; rename the package flow before importing."
            )
        self.storage.save_flow(flow)
        publish_teach_event("package_imported", flow_id=flow.id, title="Untrusted flow imported; dry run required", status="waiting")
        return flow

    def flow_card(self, flow_id: str, *, estimated_minutes_saved: int = 1) -> FlowCard:
        flow = self.storage.load_flow(flow_id)
        session = self.storage.load_session(flow.source_session_id) if flow.source_session_id else None
        duration = 0
        if session and session.started_at and session.stopped_at:
            duration = max(0, int((session.stopped_at - session.started_at).total_seconds()))
        attempts = flow.statistics.successful_replays + flow.statistics.failed_replays
        rate = flow.statistics.successful_replays / attempts if attempts else 0
        copy = (
            f"I taught Mana to {flow.name.lower()} in {duration} seconds.\n\n"
            f"{len(flow.required_applications)} applications · {len(flow.steps)} actions · "
            f"{rate:.0%} replay success · {estimated_minutes_saved} minutes saved weekly"
        )
        return FlowCard(
            title=flow.name,
            demonstration_duration_seconds=duration,
            application_count=len(flow.required_applications),
            action_count=len(flow.steps),
            verified_replays=flow.statistics.verified_replays,
            successful_replays=flow.statistics.successful_replays,
            success_rate=round(rate, 4),
            estimated_minutes_saved_per_run=estimated_minutes_saved,
            share_copy=copy,
        )

    def doctor(self) -> dict[str, Any]:
        report = doctor_report(browser_enabled=self.settings.browser_capture, voice_enabled=self.settings.voice_enabled)
        statuses = grant_status(self.grants)
        report["grants"] = [item.model_dump(mode="json") for item in statuses]
        desktop_available = all(
            item.mana_granted and item.available and item.os_granted is not False
            for item in statuses
        )
        report["recorders"]["native_desktop"] = {
            "available": desktop_available,
            "reason": (
                ""
                if desktop_available
                else "Native desktop recording requires every Teach grant, optional dependency, and OS permission."
            ),
        }
        if not desktop_available:
            report["limitations"].append(report["recorders"]["native_desktop"]["reason"])
        return report

    def _session(self, session_id: str | None) -> TeachSession:
        return self.storage.load_session(session_id) if session_id else self.active_session()
