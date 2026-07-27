"""Deterministic compilation after an explicit recording decision."""

from __future__ import annotations

import re

from .models import (
    FlowStep,
    ManaFlow,
    RecordedEvent,
    SessionState,
    TeachError,
    TeachSession,
    VerificationRule,
)
from .parameterizer import Parameterizer, replace_values


PERMISSION_BY_SOURCE = {
    "browser": "computer.browser.control",
    "accessibility": "computer.apps.control",
    "keyboard": "computer.apps.control",
    "application": "computer.apps.control",
    "filesystem": "computer.files.write",
    "voice": "computer.microphone.read",
}
SENSITIVE_ACTION_MARKERS = ("send", "delete", "purchase", "publish", "pay", "submit", "post")


class FlowCompiler:
    def __init__(self, parameterizer: Parameterizer | None = None):
        self.parameterizer = parameterizer or Parameterizer()

    def compile(self, session: TeachSession, events: list[RecordedEvent]) -> ManaFlow:
        if session.state != SessionState.COMPILING:
            raise TeachError("The session must be in compiling state.")
        inputs, replacements = self.parameterizer.infer(session, events)
        steps: list[FlowStep] = []
        permissions: set[str] = set()
        applications: set[str] = set()
        for index, event in enumerate(events, start=1):
            permission = PERMISSION_BY_SOURCE[event.source.value]
            permissions.add(permission)
            if event.application.id:
                applications.add(event.application.id)
            selectors = event.target.selectors
            confidence = selectors[0].confidence if selectors else (0.45 if event.fallback_position else 0.75)
            step_id = f"step-{index:03d}"
            arguments = replace_values(dict(event.data), replacements)
            arguments["target"] = event.target.model_dump(mode="json", exclude_none=True, exclude={"selectors"})
            if event.context:
                arguments["context"] = event.context
            if event.fallback_position:
                arguments["fallback_position"] = event.fallback_position.model_dump()
            sensitive_action = any(marker in event.action.lower() for marker in SENSITIVE_ACTION_MARKERS)
            steps.append(
                FlowStep(
                    id=step_id,
                    action=f"{event.source.value}.{event.action}",
                    **{"with": arguments},
                    depends_on=[steps[-1].id] if steps else [],
                    permissions=[permission],
                    confidence=confidence,
                    requires_review=confidence < 0.65,
                    requires_confirmation=sensitive_action,
                    selectors=selectors,
                    provenance=[event.event_id],
                    checkpoint=event.action in {"navigation", "open", "save", "download", "export"},
                )
            )
        rules = self._verification_rules(events, steps)
        slug = re.sub(r"[^a-z0-9]+", "-", session.task_name.lower()).strip("-")[:72] or session.id
        return ManaFlow(
            id=slug,
            name=session.task_name,
            description=f"Workflow learned from Teach Mode session {session.id}.",
            source_session_id=session.id,
            inputs=inputs,
            permissions=sorted(permissions),
            supported_platforms=[str(session.platform_metadata.get("platform", "unknown"))],
            required_applications=sorted(applications),
            required_capabilities=sorted(permissions),
            steps=steps,
            verify=rules,
        )

    def _verification_rules(
        self, events: list[RecordedEvent], steps: list[FlowStep]
    ) -> list[VerificationRule]:
        rules: list[VerificationRule] = []
        for event, step in zip(events, steps):
            if event.action in {"download", "export", "file_created"} and event.data.get("path"):
                rules.append(
                    VerificationRule(
                        id=f"verify-{step.id}",
                        type="file.exists",
                        arguments={"path": event.data["path"]},
                        step_id=step.id,
                    )
                )
            elif event.action in {"navigation", "navigate"} and event.context.get("url_pattern"):
                rules.append(
                    VerificationRule(
                        id=f"verify-{step.id}",
                        type="browser.url_matches",
                        arguments={"pattern": event.context["url_pattern"]},
                        step_id=step.id,
                    )
                )
            elif event.data.get("success_text"):
                rules.append(
                    VerificationRule(
                        id=f"verify-{step.id}",
                        type="ui.text_visible",
                        arguments={"text": event.data["success_text"]},
                        step_id=step.id,
                    )
                )
        return rules
