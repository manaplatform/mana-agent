"""Transactional adapter for exact computer-control actions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from mana_agent.integrations.computer_control.models import ComputerAction
from mana_agent.integrations.computer_control.policy import ACTION_SPECS
from mana_agent.integrations.computer_control.service import (
    ComputerControlService,
    default_computer_control_service,
)

from .adapters import ActionAdapter
from .models import (
    ActionIntent,
    ActionPreview,
    BlastRadius,
    DataDisclosure,
    Reversibility,
    VerificationEvidence,
)


def _run(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result: list[Any] = []
    failure: list[BaseException] = []
    thread = threading.Thread(
        target=lambda: _collect(coro, result, failure), daemon=True,
    )
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    return result[0]


def _collect(coro: Any, result: list[Any], failure: list[BaseException]) -> None:
    try:
        result.append(asyncio.run(coro))
    except BaseException as exc:  # surfaced by the synchronous adapter boundary
        failure.append(exc)


class ComputerActionAdapter(ActionAdapter):
    """Persist and execute one model-selected computer action after exact approval."""

    def __init__(
        self,
        *,
        action: ComputerAction,
        session_id: str,
        client_type: str,
        service: ComputerControlService | None = None,
    ) -> None:
        self.computer_action = action
        self.session_id = session_id
        self.client_type = client_type
        self.service = service or default_computer_control_service()

    def build_intent(self) -> ActionIntent:
        full_action_payload = self.computer_action.model_dump(mode="json")
        action_payload = json.loads(json.dumps(full_action_payload))
        if self.computer_action.operation == "screen_recording.capture":
            output_path = str((action_payload.get("arguments") or {}).get("output_path") or "")
            if output_path:
                action_payload["arguments"]["output_path"] = "<protected:" + hashlib.sha256(output_path.encode("utf-8")).hexdigest()[:24] + ">"
        material = json.dumps(full_action_payload, sort_keys=True, default=str)
        target = self.computer_action.target.model_dump(mode="json", exclude_none=True)
        target_label = next((str(value) for value in target.values() if value), self.computer_action.capability)
        spec = ACTION_SPECS[self.computer_action.operation]
        context = self.computer_action.execution_context
        parent_task_id = (
            context.task_id
            if context is not None and context.task_id
            else f"computer-session:{self.session_id}"
        )
        disclosure = DataDisclosure(spec.data_disclosure)
        reversibility = Reversibility(spec.reversibility)
        blast_radius = BlastRadius(spec.blast_radius)
        return ActionIntent(
            parent_task_id=parent_task_id,
            actor="computer_control",
            originating_agent="model_tool",
            tool_name="computer",
            operation_name=self.computer_action.operation,
            target_resources=[target_label],
            normalized_arguments={
                "computer_action": action_payload,
                "session_id": self.session_id,
                "client_type": self.client_type,
                "source_decision_id": self.computer_action.source_decision_id,
                "execution_context": context.redacted() if context else {},
            },
            requested_capabilities=[self.computer_action.permission_scope],
            expected_side_effects=[f"computer control: {self.computer_action.operation}"],
            data_disclosure=disclosure,
            blast_radius=blast_radius,
            reversibility=reversibility,
            idempotency_key="computer:" + hashlib.sha256(material.encode("utf-8")).hexdigest(),
            verification_plan=["record the computer-control provider completion receipt", "independently verify declared artifact metadata when present"],
            compensation_strategy="Computer-control actions do not have a general safe compensation.",
        )

    def protected_action_context(self) -> dict[str, Any]:
        if self.computer_action.operation != "screen_recording.capture":
            return {}
        return {"computer_action": self.computer_action.model_dump(mode="json")}

    def preview(self, action: ActionIntent) -> ActionPreview:
        spec = ACTION_SPECS[self.computer_action.operation]
        recording = self.computer_action.operation == "screen_recording.capture"
        return ActionPreview(
            summary=f"Computer control: {self.computer_action.operation}",
            resources=[{"target": item, "operation": self.computer_action.operation} for item in action.target_resources],
            exact_invocation={
                "operation": self.computer_action.operation,
                "target": self.computer_action.target.model_dump(mode="json", exclude_none=True),
                "arguments": {
                    **self.computer_action.arguments,
                    **({"output_path": "<protected-output-path>"} if recording else {}),
                },
            },
            expected_side_effects=action.expected_side_effects,
            risks=[
                "This exact computer action runs once after approval.",
                *(["Screen content is confidential. The generated file can be deleted, but capture cannot be completely undone."] if recording else []),
            ],
            externally_visible=spec.externally_visible,
            potentially_billable=spec.potentially_billable,
        )

    def execute(self, action: ActionIntent) -> dict[str, Any]:
        result = _run(self.service.execute(
            self.computer_action,
            session_id=self.session_id,
            client_type=self.client_type,
            transactional_authorized=True,
        ))
        return result.model_dump(mode="json")

    def verify(self, action: ActionIntent, result: dict[str, Any]) -> VerificationEvidence:
        complete = str(result.get("state") or "") == "completed"
        if self.computer_action.operation == "screen_recording.capture":
            required = {"artifact_path", "artifact_sha256", "artifact_bytes", "duration_seconds", "provider"}
            complete = complete and required.issubset(result.get("data") or {}) and int((result.get("data") or {}).get("artifact_bytes") or 0) > 0
        return VerificationEvidence(
            complete=complete,
            summary=str(result.get("message") or "Computer-control completion receipt was unavailable."),
            checks=[{"check": "computer_control_result", "state": result.get("state")}],
        )

    def persistable_result(self, result: dict[str, Any]) -> dict[str, Any]:
        persisted = super().persistable_result(result)
        if self.computer_action.operation != "screen_recording.capture":
            return persisted
        data = dict(persisted.get("data") or {})
        artifact_path = str(data.get("artifact_path") or "")
        if artifact_path:
            data["artifact_path"] = "<protected:" + hashlib.sha256(artifact_path.encode("utf-8")).hexdigest()[:24] + ">"
        persisted["data"] = data
        return persisted


def adapter_for_stored_action(action: ActionIntent, *, protected_context: dict[str, Any] | None = None) -> ComputerActionAdapter:
    payload = action.normalized_arguments
    computer_action = payload["computer_action"]
    if protected_context and isinstance(protected_context.get("computer_action"), dict):
        computer_action = protected_context["computer_action"]
    return ComputerActionAdapter(
        action=ComputerAction.model_validate(computer_action),
        session_id=str(payload["session_id"]),
        client_type=str(payload["client_type"]),
    )
