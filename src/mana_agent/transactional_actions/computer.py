"""Transactional adapter for exact computer-control actions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from mana_agent.integrations.computer_control.models import ComputerAction
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
        action_payload = self.computer_action.model_dump(mode="json")
        material = json.dumps(action_payload, sort_keys=True, default=str)
        target = self.computer_action.target.model_dump(mode="json", exclude_none=True)
        target_label = next((str(value) for value in target.values() if value), self.computer_action.capability)
        return ActionIntent(
            parent_task_id=self.computer_action.source_decision_id,
            actor="computer_control",
            originating_agent="model_tool",
            tool_name="computer",
            operation_name=self.computer_action.operation,
            target_resources=[target_label],
            normalized_arguments={
                "computer_action": action_payload,
                "session_id": self.session_id,
                "client_type": self.client_type,
            },
            requested_capabilities=[self.computer_action.permission_scope],
            expected_side_effects=[f"computer control: {self.computer_action.operation}"],
            data_disclosure=DataDisclosure.NONE,
            blast_radius=BlastRadius.SINGLE_RESOURCE,
            reversibility=Reversibility.UNKNOWN,
            idempotency_key="computer:" + hashlib.sha256(material.encode("utf-8")).hexdigest(),
            verification_plan=["record the computer-control provider completion receipt"],
            compensation_strategy="Computer-control actions do not have a general safe compensation.",
        )

    def preview(self, action: ActionIntent) -> ActionPreview:
        return ActionPreview(
            summary=f"Computer control: {self.computer_action.operation}",
            resources=[{"target": item, "operation": self.computer_action.operation} for item in action.target_resources],
            exact_invocation={
                "operation": self.computer_action.operation,
                "target": self.computer_action.target.model_dump(mode="json", exclude_none=True),
            },
            expected_side_effects=action.expected_side_effects,
            risks=["This exact computer action runs once after approval."],
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
        return VerificationEvidence(
            complete=complete,
            summary=str(result.get("message") or "Computer-control completion receipt was unavailable."),
            checks=[{"check": "computer_control_result", "state": result.get("state")}],
        )


def adapter_for_stored_action(action: ActionIntent) -> ComputerActionAdapter:
    payload = action.normalized_arguments
    return ComputerActionAdapter(
        action=ComputerAction.model_validate(payload["computer_action"]),
        session_id=str(payload["session_id"]),
        client_type=str(payload["client_type"]),
    )
